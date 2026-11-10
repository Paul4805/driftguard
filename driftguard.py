#!/usr/bin/env python3
"""
DriftGuard: drift-aware single-object tracking with a custom KCF core.

The tracker in this file is intentionally self-contained and does NOT use
``cv2.TrackerKCF_create``.  OpenCV's Python KCF wrapper hides the correlation
response map, which is exactly what DriftGuard needs for Peak-to-Sidelobe
Ratio (PSR) based failure detection.  The implementation below is a compact
NumPy FFT implementation of the KCF/CSK tracker described in:

    J. F. Henriques, R. Caseiro, P. Martins, J. Batista,
    "High-Speed Tracking with Kernelized Correlation Filters", TPAMI 2015.

Author: DriftGuard reference implementation
Python: 3.9+
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import matplotlib

matplotlib.use("Agg")  # Makes the program work on headless machines as well.
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.ndimage import gaussian_filter  # noqa: E402

BBox = Tuple[float, float, float, float]  # x, y, w, h

# ---------------------------------------------------------------------------
# Small geometry / image helpers
# ---------------------------------------------------------------------------


def bbox_center(bbox: BBox) -> Tuple[float, float]:
    """Return the geometric centre of an ``(x, y, w, h)`` box."""
    x, y, w, h = bbox
    return x + w / 2.0, y + h / 2.0


def clip_bbox(bbox: BBox, frame_w: int, frame_h: int, min_size: float = 8.0) -> BBox:
    """Clip a box so that it stays inside the image while preserving its size.

    If the object is near a border (as the vtest pedestrian is), the box is
    translated inwards instead of being truncated.  This keeps the KCF
    template and the colour reference at a stable size.
    """
    x, y, w, h = bbox
    w = max(min_size, float(w))
    h = max(min_size, float(h))
    w = min(w, float(frame_w))
    h = min(h, float(frame_h))

    cx = x + w / 2.0
    cy = y + h / 2.0
    cx = min(max(cx, w / 2.0), frame_w - w / 2.0)
    cy = min(max(cy, h / 2.0), frame_h - h / 2.0)
    return (cx - w / 2.0, cy - h / 2.0, w, h)


def bbox_iou(a: BBox, b: BBox) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return float(inter / union) if union > 0 else 0.0


def safe_crop(frame: np.ndarray, bbox: BBox) -> np.ndarray:
    """Crop with border replication; always returns a non-empty patch."""
    frame_h, frame_w = frame.shape[:2]
    x, y, w, h = bbox
    x0 = int(math.floor(x))
    y0 = int(math.floor(y))
    x1 = int(math.ceil(x + w))
    y1 = int(math.ceil(y + h))

    pad_l = max(0, -x0)
    pad_t = max(0, -y0)
    pad_r = max(0, x1 - frame_w)
    pad_b = max(0, y1 - frame_h)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(frame_w, x1), min(frame_h, y1)

    if x1 <= x0 or y1 <= y0:
        return np.zeros((max(2, int(round(h))), max(2, int(round(w))), 3), np.uint8)

    patch = frame[y0:y1, x0:x1]
    if pad_l or pad_t or pad_r or pad_b:
        patch = cv2.copyMakeBorder(
            patch, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_REPLICATE
        )
    return patch


def to_gray(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return frame
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


# ---------------------------------------------------------------------------
# PSR
# ---------------------------------------------------------------------------


def compute_psr(response: np.ndarray, exclusion_window: int = 11) -> float:
    """Compute the Peak-to-Sidelobe Ratio of a correlation response map.

    Math
    ----
    For a response map R, let p = max(R) at location (py, px).  The sidelobe
    is every response value outside an ``exclusion_window x exclusion_window``
    square centred on the peak.  Then

        PSR = (p - mean(R_sidelobe)) / (std(R_sidelobe) + eps).

    A high, narrow peak means the correlation filter is confident.  When the
    target is occluded or leaves the search region, energy spreads over the
    map: the peak falls and the sidelobe variance rises, so PSR collapses.

    Parameters
    ----------
    response:
        2-D real correlation output (same shape as the model patch).
    exclusion_window:
        Width/height of the square around the peak excluded from the
        sidelobe statistics.  The presentation specifies 11 pixels.
    """
    r = np.asarray(response, dtype=np.float64)
    if r.size == 0 or not np.all(np.isfinite(r)):
        return 0.0

    peak_idx = int(np.argmax(r))
    py, px = np.unravel_index(peak_idx, r.shape)
    peak = float(r[py, px])

    half = max(0, int(exclusion_window) // 2)
    y0 = max(0, py - half)
    y1 = min(r.shape[0], py + half + 1)
    x0 = max(0, px - half)
    x1 = min(r.shape[1], px + half + 1)

    mask = np.ones(r.shape, dtype=bool)
    mask[y0:y1, x0:x1] = False
    sidelobe = r[mask]
    if sidelobe.size < 2:
        return 0.0

    mean_s = float(sidelobe.mean())
    std_s = float(sidelobe.std())
    return float((peak - mean_s) / (std_s + 1e-12))


def subpixel_peak_offset(
    response: np.ndarray, py: int, px: int
) -> Tuple[float, float]:
    """Quadratic interpolation of a response-map peak (in model pixels)."""
    dy = 0.0
    dx = 0.0
    h, w = response.shape

    if 0 < px < w - 1:
        left = float(response[py, px - 1])
        centre = float(response[py, px])
        right = float(response[py, px + 1])
        denom = 2.0 * centre - left - right
        if abs(denom) > 1e-9:
            dx = 0.5 * (right - left) / denom

    if 0 < py < h - 1:
        top = float(response[py - 1, px])
        centre = float(response[py, px])
        bottom = float(response[py + 1, px])
        denom = 2.0 * centre - top - bottom
        if abs(denom) > 1e-9:
            dy = 0.5 * (bottom - top) / denom

    return dx, dy


# ---------------------------------------------------------------------------
# Custom KCF tracker (FFT based, returns the raw response map)
# ---------------------------------------------------------------------------


class KCFTracker:
    """A lightweight grayscale KCF tracker with an exposed response map.

    The model is trained in the Fourier domain.  For a feature patch ``x``:

    * Gaussian kernel correlation:
          k(x, z) = exp(-||x - z||^2 / (N * sigma^2))
      where the squared distance is computed efficiently using the inverse
      FFT of ``X * conj(Z)``.
    * Ridge-regression solution in the Fourier domain:
          alpha_f = Y_f / (K_f + lambda)
      with ``Y_f`` the FFT of a centred 2-D Gaussian label.
    * Detection:
          response = real(IFFT(alpha_f * K_f))
      The peak location gives the object translation relative to the previous
      search window; the complete map is returned for PSR calculation.

    The feature normalization and the padding/window choices follow the
    standard KCF/CSK formulation.  Grayscale plus the two Lab chroma channels
    are used as raw features.  The chroma channels are a very cheap way to
    stop a dark target from being confused with a dark pole/sign during
    occlusion, while still keeping the FFT model small enough for real time.
    """

    def __init__(
        self,
        *,
        padding: float = 2.5,
        template_size: int = 96,
        output_sigma_factor: float = 0.125,
        sigma: float = 0.2,
        regularization: float = 1e-4,
        learning_rate: float = 0.075,
        use_color: bool = True,
    ) -> None:
        self.padding = float(padding)
        self.template_size = int(max(32, template_size))
        self.output_sigma_factor = float(output_sigma_factor)
        self.sigma = float(sigma)
        self.regularization = float(regularization)
        self.learning_rate = float(learning_rate)
        self.use_color = bool(use_color)

        self._initialized = False
        self._roi: BBox = (0.0, 0.0, 1.0, 1.0)
        self._tmpl_sz: Tuple[int, int] = (64, 64)  # model width, height
        self._scale: float = 1.0  # image pixels per model pixel
        self._hann: Optional[np.ndarray] = None
        self._label_f: Optional[np.ndarray] = None
        self._alphaf: Optional[np.ndarray] = None
        self._model_f: Optional[np.ndarray] = None
        self._n_channels: int = 1

    # -- public API ---------------------------------------------------------

    @property
    def initialized(self) -> bool:
        return self._initialized

    def initialize(self, frame: np.ndarray, bbox: BBox) -> None:
        """Initialise / re-initialise the filter on ``bbox`` in ``frame``."""
        frame_h, frame_w = frame.shape[:2]
        self._roi = clip_bbox(bbox, frame_w, frame_h)

        # The model patch is a resized version of the padded search window.
        # Its largest side is fixed at template_size, which bounds FFT cost.
        padded_w = self._roi[2] * self.padding
        padded_h = self._roi[3] * self.padding
        self._scale = max(padded_w, padded_h) / float(self.template_size)
        self._scale = max(self._scale, 1e-6)

        tmpl_w = max(16, int(round(padded_w / self._scale)))
        tmpl_h = max(16, int(round(padded_h / self._scale)))
        # Even sizes make the response centre unambiguous and FFT-friendly.
        tmpl_w = max(16, tmpl_w - (tmpl_w % 2))
        tmpl_h = max(16, tmpl_h - (tmpl_h % 2))
        self._tmpl_sz = (tmpl_w, tmpl_h)

        self._hann = self._create_hanning(tmpl_h, tmpl_w)
        self._label_f = self._create_gaussian_label_f(tmpl_h, tmpl_w)

        x = self._get_features(frame, self._roi)
        self._n_channels = int(x.shape[0])
        xf = np.fft.fft2(x)
        self._model_f = xf

        k = self._gaussian_correlation_freq(xf, xf)
        kf = np.fft.fft2(k)
        self._alphaf = self._label_f / (kf + self.regularization)
        self._initialized = True

    def detect(self, frame: np.ndarray) -> Tuple[BBox, np.ndarray]:
        """Detect the target and return ``(new_bbox, response_map)``.

        The model is *not* updated here.  DriftGuard decides whether an update
        is safe after it has inspected the PSR of the returned response.
        """
        if not self._initialized or self._model_f is None or self._alphaf is None:
            raise RuntimeError("KCFTracker.detect() called before initialize()")

        z = self._get_features(frame, self._roi)
        zf = np.fft.fft2(z)
        k = self._gaussian_correlation_freq(self._model_f, zf)
        response = np.fft.ifft2(self._alphaf * np.fft.fft2(k)).real

        py, px = np.unravel_index(int(np.argmax(response)), response.shape)
        dx_model, dy_model = subpixel_peak_offset(response, int(py), int(px))
        # The response is centred on the old target position.
        offset_x_model = (float(px) + dx_model) - self._tmpl_sz[0] / 2.0
        offset_y_model = (float(py) + dy_model) - self._tmpl_sz[1] / 2.0
        # np.fft.ifft2(X * conj(Z)) returns the complex-conjugate
        # cross-correlation; its peak is on the opposite side of the true
        # translation.  Negating here maps the response peak back to image
        # motion.  (The same convention appears in the reference KCF code.)
        offset_x = -offset_x_model * self._scale
        offset_y = -offset_y_model * self._scale

        cx, cy = bbox_center(self._roi)
        new_w, new_h = self._roi[2], self._roi[3]
        new_bbox = (
            cx + offset_x - new_w / 2.0,
            cy + offset_y - new_h / 2.0,
            new_w,
            new_h,
        )
        frame_h, frame_w = frame.shape[:2]
        new_bbox = clip_bbox(new_bbox, frame_w, frame_h)

        self._roi = new_bbox
        return new_bbox, response

    def update_model(self, frame: np.ndarray, bbox: Optional[BBox] = None) -> None:
        """Update the KCF model using the appearance inside ``bbox``.

        DriftGuard only calls this on healthy frames.  On a drift flag the
        call is deliberately skipped, preventing the filter from learning the
        occluder/background.
        """
        if not self._initialized:
            raise RuntimeError("KCFTracker.update_model() called before initialize()")

        if bbox is not None:
            frame_h, frame_w = frame.shape[:2]
            self._roi = clip_bbox(bbox, frame_w, frame_h)

        x = self._get_features(frame, self._roi)
        xf = np.fft.fft2(x)
        k = self._gaussian_correlation_freq(xf, xf)
        kf = np.fft.fft2(k)
        alphaf_new = self._label_f / (kf + self.regularization)

        lr = self.learning_rate
        assert self._model_f is not None and self._alphaf is not None
        self._model_f = (1.0 - lr) * self._model_f + lr * xf
        self._alphaf = (1.0 - lr) * self._alphaf + lr * alphaf_new

    def snapshot(self) -> Dict[str, object]:
        """Return a deep copy of the filter state.

        DriftGuard uses this to roll back a re-initialised model if the
        candidate does not survive the recovery confirmation window.  This is
        the safety net against model contamination.
        """
        return {
            "roi": tuple(self._roi),
            "tmpl_sz": tuple(self._tmpl_sz),
            "scale": float(self._scale),
            "hann": None if self._hann is None else self._hann.copy(),
            "label_f": None if self._label_f is None else self._label_f.copy(),
            "alphaf": None if self._alphaf is None else self._alphaf.copy(),
            "model_f": None if self._model_f is None else self._model_f.copy(),
            "n_channels": int(self._n_channels),
            "initialized": bool(self._initialized),
        }

    def restore(self, state: Dict[str, object]) -> None:
        """Restore a state produced by :meth:`snapshot`."""
        self._roi = state["roi"]  # type: ignore[assignment]
        self._tmpl_sz = state["tmpl_sz"]  # type: ignore[assignment]
        self._scale = float(state["scale"])
        self._hann = state["hann"]  # type: ignore[assignment]
        self._label_f = state["label_f"]  # type: ignore[assignment]
        self._alphaf = state["alphaf"]  # type: ignore[assignment]
        self._model_f = state["model_f"]  # type: ignore[assignment]
        self._n_channels = int(state["n_channels"])
        self._initialized = bool(state["initialized"])

    # -- KCF internals ------------------------------------------------------

    def _create_hanning(self, height: int, width: int) -> np.ndarray:
        """Separable Hann window, suppressing circular-boundary artefacts."""
        wy = 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(height) / max(1, height - 1)))
        wx = 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(width) / max(1, width - 1)))
        return np.outer(wy, wx).astype(np.float32)

    def _create_gaussian_label_f(self, height: int, width: int) -> np.ndarray:
        """FFT of the centred Gaussian regression label used by KCF."""
        # The target occupies 1/padding of the search window, so the label
        # bandwidth is scaled accordingly (same convention as KCFcpp).
        output_sigma = (
            math.sqrt(float(height * width))
            / self.padding
            * self.output_sigma_factor
        )
        output_sigma = max(output_sigma, 1e-3)

        y, x = np.ogrid[0:height, 0:width]
        y = (y - height / 2.0) ** 2
        x = (x - width / 2.0) ** 2
        label = np.exp(-0.5 * (x + y) / (output_sigma * output_sigma))
        return np.fft.fft2(label.astype(np.float32))

    def _get_features(self, frame: np.ndarray, bbox: BBox) -> np.ndarray:
        """Extract the padded, resized, windowed feature patch.

        Returns ``(C, H, W)`` where C is 3 for colour input (L, a, b) and 1
        for true grayscale input.  All channels are in [-0.5, 0.5].
        """
        if self._hann is None:
            raise RuntimeError("KCF model has not been initialised")

        cx, cy = bbox_center(bbox)
        patch_w = max(8, int(round(self._scale * self._tmpl_sz[0])))
        patch_h = max(8, int(round(self._scale * self._tmpl_sz[1])))

        patch = cv2.getRectSubPix(
            frame, (patch_w, patch_h), (float(cx), float(cy))
        )
        if patch.shape[1] != self._tmpl_sz[0] or patch.shape[0] != self._tmpl_sz[1]:
            patch = cv2.resize(
                patch, self._tmpl_sz, interpolation=cv2.INTER_LINEAR
            )

        if self.use_color and patch.ndim == 3 and patch.shape[2] == 3:
            gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
            lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB)
            channels = [
                gray.astype(np.float32),
                lab[:, :, 1].astype(np.float32),
                lab[:, :, 2].astype(np.float32),
            ]
            features = np.stack(channels, axis=0) / 255.0 - 0.5
        else:
            features = to_gray(patch).astype(np.float32)[None, :, :] / 255.0 - 0.5

        features *= self._hann[None, :, :]
        return features.astype(np.float32)

    def _gaussian_correlation_freq(
        self, xf: np.ndarray, yf: np.ndarray
    ) -> np.ndarray:
        """Gaussian kernel correlation from two Fourier-domain patches.

        ``||x - y||^2 = ||x||^2 + ||y||^2 - 2 * x*y``.  The cross term is
        obtained with one inverse FFT, summed over feature channels.  The
        result is fftshifted so the zero lag (perfect match) is at the centre,
        matching the centred label.
        """
        channels = 1 if xf.ndim == 2 else xf.shape[0]
        n = float(xf.shape[-2] * xf.shape[-1])
        if channels == 1:
            xyf = xf * np.conj(yf)
            xx = float(np.sum(np.abs(xf) ** 2) / n)
            yy = float(np.sum(np.abs(yf) ** 2) / n)
        else:
            xyf = np.sum(xf * np.conj(yf), axis=0)
            xx = float(np.sum(np.abs(xf) ** 2) / n)
            yy = float(np.sum(np.abs(yf) ** 2) / n)

        c = np.fft.ifft2(xyf).real
        c = np.fft.fftshift(c)

        d = (xx + yy - 2.0 * c) / (n * channels)
        np.maximum(d, 0.0, out=d)
        return np.exp(-d / (self.sigma * self.sigma)).astype(np.float32)


# ---------------------------------------------------------------------------
# Recovery: multi-template matching + appearance verification
# ---------------------------------------------------------------------------


@dataclass
class ReferenceAppearance:
    """One trusted appearance snapshot of the target.

    The colour descriptor is split vertically (upper/lower body).  A global
    histogram alone can confuse pedestrians with the same jacket but different
    trousers; the lower-body histogram is a cheap and effective identity cue
    for surveillance clips like vtest.avi.
    """

    template_gray: np.ndarray
    hist: np.ndarray
    hist_upper: np.ndarray
    hist_lower: np.ndarray
    size: Tuple[int, int]
    frame_idx: int


@dataclass
class RecoveryResult:
    bbox: BBox
    ncc: float
    hist_similarity: float
    ssim: float
    score: float
    candidates_considered: int
    foreground_diff: float = 0.0


class RecoveryModule:
    """Local re-detection followed by appearance verification.

    The module maintains a small bank of *verified* target appearance
    snapshots.  During a drift episode it searches a region around the last
    known good location with multi-scale normalised cross-correlation.  It
    keeps several non-maximum-suppressed peaks (not just the global maximum),
    then verifies every candidate using colour-histogram and SSIM similarity.
    This handles the common case where the strongest template-match peak is a
    distractor and the real target is the second or third peak.
    """

    def __init__(
        self,
        *,
        max_references: int = 3,
        scale_factors: Sequence[float] = (0.9, 1.0, 1.1),
        max_candidates: int = 8,
        min_ncc: float = 0.50,
        min_hist: float = 0.30,
        min_ssim: float = 0.48,
        min_total: float = 0.66,
        min_foreground_diff: float = 8.0,
        reference_interval: int = 12,
    ) -> None:
        self.max_references = int(max_references)
        self.scale_factors = tuple(float(s) for s in scale_factors)
        self.max_candidates = int(max_candidates)
        self.min_ncc = float(min_ncc)
        self.min_hist = float(min_hist)
        self.min_ssim = float(min_ssim)
        self.min_total = float(min_total)
        self.min_foreground_diff = float(min_foreground_diff)
        self.reference_interval = int(reference_interval)

        self.references: List[ReferenceAppearance] = []
        self._last_reference_frame = -10**9
        # Static-camera background captured on the first frame.  It lets the
        # verifier reject a candidate box that is actually a sign/pole/patch of
        # pavement: such a box barely differs from the background snapshot,
        # whereas a genuine pedestrian must differ from it.
        self._background_gray: Optional[np.ndarray] = None

    # -- reference bank -----------------------------------------------------

    @property
    def has_references(self) -> bool:
        return bool(self.references)

    def reset(self, frame: np.ndarray, bbox: BBox, frame_idx: int = 0) -> None:
        """Start a fresh reference bank from the initial target box."""
        self.references = []
        self._last_reference_frame = -10**9
        self._background_gray = to_gray(frame).copy()
        self.add_reference(frame, bbox, frame_idx, force=True)

    def add_reference(
        self, frame: np.ndarray, bbox: BBox, frame_idx: int, force: bool = False
    ) -> None:
        """Add a trusted snapshot, keeping the initial target and recent ones.

        Only healthy frames should call this method.  Snapshots taken during
        drift are what cause model contamination, so DriftGuard never adds
        them.
        """
        if not force and frame_idx - self._last_reference_frame < self.reference_interval:
            return

        patch = safe_crop(frame, bbox)
        if patch.size == 0:
            return
        gray = to_gray(patch)
        if gray.shape[0] < 8 or gray.shape[1] < 8:
            return

        hist = self._histogram(patch)
        mid = max(1, patch.shape[0] // 2)
        ref = ReferenceAppearance(
            template_gray=gray.copy(),
            hist=hist,
            hist_upper=self._histogram(patch[:mid]),
            hist_lower=self._histogram(patch[mid:]),
            size=(int(gray.shape[1]), int(gray.shape[0])),
            frame_idx=int(frame_idx),
        )
        self.references.append(ref)
        self._last_reference_frame = frame_idx

        # Keep the very first (hand-initialised) snapshot plus the newest
        # snapshots.  Discarding the oldest non-initial reference prevents
        # the bank from being dominated by long-outdated poses.
        if len(self.references) > self.max_references:
            if self.max_references >= 2:
                self.references = [self.references[0]] + self.references[
                    -(self.max_references - 1) :
                ]
            else:
                self.references = [self.references[-1]]

    def _histogram(self, bgr_patch: np.ndarray) -> np.ndarray:
        if bgr_patch.ndim == 2:
            bgr_patch = cv2.cvtColor(bgr_patch, cv2.COLOR_GRAY2BGR)
        hsv = cv2.cvtColor(bgr_patch, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist(
            [hsv],
            [0, 1, 2],
            None,
            [16, 16, 8],
            [0, 180, 0, 256, 0, 256],
        )
        cv2.normalize(hist, hist, alpha=1.0, beta=0.0, norm_type=cv2.NORM_L1)
        return hist

    def _foreground_diff(self, gray_patch: np.ndarray, bbox: BBox) -> float:
        """Mean absolute difference between a box and the first-frame scene.

        This is intentionally a *static-camera* cue, which matches vtest.avi
        and most surveillance clips.  For a moving camera it simply contributes
        a weak, non-blocking signal (the threshold can be set to zero).
        """
        if self._background_gray is None:
            return float("inf")
        bg = safe_crop(self._background_gray, bbox)
        if bg.shape != gray_patch.shape:
            bg = cv2.resize(
                bg,
                (gray_patch.shape[1], gray_patch.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
        return float(np.mean(cv2.absdiff(gray_patch, bg)))

    def appearance_metrics(
        self, frame: np.ndarray, bbox: BBox
    ) -> Tuple[float, float, float]:
        """Return ``(hist_similarity, ssim, foreground_diff)`` for a box."""
        patch = safe_crop(frame, bbox)
        if patch.size == 0 or not self.references:
            return 0.0, 0.0, 0.0
        hist = self._histogram(patch)
        mid = max(1, patch.shape[0] // 2)
        hist_upper = self._histogram(patch[:mid])
        hist_lower = self._histogram(patch[mid:])

        hist_sim = 0.0
        for ref in self.references:
            whole = float(cv2.compareHist(hist, ref.hist, cv2.HISTCMP_CORREL))
            upper = float(
                cv2.compareHist(hist_upper, ref.hist_upper, cv2.HISTCMP_CORREL)
            )
            lower = float(
                cv2.compareHist(hist_lower, ref.hist_lower, cv2.HISTCMP_CORREL)
            )
            hist_sim = max(hist_sim, 0.50 * whole + 0.25 * upper + 0.25 * lower)

        cand_gray = to_gray(patch)
        ssim_sim = 0.0
        for ref in self.references:
            ref_resized = cv2.resize(
                ref.template_gray,
                (cand_gray.shape[1], cand_gray.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
            ssim_sim = max(ssim_sim, _ssim(cand_gray, ref_resized))

        return (
            float(np.clip(hist_sim, -1.0, 1.0)),
            float(np.clip(ssim_sim, 0.0, 1.0)),
            self._foreground_diff(cand_gray, bbox),
        )

    def is_plausible_track(self, frame: np.ndarray, bbox: BBox) -> bool:
        """Conservative check used during the recovery confirmation window."""
        hist_sim, ssim_sim, fg_diff = self.appearance_metrics(frame, bbox)
        return (
            fg_diff >= self.min_foreground_diff
            and hist_sim >= 0.40
            and ssim_sim >= 0.25
        )

    # -- search / verify ----------------------------------------------------

    def find(
        self,
        frame: np.ndarray,
        last_good_bbox: BBox,
        lost_frames: int = 0,
        max_lost_frames: int = 40,
    ) -> Optional[RecoveryResult]:
        """Search around ``last_good_bbox`` and return a verified candidate."""
        if not self.references:
            return None

        frame_h, frame_w = frame.shape[:2]
        x, y, w, h = last_good_bbox
        cx, cy = bbox_center(last_good_bbox)

        # Expand the search window gradually if the target is not found.  This
        # is the timeout strategy: the last known state remains the anchor,
        # while the searched area grows until it covers the whole frame.
        growth = min(float(lost_frames) / 12.0, 1.8)
        radius_x = max(w * 2.0, 72.0) * (1.0 + growth)
        radius_y = max(h * 1.5, 72.0) * (1.0 + growth)
        if lost_frames >= max_lost_frames:
            radius_x = float(frame_w)
            radius_y = float(frame_h)

        rx0 = int(max(0, math.floor(cx - radius_x - w / 2.0)))
        ry0 = int(max(0, math.floor(cy - radius_y - h / 2.0)))
        rx1 = int(min(frame_w, math.ceil(cx + radius_x + w / 2.0)))
        ry1 = int(min(frame_h, math.ceil(cy + radius_y + h / 2.0)))
        if rx1 - rx0 < 8 or ry1 - ry0 < 8:
            return None

        frame_gray = to_gray(frame)
        search_gray = frame_gray[ry0:ry1, rx0:rx1]

        candidates: List[Tuple[float, BBox, float]] = []
        for ref in self.references:
            # Match at a few scales.  The reference patch is resized to the
            # expected current target size before matching.
            for scale in self.scale_factors:
                tw = max(8, int(round(w * scale)))
                th = max(8, int(round(h * scale)))
                if tw >= search_gray.shape[1] or th >= search_gray.shape[0]:
                    continue
                tmpl = cv2.resize(
                    ref.template_gray, (tw, th), interpolation=cv2.INTER_LINEAR
                )
                result = cv2.matchTemplate(
                    search_gray, tmpl, cv2.TM_CCOEFF_NORMED
                )
                result = np.nan_to_num(result, nan=-1.0, posinf=-1.0, neginf=-1.0)

                # Keep local maxima as candidates.  A global argmax alone is
                # brittle when another pedestrian has a similar silhouette.
                for _ in range(3):
                    _, score, _, max_loc = cv2.minMaxLoc(result)
                    if not np.isfinite(score) or score < self.min_ncc:
                        break
                    bx = float(rx0 + max_loc[0])
                    by = float(ry0 + max_loc[1])
                    candidate_bbox = (bx, by, float(tw), float(th))
                    candidates.append((float(score), candidate_bbox, float(scale)))

                    # Suppress this peak before looking for the next one.
                    lx = max(0, max_loc[0] - tw // 2)
                    ly = max(0, max_loc[1] - th // 2)
                    hx = min(result.shape[1], max_loc[0] + tw // 2 + 1)
                    hy = min(result.shape[0], max_loc[1] + th // 2 + 1)
                    result[ly:hy, lx:hx] = -1.0

        if not candidates:
            return None

        # Greedy non-maximum suppression across all templates/scales.
        candidates.sort(key=lambda c: c[0], reverse=True)
        kept: List[Tuple[float, BBox, float]] = []
        for cand in candidates:
            if all(
                bbox_iou(cand[1], other[1]) < 0.30 for other in kept
            ):
                kept.append(cand)
            if len(kept) >= self.max_candidates:
                break

        # A pedestrian cannot teleport while occluded.  This gate rejects
        # distant look-alikes that happen to survive appearance verification,
        # which is especially important after the true target has left the
        # scene.  The allowance grows with the number of lost frames because
        # the target may legitimately have walked during a long occlusion.
        max_jump = max(0.8 * max(w, h), 30.0) + 18.0 * float(lost_frames)

        best: Optional[RecoveryResult] = None
        for ncc, cand_bbox, _ in kept:
            cand_bbox = clip_bbox(cand_bbox, frame_w, frame_h)
            if math.dist(bbox_center(cand_bbox), (cx, cy)) > max_jump:
                continue
            hist_sim, ssim_sim, fg_diff = self.appearance_metrics(
                frame, cand_bbox
            )
            if fg_diff < self.min_foreground_diff:
                continue

            # Weighted late-fusion score.  NCC is the strongest cue, while
            # colour and structure prevent background/other-person matches.
            total = 0.55 * float(ncc) + 0.25 * hist_sim + 0.20 * ssim_sim
            if (
                ncc >= self.min_ncc
                and hist_sim >= self.min_hist
                and ssim_sim >= self.min_ssim
                and total >= self.min_total
            ):
                if best is None or total > best.score:
                    best = RecoveryResult(
                        bbox=cand_bbox,
                        ncc=float(ncc),
                        hist_similarity=hist_sim,
                        ssim=ssim_sim,
                        score=float(total),
                        candidates_considered=len(kept),
                        foreground_diff=fg_diff,
                    )
        return best


def _ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Mean structural similarity for two single-channel uint8 images."""
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]))
    x = a.astype(np.float64)
    y = b.astype(np.float64)
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    sigma = 1.5

    mu_x = gaussian_filter(x, sigma=sigma)
    mu_y = gaussian_filter(y, sigma=sigma)
    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = gaussian_filter(x * x, sigma=sigma) - mu_x2
    sigma_y2 = gaussian_filter(y * y, sigma=sigma) - mu_y2
    sigma_xy = gaussian_filter(x * y, sigma=sigma) - mu_xy

    numerator = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
    denominator = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    ratio = numerator / np.maximum(denominator, 1e-12)
    return float(np.mean(ratio))


# ---------------------------------------------------------------------------
# DriftGuard state machine
# ---------------------------------------------------------------------------


@dataclass
class FrameResult:
    frame_idx: int
    bbox: Optional[BBox]
    psr: Optional[float]
    state: str
    model_updated: bool
    recovery_attempted: bool = False
    reinitialized: bool = False
    recovery_confirmed: bool = False


class DriftGuard:
    """KCF + PSR drift detection + frozen updates + recovery pipeline."""

    TRACKING = "TRACKING"
    DRIFT = "DRIFT_FLAGGED"
    RECOVERED = "RECOVERED"
    LOST = "LOST"

    def __init__(
        self,
        frame: np.ndarray,
        initial_bbox: BBox,
        *,
        psr_threshold: float = 7.0,
        max_lost_frames: int = 40,
        confirmation_frames: int = 3,
        recovery_delay_frames: int = 10,
        tracker: Optional[KCFTracker] = None,
        recovery: Optional[RecoveryModule] = None,
        reference_interval: int = 12,
    ) -> None:
        self.psr_threshold = float(psr_threshold)
        self.max_lost_frames = int(max_lost_frames)
        self.confirmation_frames = int(max(1, confirmation_frames))
        # Let the KCF response settle for a few frames before trusting a
        # template match.  This avoids re-initialising on a sign/pole that is
        # temporarily overlapping the target at the instant PSR collapses.
        self.recovery_delay_frames = int(max(0, recovery_delay_frames))
        self.reference_interval = int(reference_interval)

        frame_h, frame_w = frame.shape[:2]
        self.initial_bbox = clip_bbox(initial_bbox, frame_w, frame_h)
        self.bbox: BBox = self.initial_bbox
        self.last_good_bbox: BBox = self.initial_bbox

        self.tracker = tracker or KCFTracker()
        self.recovery = recovery or RecoveryModule()
        self.tracker.initialize(frame, self.initial_bbox)
        self.recovery.reset(frame, self.initial_bbox, frame_idx=0)

        self.state = self.TRACKING
        self.last_psr: Optional[float] = None
        self.frame_idx = 0

        # Statistics
        self.drift_frames = 0
        self.drift_episodes = 0
        self.recovery_attempts = 0
        self.reinitializations = 0
        self.successful_recoveries = 0
        self.model_updates = 0
        self.frozen_updates = 0
        self.lost_frames = 0

        # Recovery bookkeeping
        self.lost_count = 0
        self._in_drift_episode = False
        self._episode_had_attempt = False
        self._pending_recovery = False
        self._confirmation_count = 0
        self._frozen_healthy_count = 0
        self._timed_out = False
        self.frozen_recoveries = 0
        self._last_recovery_result: Optional[RecoveryResult] = None
        self._pre_drift_snapshot: Optional[Dict[str, object]] = None
        self._pre_recovery_bbox: Optional[BBox] = None

    @property
    def is_tracking(self) -> bool:
        return self.state in (self.TRACKING, self.RECOVERED)

    def step(self, frame: np.ndarray, frame_idx: int) -> FrameResult:
        """Process one frame and return the DriftGuard decision/state."""
        self.frame_idx = int(frame_idx)
        result = FrameResult(
            frame_idx=self.frame_idx,
            bbox=self.bbox,
            psr=self.last_psr,
            state=self.state,
            model_updated=False,
        )

        # Once the target has been missing for max_lost_frames, DriftGuard
        # keeps the last known state instead of latching onto an arbitrary
        # look-alike.  This is the timeout branch of the recovery design.
        if self._timed_out:
            self.state = self.LOST
            self.frozen_updates += 1
            self.drift_frames += 1
            self.lost_frames += 1
            result.bbox = self.last_good_bbox
            result.state = self.LOST
            return result

        if self.is_tracking:
            bbox, response = self.tracker.detect(frame)
            psr = compute_psr(response)
            self.last_psr = psr
            self.bbox = bbox

            if psr >= self.psr_threshold:
                if self._pending_recovery:
                    # A re-detected box must remain both PSR-healthy and
                    # appearance-plausible for several frames.  In particular,
                    # do NOT adapt the model during this probation period.
                    if not self.recovery.is_plausible_track(frame, bbox):
                        self._rollback_recovery()
                        self._enter_drift()
                        self.frozen_updates += 1
                        self.drift_frames += 1
                        result.bbox = self.last_good_bbox
                        result.psr = psr
                        result.state = self.DRIFT
                        result.model_updated = False
                    else:
                        self._confirmation_count += 1
                        self.last_good_bbox = bbox
                        self.lost_count = 0
                        if self._confirmation_count >= self.confirmation_frames:
                            # Recovery is now confirmed.  Resume adaptation
                            # and add this verified appearance to the bank.
                            self.tracker.update_model(frame, bbox)
                            self.model_updates += 1
                            self.recovery.add_reference(
                                frame, bbox, self.frame_idx
                            )
                            self.successful_recoveries += 1
                            self._pending_recovery = False
                            self._in_drift_episode = False
                            self._episode_had_attempt = False
                            self._pre_drift_snapshot = None
                            self._pre_recovery_bbox = None
                            self.state = self.RECOVERED
                            result.recovery_confirmed = True
                            result.model_updated = True
                        else:
                            # Probation frame: freeze adaptation but keep the
                            # PSR for display/evaluation.
                            self.frozen_updates += 1
                            self.state = self.RECOVERED
                            result.model_updated = False
                        result.bbox = bbox
                        result.psr = psr
                        result.state = self.state
                        return result
                else:
                    # Healthy: safe to adapt the model.
                    self.tracker.update_model(frame, bbox)
                    self.model_updates += 1
                    self.last_good_bbox = bbox
                    self.lost_count = 0
                    self.recovery.add_reference(frame, bbox, self.frame_idx)
                    self.state = self.TRACKING
                    result.bbox = bbox
                    result.psr = psr
                    result.state = self.state
                    result.model_updated = True
                    return result

            # --- PSR below threshold: drift/failure detected ---------------
            if self._pending_recovery:
                # The candidate re-initialised earlier did not survive the
                # confirmation window.  Restore the uncontaminated model.
                self._rollback_recovery()
            self._enter_drift()
            self.frozen_updates += 1
            self.drift_frames += 1
            result.bbox = self.last_good_bbox
            result.psr = psr
            result.state = self.DRIFT
            result.model_updated = False
        else:
            # Frozen model: keep running the detector, but NEVER update the
            # model.  This "freeze, keep looking" behaviour lets KCF itself
            # re-acquire a target that was only briefly occluded, without
            # re-initialising on a pole or a different pedestrian.
            self.lost_count += 1
            self.frozen_updates += 1
            self.drift_frames += 1

            bbox_frozen, response_frozen = self.tracker.detect(frame)
            psr_frozen = compute_psr(response_frozen)
            self.last_psr = psr_frozen
            self.bbox = bbox_frozen
            result.psr = psr_frozen

            frozen_healthy = (
                psr_frozen >= self.psr_threshold
                and self.recovery.is_plausible_track(frame, bbox_frozen)
            )
            if frozen_healthy:
                self._frozen_healthy_count += 1
            else:
                self._frozen_healthy_count = 0

            if self._frozen_healthy_count >= 2:
                # The frozen filter has re-acquired the target on two
                # consecutive appearance-plausible frames.  Resume adaptation
                # without a template re-initialisation.
                self.tracker.update_model(frame, bbox_frozen)
                self.model_updates += 1
                self.recovery.add_reference(frame, bbox_frozen, self.frame_idx)
                self.last_good_bbox = bbox_frozen
                self.bbox = bbox_frozen
                self.lost_count = 0
                self._frozen_healthy_count = 0
                self._pending_recovery = False
                self._confirmation_count = 0
                self._in_drift_episode = False
                self._episode_had_attempt = False
                self._pre_drift_snapshot = None
                self._pre_recovery_bbox = None
                self.state = self.RECOVERED
                self.successful_recoveries += 1
                self.frozen_recoveries += 1
                result.bbox = bbox_frozen
                result.psr = psr_frozen
                result.state = self.RECOVERED
                result.recovery_attempted = True
                result.recovery_confirmed = True
                result.model_updated = True
                return result

            self.state = self.DRIFT

        # --- recovery pipeline --------------------------------------------
        # Debounce: give the frozen filter time to re-acquire the target
        # before trusting a template match.  A template match during partial
        # occlusion can otherwise lock onto the occluder.
        if self.lost_count < self.recovery_delay_frames:
            result.bbox = self.last_good_bbox
            result.state = self.state
            return result

        result.recovery_attempted = True
        candidate = self.recovery.find(
            frame,
            self.last_good_bbox,
            lost_frames=self.lost_count,
            max_lost_frames=self.max_lost_frames,
        )
        if candidate is not None:
            # Verified candidate: re-initialise KCF on the new box.  Keep the
            # pre-recovery anchor and model so this attempt can be rolled back
            # if the confirmation window rejects the candidate.
            self._last_recovery_result = candidate
            self._pre_recovery_bbox = self.last_good_bbox
            frame_h, frame_w = frame.shape[:2]
            self.bbox = clip_bbox(candidate.bbox, frame_w, frame_h)
            self.tracker.initialize(frame, self.bbox)
            self.last_good_bbox = self.bbox
            self.state = self.RECOVERED
            self._pending_recovery = True
            self._confirmation_count = 0
            self._frozen_healthy_count = 0
            self.lost_count = 0
            self.reinitializations += 1
            result.state = self.RECOVERED
            result.bbox = self.bbox
            result.psr = None  # no PSR until the next frame's detection
            result.reinitialized = True
            result.model_updated = True
            return result

        result.bbox = self.last_good_bbox
        if self.lost_count >= self.max_lost_frames:
            self.state = self.LOST
            self._timed_out = True
            self.lost_frames += 1
        result.state = self.state
        return result

    # -- state transitions --------------------------------------------------

    def _enter_drift(self) -> None:
        """Open a new drift episode and freeze model adaptation."""
        if not self._in_drift_episode:
            self._in_drift_episode = True
            self.drift_episodes += 1
            self.recovery_attempts += 1
            self.lost_count = 0
            self._frozen_healthy_count = 0
            self._episode_had_attempt = False
            self._pre_drift_snapshot = self.tracker.snapshot()
        self.state = self.DRIFT

    def _rollback_recovery(self) -> None:
        """Discard a failed recovery candidate and restore the frozen model."""
        if self._pre_drift_snapshot is not None:
            self.tracker.restore(self._pre_drift_snapshot)
        if self._pre_recovery_bbox is not None:
            self.last_good_bbox = self._pre_recovery_bbox
        self.bbox = self.last_good_bbox
        self._pending_recovery = False
        self._confirmation_count = 0
        self._pre_recovery_bbox = None
        self.state = self.DRIFT


class BaselineKCF:
    """Baseline: plain KCF that updates its model on every frame."""

    def __init__(self, frame: np.ndarray, initial_bbox: BBox, *, psr_threshold: float = 7.0) -> None:
        self.psr_threshold = float(psr_threshold)
        frame_h, frame_w = frame.shape[:2]
        self.bbox = clip_bbox(initial_bbox, frame_w, frame_h)
        self.tracker = KCFTracker()
        self.tracker.initialize(frame, self.bbox)
        self.drift_episodes = 0
        self._was_bad = False

    def step(self, frame: np.ndarray) -> Tuple[BBox, float]:
        bbox, response = self.tracker.detect(frame)
        psr = compute_psr(response)
        # No failure awareness: always update, even when the response is bad.
        self.tracker.update_model(frame, bbox)
        self.bbox = bbox
        bad = psr < self.psr_threshold
        if bad and not self._was_bad:
            self.drift_episodes += 1
        self._was_bad = bad
        return bbox, psr


# ---------------------------------------------------------------------------
# Visualisation, logging and evaluation
# ---------------------------------------------------------------------------


def _status_color(state: str) -> Tuple[int, int, int]:
    return (0, 200, 0) if state in (
        DriftGuard.TRACKING,
        DriftGuard.RECOVERED,
    ) else (0, 0, 255)


def draw_overlay(
    frame: np.ndarray,
    result: FrameResult,
    *,
    fps: Optional[float] = None,
    show_search_hint: bool = False,
    psr_threshold: float = 7.0,
) -> np.ndarray:
    """Draw the tracking box and the DriftGuard heads-up display."""
    out = frame.copy()
    color = _status_color(result.state)

    if result.bbox is not None:
        x, y, w, h = [int(round(v)) for v in result.bbox]
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)

    panel = out.copy()
    cv2.rectangle(panel, (8, 8), (330, 112), (0, 0, 0), -1)
    cv2.addWeighted(panel, 0.55, out, 0.45, 0, out)

    psr_text = "n/a" if result.psr is None else f"{result.psr:5.2f}"
    status = result.state.replace("_", " ")
    cv2.putText(
        out,
        f"Frame: {result.frame_idx:04d}",
        (18, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        out,
        f"PSR: {psr_text}  (threshold {psr_threshold:.1f})",
        (18, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        out,
        f"Status: {status}",
        (18, 86),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )
    update_text = "UPDATE FROZEN" if not result.model_updated else "MODEL UPDATED"
    cv2.putText(
        out,
        update_text,
        (18, 108),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )
    if fps is not None:
        cv2.putText(
            out,
            f"{fps:4.1f} FPS",
            (out.shape[1] - 105, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    cv2.putText(
        out,
        "DriftGuard",
        (out.shape[1] - 145, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 220, 0),
        2,
        cv2.LINE_AA,
    )

    if show_search_hint and result.state in (
        DriftGuard.DRIFT,
        DriftGuard.LOST,
    ):
        hint = (
            "Re-detecting..."
            if result.recovery_attempted
            else "Lost: holding last known state"
        )
        cv2.putText(
            out,
            hint,
            (18, 132),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 165, 255),
            1,
            cv2.LINE_AA,
        )
    return out


def write_psr_plot(
    baseline_psr: Sequence[float],
    guard_psr: Sequence[Optional[float]],
    drift_flags: Sequence[bool],
    recovery_frames: Sequence[int],
    threshold: float,
    output_path: Path,
) -> None:
    """Save the baseline-vs-DriftGuard PSR comparison plot."""
    frames = np.arange(len(baseline_psr))
    base = np.asarray(baseline_psr, dtype=np.float64)
    guard = np.asarray(
        [np.nan if v is None else float(v) for v in guard_psr],
        dtype=np.float64,
    )

    fig, ax = plt.subplots(figsize=(12, 4.8), dpi=140)
    drift_idx = np.flatnonzero(np.asarray(drift_flags, dtype=bool))
    if drift_idx.size:
        # Shade contiguous drift segments.
        starts = drift_idx[np.r_[True, np.diff(drift_idx) > 1]]
        ends = drift_idx[np.r_[np.diff(drift_idx) > 1, True]]
        for a, b in zip(starts, ends):
            ax.axvspan(a, b + 1, color="#ffcccc", alpha=0.30, lw=0)

    ax.plot(frames, base, color="#1f77b4", lw=1.1, label="Baseline KCF PSR")
    ax.plot(frames, guard, color="#2ca02c", lw=1.3, label="DriftGuard PSR")
    ax.axhline(
        threshold,
        color="#d62728",
        linestyle="--",
        lw=1.2,
        label=f"Drift threshold ({threshold:.1f})",
    )
    if len(recovery_frames):
        valid = [f for f in recovery_frames if 0 <= f < len(guard)]
        if valid:
            ax.scatter(
                valid,
                [max(threshold, np.nan_to_num(guard[f], nan=threshold)) for f in valid],
                marker="v",
                s=30,
                color="#2ca02c",
                edgecolor="black",
                linewidth=0.4,
                zorder=5,
                label="Confirmed recovery",
            )
    ax.set_xlabel("Frame")
    ax.set_ylabel("PSR")
    ax.set_title("Peak-to-Sidelobe Ratio: baseline KCF vs DriftGuard")
    ax.set_xlim(0, max(1, len(frames) - 1))
    ymax = float(np.nanmax(np.r_[base[np.isfinite(base)], 10.0]))
    ax.set_ylim(0, min(max(ymax * 1.08, threshold * 1.4), 80.0))
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def write_gt_plot(
    guard_gt: Dict[str, object],
    baseline_gt: Dict[str, object],
    output_path: Path,
) -> None:
    """Save OTB-style success and precision plots when GT is available."""
    if not guard_gt or not baseline_gt:
        return

    gs = guard_gt["summary"]  # type: ignore[index]
    bs = baseline_gt["summary"]  # type: ignore[index]
    iou_t = np.asarray(guard_gt["iou_thresholds"])
    dist_t = np.asarray(guard_gt["distance_thresholds"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), dpi=140)
    ax = axes[0]
    ax.plot(
        iou_t,
        np.asarray(guard_gt["success_curve"]),
        color="#2ca02c",
        lw=1.8,
        label=f"DriftGuard (AUC={gs['success_auc']:.3f})",
    )
    ax.plot(
        iou_t,
        np.asarray(baseline_gt["success_curve"]),
        color="#1f77b4",
        lw=1.6,
        label=f"Baseline KCF (AUC={bs['success_auc']:.3f})",
    )
    ax.set_xlabel("IoU threshold")
    ax.set_ylabel("Success rate")
    ax.set_title("Success plot")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower left", fontsize=8)

    ax = axes[1]
    ax.plot(
        dist_t,
        np.asarray(guard_gt["precision_curve"]),
        color="#2ca02c",
        lw=1.8,
        label=f"DriftGuard (20px={gs['precision_center_20px']:.3f})",
    )
    ax.plot(
        dist_t,
        np.asarray(baseline_gt["precision_curve"]),
        color="#1f77b4",
        lw=1.6,
        label=f"Baseline KCF (20px={bs['precision_center_20px']:.3f})",
    )
    ax.axvline(20.0, color="#d62728", linestyle="--", lw=1.0)
    ax.set_xlabel("Centre-error threshold (px)")
    ax.set_ylabel("Precision")
    ax.set_title("Precision plot")
    ax.set_xlim(0, 50)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize=8)

    fig.suptitle("Ground-truth evaluation on the annotated sequence", fontsize=11)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def write_frame_log(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_gt_csv(path: Path) -> Dict[int, BBox]:
    """Read optional ground truth CSV: frame,x,y,w,h (frame 0-based)."""
    gt: Dict[int, BBox] = {}
    with path.open("r", newline="", encoding="utf-8-sig") as fh:
        sample = fh.read(2048)
        fh.seek(0)
        has_header = "frame" in sample.lower().splitlines()[0]
        reader = csv.reader(fh)
        if has_header:
            next(reader, None)
        for row in reader:
            if len(row) < 5:
                continue
            try:
                idx = int(float(row[0]))
                gt[idx] = tuple(float(v) for v in row[1:5])  # type: ignore[assignment]
            except (ValueError, TypeError):
                continue
    return gt


def evaluate_gt_full(
    gt: Dict[int, BBox], bboxes: Dict[int, BBox], threshold_iou: float = 0.5
) -> Dict[str, object]:
    """Compute OTB-style success/precision curves and summary statistics."""
    common = sorted(set(gt) & set(bboxes))
    if not common:
        return {}
    ious = np.array([bbox_iou(gt[i], bboxes[i]) for i in common], dtype=float)
    errors = np.array(
        [
            math.dist(bbox_center(gt[i]), bbox_center(bboxes[i]))
            for i in common
        ],
        dtype=float,
    )

    iou_thresholds = np.linspace(0.0, 1.0, 101)
    success_curve = np.array(
        [float(np.mean(ious >= t)) for t in iou_thresholds], dtype=float
    )
    distance_thresholds = np.arange(0.0, 51.0, 1.0)
    precision_curve = np.array(
        [float(np.mean(errors <= t)) for t in distance_thresholds], dtype=float
    )

    summary = {
        "gt_frames": float(len(common)),
        "success_rate_iou_0.5": float(np.mean(ious >= threshold_iou)),
        "success_auc": float(np.mean(success_curve)),
        "mean_iou": float(np.mean(ious)),
        "precision_center_20px": float(np.mean(errors <= 20.0)),
        "precision_auc_50px": float(np.mean(precision_curve)),
        "mean_center_error_px": float(np.mean(errors)),
    }
    return {
        "summary": summary,
        "iou_thresholds": iou_thresholds,
        "success_curve": success_curve,
        "distance_thresholds": distance_thresholds,
        "precision_curve": precision_curve,
    }


def evaluate_gt(
    gt: Dict[int, BBox], bboxes: Dict[int, BBox], threshold_iou: float = 0.5
) -> Dict[str, float]:
    """Backward-compatible summary-only wrapper around evaluate_gt_full()."""
    result = evaluate_gt_full(gt, bboxes, threshold_iou=threshold_iou)
    return result.get("summary", {})  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Video I/O, download and CLI
# ---------------------------------------------------------------------------


VTEST_URL = (
    "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/vtest.avi"
)
DEFAULT_BBOX: BBox = (638.0, 238.0, 48.0, 86.0)


def download_vtest(destination: Path = Path("vtest.avi")) -> Path:
    """Download OpenCV's sample surveillance clip (no OpenCV install needed)."""
    destination = Path(destination)
    if destination.exists():
        return destination
    print(f"Downloading vtest.avi -> {destination} ...")
    urllib.request.urlretrieve(VTEST_URL, destination)
    print(f"Saved {destination} ({destination.stat().st_size / 1e6:.1f} MB)")
    return destination


def parse_bbox(value: str) -> BBox:
    parts = [p.strip() for p in value.replace(";", ",").split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("bbox must be x,y,w,h")
    try:
        x, y, w, h = (float(p) for p in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("bbox values must be numeric") from exc
    if w <= 0 or h <= 0:
        raise argparse.ArgumentTypeError("bbox width/height must be positive")
    return x, y, w, h


def open_source(source: str) -> cv2.VideoCapture:
    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
    else:
        cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video source {source!r}. "
            "For the benchmark clip, run `python driftguard.py --download` "
            "or pass a valid video path / webcam index."
        )
    return cap


def run(
    *,
    source: str,
    bbox_arg: Optional[BBox],
    output: Path,
    plot_path: Path,
    metric_plot_path: Path,
    log_path: Path,
    report_path: Path,
    psr_threshold: float,
    max_lost_frames: int,
    confirmation_frames: int,
    recovery_delay_frames: int,
    show_display: bool,
    max_frames: Optional[int],
    gt_path: Optional[Path],
) -> int:
    cap = open_source(source)
    fps_in = float(cap.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps_in) or fps_in <= 0 or fps_in > 240:
        fps_in = 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ok, frame0 = cap.read()
    if not ok or frame0 is None:
        raise RuntimeError("Video source returned no frames")
    if frame0.ndim == 2:
        frame0 = cv2.cvtColor(frame0, cv2.COLOR_GRAY2BGR)
    height, width = frame0.shape[:2]

    if bbox_arg is None:
        if Path(source).name.lower() == "vtest.avi":
            bbox = DEFAULT_BBOX
            print(f"Using vtest.avi default bbox {bbox}")
        elif show_display:
            # Interactive fallback for arbitrary videos / webcams.
            roi = cv2.selectROI("Select target", frame0, showCrosshair=True, fromCenter=False)
            cv2.destroyWindow("Select target")
            if roi[2] <= 0 or roi[3] <= 0:
                raise RuntimeError("No target ROI selected")
            bbox = tuple(float(v) for v in roi)  # type: ignore[assignment]
        else:
            raise RuntimeError(
                "No --bbox supplied. Provide --bbox x,y,w,h, or run with "
                "--display to select an ROI interactively."
            )
    else:
        bbox = bbox_arg

    frame_h, frame_w = frame0.shape[:2]
    bbox = clip_bbox(bbox, frame_w, frame_h)
    print(f"Source: {source}  {frame_w}x{frame_h}  @ {fps_in:.1f} FPS")
    print(f"Initial bbox: ({bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f})")

    output.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(output), fourcc, fps_in, (width, height))
    if not writer.isOpened():
        # Some OpenCV builds lack MJPG; XVID is a widely available fallback.
        writer.release()
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(str(output), fourcc, fps_in, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open output video writer: {output}")

    baseline = BaselineKCF(frame0, bbox, psr_threshold=psr_threshold)
    guard = DriftGuard(
        frame0,
        bbox,
        psr_threshold=psr_threshold,
        max_lost_frames=max_lost_frames,
        confirmation_frames=confirmation_frames,
        recovery_delay_frames=recovery_delay_frames,
    )

    # First frame: both trackers are already initialised.
    first_result = FrameResult(
        frame_idx=0,
        bbox=guard.bbox,
        psr=None,
        state=guard.state,
        model_updated=True,
    )
    writer.write(draw_overlay(frame0, first_result, fps=None))
    if show_display:
        cv2.imshow("DriftGuard", draw_overlay(frame0, first_result))
        if cv2.waitKey(1) & 0xFF == 27:
            writer.release()
            cap.release()
            return 0

    baseline_psr: List[float] = [np.nan]  # no PSR on frame 0
    guard_psr: List[Optional[float]] = [None]
    drift_flags: List[bool] = [False]
    frame_rows: List[Dict[str, object]] = []
    guard_bboxes: Dict[int, BBox] = {0: guard.bbox}
    baseline_bboxes: Dict[int, BBox] = {0: baseline.bbox}
    baseline_drift_flags = 0
    recovery_frames: List[int] = []

    t_baseline = 0.0
    t_guard = 0.0
    overall_start = time.perf_counter()
    frame_idx = 1
    last_psr_display: Optional[float] = None

    while True:
        if max_frames is not None and frame_idx > max_frames:
            break
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        t0 = time.perf_counter()
        base_bbox, base_psr = baseline.step(frame)
        t_baseline += time.perf_counter() - t0

        t0 = time.perf_counter()
        res = guard.step(frame, frame_idx)
        t_guard += time.perf_counter() - t0

        if base_psr < psr_threshold:
            baseline_drift_flags += 1

        baseline_psr.append(float(base_psr))
        guard_psr.append(None if res.psr is None else float(res.psr))
        drift_flags.append(res.state in (DriftGuard.DRIFT, DriftGuard.LOST))
        if res.recovery_confirmed:
            recovery_frames.append(frame_idx)

        guard_bboxes[frame_idx] = (
            res.bbox if res.bbox is not None else guard.last_good_bbox
        )
        baseline_bboxes[frame_idx] = base_bbox
        if res.reinitialized:
            last_psr_display = None
        elif res.psr is not None:
            last_psr_display = res.psr

        # Keep a single, stable overlay PSR during a search episode.
        display_result = FrameResult(
            frame_idx=frame_idx,
            bbox=res.bbox,
            psr=res.psr if res.psr is not None else last_psr_display,
            state=res.state,
            model_updated=res.model_updated,
            recovery_attempted=res.recovery_attempted,
            reinitialized=res.reinitialized,
            recovery_confirmed=res.recovery_confirmed,
        )
        elapsed = max(1e-9, time.perf_counter() - overall_start)
        live_fps = (frame_idx + 1) / elapsed
        annotated = draw_overlay(
            frame,
            display_result,
            fps=live_fps,
            show_search_hint=True,
            psr_threshold=psr_threshold,
        )
        writer.write(annotated)

        if show_display:
            cv2.imshow("DriftGuard", annotated)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        frame_rows.append(
            {
                "frame": frame_idx,
                "state": res.state,
                "psr": "" if res.psr is None else f"{res.psr:.6f}",
                "bbox_x": "" if res.bbox is None else f"{res.bbox[0]:.2f}",
                "bbox_y": "" if res.bbox is None else f"{res.bbox[1]:.2f}",
                "bbox_w": "" if res.bbox is None else f"{res.bbox[2]:.2f}",
                "bbox_h": "" if res.bbox is None else f"{res.bbox[3]:.2f}",
                "model_updated": int(bool(res.model_updated)),
                "recovery_attempted": int(bool(res.recovery_attempted)),
                "reinitialized": int(bool(res.reinitialized)),
                "recovery_confirmed": int(bool(res.recovery_confirmed)),
            }
        )
        frame_idx += 1

    cap.release()
    writer.release()
    if show_display:
        cv2.destroyAllWindows()

    total_frames = frame_idx
    wall_seconds = max(1e-9, time.perf_counter() - overall_start)
    # Overall pipeline throughput is the user-visible real-time number.
    overall_fps = total_frames / wall_seconds
    baseline_fps = total_frames / t_baseline if t_baseline > 0 else 0.0
    processing_fps = total_frames / t_guard if t_guard > 0 else 0.0

    # The PSR arrays start at frame 0.  If the source ended early, trim to the
    # actual number of processed frames.
    baseline_psr = baseline_psr[:total_frames]
    guard_psr = guard_psr[:total_frames]
    drift_flags = drift_flags[:total_frames]

    write_psr_plot(
        baseline_psr,
        guard_psr,
        drift_flags,
        recovery_frames,
        psr_threshold,
        plot_path,
    )
    write_frame_log(log_path, frame_rows)

    gt_metrics: Dict[str, float] = {}
    baseline_gt_metrics: Dict[str, float] = {}
    guard_gt_full: Dict[str, object] = {}
    baseline_gt_full: Dict[str, object] = {}
    if gt_path is not None and gt_path.exists():
        gt = read_gt_csv(gt_path)
        guard_gt_full = evaluate_gt_full(gt, guard_bboxes)
        baseline_gt_full = evaluate_gt_full(gt, baseline_bboxes)
        gt_metrics = guard_gt_full.get("summary", {})  # type: ignore[assignment]
        baseline_gt_metrics = baseline_gt_full.get(
            "summary", {}
        )  # type: ignore[assignment]
        write_gt_plot(guard_gt_full, baseline_gt_full, metric_plot_path)

    valid_guard_psr = np.array(
        [v for v in guard_psr if v is not None], dtype=np.float64
    )
    valid_baseline_psr = np.asarray(baseline_psr, dtype=np.float64)
    healthy_guard_frames = int(
        sum(
            1
            for idx, v in enumerate(guard_psr)
            if (not drift_flags[idx]) and v is not None and v >= psr_threshold
        )
    )
    healthy_baseline_frames = int(np.sum(valid_baseline_psr >= psr_threshold))
    evaluated_frames = max(1, total_frames - 1)  # frame 0 has no PSR measurement

    report = {
        "video": str(source),
        "total_frames": total_frames,
        "fps_input": fps_in,
        "psr_threshold": psr_threshold,
        "recovery_delay_frames": recovery_delay_frames,
        "driftguard": {
            "drift_flagged_frames": int(guard.drift_frames),
            "drift_episodes": int(guard.drift_episodes),
            "recovery_attempts": int(guard.recovery_attempts),
            "reinitializations": int(guard.reinitializations),
            "frozen_model_recoveries": int(guard.frozen_recoveries),
            "successful_recoveries": int(guard.successful_recoveries),
            "recovery_success_rate": (
                float(guard.successful_recoveries / guard.drift_episodes)
                if guard.drift_episodes
                else 0.0
            ),
            "lost_timeout_frames": int(guard.lost_frames),
            "model_updates": int(guard.model_updates),
            "frozen_updates": int(guard.frozen_updates),
            "healthy_psr_frames": healthy_guard_frames,
            "healthy_psr_rate": float(healthy_guard_frames / evaluated_frames),
            "mean_psr": float(np.nanmean(valid_guard_psr))
            if np.isfinite(valid_guard_psr).any()
            else None,
            "avg_processing_fps": float(processing_fps),
            "overall_pipeline_fps": float(overall_fps),
        },
        "baseline": {
            "drift_flagged_frames": int(baseline_drift_flags),
            "drift_episodes": int(baseline.drift_episodes),
            "healthy_psr_frames": healthy_baseline_frames,
            "healthy_psr_rate": float(healthy_baseline_frames / evaluated_frames),
            "mean_psr": float(np.nanmean(valid_baseline_psr))
            if np.isfinite(valid_baseline_psr).any()
            else None,
            "avg_processing_fps": float(baseline_fps),
        },
        "ground_truth": {
            "driftguard": gt_metrics,
            "baseline": baseline_gt_metrics,
        }
        if gt_metrics
        else None,
        "runtime_seconds": float(wall_seconds),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, allow_nan=False)

    print("\n" + "=" * 64)
    print("DRIFTGUARD REPORT")
    print("=" * 64)
    print(f"Source                    : {source}")
    print(f"Frames processed          : {total_frames}")
    print(f"Video resolution/fps      : {frame_w}x{frame_h} @ {fps_in:.1f}")
    print(f"PSR threshold             : {psr_threshold:.1f}")
    print("-" * 64)
    print("DriftGuard")
    print(f"  Drift-flagged frames    : {guard.drift_frames}")
    print(f"  Drift episodes          : {guard.drift_episodes}")
    print(f"  Recovery attempts       : {guard.recovery_attempts}")
    print(f"  Re-initialisations      : {guard.reinitializations}")
    print(f"  Frozen-model recoveries : {guard.frozen_recoveries}")
    print(f"  Successful recoveries   : {guard.successful_recoveries}")
    rate = (
        guard.successful_recoveries / guard.drift_episodes
        if guard.drift_episodes
        else 0.0
    )
    print(f"  Recovery success rate   : {rate * 100.0:.1f}%")
    print(f"  Frames held as LOST     : {guard.lost_frames}")
    print(f"  Model updates           : {guard.model_updates}")
    print(f"  Frozen updates          : {guard.frozen_updates}")
    print(
        f"  PSR-healthy frames      : {healthy_guard_frames} "
        f"({100.0 * healthy_guard_frames / evaluated_frames:.1f}%)"
    )
    print(f"  Average processing FPS  : {processing_fps:.1f}")
    print(f"  Overall pipeline FPS    : {overall_fps:.1f}")
    print("-" * 64)
    print("Baseline KCF (always updates)")
    print(f"  PSR-flagged frames      : {baseline_drift_flags}")
    print(f"  Drift episodes          : {baseline.drift_episodes}")
    print(
        f"  PSR-healthy frames      : {healthy_baseline_frames} "
        f"({100.0 * healthy_baseline_frames / evaluated_frames:.1f}%)"
    )
    print(f"  Average processing FPS  : {baseline_fps:.1f}")
    if gt_metrics:
        print("-" * 64)
        print("Ground-truth evaluation (optional CSV)")
        for name, metrics in (("DriftGuard", gt_metrics), ("Baseline", baseline_gt_metrics)):
            print(
                f"  {name:<10}: success@0.5={metrics['success_rate_iou_0.5']:.3f} "
                f"AUC={metrics['success_auc']:.3f} "
                f"meanIoU={metrics['mean_iou']:.3f} "
                f"precision@20px={metrics['precision_center_20px']:.3f}"
            )
    else:
        print("-" * 64)
        print(
            "No ground-truth CSV supplied.  vtest.avi ships with only the\n"
            "initial box, so PSR-health, drift and recovery metrics are the\n"
            "available accuracy proxies.  Use --gt frame,x,y,w,h for OTB-style\n"
            "success/precision evaluation."
        )
    print("=" * 64)
    print(f"Output video              : {output}")
    print(f"PSR plot                  : {plot_path}")
    if gt_metrics:
        print(f"Success/precision plot    : {metric_plot_path}")
    print(f"Frame log                 : {log_path}")
    print(f"JSON report               : {report_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="DriftGuard: PSR-aware KCF tracking with recovery."
    )
    parser.add_argument(
        "--source",
        default="vtest.avi",
        help="Video path or webcam index (e.g. 0). Default: vtest.avi",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download vtest.avi into the current directory and exit.",
    )
    parser.add_argument(
        "--bbox",
        type=parse_bbox,
        default=None,
        help="Initial target box as x,y,w,h. Defaults to (638,238,48,86) for vtest.avi.",
    )
    parser.add_argument("--psr-threshold", type=float, default=7.0)
    parser.add_argument("--max-lost-frames", type=int, default=40)
    parser.add_argument("--confirmation-frames", type=int, default=3)
    parser.add_argument(
        "--recovery-delay-frames",
        type=int,
        default=10,
        help="Frames to wait after PSR collapse before template re-detection.",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("driftguard_output.avi")
    )
    parser.add_argument("--plot", type=Path, default=Path("psr_comparison.png"))
    parser.add_argument(
        "--metric-plot",
        type=Path,
        default=Path("success_precision.png"),
        help="OTB-style success/precision plot (written only with --gt).",
    )
    parser.add_argument("--log", type=Path, default=Path("driftguard_log.csv"))
    parser.add_argument("--report", type=Path, default=Path("driftguard_report.json"))
    parser.add_argument("--gt", type=Path, default=None, help="Optional GT CSV")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Process only the first N frames (useful for quick smoke tests).",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Show a live OpenCV window (press Esc to stop).",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.download:
        download_vtest()
        return 0

    try:
        return run(
            source=args.source,
            bbox_arg=args.bbox,
            output=args.output,
            plot_path=args.plot,
            metric_plot_path=args.metric_plot,
            log_path=args.log,
            report_path=args.report,
            psr_threshold=args.psr_threshold,
            max_lost_frames=args.max_lost_frames,
            confirmation_frames=args.confirmation_frames,
            recovery_delay_frames=args.recovery_delay_frames,
            show_display=args.display,
            max_frames=args.max_frames,
            gt_path=args.gt,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
