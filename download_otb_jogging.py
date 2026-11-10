#!/usr/bin/env python3
"""Download OTB-2015 Jogging and build the benchmark assets.

The 307-frame sequence is not committed to the repository because it is a
research dataset. This script downloads the archived OTB Jogging zip, converts
the frames to the MJPG AVI used by DriftGuard, and writes the two official
ground-truth annotations as CSV files.

Source:
    https://web.archive.org/web/20220210102816/
    http://cvlab.hanyang.ac.kr/tracker_benchmark/seq/Jogging.zip

OTB citation:
    Yi Wu, Jongwoo Lim, Ming-Hsuan Yang.
    "Object Tracking Benchmark." IEEE TPAMI, 2015.
"""

from __future__ import annotations

import csv
import io
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "datasets"
URL = (
    "https://web.archive.org/web/20220210102816/"
    "http://cvlab.hanyang.ac.kr/tracker_benchmark/seq/Jogging.zip"
)


def download_zip() -> bytes:
    request = urllib.request.Request(
        URL,
        headers={"User-Agent": "DriftGuard-dataset-downloader/1.0"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def write_gt(lines: list[str], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["frame", "x", "y", "w", "h"])
        for frame, line in enumerate(lines):
            values = [v for v in line.replace("\t", ",").replace(" ", ",").split(",") if v.strip()]
            if len(values) >= 4:
                writer.writerow([frame] + [float(v) for v in values[:4]])


def main() -> int:
    DATA.mkdir(parents=True, exist_ok=True)
    video_path = DATA / "jogging_1.avi"
    gt1_path = DATA / "jogging_1_gt.csv"
    gt2_path = DATA / "jogging_2_gt.csv"

    if video_path.exists() and gt1_path.exists() and gt2_path.exists():
        print("OTB Jogging assets already exist in datasets/; nothing to do.")
        return 0

    print("Downloading OTB-2015 Jogging archive ...")
    payload = download_zip()
    print(f"Downloaded {len(payload) / 1e6:.1f} MB")

    with tempfile.TemporaryDirectory(prefix="driftguard-jogging-") as tmp:
        tmp_path = Path(tmp)
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            archive.extractall(tmp_path)

        jpg_files = sorted(tmp_path.rglob("*.jpg"))
        if not jpg_files:
            raise RuntimeError("The archive did not contain Jogging JPG frames")

        first = cv2.imread(str(jpg_files[0]))
        if first is None:
            raise RuntimeError(f"Could not read {jpg_files[0]}")
        height, width = first.shape[:2]

        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"MJPG"),
            30.0,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Could not create {video_path}")
        for path in jpg_files:
            frame = cv2.imread(str(path))
            if frame is not None:
                writer.write(frame)
        writer.release()

        gt1 = next(tmp_path.rglob("groundtruth_rect.1.txt"), None)
        gt2 = next(tmp_path.rglob("groundtruth_rect.2.txt"), None)
        if gt1 is None or gt2 is None:
            raise RuntimeError("The archive did not contain the two OTB annotation files")
        write_gt(gt1.read_text(encoding="utf-8", errors="replace").splitlines(), gt1_path)
        write_gt(gt2.read_text(encoding="utf-8", errors="replace").splitlines(), gt2_path)

    print(f"Wrote {video_path} ({video_path.stat().st_size / 1e6:.1f} MB)")
    print(f"Wrote {gt1_path.name} and {gt2_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
