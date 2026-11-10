#!/usr/bin/env python3
"""Reproduce the DriftGuard vs baseline evaluation on OTB Jogging.

This is a thin wrapper around ``driftguard.py`` that runs both annotated
targets, reads the generated JSON reports, and prints a frame-weighted
aggregate plus per-sequence results.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "datasets"

CASES = [
    {
        "name": "Jogging-1",
        "bbox": "111,98,25,101",
        "gt": DATA / "jogging_1_gt.csv",
        "prefix": "jogging1",
    },
    {
        "name": "Jogging-2",
        "bbox": "180,79,37,114",
        "gt": DATA / "jogging_2_gt.csv",
        "prefix": "jogging2",
    },
]

METRICS = (
    "success_rate_iou_0.5",
    "success_auc",
    "mean_iou",
    "precision_center_20px",
    "precision_auc_50px",
    "mean_center_error_px",
)


def run_case(case: dict) -> dict:
    prefix = case["prefix"]
    command = [
        sys.executable,
        str(ROOT / "driftguard.py"),
        "--source",
        str(DATA / "jogging_1.avi"),
        "--bbox",
        case["bbox"],
        "--gt",
        str(case["gt"]),
        "--output",
        str(DATA / f"{prefix}_output.avi"),
        "--plot",
        str(DATA / f"{prefix}_psr.png"),
        "--metric-plot",
        str(DATA / f"{prefix}_success_precision.png"),
        "--log",
        str(DATA / f"{prefix}_log.csv"),
        "--report",
        str(DATA / f"{prefix}_report.json"),
    ]
    subprocess.run(command, check=True)
    with (DATA / f"{prefix}_report.json").open(encoding="utf-8") as fh:
        return json.load(fh)


def main() -> int:
    reports = []
    for case in CASES:
        print(f"\n=== {case['name']} ===")
        reports.append(run_case(case))

    aggregate: dict[str, dict[str, float]] = {"driftguard": {}, "baseline": {}}
    total_frames = sum(
        int(report["ground_truth"]["driftguard"]["gt_frames"]) for report in reports
    )
    for tracker in ("driftguard", "baseline"):
        for metric in METRICS:
            weighted = sum(
                float(report["ground_truth"][tracker][metric])
                * int(report["ground_truth"]["driftguard"]["gt_frames"])
                for report in reports
            )
            aggregate[tracker][metric] = weighted / max(1, total_frames)

    summary = {
        "sequences": [case["name"] for case in CASES],
        "total_gt_frames": total_frames,
        "per_sequence": {
            case["name"]: report["ground_truth"]
            for case, report in zip(CASES, reports)
        },
        "aggregate": aggregate,
    }
    out_path = DATA / "benchmark_summary.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print("\n" + "=" * 82)
    print("OTB JOGGING BENCHMARK SUMMARY")
    print("=" * 82)
    header = (
        f"{'Sequence':<12} {'Tracker':<12} {'Succ@.5':>8} {'AUC':>7} "
        f"{'mIoU':>7} {'Prec@20':>8} {'PrAUC':>7}"
    )
    print(header)
    print("-" * 82)
    for case, report in zip(CASES, reports):
        for tracker in ("driftguard", "baseline"):
            m = report["ground_truth"][tracker]
            name = "DriftGuard" if tracker == "driftguard" else "Baseline"
            print(
                f"{case['name']:<12} {name:<12} "
                f"{m['success_rate_iou_0.5']:8.3f} {m['success_auc']:7.3f} "
                f"{m['mean_iou']:7.3f} {m['precision_center_20px']:8.3f} "
                f"{m['precision_auc_50px']:7.3f}"
            )
    print("-" * 82)
    for tracker in ("driftguard", "baseline"):
        m = aggregate[tracker]
        name = "DriftGuard" if tracker == "driftguard" else "Baseline"
        print(
            f"{'AGGREGATE':<12} {name:<12} "
            f"{m['success_rate_iou_0.5']:8.3f} {m['success_auc']:7.3f} "
            f"{m['mean_iou']:7.3f} {m['precision_center_20px']:8.3f} "
            f"{m['precision_auc_50px']:7.3f}"
        )
    print("=" * 82)
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
