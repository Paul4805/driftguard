# OTB Jogging ground-truth benchmark

This directory adds a real, annotated single-object tracking benchmark to
DriftGuard. It is the **OTB-2015 Jogging** sequence from the Visual Tracker
Benchmark (OTB100).

## Source

- Dataset: OTB-2015 / Visual Tracker Benchmark
- Sequence: `Jogging`
- Authors: Yi Wu, Jongwoo Lim, Ming-Hsuan Yang
- Paper: *Object Tracking Benchmark*, TPAMI 2015
- Original project page: `http://cvlab.hanyang.ac.kr/tracker_benchmark/`
- Download used: the archived `Jogging.zip` copy from the Wayback Machine
  (`https://web.archive.org/web/20220210102816/http://cvlab.hanyang.ac.kr/tracker_benchmark/seq/Jogging.zip`),
  the same archive used by the public `OTB100_Dataset` download script.

The OTB data is intended for research use. Cite the OTB paper if you publish
results obtained with it.

## Files

| File | Description |
|---|---|
| `jogging_1.avi` | All 307 Jogging frames, 352x288, 30 FPS, MJPG (download; not committed) |
| `jogging_1_gt.csv` | Official `groundtruth_rect.1.txt` converted to `frame,x,y,w,h` |
| `jogging_2_gt.csv` | Official `groundtruth_rect.2.txt` converted to `frame,x,y,w,h` |
| `jogging1_output.avi` | DriftGuard output for target 1 |
| `jogging2_output.avi` | DriftGuard output for target 2 |
| `jogging1_psr.png` | PSR comparison for target 1 |
| `jogging2_psr.png` | PSR comparison for target 2 |
| `jogging1_success_precision.png` | OTB success/precision curves for target 1 |
| `jogging2_success_precision.png` | OTB success/precision curves for target 2 |
| `jogging1_log.csv`, `jogging2_log.csv` | Per-frame logs |
| `jogging1_report.json`, `jogging2_report.json` | Full reports and GT metrics |
| `benchmark_summary.json` | Frame-weighted aggregate produced by `benchmark_otb.py` |

Jogging contains two independently annotated runners in the same video:

- `jogging_1_gt.csv`: the left runner, initial box `(111, 98, 25, 101)`.
- `jogging_2_gt.csv`: the right runner, initial box `(180, 79, 37, 114)`.

The left runner is briefly occluded by a large pole. The right runner is also
hidden by the same pole while the camera pans. This is a good test of update
freezing and recovery.

## Reproduce

The 307-frame AVI is not committed to the repository (large binary). Download
and build it with:

```bash
python download_otb_jogging.py
```

Then, from the project root:

```bash
python driftguard.py \
  --source datasets/jogging_1.avi \
  --bbox 111,98,25,101 \
  --gt datasets/jogging_1_gt.csv \
  --output datasets/jogging1_output.avi \
  --plot datasets/jogging1_psr.png \
  --metric-plot datasets/jogging1_success_precision.png \
  --log datasets/jogging1_log.csv \
  --report datasets/jogging1_report.json

python driftguard.py \
  --source datasets/jogging_1.avi \
  --bbox 180,79,37,114 \
  --gt datasets/jogging_2_gt.csv \
  --output datasets/jogging2_output.avi \
  --plot datasets/jogging2_psr.png \
  --metric-plot datasets/jogging2_success_precision.png \
  --log datasets/jogging2_log.csv \
  --report datasets/jogging2_report.json
```

## Results

| Sequence | Tracker | Success@0.5 | Success AUC | Mean IoU | Precision@20px | Precision AUC |
|---|---|---:|---:|---:|---:|---:|
| Jogging-1 | **DriftGuard** | **0.967** | **0.770** | **0.773** | **1.000** | **0.873** |
| Jogging-1 | Baseline KCF | 0.225 | 0.185 | 0.178 | 0.231 | 0.218 |
| Jogging-2 | **DriftGuard** | **1.000** | **0.777** | **0.780** | **1.000** | **0.925** |
| Jogging-2 | Baseline KCF | 1.000 | 0.741 | 0.743 | 1.000 | 0.910 |
| **Aggregate** | **DriftGuard** | **0.984** | **0.773** | **0.776** | **1.000** | **0.899** |
| **Aggregate** | Baseline KCF | 0.612 | 0.463 | 0.461 | 0.616 | 0.564 |

Aggregate is frame-weighted over the two 307-frame evaluations.

Recovery behaviour:

- Jogging-1: 1 drift episode, 1 confirmed recovery via template re-init.
- Jogging-2: 1 drift episode, 1 confirmed recovery via **frozen-KCF
  reacquisition** (the pole occlusion only lasted a few frames; the frozen
  filter re-locked before template re-detection was needed).

The baseline is not failure-aware. On Jogging-1 it permanently updates its
model while the pole covers the target and never recovers. DriftGuard freezes
the model, reacquires the target, and reaches 96.7% success with 100%
precision.
