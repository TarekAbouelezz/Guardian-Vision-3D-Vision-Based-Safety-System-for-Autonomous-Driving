# 03 - Fine-tuning and evaluation

## Main files

| Purpose | File |
|---|---|
| YOLO training/evaluation entry point | [finetune.py](../../finetune.py) |
| Full-dataset audit | [evaluate_full.py](../../evaluate_full.py) |
| Corridor evaluation | [evaluate_corridor.py](../../evaluate_corridor.py) |
| Test-only v3 evaluation | [evaluate_test_v3.py](../../evaluate_test_v3.py) |
| Critical-zone diagnosis | [diagnose_critical.py](../../diagnose_critical.py) |
| Pretrained weights | [yolo11n.pt](../../yolo11n.pt) |
| Fine-tuned checkpoint | [best.pt](../../runs/detect/outputs/training/guardian_yolo11n-3/weights/best.pt) |
| Training arguments | [args.yaml](../../runs/detect/outputs/training/guardian_yolo11n-3/args.yaml) |
| Training curves | [results.csv](../../runs/detect/outputs/training/guardian_yolo11n-3/results.csv) |

## Current test-only comparison

The final comparison uses 994 test frames and does not retrain or modify either
checkpoint.

| Metric | Pretrained | Fine-tuned |
|---|---:|---:|
| Ultralytics precision | 83.25% | 96.30% |
| Ultralytics recall | 53.75% | 85.13% |
| mAP50 | 45.64% | 85.30% |
| mAP50-95 | 14.16% | 58.32% |
| Custom F1 | 65.61% | 90.21% |
| Distance MAE | 2.34 m | 3.40 m |

Evidence: [ultralytics_val_metrics.json](../../outputs/results_v3/ultralytics_val_metrics.json),
[pretrained_report.json](../../outputs/results_v3/pretrained_report.json), and
[finetuned_best_report.json](../../outputs/results_v3/finetuned_best_report.json).

## Presentation message

Fine-tuning substantially improves detection, particularly for distant
pedestrians, but it does not improve the geometric distance estimate.
