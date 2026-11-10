# 01 - Model and safety pipeline

## Main files

| Area | File |
|---|---|
| Runtime entry point | [main.py](../../main.py) |
| Detection factory | [src/detection/](../../src/detection) |
| Depth estimators | [src/depth/](../../src/depth) |
| 3D localization | [src/localization/localizer.py](../../src/localization/localizer.py) |
| Risk zones and actions | [src/risk/__init__.py](../../src/risk/__init__.py) |
| Runtime configuration | [src/config/config.yaml](../../src/config/config.yaml) |
| Video generation | [make_video.py](../../make_video.py) |

## Pipeline

1. YOLO pedestrian detection.
2. Geometric or relative-depth distance estimation.
3. Pinhole 3D back-projection.
4. 3D Gaussian uncertainty.
5. Monitoring, warning, or critical-danger classification.
6. Most-critical vehicle action.

## Presentation message

The model has strong pedestrian detection after fine-tuning, but geometric
distance estimation remains the main safety limitation.
