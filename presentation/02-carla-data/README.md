# 02 - CARLA data

## Main files

| Purpose | File |
|---|---|
| CARLA collection | [collect_data.py](../../collect_data.py) |
| Scenario collection | [Collect_carla_scenarios.py](../../Collect_carla_scenarios.py) |
| KITTI-style to YOLO conversion | [convert_labels.py](../../convert_labels.py) |
| Sequence/scenario split | [split_dataset.py](../../split_dataset.py) |
| Original annotations | [dataset/labels/](../../dataset/labels) |
| Source images | [dataset/images/](../../dataset/images) |
| Dataset metadata | [dataset/metadata.csv](../../dataset/metadata.csv) |
| YOLO dataset configuration | [dataset/yolo/carla.yaml](../../dataset/yolo/carla.yaml) |

## Dataset summary

- 6,567 CARLA frames.
- 9,573 pedestrian boxes.
- Train/validation/test split: 4,580/993/994 frames.
- Splitting is by `run_id::scenario`, not by adjacent random frames.
- Camera: 1242x375, 90-degree FOV, `fx=fy=621`.

## Important rule

`dataset/vis/` contains visualization images with boxes already drawn. It is
not used for inference or training.
