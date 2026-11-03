# Guardian Vision

Guardian Vision is a monocular pedestrian-safety pipeline for the Industrial
Applications of Computer Vision course at the University of Padova. It keeps
the working Ultralytics YOLO detector and adds metric distance, 3D localization
with uncertainty, risk zones, and a vehicle action policy.

## Pipeline

1. YOLO pedestrian detection.
2. Distance estimation: `geometric`, `depth_anything`, `midas`, or `metric3d`.
3. Central-region distance aggregation and pinhole back-projection.
4. A 3D Gaussian `(mean, covariance)` for every detection.
5. Zone classification: monitoring (alarm), warning (reduce speed), and
   critical danger (emergency brake).
6. The system action is the most critical action among all detections.

The CARLA camera is 1242x375 with horizontal FOV 90 degrees:
`fx=fy=621`, `cx=621`, and `cy=187.5`.

## Repository structure

```text
convert_labels.py       KITTI-style CARLA -> YOLO conversion
split_dataset.py        sequence/scenario-disjoint train/val/test split
finetune.py             YOLO11n training and distance-binned evaluation
dataset/
  labels/               original KITTI-style annotations
  images/               source images
  vis/                  human visualization only; never inference/training
src/
  detection/            detector interface, factory, and YOLO backend
  depth/                geometric and calibrated relative-depth backends
  localization/         camera back-projection and Gaussian uncertainty
  risk/                 zones, stopping distance, and action policy
  evaluation/           metrics, CSV export, and ablation plots
tests/                  unit tests for labels, geometry, risk, and metrics
```

## Install and run

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py --image dataset/images/000098.png
```

The output is written to `outputs/images/*_guardian.jpg`. The overlay contains
zone-coloured boxes, distance labels, and a system-action banner. The runtime
configuration is [src/config/config.yaml](src/config/config.yaml).

Select the geometric baseline without changing code:

```yaml
depth:
  model: geometric
```

Relative-depth models must use a calibration JSON produced from the training
split. The test split must not be used to fit it:

```json
{"scale": 12.4, "offset": 0.8}
```

## Dataset annotations and splits

Each non-empty row in `dataset/labels/*.txt` is a KITTI object record:

```text
type truncation occlusion alpha xmin ymin xmax ymax
height width length location_x location_y location_z rotation_y
```

`type` is `Pedestrian`, `Car`, etc.; the pedestrian metric distance is
`location_z` (field 14, one-based). The pixel boxes and metric distances are
real values, not placeholders. The conversion keeps only `Pedestrian` rows,
assigns class 0, and normalizes `x_center y_center width height`.

```powershell
python convert_labels.py
python split_dataset.py
```

Expected output for the supplied dataset is 6,567 converted files, 9,573
pedestrian boxes, and sequence/scenario-disjoint splits of 4,580/993/994
frames. The generated YOLO dataset is `dataset/yolo/` and its data file is
`dataset/yolo/carla.yaml`. Splits are made by `run_id::scenario`, never by
randomly distributing adjacent frames.

The dataset was collected from CARLA
([official simulator documentation](https://carla.readthedocs.io/)); it is
local course data rather than a redistributed public benchmark. `dataset/vis`
contains boxes already drawn and is deliberately excluded from all commands.

## Training and evaluation

Fine-tune YOLO11n at the requested resolution:

```powershell
python finetune.py --train --weights yolo11n.pt --imgsz 1280 --epochs 50
```

Evaluate a checkpoint by ground-truth distance bin:

```powershell
python finetune.py --evaluate --weights yolo11n.pt --imgsz 1280 `
  --output outputs/results/pretrained_distance.csv
python finetune.py --evaluate --weights outputs/training/guardian_yolo11n/weights/best.pt `
  --imgsz 1280 --output outputs/results/finetuned_distance.csv
```

Distance evaluation reports recall and mAP50 for 0-10, 10-20, 20-30, and 30+
metres. A prediction CSV can be passed to the full evaluation exporter:

```powershell
python -m src.evaluation.run predictions.csv --output outputs/results
```

It writes metric tables and a critical-recall/false-brake threshold-ablation
plot under `outputs/results/`.

## Tests

```powershell
python -m unittest discover -s tests -p "test_*.py"
```

The tests cover label conversion, camera geometry, squared covariance
variances, stopping-distance zones, action priority, and distance metrics.

## Third-party code and licenses

- Ultralytics YOLO: AGPL-3.0, https://github.com/ultralytics/ultralytics
- PyTorch: BSD-style license, https://github.com/pytorch/pytorch
- Transformers: Apache-2.0, https://github.com/huggingface/transformers
- OpenCV: Apache-2.0, https://opencv.org/license/
- NumPy, Matplotlib, PyYAML: their respective permissive open-source licenses
- CARLA: MIT, https://github.com/carla-simulator/carla

Model weights remain separate from source control and retain their upstream
license/model-card obligations.

## AI/LLM declaration

AI/LLM assistance was used to help inspect the existing repository, draft
small Python utilities, suggest tests, and improve documentation. The project
authors reviewed, tested, and are responsible for the final algorithms,
configuration, results, and claims. No private dataset or credentials were
submitted to a third party.
