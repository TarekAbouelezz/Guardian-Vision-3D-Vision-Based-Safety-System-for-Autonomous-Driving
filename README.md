# Guardian Vision

Guardian Vision is a monocular pedestrian-safety pipeline for the Industrial
Applications of Computer Vision course at the University of Padova. It keeps
the working Ultralytics YOLO detector and adds metric distance, 3D localization
with uncertainty, risk zones, and a vehicle action policy.

## Project status

### Implemented and verified

- CARLA KITTI-style annotation inspection and conversion to YOLO format.
- Sequence/scenario-disjoint train/validation/test splitting.
- YOLO detector factory and pedestrian-only detection.
- Geometric distance baseline using `fy * 1.7 / bbox_height_px`.
- Depth Anything V2, MiDaS/DPT, and Metric3D-named factory backends with
  optional training-split affine calibration.
- Central-region depth aggregation.
- Pinhole 3D back-projection and 3D Gaussian covariance with squared
  standard deviations.
- Static configurable risk zones and most-critical system action.
- Speed-derived stopping-distance and threshold-ablation helper functions.
- Evaluation helpers for distance metrics, zone confusion, critical recall,
  false-brake rate, timing hooks, CSVs, and plots.
- Zone-coloured output images with distance labels and action banners.
- Unit tests for labels, geometry, risk logic, and evaluation metrics.

The latest geometric pipeline was executed successfully on:

| Smoke test | Input images | Output images | Exit code |
|---|---:|---:|---:|
| 100-image run | 100 | 100 | 0 |
| 200-image run | 200 | 200 | 0 |

These smoke tests used `yolo11n.pt`, the geometric estimator, and CPU
inference. They verify runtime integration and output generation; they are not
the final detector or depth-model benchmark.

The complete source image set was also processed successfully:

```powershell
python run_fulltest.py
```

This run processed **6,567 images**, generated **6,567 annotated outputs** in
`outputs/fulltest_6566`, and completed with exit code 0. The pretrained
detector reported at least one person in 2,986 frames. This is a pipeline
smoke test, not a recall/mAP result: frames with no detections are still
rendered, and the run used the geometric estimator rather than a calibrated
learned-depth model. The detailed log is written to
`outputs/fulltest_6566.log`.

### Still required for the final research submission

- Run the complete YOLO fine-tuning job and save the resulting checkpoint.
- Run pretrained **and** fine-tuned detector evaluation over the complete test
  split and report the actual recall/mAP50 values for every distance bin.
- Fit and persist calibration parameters for each relative-depth model using
  training data only. The repository currently provides the calibration class,
  but does not provide a complete dataset calibration command.
- Run Depth Anything V2, MiDaS/DPT, and Metric3D experiments end-to-end and
  record their MAE, AbsRel, inference time, zone metrics, and plots.
- Validate the final thresholds against labelled ground-truth zones and
  vehicle-speed scenarios.
- Integrate the Gaussian uncertainty into a probabilistic risk decision. The
  current runtime constructs the Gaussian, but zone classification currently
  uses the point distance only.
- Replace or explicitly document the `Metric3D` and `ZoeDepth` compatibility
  backends: they currently use transformer relative-depth implementations,
  not dedicated Metric3D/ZoeDepth model architectures.
- Implement RT-DETR and GroundingDINO if alternative detector comparisons are
  required; those classes remain explicit `NotImplementedError` placeholders.
- Add a reproducible results table and plots generated from the final test
  split to the report.

## Pipeline

1. YOLO pedestrian detection.
2. Distance estimation: `geometric`, `depth_anything`, `midas`, or
   `metric3d` through the factory. The geometric estimator is the only one
   that is immediately metric without a calibration file.
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
split. The test split must not be used to fit it. Until such a file exists,
the relative-depth output is not a valid metric distance and should not be
used for final zone or safety claims:

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

The current evaluation code is an implementation of the reporting workflow;
it does not imply that final benchmark numbers have already been produced.

## Tests

```powershell
python -m unittest discover -s tests -p "test_*.py"
```

The tests cover label conversion, camera geometry, squared covariance
variances, stopping-distance zones, action priority, and distance metrics.

The current focused suite contains 11 tests:

```text
Ran 11 tests
OK
```

The 100- and 200-image smoke tests were run separately from the unit suite.

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
