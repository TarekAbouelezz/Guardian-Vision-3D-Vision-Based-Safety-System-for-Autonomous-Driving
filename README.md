# Guardian Vision

Guardian Vision is a monocular pedestrian-safety pipeline for the Industrial
Applications of Computer Vision course at the University of Padova. It keeps
the working Ultralytics YOLO detector and adds metric distance, 3D localization
with uncertainty, risk zones, and a vehicle action policy.

For presentation navigation, see [presentation/](presentation/), organized
into model, CARLA data, and fine-tuning sections.

## Project status

### Implemented

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

Depth Anything V2, MiDaS/DPT, and Metric3D-named backends are implemented
behind the factory, but they have not been run end to end.

### Verified by tests or runs

- Label conversion from the CARLA annotations.
- Camera back-projection geometry.
- Covariance construction.
- Risk-zone logic and action selection.
- Evaluation metric helpers.
- Geometric pipeline smoke tests on 100 and 200 images.
- Complete geometric pipeline run on 6,567 images.
- Unit test suite covering labels, geometry, risk logic, and evaluation.

The latest geometric pipeline was executed successfully on:


| Smoke test    | Input images | Output images | Exit code |
| ------------- | -----------: | ------------: | --------: |
| 100-image run |          100 |           100 |         0 |
| 200-image run |          200 |           200 |         0 |

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

The annotated frames can be assembled into a video:

```powershell
python make_video.py --input outputs/fulltest_6566 `
  --output outputs/fulltest_6566_guardian.mp4 --fps 20
```

The resulting video contains 6,567 frames at 20 FPS.

## Full-dataset evaluation results

The pretrained `yolo11n.pt` model was evaluated on all 6,567 images against
the pedestrian rows in `dataset/labels`. A prediction was considered a true
positive when its IoU with a ground-truth pedestrian box was at least 0.50.
Distance was estimated with the geometric baseline from the predicted box
height.


| Metric                   |             Result |
| ------------------------ | -----------------: |
| True positives           |              5,538 |
| False positives          |              1,332 |
| False negatives          |              4,035 |
| Precision                |         **80.61%** |
| Recall                   |         **57.85%** |
| F1 score                 |         **67.36%** |
| Frame presence accuracy  |         **86.51%** |
| Matched distance samples |              5,538 |
| Distance MAE             |         **2.15 m** |
| Distance AbsRel          |         **11.78%** |
| Critical-zone recall     |         **90.71%** |
| False-brake rate         |          **3.21%** |
| Mean inference time      | **70.13 ms/image** |

Run the evaluation again with:

```powershell
python evaluate_full.py --weights yolo11n.pt `
  --images dataset/images --labels dataset/labels `
  --output outputs/results/full --imgsz 1280
```

Results are exported to `outputs/results/full/full_evaluation.json` and
`outputs/results/full/matched_distances.csv`.

These are baseline results from the pretrained detector and geometric depth;
they are not fine-tuned-model results. Frame presence accuracy should not be
confused with object recall: it only measures whether a frame contains at
least one predicted pedestrian when the labels contain at least one.

### How to read these numbers

- Frame presence accuracy is not object recall; it measures frame-level
  pedestrian presence rather than individual pedestrian detections.
- Relative-depth values are not metres without a calibration file.

v1 critical-zone recall (90.71%) is superseded: v2 computes it with all
ground-truth critical pedestrians as the denominator and gives 65.96%.

### Evaluation v2 (corridor protocol)

The v2 protocol reports all pedestrians, the primary 2.0 m corridor, and
separate 1.5 m and 3.0 m sensitivity corridors. The primary corridor uses
ground-truth `|lateral| <= 2.0 m` and `forward <= 30 m`; pedestrians outside
the selected corridor are ignored.


| View                   | Precision | Recall |     F1 | Distance MAE | AbsRel | Critical recall | False-brake/frame | False-brake/object |
| ---------------------- | --------: | -----: | -----: | -----------: | -----: | --------------: | ----------------: | -----------------: |
| All pedestrians        |    80.57% | 57.82% | 67.32% |       2.15 m | 11.79% |          65.96% |             2.82% |              2.50% |
| Primary corridor 2.0 m |    46.43% | 69.66% | 55.72% |       1.20 m | 13.13% |          27.75% |             2.82% |             26.20% |
| Corridor 1.5 m         |    39.73% | 69.07% | 50.44% |       1.22 m | 13.33% |          20.11% |             2.82% |             51.56% |
| Corridor 3.0 m         |    55.26% | 69.26% | 61.47% |       1.17 m | 13.02% |          45.17% |             2.82% |             14.08% |


| Empty-frame metric              | Result |
| ------------------------------- | -----: |
| Empty frames                    |  1,862 |
| False positives                 |     22 |
| False positives per empty frame | 0.0118 |
| Empty frames with a false alarm |  0.97% |
| Frame precision                 | 99.53% |
| Frame recall                    | 81.55% |
| Frame specificity               | 99.03% |

The primary corridor contains 1,661 ground-truth pedestrians; the 1.5 m and
3.0 m sensitivity corridors contain 1,274 and 2,381 respectively. The v2
report is written to `outputs/results_v2/`.

### Test-only v3 comparison

The following comparison uses the 994-frame test split only
(`dataset/yolo/test.txt`), with the existing `guardian_yolo11n-3` checkpoint
used unchanged. The older 6,567-frame results include training data and are
not directly comparable with this test-only evaluation.

| Metric | Old pretrained full-dataset result | New pretrained test result | New fine-tuned test result |
| ------ | ---------------------------------: | -------------------------: | -------------------------: |
| Precision | 80.61% | 83.25% | 96.30% |
| Recall | 57.85% | 53.75% | 85.13% |
| F1 score | 67.36% | 65.61% | 90.21% |
| mAP50 | Not reported | 45.64% | 85.30% |
| mAP50-95 | Not reported | 14.16% | 58.32% |
| Distance MAE | 2.15 m | 2.34 m | 3.40 m |
| Distance AbsRel | 11.78% | 13.01% | 15.18% |
| Critical-zone recall | 90.71% (superseded) | 55.00% | 60.00% |
| Mean inference time | 70.13 ms/image | 49.22 ms/image | 53.45 ms/image |

The earlier Ultralytics result of mAP50 91.96% and mAP50-95 61.95% is
superseded and untrusted because no surviving artifact identifies its YAML,
split, or checkpoint. The reproducible test-only fine-tuned result is mAP50
85.30% and mAP50-95 58.32%.

Evidence: `outputs/results/full/full_evaluation.json`,
`outputs/results_v3/ultralytics_val_metrics.json`,
`outputs/results_v3/finetuned_best_report.json`, and
`outputs/results_v3/pretrained_report.json`.

## Strengths

- **Own evaluation dataset.** The project uses 6,567 CARLA frames and 9,573
  pedestrian boxes with exact ground-truth distance and occlusion values,
  including day/night and varied weather conditions. The dataset was created
  from scratch for the course rather than copied from an external benchmark.
- **Leakage-aware protocol.** Splits are made by `run_id::scenario`, so
  adjacent frames from the same sequence are never distributed across
  training, validation, and test sets.
- **Safety-oriented evaluation.** The evaluation reports distance error,
  critical-zone recall, and false-brake rate in addition to ordinary
  detection precision and recall.
- **Transparent baseline.** The geometric estimator is metric without
  training, using the camera model and assumed pedestrian height, and gives a
  clear reference for learned depth models.
- **Explicit uncertainty.** Each detection can be represented by a 3D Gaussian,
  while the action policy is configurable and can be derived from stopping
  distance.
- **Modular and reproducible.** Detector and depth factories, documented
  commands, and unit tests make the experiments repeatable and extensible.
- **Honest reporting.** Results from the pretrained detector and geometric
  depth estimator are explicitly labelled as baseline results.

## Limitations

- **No pedestrian intent or path awareness.** The system has no road geometry,
  drivable-area mask, crosswalk map, trajectory, pose, velocity, or
  time-to-collision model. It can therefore warn about pedestrians beside the
  road or facing away who will not cross the vehicle path. A corridor-based
  evaluation is planned as a geometric relevance proxy, not as an intent
  classifier.
- **Small and distant pedestrians.** The baseline object recall is 57.85%,
  with especially weak performance on small and distant boxes. Fine-tuning is
  intended to improve this.
- **Geometric distance assumptions.** The baseline assumes an approximately
  1.7 m pedestrian height and a reliable predicted box height. Occlusion,
  truncation, detector errors, and unusual poses can produce large distance
  errors. Relative-depth models additionally require calibration on training
  data before their values can be interpreted as metres.
- **Uncertainty is not yet used in decisions.** The Gaussian is constructed
  for uncertainty representation, but zone decisions currently use point
  distance rather than propagating covariance probabilistically.
- **Simulation only.** CARLA imagery and synthetic labels leave a sim-to-real
  gap for lighting, sensors, weather, traffic, and pedestrian behaviour.
- **Runtime.** CPU inference is about 70 ms/image and has not been validated
  for real-time vehicle use, especially at higher input resolutions or with
  transformer depth models.

## Roadmap (remaining work)

- [X]  Fine-tune YOLO and compare pretrained versus fine-tuned performance by
  distance bin.
- [ ]  Fit depth calibration on the training split and run all depth models
  end to end.
- [ ]  Replace or document the Metric3D/ZoeDepth compatibility backends.
- [ ]  Validate zone thresholds against ground-truth zones and speed scenarios.
- [ ]  Use Gaussian uncertainty in the risk decision.
- [ ]  Add the corridor-based evaluation and final results tables/plots.
- [ ]  Implement RT-DETR/GroundingDINO only if needed.

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
