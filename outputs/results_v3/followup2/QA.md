# Questions and answers

## 1. Danger zones

Thresholds are critical `< 7.0 m`, warning `< 15.0 m`, otherwise monitoring;
the runtime config uses reaction time 1.0 s, deceleration 6.0 m/s², and
warning buffer 5.0 m ([src/config/config.yaml](../../../src/config/config.yaml),
[src/risk/__init__.py](../../../src/risk/__init__.py)). The speed-derived formula
exists, but the evaluated reports use fixed thresholds. Status: **Done**.

## 2. Distance

The geometric estimator is `Z = 621 * 1.7 / box_height`
([src/depth/geometric.py](../../../src/depth/geometric.py)). Existing relative
depth backends require calibration; no completed metric depth comparison exists.
LiDAR is not used in the evaluated pipeline. Status: **Done for geometric;
PENDING for depth-model comparison**.

## 3. Final results

See Table C in [RESULTS.md](RESULTS.md). Status: **Done for existing v3;
PENDING for follow-up-2 selected settings**.

## 4. YOLO version and size

The saved run uses YOLO11n at image size 1280
([guardian_yolo11n-3_args.yaml](../followup/guardian_yolo11n-3_args.yaml)).
The rationale recorded by the project is real-time suitability, modest model
size, pretrained weights, and a simple API. Other model sizes were not
benchmarked. Status: **Done**.

## 5. Training details

The run used 50 epochs, batch 2, `optimizer: auto`, `lr0: 0.01`, `lrf: 0.01`,
`freeze: null`, seed 0, and validation split
([guardian_yolo11n-3_args.yaml](../followup/guardian_yolo11n-3_args.yaml)).
Status: **Done**.

## 6. COCO person to Pedestrian mapping

Only class 0 is retained by the YOLO conversion/detection protocol; CARLA
Pedestrian labels are used for training ([README.md](../../../README.md),
[dataset/yolo/carla.yaml](../../../dataset/yolo/carla.yaml)). Status: **Done**.

## 7. KITTI and CARLA split

KITTI is the annotation/protocol reference, not a reported detector-training
dataset. CARLA splits are by `run_id::scenario`
([docs/evaluation_protocol.md](../../../docs/evaluation_protocol.md)). Status:
**Done**.

## 8. Held-out KITTI evaluation

Not performed: no held-out KITTI evaluation output exists. Status: **Not
performed**.

## 9. CARLA images

The dataset contains 6,567 frames and conditions are listed in
[condition_summary.json](../condition_summary.json). The exact full
pedestrian distance range was not found in the inspected evidence. Status:
**PENDING (compute from labels first)**.

## 10. Camera comparison

CARLA uses 1242x375 and 90 degrees
([collect_data.py](../../../collect_data.py)); the project documentation records
`fx=621`, `fy=621`, `cx=621`, and `cy=187.5`. A verified KITTI focal length
comparison was not found in the inspected artifacts. Mounting height was not
verified. Status: **PENDING for KITTI comparison/mounting height**.

## 11. CARLA versus KITTI performance drop

Not measurable without the held-out KITTI evaluation. Status: **Not performed**.

## 12. Failure cases

The existing diagnosis records 42 of 58 close-range misses overlapping another
pedestrian and 35 of 58 extending past the image border
([zero_to_ten_diagnosis.json](../followup/zero_to_ten_diagnosis.json)).
The 30 annotated images are in [worst_misses](../followup/worst_misses).
Status: **Done for existing diagnosis**.

## 13. Real-time

Existing v3 mean inference time is 49.22 ms/image pretrained and 53.45
ms/image fine-tuned ([FINDINGS.md](../followup/FINDINGS.md)). Hardware is CPU
in the existing protocol; p95 timing and a target FPS were not saved. Status:
**PENDING for p95/target comparison**.

## 14. Why CARLA, KITTI, and YOLO

CARLA supplies exact synthetic ground truth and safe scenarios; KITTI supplies
the label-format/protocol reference; YOLO supplies the detector ecosystem and
speed. Alternatives were not benchmarked. Status: **Done as rationale; not a
comparative benchmark**.

## 15. Checkpoint

`best.pt` is `runs/detect/outputs/training/guardian_yolo11n-3/weights/best.pt`;
the saved run selected best by validation mAP50 at epoch 22
([provenance.json](../followup/provenance.json),
[train_validation_gap.json](../training_curves/train_validation_gap.json)).
Status: **Done**.

## 16. Training curves

The existing artifacts are `results.png` and
`loss_and_val_map_curves.png`. The saved validation-minus-training box-loss
gap is 0.20511999999999997
([train_validation_gap.json](../training_curves/train_validation_gap.json)).
Training loss finished at 0.63443 versus validation loss 0.83955. This is a
gap, not proof of catastrophic over-fitting; the test result is lower than
validation, and the condition distribution differs. Status: **Done**.

## 17. Corridor weak points

Fine-tuned corridor critical recall is 28/54 = 0.5185185185185185 and
false-brake/object is 22/50 = 0.44, with CIs in
[critical_denominators.json](../followup/critical_denominators.json). Causes
such as projection error, groups, truncation, and corridor filtering are
hypotheses unless directly supported by the miss diagnosis. Status: **Done
for measurements; hypotheses clearly labelled**.
