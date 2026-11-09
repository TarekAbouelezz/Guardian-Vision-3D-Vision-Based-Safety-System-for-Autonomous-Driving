# Results

Every reported value below is copied from an existing artifact. Values for
the requested new validation sweeps and depth comparisons are marked pending.

## Table A: v1 versus v2, pretrained all-pedestrian view

| Version | Precision | Recall | F1 | MAE | AbsRel | Critical recall | False-brake |
|---|---:|---:|---:|---:|---:|---:|---:|
| v1 | 80.61% | 57.85% | 67.36% | 2.15 m | 11.78% | 90.71% | 3.21% |
| v2 | 80.57% | 57.82% | 67.32% | 2.15 m | 11.79% | 65.96% | PENDING (v2 report separates object/frame rates) |

Evidence: [full_evaluation.json](../../results/full/full_evaluation.json),
[corridor_report.json](../../results_v2/corridor_report.json).
The 0.03-point precision/recall/F1 differences have cause **not yet traced**.
The v1 critical recall 90.71% is superseded: v2 uses all ground-truth
critical pedestrians as denominator.

## Table B: v2 views

| View | Precision | Recall | F1 | MAE | AbsRel | Critical recall | False-brake/object |
|---|---:|---:|---:|---:|---:|---:|---:|
| All | 80.5676855895196% | 57.81886555938577% | 67.32348111658456% | 2.145282775191047 m | 11.79406740680678% | 65.95505617977528% | 2.499136243339053% |
| Corridor 1.5 m | 39.72911963582618% | 69.07378335949764% | 50.44425336772714% | 1.2219557663534275 m | 13.33361726209612% | 20.112359550561798% | 51.5625% |
| Corridor 2.0 m | 46.42857142857143% | 69.65683323299218% | 55.71875752468064% | 1.1964701917351286 m | 13.126706261316187% | 27.752808988764044% | 26.19974059662776% |
| Corridor 3.0 m | 55.26139410187667% | 69.25661486770265% | 61.4725069897437% | 1.1727518690944239 m | 13.02093646711351% | 45.1685393258427% | 14.084507042253522% |

The 2.0 m critical-recall CI is [0.19172281940210706,
0.3602464226089727]; its false-brake/object CI is
[0.1400906653089876, 0.44418961754313024]. Empty-frame metrics are 1,862
frames, 22 false positives, 0.011815252416756176 false positives/frame,
0.966702470461869% false-alarm frames, 0.9953307392996109 frame precision,
0.8155154091392136 frame recall, and 0.9903329752953813 specificity.
Evidence: [corridor_report.json](../../results_v2/corridor_report.json).

## Table C: v3 test comparison

| Model | Ultralytics precision | Ultralytics recall | mAP50 | mAP50-95 | Custom precision | Custom recall | Custom F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Pretrained | 0.8324607329842932 | 0.5374647887323943 | 0.4564127945390283 | 0.1415621067943813 | 0.83391 | 0.54074 | 0.65606 |
| Fine-tuned | 0.9630337794773741 | 0.8512676056338028 | 0.8529849261580128 | 0.583229556692938 | 0.96622 | 0.84598 | 0.90211 |

Evidence: [ultralytics_val_metrics.json](../ultralytics_val_metrics.json) and
[FINDINGS.md](../followup/FINDINGS.md). The fine-tuned distance-bin matched
counts are 276/347/429/464 for 0-10/10-20/20-30/30+ m, with unmatched GT
counts 58/33/31/154 and unmatched predicted-distance counts 13/5/12/23
([distance_bin_protocol.json](../followup/distance_bin_protocol.json)).
Mean inference times are 49.22 ms/image pretrained and 53.45 ms/image
fine-tuned ([FINDINGS.md](../followup/FINDINGS.md)).

The 91.96/61.95 result is superseded and untrusted. Any fine-tuned result on
all 6,567 frames includes training data and is not a generalisation result.
The v1/v2 baseline numbers use all 6,567 frames and are not directly
comparable with fine-tuned test results. Fine-tuning found far pedestrians
that pretrained inference did not, so unpaired distance MAE is not comparable;
the paired comparison uses 946 pedestrians.

## Table D: paired distance and geometry

| Comparison | MAE | AbsRel | Signed bias |
|---|---:|---:|---:|
| Paired pretrained, 946 | 2.355542983363429 m | 0.1315299079568675 | -0.21379086159140415 m |
| Paired fine-tuned, 946 | 2.4358639134803672 m | 0.13378262957868597 | -1.3325554392823102 m |
| Fixed 1.7 m | 3.400756671750405 m | 0.15178284548686058 | -0.6908021468515079 m |
| Learned 1.8760370370370374 m | 2.490418225702719 m | 0.11859784283384257 | +1.6484456473718914 m |
| Affine | 3.5657927996479937 m | 0.19660543776963638 | -0.02074556427914656 m |

Evidence: [paired_distance.json](../followup/paired_distance.json) and
[geometry_bias.json](../followup/geometry_bias.json). The learned height's
positive bias is the unsafe overestimation direction for braking.

## Table E: critical denominators

| Model | View | Detected critical / denominator | Recall | Recall CI |
|---|---|---:|---:|---:|
| Pretrained | All | 88 / 160 | 0.55 | [0.3583269650655022, 0.7805011572013529] |
| Fine-tuned | All | 96 / 160 | 0.6 | [0.3893682399213373, 0.8507462686567164] |
| Pretrained | Corridor | 23 / 54 | 0.42592592592592593 | [0.0, 0.75] |
| Fine-tuned | Corridor | 28 / 54 | 0.5185185185185185 | [0.0, 0.875] |

The CIs overlap in both views. The differences are small and may not be
significant. In `critical_denominators.json`, “detected” means a ground-truth
critical pedestrian has an IoU >= 0.5 one-to-one match and its matched
prediction is classified critical by predicted geometric distance.

## Table F: training summary and conditions

| Setting | Value |
|---|---|
| Epochs | 50 |
| Image size | 1280 |
| Batch | 2 |
| Optimizer | auto |
| lr0 / lrf | 0.01 / 0.01 |
| Freeze | null |
| Seed | 0 |
| Best epoch by validation mAP50 | 22 |
| Best validation mAP50 | 0.93874 |
| Validation mAP50-95 at best epoch | 0.62848 |
| Final train box loss | 0.63443 |
| Final validation box loss | 0.83955 |
| Validation-minus-training gap | 0.20511999999999997 |

Evidence: [guardian_yolo11n-3_args.yaml](../followup/guardian_yolo11n-3_args.yaml),
[results.csv](../../../runs/detect/outputs/training/guardian_yolo11n-3/results.csv),
[train_validation_gap.json](../training_curves/train_validation_gap.json).

Condition composition is documented in
[condition_summary.json](../condition_summary.json): train 4,580 frames,
validation 993, test 994; weather counts are respectively
clear_day/fog/night/rain = 990/1251/1361/978,
469/201/47/276, and 137/339/286/232.

## Requested follow-up experiments

| Part | Status |
|---|---|
| Validation-only calibration selection | PENDING (validation predictions/settings not present) |
| Critical-zone decomposition | PENDING (no decomposition artifact present) |
| NMS/imgsz/confidence sweep | Not performed; hard rule forbids new experiments |
| One-time selected-setting test run | PENDING |
| Depth Anything V2 and second estimator on same detections | Not performed; no output exists |
| Sliced inference | Not performed; untested option only |
