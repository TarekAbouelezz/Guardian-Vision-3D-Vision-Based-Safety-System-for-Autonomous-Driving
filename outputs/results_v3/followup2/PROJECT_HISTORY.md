# Project history

This chronology is based on the existing repository documentation and saved
artifacts. New experiments requested in follow-up 2 were not run.

1. **Guidelines and idea.** Pedestrian detection, 3D position, three safety
   zones, and vehicle actions were defined in the project brief and existing
   README.
2. **Phased plan.** Conversion, sequence-disjoint splits, detector training,
   depth, localization, risk, evaluation, visualization, and documentation
   were implemented in phases ([README.md](../../../README.md)).
3. **KITTI DataLoader pipeline.** KITTI was used as the label-format/protocol
   reference; a separate held-out KITTI detector evaluation is **Not
   performed** because no output exists.
4. **CARLA setup.** Collection scripts configure a 1242x375, 90-degree camera
   and include CARLA sensors ([collect_data.py](../../../collect_data.py)).
5. **Dataset.** The documented dataset contains 6,567 frames and 9,573
   pedestrian boxes; the existing README also records 1,862 empty frames and
   condition coverage ([README.md](../../../README.md),
   [condition_summary.json](../condition_summary.json)).
6. **Repository architecture.** Detection, depth, localization, risk,
   evaluation, tests, conversion, splitting, and runtime modules are described
   in [README.md](../../../README.md).
7. **Baseline diagnosis.** Small and distant pedestrians were missed by the
   pretrained detector ([README.md](../../../README.md)).
8. **Label conversion and split.** The sequence-disjoint split contains
   4,580/993/994 train/validation/test frames
   ([README.md](../../../README.md), [condition_summary.json](../condition_summary.json)).
9. **v1 full-dataset evaluation.** The all-6,567-frame pretrained baseline
   reported precision 80.61%, recall 57.85%, F1 67.36%, MAE 2.15 m, AbsRel
   11.78%, critical recall 90.71%, and false-brake rate 3.21%
   ([full_evaluation.json](../../results/full/full_evaluation.json)).
10. **Audit.** Audit artifacts exist under `outputs/audit/`; no new audit was
    run here.
11. **v2 corridor evaluation.** The protocol used one-to-one matching,
    sequence-bootstrap CIs, and 2.0 m/1.5 m/3.0 m views
    ([evaluation_protocol.md](../../../docs/evaluation_protocol.md),
    [corridor_report.json](../../results_v2/corridor_report.json)).
12. **Fine-tuning.** The saved run is `guardian_yolo11n-3`, 50 epochs,
    `imgsz: 1280`, batch 2, and validation split
    ([guardian_yolo11n-3_args.yaml](../followup/guardian_yolo11n-3_args.yaml)).
13. **Test-split comparison v3.** Inference-only comparison used 994 test
    frames and the existing checkpoints
    ([provenance.json](../followup/provenance.json)).
14. **Follow-up analysis.** Existing follow-up artifacts cover paired distance
    on 946 pedestrians, geometry bias, critical denominators, 0-10 m misses,
    and corrected distance-bin accounting
    ([FINDINGS.md](../followup/FINDINGS.md)).
15. **Remaining work.** Validation-only calibration selection, critical
    decomposition, detector-setting sweep, same-detection depth comparison,
    and held-out KITTI evaluation remain pending.
