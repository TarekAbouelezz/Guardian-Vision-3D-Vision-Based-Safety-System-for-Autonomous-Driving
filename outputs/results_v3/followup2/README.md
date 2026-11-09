# Guardian Vision documentation package: follow-up analysis v3

This package records the requested follow-up documentation without running new
experiments. Existing `README.md`, scripts, checkpoints, and result files were
preserved verbatim.

## Contents

- [PROJECT_HISTORY.md](PROJECT_HISTORY.md)
- [RESULTS.md](RESULTS.md)
- [QA.md](QA.md)
- [evaluation_protocol_addendum.md](evaluation_protocol_addendum.md)
- [BUG_FIXES.txt](BUG_FIXES.txt)

## Overview and pipeline

Guardian Vision is a monocular CARLA pedestrian-safety pipeline:

1. YOLO pedestrian detection.
2. Geometric or relative-depth distance estimation.
3. Pinhole 3D back-projection and Gaussian uncertainty.
4. Monitoring, warning, and critical-danger zones.
5. Most-critical vehicle action.

The existing runtime camera is 1242x375 with `fx=fy=621`, `cx=621`, and
`cy=187.5` ([README.md](../../../README.md), [collect_data.py](../../../collect_data.py)).

## Repository structure

The source structure is documented in the existing [README.md](../../../README.md).
The new follow-up documentation is isolated in this directory.

## Install and run

Existing install, runtime, conversion, training, evaluation, and test commands
remain unchanged in [README.md](../../../README.md). No command was changed or
executed by this documentation-only task.

## Inputs and outputs

The evaluated v3 artifacts identify `dataset/yolo/carla.yaml`,
`dataset/yolo/test.txt` with 994 frames, `yolo11n.pt`, and
`guardian_yolo11n-3/weights/best.pt` ([provenance.json](../followup/provenance.json)).
Existing output locations remain unchanged.

## Dataset and links

The dataset is local CARLA course data; the existing README links the official
CARLA documentation and describes the KITTI-style labels. No public dataset
link or KITTI detector evaluation was added because no corresponding result
exists.

## Key results

On the 994-frame test split, Ultralytics reports:

| Model | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| Pretrained | 0.8324607329842932 | 0.5374647887323943 | 0.4564127945390283 | 0.1415621067943813 |
| Fine-tuned `guardian_yolo11n-3` | 0.9630337794773741 | 0.8512676056338028 | 0.8529849261580128 | 0.583229556692938 |

Evidence: [ultralytics_val_metrics.json](../ultralytics_val_metrics.json).

Custom fixed-threshold results are documented in [RESULTS.md](RESULTS.md).
The main limitation is that fine-tuning improved detection but not geometric
distance estimation ([FINDINGS.md](../followup/FINDINGS.md)).

The older 91.96/61.95 mAP result is superseded and untrusted because no
surviving artifact identifies its YAML, split, or checkpoint
([provenance.json](../followup/provenance.json)).

## Strengths

The existing strengths are preserved in [README.md](../../../README.md):
own dataset, sequence-disjoint evaluation, safety-oriented metrics,
transparent geometric baseline, explicit uncertainty, modularity, and honest
reporting.

## Limitations

The existing limitations are preserved: no path awareness, small/distant
pedestrians, geometric-distance assumptions, uncertainty not used for
decisions, simulation-only data, and runtime constraints
([README.md](../../../README.md)).

## Roadmap

- [ ] Select calibration on validation using the no-overestimate rule.
- [ ] Add critical-zone decomposition.
- [ ] Compare depth models on identical fine-tuned detections.
- [ ] Add temporal voting for false brakes.
- [ ] Sweep NMS and inference settings on validation only.
- [ ] Perform held-out KITTI evaluation.

These are roadmap items, not completed experiments.

## Third-party code and AI/LLM declaration

Third-party licenses and the existing AI/LLM declaration remain in
[README.md](../../../README.md). The documentation work also used AI assistance
to organize existing audit artifacts and explicitly label unperformed work;
the project authors remain responsible for claims.

## References and documentation

See [evaluation_protocol.md](../../../docs/evaluation_protocol.md),
[FINDINGS.md](../followup/FINDINGS.md), [BUG_FIXES.txt](../followup/BUG_FIXES.txt),
and [PROJECT_HISTORY.md](PROJECT_HISTORY.md).

## Bug fixes

Listed separately in [BUG_FIXES.txt](BUG_FIXES.txt).
