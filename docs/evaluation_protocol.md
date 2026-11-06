# Corridor Evaluation Protocol (v2)

This protocol defines the separate corridor-based evaluation. It does not
modify `evaluate_full.py` or any v1 result under `outputs/results/full`.
The v2 evaluator writes only to `outputs/results_v2/`.

## Label coordinate confirmation

The annotation rows are KITTI-style:

```text
type truncation occlusion alpha xmin ymin xmax ymax
height width length location_x location_y location_z rotation_y
```

In the collector, `make_label()` converts CARLA points to KITTI camera
coordinates with the convention:

- `x`: lateral, positive right
- `y`: vertical, positive down
- `z`: forward, positive away from the camera

This is stated in `collect_data.py:182` and implemented by `to_cam()`.
`make_label()` writes the bottom-centre camera position as `x y z` at
`collect_data.py:205` and formats it at `collect_data.py:247-249`.
Therefore:

- KITTI `location_x` (field 12, one-based) is the ground-truth lateral offset.
- KITTI `location_z` (field 14, one-based) is the ground-truth forward distance.
- KITTI `location_y` is vertical and is not used for corridor membership.

Representative labels, including `dataset/labels/000036.txt`, confirm that
these fields contain non-zero metric values.

The collector creates `walker.pedestrian.*` actors and
`controller.ai.walker` controllers (`collect_data.py:346-360`). The saved
label rows do not preserve actor IDs or controller provenance, so corridor
evaluation cannot identify the crossing intention of an individual label.

## Fixed corridor definitions

The **primary corridor** is:

```text
abs(ground_truth_location_x) <= 2.0 metres
and ground_truth_location_z <= 30.0 metres
```

Sensitivity corridors are reported separately, never as the headline:

```text
abs(ground_truth_location_x) <= 1.5 m and location_z <= 30 m
abs(ground_truth_location_x) <= 3.0 m and location_z <= 30 m
```

The primary corridor is the headline result. It is a geometric relevance
proxy, not a pedestrian-intention label or a collision-probability model.

Ground-truth pedestrians outside the selected corridor are ignored:

- a missed outside-corridor pedestrian is not an FN;
- a prediction matched to an outside-corridor pedestrian is not an FP;
- predictions not matched to an included ground-truth pedestrian remain FPs.

Empty frames are never removed silently. They are reported in their own
section with false-positive counts and frame-level false-alarm rates.

## Required reporting views

The report presents three views side by side:

1. **All pedestrians**: every labelled pedestrian.
2. **Primary corridor**: the 2.0 m corridor.
3. **Empty frames**: frames with zero labelled pedestrians.

The 1.5 m and 3.0 m corridors are sensitivity analyses.

For all-pedestrian and corridor views, report:

- precision, recall, and F1;
- distance MAE and AbsRel;
- critical-zone recall, with **all ground-truth critical pedestrians** as the
  denominator;
- false-brake rate per frame and per object.

For empty frames, report:

- false positives per empty frame;
- percentage of empty frames with at least one false alarm;
- frame-level precision, recall, and specificity.

Before metrics, print:

- the number of empty frames;
- the number of ground-truth pedestrians in each corridor.

## Uncertainty and reproducibility

Report 95% bootstrap confidence intervals. Resample complete sequences
(`metadata.csv:run_id`), not individual frames, so adjacent frames from one
CARLA run cannot appear independently in both a bootstrap sample and its
original estimate.

No threshold is tuned on the test split. The evaluator uses the fixed
confidence, matching, zone, and geometric-depth settings supplied on the
command line or in its documented defaults. Any future threshold selection
must use training/validation data only and be recorded separately.

