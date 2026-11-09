# Evaluation protocol addendum (v3 follow-up 2)

This is a documentation-only addendum. The repository's existing
`docs/evaluation_protocol.md` was not modified because the task's hard rule
requires new files only under `outputs/results_v3/followup2/`.

## Decision rules

1. Calibration is selected on the **VALIDATION** split only.
2. Select the setting with the highest validation critical-zone recall among
   settings whose validation false-brake per object is `<= 5%` and whose mean
   signed distance error is `<= +0.5 m` (no large overestimation).
3. Use the **TEST** split once, for the final report of the selected settings.
4. No calibration, threshold, NMS setting, image size, confidence, or model
   choice may be fitted or tuned using test results.

## Status

The requested validation calibration selection, detector sweep, and depth
comparison were **not performed** in this documentation-only task. No
validation predictions for those settings exist in the inspected artifacts, so
the selected setting is:

`PENDING (run validation-only calibration and detector sweep first)`

The requested final test report is:

`PENDING (run the selected validation setting once on test)`
