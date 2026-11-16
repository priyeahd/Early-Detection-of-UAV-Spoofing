# sail_2p manual run

This experiment uses all 360 audited merged flights: 288 train and 72 test,
stratified by scenario and condition with seed 42. Entire flights stay together.
80/20 refers to flights; sample counts vary because flight lengths differ.

It creates a new XGBClassifier and fits 200 trees, depth 4, learning rate 0.08,
using the same 42 causal-by-construction telemetry features as Sail 1. No old
model is loaded during training. No validation set, early stopping, or tuning
is used. The probability cutoff is fixed at 0.5 and original CUSUM uses drift
0.5, threshold 5, with resets at flight boundaries and gaps above 250 ms.
These differ from Sail 1's validation-selected operating points. The
selection.json field named max_f1 is retained for evaluator compatibility;
it does not imply threshold tuning in this experiment.

## PowerShell commands

Run each command separately and stop if it fails. Existing outputs are refused.

```powershell
Set-Location 'C:\Users\souha\Documents\XGBoost\Early-Detection-of-UAV-Spoofing'
$python = '..\.venv\Scripts\python.exe'
& $python --version
& $python -m scripts.run_sail_2p prepare

# Confirm flight counts before training: train=288, test=72.
Import-Csv '../UAV-GPS-Spoofing-Dataset/sail_2p/splits.csv' | Group-Object split | Select-Object Name,Count

& $python -m scripts.run_sail_2p train
Get-Content 'results/runs/sail_2p/selection.json'

# Only after settings are frozen, evaluate once.
& $python -m scripts.run_sail_2p evaluate
```

Preparation exports features and labels into separate train and test CSVs.
Training reads only the training CSV for fitting. It hashes the test CSV as
opaque bytes to freeze the input; those bytes are not passed to the model.
Evaluation loads only the newly saved sail_2p booster and does not train it.

## Inspect results

```powershell
$run = 'results/runs/sail_2p'
$metrics = Get-Content "$run/test_metrics.json" -Raw | ConvertFrom-Json
$metrics.xgboost | Format-List
$metrics.cusum_pulse | Format-List

# Find missed attacks and flights with alarms while spoofing was off.
Import-Csv "$run/test_flight_events.csv" |
    Where-Object { ($_.condition -eq 'Attacked' -and $_.alerted_during_attack -eq 'False') -or [int]$_.off_alarm_samples -gt 0 } |
    Format-Table flight_id,detector,condition,alerted_during_attack,off_alarm_samples

Get-Content "$run/test_xgboost.csv" -TotalCount 6
Invoke-Item "$run/test_flight_events.csv"
```

F1, precision, recall, and accuracy are fractions (multiply by 100 for percent).
FP counts spoofing-off samples incorrectly alarmed; FN counts active samples
missed. Inspect attacks_alerted / attacks, normal_flights_with_alarm, and
median_delay_s_detected together. Delay is in seconds and excludes undetected
attacks. CUSUM pulse is an intermittent alarm, so its sample F1 does not measure
the same sustained status behavior as XGBoost.

- booster.ubj: newly trained model.
- selection.json: fixed settings and input/model hashes.
- test_metrics.json: aggregate sample and flight metrics.
- test_xgboost.csv: timestamped probabilities, margins, predictions, labels.
- test_alarms.csv: paired XGBoost decisions and CUSUM pulses.
- test_flight_events.csv: per-flight detection, false alarms, and delay.

## Leakage interpretation

The existing code fits fresh estimators for Lighthouse, Ocean, and Sail 1;
their archived model hashes differ. Sail 1 preserves all Ocean assignments.
Its 36 fresh test flights exclude both earlier manifests. Lighthouse pilot
test flights were later reused in Sail 1 development (3 train, 3 validation),
so those six flights are no longer untouched holdouts for later experiments.

Repartitioning all 360 flights for sail_2p can put prior development flights
into the new test set. A fresh booster has no inherited learned trees, but
features and architecture were developed using earlier experiments. Report
sail_2p as a fresh-fit resplit experiment, not a never-before-used external
test. The current manifests rule out exact flight/hash overlap within the
split; they do not establish independence of related simulator runs or real
online sensor availability in upstream merged telemetry. Do not adjust
settings based on this test score and then report it as untouched.

The actual sail_2p training and evaluation are left for you to run manually.
