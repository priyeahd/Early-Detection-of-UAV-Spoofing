# UAV GPS spoofing detection: XGBoost baseline

This milestone trains a standard, per-sample XGBoost classifier for detecting active spoofing from approved telemetry. The real-data run awaits the team's input contract. See the [PX4 audit](reports/px4_dataset_audit.md) for what was verified. The runnable fixture checks the pipeline and does not represent PX4 telemetry or detector performance.

## Reproduce the software check

From the repository root with Python 3.12:

```powershell
python -m pip install -r requirements.txt
python -m unittest discover -v
python -m tests.xgboost_fixture results/runs/fixture
python -m src.baselines.xgboost_baseline train --data results/runs/fixture/samples.csv --splits results/runs/fixture/splits.csv --schema results/runs/fixture/schema.json --config configs/xgboost.json --output results/runs/xgboost-smoke
python -m src.baselines.xgboost_baseline replay --data results/runs/fixture/samples.csv --splits results/runs/fixture/splits.csv --bundle results/runs/xgboost-smoke --output results/runs/xgboost-smoke/replayed.csv
Get-FileHash -Path results/runs/xgboost-smoke/*csv
```

Use unused output paths for a new run. The two CSV hashes should match. Test data, model bundles and predictions generated under `results/runs/` are local outputs.

## Proposed input and output contract

- **Samples CSV:** `flight_id`, `timestamp_s` (nonnegative seconds from the agreed per-flight origin), `attack_label` (literal `0` or `1`), `split`, and the approved feature columns. Each `(flight_id, timestamp_s)` must be unique. Both train and validation need examples of both labels. A blank feature value is treated as missing. Keep any additional metadata outside the feature list.
- **Split manifest CSV:** one `flight_id,split` row per flight, using `train`, `validation`, or `test`. If the manifest includes `group_id` for dependent or related runs, each group must stay in one partition. The team manifest should also record scenario, source hash, event count and coverage. Keep test rows out of the samples CSV until final evaluation; only their assignment needs to appear in this manifest.
- **Schema JSON:** a `version`, `data_kind` (`approved_telemetry` for the research run), and ordered `features`; each feature specifies `name`, `unit`, `provenance`, and `causal: true`. Column and metadata guards help prevent leakage; the team still reviews whether each feature and the synchronization are genuinely causal. The CLI uses only this allowlist, regardless of the CSV column order.
- **Predictions CSV:** `flight_id,timestamp_s,attack_label,attack_probability,predicted_alarm,model_name,split,run_id,raw_score,score_type`. Output is sorted by flight ID and timestamp, with one row per validation sample. `attack_probability` is the classifier's uncalibrated positive-class probability; `raw_score` is its logistic margin. `predicted_alarm` is a per-row threshold decision.

To train on approved inputs, use the same `train` command with the team samples, split manifest and schema. Only train flights are used to fit trees. Model selection minimizes validation sample log loss, and the provisional threshold maximizes validation sample F1. These criteria are for the standard baseline; an event definition and false-alarm budget must be agreed before comparing operating points. This CLI does not evaluate the test set.

Every run writes `booster.ubj`, `schema.json`, `preprocessing.json`, `config.json`, `manifest.json` and `validation_predictions.csv`. XGBoost handles missing values directly; no imputer, scaler, or resampler is fitted here. Replay verifies original input and split hashes, the saved schema and booster, and the booster feature order. It uses the saved threshold and run ID. The manifest records run provenance and the exact baseline source hash, including when there is no Git commit.

## Add CUSUM to saved XGBoost scores

After running the fixture commands above, apply CUSUM to the saved validation scores:

```powershell
python -m src.sequential.cusum --predictions results/runs/xgboost-smoke/validation_predictions.csv --output results/runs/xgboost-smoke/cusum_predictions.csv --drift 0.5 --threshold 5
```

CUSUM accumulates the positive-class logistic margin in timestamp order within each flight. It updates `state = max(0, state + raw_score - drift)` and emits an alarm when the state reaches or exceeds the threshold. The output records the state before the alarm reset. Accumulation restarts at zero after an alarm and at the start of each flight. A sustained signal can produce multiple alarms. By default the state carries across gaps within a flight. Set `--max-gap-s` to reset before a sample whose gap exceeds that duration.

The output uses the same timestamps, probabilities, and source run ID for paired comparison. It sets `model_name` to `xgboost_cusum` and adds `cusum_state`. The drift and threshold above are examples for the fixture. Before evaluating a held-out test set, select these settings on real validation flights using the agreed alarm definition and false-alarm budget. With mixed sampling rates, agree on the sampling policy before comparing alarms because the state updates once per sample.
