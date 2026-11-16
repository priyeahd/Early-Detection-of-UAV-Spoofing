"""Train and tune on Lighthouse Ocean development flights, then score the frozen test split."""

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, log_loss
from xgboost import XGBClassifier

from scripts.prepare_lighthouse_ocean import BASE_FEATURES, CADENCE_US, GAP_US, INTEGRATED_FEATURES
from src.baselines.xgboost_baseline import FIXED_PARAMS, METADATA, feature_order, predictions
from src.sequential.cusum import alarm_status, score_flights


CADENCE_S = CADENCE_US / 1e6
MAX_GAP_S = GAP_US / 1e6
DRIFTS = (-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3)
THRESHOLDS = (0.25, 0.5, 1, 2, 5, 10, 20)
ALARM_BUDGET = 0.01


def digest(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def write_json(path, value):
    with path.open("x", encoding="utf-8") as file:
        json.dump(value, file, indent=2, allow_nan=False)
        file.write("\n")


def read_flights(path, assignments, splits, features):
    data = pd.read_csv(path, dtype={feature: "float32" for feature in features})
    keys = ["flight_id", "timestamp_s"]
    expected = assignments.loc[assignments["split"].isin(splits)].set_index("flight_id")
    if (not data.groupby("flight_id").size().sort_index().equals(expected["sample_count"].sort_index())
            or not data["flight_id"].map(expected["split"]).eq(data["split"]).all()
            or data.duplicated(keys).any()
            or not np.isfinite(data["timestamp_s"]).all()
            or data["timestamp_s"].lt(0).any()
            or set(data["attack_label"]) != {0, 1}
            or np.isinf(data[features].to_numpy()).any()):
        raise ValueError(f"Flight assignments or values are invalid in {path}")
    return data.sort_values(keys, ignore_index=True)


def counts(labels, alarms):
    tp = int(np.count_nonzero(labels & alarms))
    fp = int(np.count_nonzero(~labels & alarms))
    fn = int(np.count_nonzero(labels)) - tp
    tn = len(labels) - tp - fp - fn
    return {
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "accuracy": (tp + tn) / len(labels),
        "recall": tp / max(1, tp + fn),
        "precision": tp / max(1, tp + fp),
        "f1": 2 * tp / max(1, 2 * tp + fp + fn),
        "false_alarm_fraction": fp / max(1, fp + tn),
    }


def best_settings(trials):
    rank = lambda row: (row["f1"], -row["false_alarm_fraction"], row["accuracy"])
    return {
        "max_f1": max(trials, key=rank),
        "one_percent_budget": max(
            (row for row in trials if row["false_alarm_fraction"] <= ALARM_BUDGET),
            key=rank, default=None,
        ),
    }


def flight_slices(data):
    return list(data.groupby("flight_id", sort=False).indices.values())


def status_predictions(timestamps, margins, flights, drift, threshold):
    result = np.empty(len(margins), dtype=bool)
    for positions in flights:
        result[positions] = alarm_status(
            timestamps[positions], margins[positions], drift=drift, threshold=threshold,
            cadence_s=CADENCE_S, max_gap_s=MAX_GAP_S,
        )
    return result


def model_trials(train, validation, features):
    base = list(BASE_FEATURES)
    integrated = [name for name in features if name in INTEGRATED_FEATURES]
    short = [name for name in features if name not in INTEGRATED_FEATURES]
    motion = [name for name in short if "residual" not in name]
    candidates = [
        ("six_current_features", base, 3, 100, 0.1, False),
        ("motion_windows", motion, 5, 200, 0.08, False),
        ("sensor_consistency_shallow", short, 3, 200, 0.08, False),
        ("sensor_consistency_deep", short, 6, 250, 0.08, False),
        ("sensor_consistency_flight_weighted", short, 6, 250, 0.08, True),
    ]
    if integrated:
        candidates.extend([
            ("cumulative_error_only", integrated, 3, 150, 0.08, False),
            ("cumulative_and_short_consistency", features, 4, 200, 0.08, False),
            ("cumulative_and_short_flight_weighted", features, 4, 200, 0.08, True),
        ])
    y = validation["attack_label"].to_numpy(dtype=bool)
    weights = train["flight_id"].map(1 / train["flight_id"].value_counts())
    weights = weights.to_numpy() / weights.mean()
    trials = []
    best = None
    for name, inputs, depth, trees, rate, weighted in candidates:
        params = {**FIXED_PARAMS, "n_jobs": 6, "random_state": 42, "max_depth": depth,
                  "n_estimators": trees, "learning_rate": rate}
        model = XGBClassifier(**params)
        model.fit(train[inputs], train["attack_label"], sample_weight=weights if weighted else None)
        scores = model.predict_proba(validation[inputs])[:, 1].astype("float64")
        operating_points = [
            {"probability_threshold": float(t), **counts(y, scores >= t)}
            for t in np.linspace(0.01, 0.99, 99)
        ]
        trial = {
            "name": name, "features": inputs, "parameters": params, "flight_weighted": weighted,
            "log_loss": float(log_loss(y, scores)),
            "average_precision": float(average_precision_score(y, scores)),
            **best_settings(operating_points),
        }
        trials.append(trial)
        print(name, "validation F1", round(trial["max_f1"]["f1"], 4), flush=True)
        rank = (trial["max_f1"]["f1"], -trial["log_loss"])
        if best is None or rank > best[0]:
            best = (rank, model, trial)
    return best[1], best[2], trials


def event_results(data, assignments, alarm_columns):
    lookup = assignments.set_index("flight_id")
    events = []
    for flight_id, flight in data.groupby("flight_id", sort=False):
        info = lookup.loc[flight_id]
        times = flight["timestamp_s"].to_numpy()
        active = flight["attack_label"].to_numpy(dtype=bool)
        segment_start = np.r_[True, np.diff(times) > MAX_GAP_S]
        for column in alarm_columns:
            alarmed = flight[column].to_numpy(dtype=bool)
            previous = np.r_[False, alarmed[:-1]]
            starts = alarmed if column == "cusum_pulse" else alarmed & (segment_start | ~previous)
            prior = np.flatnonzero(active)[0] if active.any() else 0
            first = np.flatnonzero(active & alarmed)
            fresh = np.flatnonzero(active & starts)
            event_onset = float(info["attack_onset_s"]) if active.any() else None
            events.append({
                "flight_id": flight_id, "scenario": info["scenario"], "condition": info["condition"],
                "detector": column, "attack_onset_s": event_onset,
                "alerted_during_attack": bool(first.size),
                "already_alerting_before_first_active_sample": bool(
                    active.any() and not segment_start[prior] and previous[prior]
                    and column != "cusum_pulse"
                ),
                "first_active_alert_delay_s": float(times[first[0]] - event_onset) if first.size else None,
                "new_alert_during_attack": bool(fresh.size),
                "first_new_active_alert_delay_s": float(times[fresh[0]] - event_onset) if fresh.size else None,
                "off_alarm_samples": int((~active & alarmed).sum()),
                "off_alarm_starts": int((~active & starts).sum()),
            })
    return pd.DataFrame(events)


def save_results(model, data, selection, assignments, output, split):
    raw = predictions(model, data, selection["features"], selection["xgboost"]["max_f1"]["probability_threshold"],
                      selection["run_id"])
    raw.to_csv(output / f"{split}_xgboost.csv", index=False, mode="x")
    pulse = score_flights(raw, drift=0.5, threshold=5, max_gap_s=MAX_GAP_S)
    alerts = raw[METADATA].assign(
        xgboost=raw["predicted_alarm"].astype(bool),
        cusum_pulse=pulse["predicted_alarm"].astype(bool),
        cusum_pulse_state=pulse["cusum_state"],
    )
    times, margins = raw["timestamp_s"].to_numpy(), raw["raw_score"].to_numpy(dtype="float64")
    flights = flight_slices(raw)
    columns = ["xgboost", "cusum_pulse"]
    for operating_point, chosen in selection["cusum"].items():
        if chosen:
            name = f"cusum_{operating_point}"
            alerts[name] = status_predictions(times, margins, flights, chosen["drift"], chosen["threshold"])
            columns.append(name)
    constrained = selection["xgboost"]["one_percent_budget"]
    if constrained:
        alerts["xgboost_one_percent_budget"] = raw["attack_probability"].to_numpy(dtype="float64") >= constrained["probability_threshold"]
        columns.append("xgboost_one_percent_budget")
    alerts.to_csv(output / f"{split}_alarms.csv", index=False, mode="x")
    events = event_results(alerts, assignments, columns)
    events.to_csv(output / f"{split}_flight_events.csv", index=False, mode="x")
    metrics = summarize_alarms(alerts, events, columns)
    write_json(output / f"{split}_metrics.json", metrics)
    print(split, json.dumps(metrics), flush=True)


def summarize_alarms(alerts, events, columns):
    labels = alerts["attack_label"].to_numpy(dtype=bool)
    metrics = {}
    for column in columns:
        event = events.loc[events["detector"].eq(column)]
        attacked = event.loc[event["condition"].eq("Attacked")]
        normal = event.loc[event["condition"].eq("Normal")]
        median = attacked["first_active_alert_delay_s"].median()
        new_median = attacked["first_new_active_alert_delay_s"].median()
        metrics[column] = {
            **counts(labels, alerts[column].to_numpy(dtype=bool)),
            "attacks": len(attacked),
            "attacks_alerted": int(attacked["alerted_during_attack"].sum()),
            "attacks_with_new_alert": int(attacked["new_alert_during_attack"].sum()),
            "attacks_already_alerting": int(attacked["already_alerting_before_first_active_sample"].sum()),
            "normal_flights": len(normal),
            "normal_flights_with_alarm": int(normal["off_alarm_samples"].gt(0).sum()),
            "off_alarm_starts": int(event["off_alarm_starts"].sum()),
            "median_delay_s_detected": float(median) if pd.notna(median) else None,
            "median_delay_s_new_alert_detected": float(new_median) if pd.notna(new_median) else None,
        }
    return metrics


def train(dataset, output, assignments, features):
    data = read_flights(dataset / "samples.csv", assignments, {"train", "validation"}, features)
    training = data.loc[data["split"].eq("train")].reset_index(drop=True)
    validation = data.loc[data["split"].eq("validation")].reset_index(drop=True)
    model, winner, ablations = model_trials(training, validation, features)
    margins = model.predict(validation[winner["features"]], output_margin=True).astype("float64")
    timestamps = validation["timestamp_s"].to_numpy()
    flights = flight_slices(validation)
    y = validation["attack_label"].to_numpy(dtype=bool)
    trials = []
    for drift, threshold in itertools.product(DRIFTS, THRESHOLDS):
        alarmed = status_predictions(timestamps, margins, flights, drift, threshold)
        trials.append({"drift": drift, "threshold": threshold, **counts(y, alarmed)})
    selection = {
        "run_id": output.name, "model": winner["name"], "features": winner["features"],
        "xgboost": {key: winner[key] for key in ("max_f1", "one_percent_budget")},
        "cusum": best_settings(trials), "validation_false_alarm_budget": ALARM_BUDGET,
        "cadence_s": CADENCE_S, "max_gap_s": MAX_GAP_S,
        "dataset_sha256": {file: digest(dataset / file) for file in ("samples.csv", "splits.csv", "schema.json")},
    }
    output.mkdir(parents=True, exist_ok=False)
    model.save_model(output / "booster.ubj")
    selection["model_sha256"] = digest(output / "booster.ubj")
    write_json(output / "selection.json", selection)
    write_json(output / "model_trials.json", ablations)
    pd.DataFrame(trials).to_csv(output / "cusum_trials.csv", index=False, mode="x")
    save_results(model, validation, selection, assignments, output, "validation")


def evaluate(dataset, output, assignments, features):
    selection = json.loads((output / "selection.json").read_text(encoding="utf-8"))
    selection.setdefault("run_id", "lighthouse-ocean-v1")
    if not all(digest(dataset / file) == sha for file, sha in selection["dataset_sha256"].items()):
        raise ValueError("Development inputs changed after model selection")
    if digest(output / "booster.ubj") != selection["model_sha256"]:
        raise ValueError("Frozen model changed")
    if (output / "test_metrics.json").exists():
        raise FileExistsError("Test evaluation already exists in this directory")
    model = XGBClassifier(n_jobs=6)
    model.load_model(output / "booster.ubj")
    data = read_flights(dataset / "test_samples.csv", assignments, {"test"}, features)
    save_results(model, data, selection, assignments, output, "test")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("train", "evaluate"))
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assignments = pd.read_csv(args.dataset / "splits.csv")
    schema = json.loads((args.dataset / "schema.json").read_text(encoding="utf-8"))
    action = train if args.stage == "train" else evaluate
    action(args.dataset, args.output, assignments, feature_order(schema))


if __name__ == "__main__":
    main()
