"""Compare direct XGBoost and CUSUM-filtered transitions for persistent flight alerts."""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from scripts.run_lighthouse_ocean import (
    FIXED_PARAMS, MAX_GAP_S, METADATA, counts, digest, event_results, feature_order,
    flight_slices, read_flights, status_predictions, write_json,
)


PROBABILITIES = (0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999)
CUSUM_DRIFTS = (-2, -1, 0, 1, 2, 3, 4)
CUSUM_THRESHOLDS = (0.5, 2, 5, 10)


def transition_labels(data, horizon_s):
    time = data["timestamp_s"].to_numpy()
    active = data["attack_label"].to_numpy(dtype=bool)
    flight = data["flight_id"].to_numpy()
    new_flight = np.r_[True, flight[1:] != flight[:-1]]
    previous = np.r_[False, active[:-1]]
    enters = active & (new_flight | ~previous)
    leaves = ~new_flight & previous & ~active
    targets = {event: np.zeros(len(data), dtype=bool) for event in ("start", "stop")}
    for positions in flight_slices(data):
        local = time[positions]
        is_active = active[positions]
        for event, transitions, state in (("start", enters, is_active), ("stop", leaves, ~is_active)):
            changed_at = np.maximum.accumulate(np.where(transitions[positions], local, -np.inf))
            targets[event][positions] = state & (local - changed_at <= horizon_s)
    return targets


def transition_model(inputs, target, positive_weight):
    model = XGBClassifier(
        **{**FIXED_PARAMS, "n_jobs": 6, "random_state": 42, "max_depth": 4,
           "n_estimators": 220, "learning_rate": 0.07, "min_child_weight": 5},
    )
    model.fit(inputs, target, sample_weight=np.where(target, positive_weight, 1.0))
    return model


def trigger_candidates(data, scores, margins, method):
    if method == "xgboost_transitions":
        return [
            ({"probability_threshold": probability}, scores >= probability)
            for probability in PROBABILITIES
        ]
    times = data["timestamp_s"].to_numpy()
    flights = flight_slices(data)
    return [
        ({"drift": drift, "threshold": threshold},
         status_predictions(times, margins, flights, drift, threshold))
        for drift, threshold in itertools.product(CUSUM_DRIFTS, CUSUM_THRESHOLDS)
    ]


def switch_indices(data, evidence):
    flight = data["flight_id"].to_numpy()
    times = data["timestamp_s"].to_numpy()
    new_flight = np.r_[True, flight[1:] != flight[:-1]]
    previous = np.r_[False, evidence[:-1]]
    starts = evidence & (new_flight | ~previous | (np.diff(times, prepend=times[0]) > MAX_GAP_S))
    index = np.arange(len(data))
    # The last flight boundary acts as a clear. A real start at that row can open an alert.
    cleared = np.maximum.accumulate(np.where(new_flight, 2 * index, -1))
    switches = np.maximum.accumulate(np.where(starts, 2 * index + 1, -1))
    return cleared, switches


def rank(row):
    return row["f1"], -row["false_alarm_fraction"], row["accuracy"]


def choose_switches(data, signals, method):
    options = {}
    for event, (scores, margins) in signals.items():
        options[event] = [
            (params, *switch_indices(data, evidence))
            for params, evidence in trigger_candidates(data, scores, margins, method)
        ]
    cleared = options["start"][0][1]
    labels = data["attack_label"].to_numpy(dtype=bool)
    best = None
    for (start_params, _, starts), (stop_params, _, stops) in itertools.product(
        options["start"], [({"disabled": True}, cleared, cleared), *options["stop"]],
    ):
        status = starts > np.maximum(cleared, stops)
        trial = {"start": start_params, "stop": stop_params, **counts(labels, status)}
        if best is None or rank(trial) > rank(best):
            best = trial
    return best


def scores_for(model, features):
    return (model.predict_proba(features)[:, 1].astype("float64"),
            model.predict(features, output_margin=True).astype("float64"))


def train(dataset, output, assignments, features):
    data = read_flights(dataset / "samples.csv", assignments, {"train", "validation"}, features)
    training = data.loc[data["split"].eq("train")].reset_index(drop=True)
    validation = data.loc[data["split"].eq("validation")].reset_index(drop=True)
    methods = ("xgboost_transitions", "xgboost_cusum_transitions")
    trials, winners = [], {}
    for horizon in (0.6, 1.2, 2.0):
        targets = transition_labels(training, horizon)
        models = {
            event: transition_model(training[features], target, 18 if event == "start" else 36)
            for event, target in targets.items()
        }
        signals = {event: scores_for(model, validation[features]) for event, model in models.items()}
        for method in methods:
            result = {"method": method, "target_window_s": horizon, **choose_switches(validation, signals, method)}
            trials.append(result)
            if method not in winners or rank(result) > rank(winners[method][0]):
                winners[method] = (result, models)
            print(method, horizon, "validation F1", round(result["f1"], 5),
                  "false alarm fraction", round(result["false_alarm_fraction"], 5), flush=True)
    output.mkdir(parents=True, exist_ok=False)
    frozen = {
        "features": features, "methods": {},
        "dataset_sha256": {file: digest(dataset / file) for file in ("samples.csv", "splits.csv", "schema.json")},
    }
    for method, (selection, models) in winners.items():
        frozen["methods"][method] = dict(selection, model_sha256={})
        for event, model in models.items():
            path = output / f"{method}_{event}.ubj"
            model.save_model(path)
            frozen["methods"][method]["model_sha256"][event] = digest(path)
    write_json(output / "selection.json", frozen)
    write_json(output / "trials.json", trials)
    score(validation, assignments, output, "validation", frozen)


def score(data, assignments, output, split, frozen):
    alerts = data[METADATA].copy()
    for method, settings in frozen["methods"].items():
        switches = {}
        for event in ("start", "stop"):
            path = output / f"{method}_{event}.ubj"
            if digest(path) != settings["model_sha256"][event]:
                raise ValueError(f"Frozen model changed: {path}")
            model = XGBClassifier(n_jobs=6)
            model.load_model(path)
            scores, margins = scores_for(model, data[frozen["features"]])
            alerts[f"{method}_{event}_score"] = scores
            params = settings[event]
            if params.get("disabled"):
                evidence = np.zeros(len(data), dtype=bool)
            elif "probability_threshold" in params:
                evidence = scores >= params["probability_threshold"]
            else:
                evidence = status_predictions(
                    data["timestamp_s"].to_numpy(), margins, flight_slices(data),
                    params["drift"], params["threshold"],
                )
            cleared, switches[event] = switch_indices(data, evidence)
            alerts[f"{method}_{event}_evidence"] = evidence
        alerts[method] = switches["start"] > np.maximum(switches["stop"], cleared)
    alerts.to_csv(output / f"{split}_alarms.csv", index=False, mode="x")
    events = event_results(alerts, assignments, list(frozen["methods"]))
    events.to_csv(output / f"{split}_flight_events.csv", index=False, mode="x")
    labels = alerts["attack_label"].to_numpy(dtype=bool)
    results = {}
    for method in frozen["methods"]:
        rows = events.loc[events["detector"].eq(method)]
        attacked = rows.loc[rows["condition"].eq("Attacked")]
        normal = rows.loc[rows["condition"].eq("Normal")]
        median = attacked["first_new_active_alert_delay_s"].median()
        length = attacked["first_new_active_alert_delay_s"].count()
        results[method] = {
            **counts(labels, alerts[method].to_numpy(dtype=bool)),
            "attacks": len(attacked), "attacks_with_new_alert": int(length),
            "attacks_alerted": int(attacked["alerted_during_attack"].sum()),
            "attacks_already_alerting": int(attacked["already_alerting_before_first_active_sample"].sum()),
            "normal_flights": len(normal), "normal_flights_with_alarm": int(normal["off_alarm_samples"].gt(0).sum()),
            "off_alarm_starts": int(rows["off_alarm_starts"].sum()),
            "median_delay_s_new_alert_detected": float(median) if pd.notna(median) else None,
        }
    write_json(output / f"{split}_metrics.json", results)
    print(split, json.dumps(results), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("train", "evaluate"))
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assignments = pd.read_csv(args.dataset / "splits.csv")
    schema = json.loads((args.dataset / "schema.json").read_text(encoding="utf-8"))
    features = feature_order(schema)
    if args.stage == "train":
        train(args.dataset, args.output, assignments, features)
    else:
        frozen = json.loads((args.output / "selection.json").read_text(encoding="utf-8"))
        if not all(digest(args.dataset / file) == sha for file, sha in frozen["dataset_sha256"].items()):
            raise ValueError("Development inputs changed after model selection")
        if (args.output / "test_metrics.json").exists():
            raise FileExistsError("Test evaluation already exists in this directory")
        test = read_flights(args.dataset / "test_samples.csv", assignments, {"test"}, features)
        score(test, assignments, args.output, "test", frozen)


if __name__ == "__main__":
    main()
