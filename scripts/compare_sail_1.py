"""Evaluate a frozen Ocean model on the same test cohorts as Sail 1."""

import argparse
import json
from pathlib import Path

import pandas as pd
from xgboost import XGBClassifier

from scripts.run_lighthouse_ocean import digest, read_flights, save_results, summarize_alarms, write_json


DETECTORS = ("xgboost", "cusum_max_f1", "cusum_pulse")
KEY = ["flight_id", "timestamp_s", "attack_label"]


def read_result_tables(directory):
    return (pd.read_csv(directory / "test_alarms.csv"),
            pd.read_csv(directory / "test_flight_events.csv"))


def compare(dataset, sail, ocean):
    reference_dir = sail / "ocean_reference"
    comparison = sail / "comparison.json"
    if comparison.exists() or reference_dir.exists():
        raise FileExistsError("Sail 1 reference comparison already exists")
    sail_selection = json.loads((sail / "selection.json").read_text(encoding="utf-8"))
    for name, expected in sail_selection["dataset_sha256"].items():
        if digest(dataset / name) != expected:
            raise ValueError(f"Sail 1 input changed: {name}")
    ocean_selection = json.loads((ocean / "selection.json").read_text(encoding="utf-8"))
    if digest(ocean / "booster.ubj") != ocean_selection["model_sha256"]:
        raise ValueError("Frozen Ocean model changed")

    assignments = pd.read_csv(dataset / "splits.csv")
    test = read_flights(dataset / "test_samples.csv", assignments, {"test"}, sail_selection["features"])
    sail_alerts, sail_events = read_result_tables(sail)
    pd.testing.assert_frame_equal(test[KEY], sail_alerts[KEY])
    model = XGBClassifier(n_jobs=6)
    model.load_model(ocean / "booster.ubj")
    reference_dir.mkdir()
    save_results(model, test, ocean_selection, assignments, reference_dir, "test")
    ocean_alerts, ocean_events = read_result_tables(reference_dir)
    pd.testing.assert_frame_equal(sail_alerts[KEY], ocean_alerts[KEY])
    replay_ids = assignments.loc[assignments["test_cohort"].eq("ocean_replay"), "flight_id"]
    archived_alerts, _ = read_result_tables(ocean)
    replay_alerts = ocean_alerts.loc[ocean_alerts["flight_id"].isin(replay_ids)]
    pd.testing.assert_frame_equal(
        replay_alerts[KEY + list(DETECTORS)].reset_index(drop=True),
        archived_alerts[KEY + list(DETECTORS)],
    )

    results = {}
    test_flights = assignments.loc[assignments["split"].eq("test")]
    cohorts = {"all": test_flights, **dict(tuple(test_flights.groupby("test_cohort")))}
    for cohort, flights in cohorts.items():
        cohort_result = {"flights": len(flights), "samples": int(flights["sample_count"].sum())}
        for name, alerts, events in (("sail_1", sail_alerts, sail_events), ("ocean", ocean_alerts, ocean_events)):
            cohort_result[name] = summarize_alarms(
                alerts.loc[alerts["flight_id"].isin(flights["flight_id"])],
                events.loc[events["flight_id"].isin(flights["flight_id"])],
                DETECTORS,
            )
        results[cohort] = cohort_result
    write_json(comparison, {
        "test_data_sha256": digest(dataset / "test_samples.csv"),
        "ocean_model_sha256": ocean_selection["model_sha256"],
        "sail_1_model_sha256": sail_selection["model_sha256"],
        "cohorts": results,
    })
    print(f"Compared {len(test_flights)} test flights and reproduced archival Ocean alarms.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--sail", type=Path, default=Path("results/runs/sail_1"))
    parser.add_argument("--ocean", type=Path, default=Path("results/runs/lighthouse-ocean-integrated"))
    args = parser.parse_args()
    compare(args.dataset, args.sail, args.ocean)


if __name__ == "__main__":
    main()
