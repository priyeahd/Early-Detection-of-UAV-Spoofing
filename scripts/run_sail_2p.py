"""Manual 80/20 whole-flight experiment; fresh fitting with fixed settings."""

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from scripts.prepare_lighthouse_ocean import export_subset
from scripts.run_lighthouse_ocean import digest, evaluate, read_flights, write_json
from src.baselines.xgboost_baseline import FIXED_PARAMS, feature_order


def assign_flights(inventory):
    data = inventory.sort_values("flight_id").reset_index(drop=True).copy()
    if len(data) != 360 or not data["flight_id"].is_unique or not data["sha256"].is_unique:
        raise ValueError("Expected 360 unique audited flights and content hashes")
    training, testing = train_test_split(
        data.index, test_size=0.2, random_state=42,
        stratify=data["scenario"] + "/" + data["condition"],
    )
    data["split"] = "train"
    data.loc[testing, "split"] = "test"
    assert len(training) == 288 and len(testing) == 72
    return data


def prepare(source, inventory_path, dataset):
    inventory = pd.read_csv(inventory_path)
    source_root = source / "PX4_Simulation_Data"
    files = {p.relative_to(source_root).as_posix() for p in (source_root / "Merged").glob("*/*/*.csv")}
    if files != set(inventory["file"]):
        raise ValueError("Source files differ from the audited inventory")
    export_subset(source, assign_flights(inventory), dataset, "sail_2p-v1", integrated=True)


def train(dataset, output):
    if output.exists():
        raise FileExistsError(output)
    assignments = pd.read_csv(dataset / "splits.csv")
    if (assignments["split"].value_counts().to_dict() != {"train": 288, "test": 72}
            or not assignments["flight_id"].is_unique or not assignments["sha256"].is_unique):
        raise ValueError("Expected disjoint 288/72 flight split with unique source hashes")
    features = feature_order(json.loads((dataset / "schema.json").read_text(encoding="utf-8")))
    # Only training payload is opened. Test payload is hashed to lock it, never used by fit.
    hashes = {name: digest(dataset / name) for name in
              ("samples.csv", "test_samples.csv", "splits.csv", "schema.json")}
    training = read_flights(dataset / "samples.csv", assignments, {"train"}, features)
    params = {**FIXED_PARAMS, "n_jobs": 6, "random_state": 42,
              "max_depth": 4, "n_estimators": 200, "learning_rate": 0.08}
    model = XGBClassifier(**params)
    model.fit(training[features], training["attack_label"])
    output.mkdir(parents=True, exist_ok=False)
    model.save_model(output / "booster.ubj")
    write_json(output / "selection.json", {
        "run_id": output.name, "model": "fresh_fixed_80_20", "features": features,
        "parameters": params, "initialization": "new XGBClassifier; no prior booster",
        "protocol": "288 train / 72 test flights; seed 42; scenario/condition stratification",
        "settings_provenance": "Sail 1 architecture/features; fixed probability 0.5; original CUSUM 0.5/5. No tuning.",
        "xgboost": {"max_f1": {"probability_threshold": 0.5}, "one_percent_budget": None},
        "cusum": {}, "dataset_sha256": hashes,
        "model_sha256": digest(output / "booster.ubj"),
        "train_flights": 288, "test_flights": 72, "train_rows": len(training),
        "note": "max_f1 is a compatibility field name, not a tuned threshold in this run.",
    })
    print(f"Fresh model saved to {output}; test evaluation has not run.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "train", "evaluate"))
    parser.add_argument("--source", type=Path, default=Path("../UAV-GPS-Spoofing-Dataset"))
    parser.add_argument("--inventory", type=Path, default=Path("reports/px4_merged_inventory.csv"))
    parser.add_argument("--dataset", type=Path, default=Path("../UAV-GPS-Spoofing-Dataset/sail_2p"))
    parser.add_argument("--output", type=Path, default=Path("results/runs/sail_2p"))
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare(args.source, args.inventory, args.dataset)
    elif args.stage == "train":
        train(args.dataset, args.output)
    else:
        assignments = pd.read_csv(args.dataset / "splits.csv")
        schema = json.loads((args.dataset / "schema.json").read_text(encoding="utf-8"))
        evaluate(args.dataset, args.output, assignments, feature_order(schema))


if __name__ == "__main__":
    main()
