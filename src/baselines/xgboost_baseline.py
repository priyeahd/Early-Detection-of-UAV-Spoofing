"""Train and replay the standard XGBoost validation baseline."""

import argparse
import csv
import hashlib
import json
import platform
import re
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, log_loss
from xgboost import XGBClassifier

KEY = ["flight_id", "timestamp_s"]
METADATA = [*KEY, "attack_label", "split"]
MODEL = "xgboost_baseline"
PARTITIONS = {"train", "validation", "test"}
FIXED_PARAMS = {
    "objective": "binary:logistic",
    "booster": "gbtree",
    "tree_method": "hist",
    "device": "cpu",
    "n_jobs": 1,
    "eval_metric": "logloss",
    "subsample": 1.0,
    "colsample_bytree": 1.0,
}


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_csv(path, columns):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        header = next(csv.reader(stream), [])
    if len(header) != len(set(header)) or not set(columns).issubset(header):
        raise ValueError(f"{path.name}: duplicate header or missing required columns {columns}")
    # Keep flight IDs and binary labels literal. Convert numeric fields after splitting.
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")


def feature_order(schema):
    if not isinstance(schema.get("version"), str) or not schema["version"].strip():
        raise ValueError("Schema needs a nonempty version")
    if schema.get("data_kind") not in {"synthetic_fixture", "approved_telemetry"}:
        raise ValueError("Schema must identify synthetic_fixture or approved_telemetry")
    features = schema.get("features")
    if not isinstance(features, list) or not features or any(
        not isinstance(f, dict) or not all(
            isinstance(f.get(k), str) and f[k].strip() for k in ("name", "unit", "provenance")
        ) or f.get("causal") is not True for f in features
    ):
        raise ValueError("Specify ordered features with name, unit, provenance and causal=true")
    names = [f["name"] for f in features]
    normalized = [re.sub(r"[^a-z0-9]", "", name.lower()) for name in names]
    reserved = {"label", "target", "groupid", "runid", "modelname", "timeusec",
                "condition", "scenario", "rawscore", "predictedalarm"}
    if len(set(normalized)) != len(names) or any(
        not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name) or token in reserved
        or token == "split" or token.startswith(("attack", "flight", "timestamp", "groundtruth",
                                                 "simulator", "future", "rawgps"))
        for name, token in zip(names, normalized, strict=True)
    ):
        raise ValueError("Duplicate, invalid or forbidden feature name in schema")
    return names


def partition_rows(data_path, split_path, features, requested):
    manifest = read_csv(split_path, ["flight_id", "split"])
    ids = manifest["flight_id"]
    if ids.eq("").any() or ids.str.strip().ne(ids).any() or ids.duplicated().any():
        raise ValueError("Split manifest requires one nonempty, unique row per flight")
    if set(manifest["split"]) != PARTITIONS:
        raise ValueError("Split manifest must contain train, validation and test flights")
    if "group_id" in manifest and (
        manifest["group_id"].str.strip().eq("").any()
        or manifest.groupby("group_id")["split"].nunique().gt(1).any()
    ):
        raise ValueError("Each group_id must belong to one partition")
    data = read_csv(data_path, [*METADATA, *features])
    assigned = data["flight_id"].map(manifest.set_index("flight_id")["split"])
    required_ids = ids[manifest["split"].isin({"train", "validation"})]
    if (not set(required_ids).issubset(data["flight_id"]) or assigned.isna().any()
            or assigned.ne(data["split"]).any()):
        raise ValueError("Table must cover all train and validation flights and agree on each split")
    # Select before converting timestamps, outcome labels or features. Test payload is unused.
    for split in requested:
        rows = data.loc[assigned.eq(split), [*METADATA, *features]].copy()
        rows["timestamp_s"] = pd.to_numeric(rows["timestamp_s"], errors="raise").astype("float64")
        if (not np.isfinite(rows["timestamp_s"]).all() or rows["timestamp_s"].lt(0).any()
                or rows.duplicated(KEY).any()):
            raise ValueError(f"{split}: timestamps must be finite, nonnegative and unique per flight")
        if set(rows["attack_label"]) != {"0", "1"}:
            raise ValueError(f"{split}: attack_label must be binary with both classes present")
        rows["attack_label"] = rows["attack_label"].astype("int8")
        for name in features:
            rows[name] = pd.to_numeric(rows[name].replace("", np.nan), errors="raise").astype("float64")
            if np.isinf(rows[name]).any():
                raise ValueError(f"{split}: infinite values in feature {name}")
        yield rows.sort_values(KEY).reset_index(drop=True)


def predictions(model, rows, features, threshold, run_id):
    x = rows[features]
    scores = model.predict_proba(x)[:, 1]
    return rows[KEY + ["attack_label"]].assign(
        attack_probability=scores,
        predicted_alarm=(scores.astype("float64") >= threshold).astype("int8"),
        model_name=MODEL,
        split=rows["split"],
        run_id=run_id,
        raw_score=model.predict(x, output_margin=True),
        score_type="logistic_margin",
    )


def fit(train, validation, features, config):
    x_train, y_train = train[features], train["attack_label"]
    x_val, y_val = validation[features], validation["attack_label"]
    trials = []
    best = None
    for candidate in config["candidates"]:
        if set(candidate) != {"n_estimators", "max_depth", "learning_rate"}:
            raise ValueError("Candidates must specify n_estimators, max_depth and learning_rate only")
        model = XGBClassifier(**FIXED_PARAMS, **candidate, random_state=config["seed"])
        model.fit(x_train, y_train)
        scores = model.predict_proba(x_val)[:, 1].astype("float64")
        loss = log_loss(y_val, scores)
        trials.append({"params": candidate, "validation_log_loss": loss})
        if best is None or loss < best[0]:
            best = (loss, model, scores)
    if best is None:
        raise ValueError("Specify at least one model candidate")
    thresholds = config["threshold_candidates"]
    if not thresholds or not all(type(t) in (int, float) and 0 < t < 1 for t in thresholds):
        raise ValueError("Specify threshold candidates between zero and one")
    threshold = max(thresholds, key=lambda t: (f1_score(y_val, best[2] >= t), t))
    return best[1], threshold, trials


def git_commit():
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--verify", "HEAD"], cwd=Path(__file__).resolve().parents[2],
            capture_output=True, text=True, check=False,
        )
    except FileNotFoundError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def train(data_path, split_path, schema_path, config_path, output):
    started = datetime.now(timezone.utc).isoformat()
    schema, config = read_json(schema_path), read_json(config_path)
    features = feature_order(schema)
    training, validation = partition_rows(data_path, split_path, features, ["train", "validation"])
    model, threshold, trials = fit(training, validation, features, config)
    run_id = uuid.uuid4().hex
    scores = predictions(model, validation, features, threshold, run_id)
    output.mkdir(parents=True, exist_ok=False)
    booster = output / "booster.ubj"
    model.save_model(booster)
    save_json(output / "schema.json", schema)
    save_json(output / "config.json", config)
    save_json(output / "preprocessing.json", {
        "feature_order": features, "numeric_dtype": "float64",
        "missing": "empty numeric cells become NaN; XGBoost learns missing branches",
        "fitted_transforms": [],
    })
    scores.to_csv(output / "validation_predictions.csv", index=False)
    save_json(output / "manifest.json", {
        "run_id": run_id, "started_utc": started, "model_name": MODEL, "data_kind": schema["data_kind"],
        "schema_version": schema["version"], "git_commit": git_commit(),
        "baseline_source_sha256": sha256(Path(__file__)),
        "input_sha256": sha256(data_path), "split_manifest_sha256": sha256(split_path),
        "schema_sha256": sha256(output / "schema.json"),
        "config_sha256": sha256(output / "config.json"), "model_sha256": sha256(booster),
        "seed": config["seed"], "model_params": {
            k: v for k, v in model.get_xgb_params().items() if v is not None
        },
        "threshold": threshold, "threshold_rule": "max validation sample F1; ties favor higher threshold",
        "model_selection": "min validation sample log loss; ties favor first candidate",
        "selection_trials": trials, "score": "binary:logistic positive-class score, uncalibrated",
        "train_rows": len(training), "validation_rows": len(validation),
        "train_flights": training["flight_id"].nunique(),
        "validation_flights": validation["flight_id"].nunique(),
        "python": sys.version.split()[0],
        "packages": {p: version(p) for p in ("xgboost", "numpy", "pandas", "scikit-learn", "scipy")},
        "platform": platform.platform(), "cpu": platform.processor() or platform.machine(),
    })
    return output


def replay(data_path, split_path, bundle, output):
    record = read_json(bundle / "manifest.json")
    schema_path = bundle / "schema.json"
    booster = bundle / "booster.ubj"
    for file, field in [(schema_path, "schema_sha256"), (booster, "model_sha256"),
                        (split_path, "split_manifest_sha256"), (data_path, "input_sha256")]:
        if sha256(file) != record[field]:
            raise ValueError(f"Validation replay requires the original {file.name}")
    features = feature_order(read_json(schema_path))
    (rows,) = partition_rows(data_path, split_path, features, ["validation"])
    model = XGBClassifier(n_jobs=1)
    model.load_model(booster)
    if model.get_booster().feature_names != features:
        raise ValueError("Booster feature order does not match the saved schema")
    with output.open("x", encoding="utf-8", newline="") as stream:
        predictions(model, rows, features, record["threshold"], record["run_id"]).to_csv(
            stream, index=False
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for action in ("train", "replay"):
        command = commands.add_parser(action)
        for option in ("data", "splits", "output"):
            command.add_argument(f"--{option}", type=Path, required=True)
        for option in (("schema", "config") if action == "train" else ("bundle",)):
            command.add_argument(f"--{option}", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "train":
        train(args.data, args.splits, args.schema, args.config, args.output)
    else:
        replay(args.data, args.splits, args.bundle, args.output)


if __name__ == "__main__":
    main()
