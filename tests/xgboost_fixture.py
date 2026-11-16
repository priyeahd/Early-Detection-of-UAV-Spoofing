"""Create a deterministic software fixture; it does not simulate GPS spoofing."""

import csv
import sys
from pathlib import Path

from src.baselines.xgboost_baseline import save_json


def write_fixture(root):
    root.mkdir(parents=True, exist_ok=True)
    table, splits, schema = (root / name for name in ("samples.csv", "splits.csv", "schema.json"))
    with (table.open("x", newline="", encoding="utf-8") as samples,
          splits.open("x", newline="", encoding="utf-8") as manifest):
        rows, assignments = csv.writer(samples), csv.writer(manifest)
        rows.writerow(["fixture_b", "split", "timestamp_s", "attack_label", "flight_id",
                       "fixture_a", "attack_enabled", "simulator_truth"])
        assignments.writerow(["flight_id", "split", "group_id"])
        for index, split in enumerate(("validation", "train", "test", "train", "validation")):
            flight_id = f"fixture_{index}"
            assignments.writerow([flight_id, split, flight_id])
            for sample in reversed(range(24)):
                label = int(sample >= 12)
                a = label * 3 + sample % 5 / 4 + index / 20
                b = (sample - index) % 7 / 3 if sample % 6 else ""
                rows.writerow([b, split, sample / 4, label, flight_id, a, label, label])
    save_json(schema, {
        "version": "xgboost-smoke-v1", "data_kind": "synthetic_fixture",
        "features": [
            {"name": name, "unit": "dimensionless", "causal": True,
             "provenance": "Generated numeric test values with deliberate artificial separation."}
            for name in ("fixture_a", "fixture_b")
        ],
    })
    return table, splits, schema


if __name__ == "__main__":
    write_fixture(Path(sys.argv[1]))
