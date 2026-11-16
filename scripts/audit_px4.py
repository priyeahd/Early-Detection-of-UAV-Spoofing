"""Inventory the existing merged PX4 logs without rewriting the dataset."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import pandas as pd


def audit(root):
    merged = root / "Merged"
    files = sorted(merged.glob("*/*/*.csv"))
    if not files:
        raise ValueError(f"No merged flight CSVs under {merged}")
    for path in files:
        data = pd.read_csv(path)
        file_order_steps = data["time_usec"].diff()
        ordered = data.sort_values("time_usec", kind="stable")
        time = ordered["time_usec"]
        label = ordered["attack_enabled"]
        steps = time.diff()
        positives = label.eq(1)
        onsets = positives & ~positives.shift(fill_value=False)
        gaps = steps[steps.gt(0)]
        median_step = gaps.median()
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        scenario, condition, _ = path.relative_to(merged).parts
        yield {
            "file": path.relative_to(root).as_posix(),
            "flight_id": path.stem,
            "scenario": scenario,
            "condition": condition,
            "sha256": digest,
            "columns": "|".join(data.columns),
            "rows": len(data),
            "missing_by_column": json.dumps(data.isna().sum().loc[lambda s: s.gt(0)].to_dict()),
            "duplicate_rows": int(data.duplicated().sum()),
            "duplicate_timestamps": int(time.duplicated().sum()),
            "file_order_backwards_steps": int(file_order_steps.lt(0).sum()),
            "first_usec": int(time.iloc[0]),
            "last_usec": int(time.iloc[-1]),
            "median_positive_step_usec": median_step,
            "max_step_usec": gaps.max(),
            "steps_over_twice_median": int(gaps.gt(2 * median_step).sum()),
            "nonbinary_or_missing_labels": int((~label.isin([0, 1])).sum()),
            "positive_rows": int(positives.sum()),
            "starts_positive": int(positives.iloc[0]),
            "positive_segments": int(onsets.sum()),
            "segment_starts_s_from_first_sample": json.dumps(((time[onsets] - time.iloc[0]) / 1e6).tolist()),
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Directory containing Merged/")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    records = list(audit(args.source))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


if __name__ == "__main__":
    main()
