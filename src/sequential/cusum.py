"""
Cumulative Sum (CUSUM) Sequential Change-Point Detection.
"""

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.baselines.xgboost_baseline import KEY, METADATA, MODEL, PARTITIONS, read_csv


PREDICTION_COLUMNS = [
    *METADATA, "attack_probability", "predicted_alarm", "model_name", "run_id",
    "raw_score", "score_type",
]


class CUSUM:
    """
    Cumulative Sum (CUSUM) detector for sequential change-point detection.
    """
    def __init__(self, threshold: float = 5.0, drift: float = 0.5):
        self.threshold = threshold
        self.drift = drift
        self.reset()

    def reset(self):
        self.pos_sum = 0.0
        self.neg_sum = 0.0

    def update(self, value: float) -> bool:
        """
        Update CUSUM with a new observation and check if threshold is exceeded.
        """
        self.pos_sum = max(0.0, self.pos_sum + value - self.drift)
        self.neg_sum = max(0.0, self.neg_sum - value - self.drift)
        return self.pos_sum > self.threshold or self.neg_sum > self.threshold


def score_flights(predictions, *, drift, threshold, max_gap_s=None):
    """Apply upper-sided CUSUM to XGBoost's logistic margins."""
    if not math.isfinite(drift) or drift < 0:
        raise ValueError("drift must be finite and nonnegative")
    if not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("threshold must be finite and positive")
    if max_gap_s is not None and (not math.isfinite(max_gap_s) or max_gap_s <= 0):
        raise ValueError("max_gap_s must be finite and positive")

    rows = predictions.reset_index(drop=True)
    if set(rows["model_name"]) != {MODEL} or set(rows["score_type"]) != {"logistic_margin"}:
        raise ValueError("Expected standard XGBoost predictions with logistic margins")
    if rows["flight_id"].str.strip().eq("").any() or rows["run_id"].str.strip().eq("").any():
        raise ValueError("Flight and run IDs must be nonempty")
    splits = set(rows["split"])
    if rows["run_id"].nunique() != 1 or len(splits) != 1 or not splits.issubset(PARTITIONS):
        raise ValueError("Use predictions from one XGBoost run and one split")

    timestamps = pd.to_numeric(rows["timestamp_s"], errors="raise").astype("float64")
    margins = pd.to_numeric(rows["raw_score"], errors="raise").astype("float64")
    if not np.isfinite(timestamps).all() or timestamps.lt(0).any() or not np.isfinite(margins).all():
        raise ValueError("Timestamps must be finite and nonnegative, and margins must be finite")
    # Sort by numeric time while preserving the original timestamp values for export.
    ordered = rows[["flight_id"]].assign(timestamp_s=timestamps, margin=margins)
    if ordered.duplicated(KEY).any():
        raise ValueError("Timestamps must be unique within a flight")
    ordered = ordered.sort_values(KEY)

    states, alarms = [], []
    for _, flight in ordered.groupby("flight_id", sort=False):
        state = 0.0
        previous_time = None
        for timestamp, margin in zip(flight["timestamp_s"], flight["margin"], strict=True):
            if max_gap_s is not None and previous_time is not None and timestamp - previous_time > max_gap_s:
                state = 0.0
            state = max(0.0, state + margin - drift)
            alarm = state >= threshold
            states.append(state)
            alarms.append(int(alarm))
            if alarm:
                state = 0.0
            previous_time = timestamp

    result = rows.loc[ordered.index].reset_index(drop=True)
    result["predicted_alarm"] = alarms
    result["model_name"] = "xgboost_cusum"
    result["cusum_state"] = states
    return result


def alarm_status(timestamps, margins, *, drift, threshold, cadence_s, max_gap_s):
    """Track an alarm state for one time-ordered flight at a nominal sample cadence."""
    active = np.empty(len(margins), dtype=bool)
    evidence = 0.0
    alarming = False
    release = threshold / 4
    gaps = np.diff(timestamps, prepend=timestamps[0])
    weights = np.minimum(gaps / cadence_s, 2.0)
    reset = gaps > max_gap_s
    reset[0] = True
    weights[reset] = 1.0
    changes = (margins - drift) * weights
    for index, (change, start) in enumerate(zip(changes, reset, strict=True)):
        if start:
            evidence = 0.0
            alarming = False
        evidence = min(threshold, max(0.0, evidence + change))
        if alarming:
            alarming = evidence > release
        else:
            alarming = evidence >= threshold
        active[index] = alarming
    return active


def main():
    parser = argparse.ArgumentParser(description="Run CUSUM on saved standard XGBoost predictions.")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--drift", type=float, required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--max-gap-s", type=float, help="Reset before a sample after a gap longer than this")
    args = parser.parse_args()
    rows = read_csv(args.predictions, PREDICTION_COLUMNS)
    result = score_flights(rows, drift=args.drift, threshold=args.threshold, max_gap_s=args.max_gap_s)
    with args.output.open("x", encoding="utf-8", newline="") as stream:
        result.to_csv(stream, index=False)


if __name__ == "__main__":
    main()

