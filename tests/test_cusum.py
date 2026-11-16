import unittest

import pandas as pd

from src.sequential.cusum import score_flights


def predictions(samples):
    return pd.DataFrame(samples, columns=["flight_id", "timestamp_s", "raw_score"]).assign(
        attack_probability=0.5,
        predicted_alarm=0,
        score_type="logistic_margin",
        model_name="xgboost_baseline",
        run_id="fixture",
        split="validation",
    )


class CusumTest(unittest.TestCase):
    def test_accumulates_in_time_order_and_resets_after_alarm_and_at_flight_boundary(self):
        rows = predictions([
            ("B", "0", 2), ("A", "11", 2), ("A", "10", 2), ("A", "2", -8), ("A", "12", 2),
        ]).assign(attack_label=0)
        actual = score_flights(rows, drift=1, threshold=2)
        self.assertEqual(actual["timestamp_s"].tolist(), ["2", "10", "11", "12", "0"])
        self.assertEqual(actual["cusum_state"].tolist(), [0, 1, 2, 1, 1])
        self.assertEqual(actual["predicted_alarm"].tolist(), [0, 0, 1, 0, 0])
        self.assertEqual(set(actual["model_name"]), {"xgboost_cusum"})
        pd.testing.assert_frame_equal(
            actual, score_flights(rows.assign(attack_label=1), drift=1, threshold=2).assign(attack_label=0),
        )

    def test_optional_gap_reset(self):
        rows = predictions([("A", "0", 1), ("A", "1", 1), ("A", "9", 1)])
        self.assertEqual(score_flights(rows, drift=0, threshold=3)["predicted_alarm"].tolist(), [0, 0, 1])
        actual = score_flights(rows, drift=0, threshold=3, max_gap_s=1)
        self.assertEqual(actual["cusum_state"].tolist(), [1, 2, 1])

    def test_rejects_ambiguous_timestamps_and_nonfinite_margins(self):
        rows = predictions([("A", "1.0", 2), ("A", "1", 2)])
        with self.assertRaisesRegex(ValueError, "unique within a flight"):
            score_flights(rows, drift=1, threshold=2)
        rows = predictions([("A", "0", float("inf"))])
        with self.assertRaisesRegex(ValueError, "margins must be finite"):
            score_flights(rows, drift=1, threshold=2)
