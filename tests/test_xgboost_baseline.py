import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from src.baselines import xgboost_baseline as baseline
from tests.xgboost_fixture import write_fixture

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "xgboost.json"


class BaselineTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.data, self.splits, self.schema = write_fixture(self.root)
        self.features = ["fixture_a", "fixture_b"]

    def test_reload_matches_and_preserves_validation_rows_and_feature_order(self):
        bundle = baseline.train(self.data, self.splits, self.schema, CONFIG, self.root / "run")
        replay = self.root / "replayed.csv"
        baseline.replay(self.data, self.splits, bundle, replay)
        self.assertEqual(replay.read_bytes(), (bundle / "validation_predictions.csv").read_bytes())
        actual = pd.read_csv(replay)
        expected = pd.read_csv(self.data).query("split == 'validation'").sort_values(baseline.KEY)
        pd.testing.assert_frame_equal(
            actual[baseline.KEY + ["attack_label"]],
            expected[baseline.KEY + ["attack_label"]].reset_index(drop=True),
        )
        model = XGBClassifier()
        model.load_model(bundle / "booster.ubj")
        self.assertEqual(model.get_booster().feature_names, self.features)
        with self.assertRaisesRegex(ValueError, "feature_names mismatch"):
            model.predict_proba(expected[self.features[::-1]])
        np.testing.assert_allclose(
            actual["attack_probability"], 1 / (1 + np.exp(-actual["raw_score"])), atol=1e-7,
        )

    def test_rejects_metadata_and_ground_truth_features(self):
        schema = baseline.read_json(self.schema)
        for column in ("Attack Enabled", "attack_label", "attack_enabled", "flight_id",
                       "time_usec", "timestamp_s", "split", "GroundTruth_Altitude", "future_speed"):
            with self.subTest(column=column):
                schema["features"][0]["name"] = column
                with self.assertRaisesRegex(ValueError, "forbidden"):
                    baseline.feature_order(schema)

    def test_rejects_duplicate_flight_and_partition_disagreement(self):
        manifest = self.splits.read_text(encoding="utf-8")
        self.splits.write_text(manifest + "fixture_0,train,fixture_0\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "unique row per flight"):
            list(baseline.partition_rows(self.data, self.splits, self.features, ["train"]))
        self.splits.write_text(manifest, encoding="utf-8")
        data = self.data.read_text(encoding="utf-8")
        self.data.write_text(data.replace(",validation,", ",train,", 1), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "agree on each split"):
            list(baseline.partition_rows(self.data, self.splits, self.features, ["train"]))

    def test_rejects_cross_partition_group_and_duplicate_timestamp(self):
        manifest = self.splits.read_text(encoding="utf-8")
        self.splits.write_text(manifest.replace("train,fixture_1", "train,fixture_0"), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "group_id"):
            list(baseline.partition_rows(self.data, self.splits, self.features, ["train"]))
        self.splits.write_text(manifest, encoding="utf-8")
        data = self.data.read_text(encoding="utf-8")
        self.data.write_text(data + data.splitlines()[1] + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "unique per flight"):
            list(baseline.partition_rows(self.data, self.splits, self.features, ["validation"]))

    def test_test_payload_cannot_affect_training_or_selection(self):
        a = baseline.train(self.data, self.splits, self.schema, CONFIG, self.root / "first")
        data = pd.read_csv(self.data, dtype=str, keep_default_na=False)
        data.loc[data["split"] == "test", ["attack_label", "timestamp_s", *self.features]] = "unused"
        data.to_csv(self.data, index=False)
        b = baseline.train(self.data, self.splits, self.schema, CONFIG, self.root / "second")
        data[data["split"] != "test"].to_csv(self.data, index=False)
        c = baseline.train(self.data, self.splits, self.schema, CONFIG, self.root / "sealed")
        first = baseline.read_json(a / "manifest.json")
        for other in (b, c):
            record = baseline.read_json(other / "manifest.json")
            self.assertNotEqual(first["input_sha256"], record["input_sha256"])
            for key in ("model_sha256", "selection_trials", "threshold"):
                self.assertEqual(first[key], record[key])
            pd.testing.assert_frame_equal(*(
                pd.read_csv(path / "validation_predictions.csv").drop(columns="run_id")
                for path in (a, other)
            ))
