import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from scripts.run_sail_2p import assign_flights, train


class Sail2pTests(unittest.TestCase):
    def inventory(self):
        return pd.DataFrame({
            "flight_id": [f"flight_{i:03d}" for i in range(360)],
            "sha256": [f"hash_{i}" for i in range(360)],
            "scenario": [str(i // 120) for i in range(360)],
            "condition": [str((i // 60) % 2) for i in range(360)],
        })

    def test_split_stable_disjoint_and_stratified(self):
        inventory = self.inventory()
        result = assign_flights(inventory)
        pd.testing.assert_frame_equal(result, assign_flights(inventory.sample(frac=1)))
        self.assertEqual(result["split"].value_counts().to_dict(), {"train": 288, "test": 72})
        self.assertTrue(result.groupby(["scenario", "condition", "split"]).size().isin([48, 12]).all())
        inventory.loc[1, "sha256"] = inventory.loc[0, "sha256"]
        with self.assertRaises(ValueError):
            assign_flights(inventory)

    def test_fit_uses_training_only_and_never_loads_old_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            assign_flights(self.inventory()).to_csv(root / "splits.csv", index=False)
            (root / "schema.json").write_text(json.dumps({"version": "test", "data_kind": "synthetic_fixture",
                "features": [{"name": "velocity_east", "unit": "m/s", "provenance": "fixture", "causal": True}]}))
            (root / "samples.csv").write_text("fixture")
            (root / "test_samples.csv").write_text("test sentinel; must not be parsed during training")
            rows = pd.DataFrame({"velocity_east": [0, 1], "attack_label": [0, 1]})
            with patch("scripts.run_sail_2p.read_flights", return_value=rows) as reader, patch(
                "scripts.run_sail_2p.XGBClassifier"
            ) as classifier:
                classifier.return_value.save_model.side_effect = lambda path: path.write_bytes(b"new model")
                train(root, root / "run")
                self.assertEqual(reader.call_args.args[0], root / "samples.csv")
                self.assertEqual(reader.call_args.args[2], {"train"})
                classifier.return_value.load_model.assert_not_called()
                args, kwargs = classifier.return_value.fit.call_args
                pd.testing.assert_frame_equal(args[0], rows[["velocity_east"]])
                self.assertEqual(kwargs, {})
                selection = json.loads((root / "run/selection.json").read_text())
                self.assertIn("test_samples.csv", selection["dataset_sha256"])
                with self.assertRaises(FileExistsError):
                    train(root, root / "run")


if __name__ == "__main__":
    unittest.main()
