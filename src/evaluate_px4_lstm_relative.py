
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

from lstm_model import MSVStackedLSTM


DATA_DIR = Path("data/resampled/px4")

MANIFEST_FILE = Path(
    "data/sequences/px4/sequence_manifest.csv"
)

MODEL_FILE = Path(
    "models/msv_stlstm_px4_relative_best.pth"
)

RESULTS_DIR = Path("results")

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

BATCH_SIZE = 512


def evaluate_group(model, records, checkpoint):

    feature_columns = checkpoint["feature_columns"]

    mean = np.asarray(checkpoint["feature_mean"])
    std = np.asarray(checkpoint["feature_std"])

    input_steps = checkpoint["input_steps"]
    output_steps = checkpoint["output_steps"]

    scale = checkpoint.get("displacement_scale", 10.0)

    actual_list = []
    predicted_list = []

    for path, flight_records in records.groupby("file_path"):

        df = pd.read_csv(
            DATA_DIR / path,
            usecols=feature_columns
        )

        values = df[feature_columns].to_numpy(
            dtype=np.float32
        )

        starts = flight_records[
            "start_index"
        ].to_numpy(dtype=np.int64)

        for batch_start in range(
            0, len(starts), BATCH_SIZE
        ):

            batch_indices = starts[
                batch_start:batch_start + BATCH_SIZE
            ]

            X = np.stack([
                values[s:s + input_steps].copy()
                for s in batch_indices
            ])

            y = np.stack([
                values[
                    s + input_steps:
                    s + input_steps + output_steps,
                    :3
                ]
                for s in batch_indices
            ])

            # Store the last observed UAV position
            last_positions = X[:, -1, :3].copy()

            # Convert positions to relative coordinates
            X[:, :, :3] -= last_positions[:, None, :]

            # Scale relative positions
            X[:, :, :3] /= scale

            # Normalize remaining motion features
            X[:, :, 3:] = (
                X[:, :, 3:] - mean[3:]
            ) / std[3:]

            X_tensor = torch.from_numpy(
                X.astype(np.float32)
            ).to(DEVICE)

            # Predict future displacements
            with torch.no_grad():
                prediction = model(
                    X_tensor
                ).cpu().numpy()

            # Convert predictions back to meters
            prediction = prediction * scale

            # Recover absolute UAV positions
            prediction = (
                prediction + last_positions[:, None, :]
            )

            actual_list.append(y)
            predicted_list.append(prediction)

    if not actual_list:
        return None

    actual = np.concatenate(
        actual_list
    ).reshape(-1, 3)

    predicted = np.concatenate(
        predicted_list
    ).reshape(-1, 3)

    rmse = np.sqrt(
        mean_squared_error(actual, predicted)
    )

    mae = mean_absolute_error(
        actual, predicted
    )

    r2 = r2_score(
        actual, predicted
    )

    return {
        "sequences": len(records),
        "RMSE_m": float(rmse),
        "MAE_m": float(mae),
        "R2": float(r2)
    }


def main():

    print("EVALUATE REAL PX4 RELATIVE LSTM")
    print("Device:", DEVICE)

    RESULTS_DIR.mkdir(
        parents=True, exist_ok=True
    )

    # Load the saved trained model
    checkpoint = torch.load(
        MODEL_FILE,
        map_location=DEVICE,
        weights_only=False
    )

    model = MSVStackedLSTM(
        input_size=len(checkpoint["feature_columns"]),
        hidden_size=128,
        num_layers=2,
        output_steps=checkpoint["output_steps"],
        dropout=0.2
    ).to(DEVICE)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    # Load only test sequences
    manifest = pd.read_csv(MANIFEST_FILE)

    test_records = manifest[
        manifest["split"] == "test"
    ]

    print("Test sequences:", len(test_records))

    results = []

    for attack_label, name in [
        (0, "Normal"),
        (1, "Attacked")
    ]:

        group = test_records[
            test_records["attack_label"] == attack_label
        ]

        print(f"\nEvaluating {name} flights...")

        metrics = evaluate_group(
            model, group, checkpoint
        )

        if metrics is not None:

            metrics["condition"] = name
            results.append(metrics)

            print("Sequences:", metrics["sequences"])
            print("RMSE (m):", round(metrics["RMSE_m"], 6))
            print("MAE (m):", round(metrics["MAE_m"], 6))
            print("R2:", round(metrics["R2"], 6))

    output_file = (
        RESULTS_DIR / "px4_lstm_relative_test_metrics.csv"
    )

    pd.DataFrame(results).to_csv(
        output_file,
        index=False
    )

    print("\nResults saved:", output_file)
    print("EVALUATION COMPLETE")


if __name__ == "__main__":
    main()
