
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path("data/resampled/px4")
MANIFEST = Path("data/sequences/px4/sequence_manifest.csv")
OUTPUT = Path("results/px4_constant_velocity_metrics.csv")

INPUT_STEPS = 20
OUTPUT_STEPS = 5
DT = 0.1

manifest = pd.read_csv(MANIFEST)
test = manifest[manifest["split"] == "test"]

results = []

for label, condition in [(0, "Normal"), (1, "Attacked")]:
    group = test[test["attack_label"] == label]

    sum_squared = 0.0
    sum_absolute = 0.0
    total_values = 0

    for path, flight_group in group.groupby("file_path"):
        df = pd.read_csv(
            DATA_DIR / path,
            usecols=["px", "py", "pz", "vx", "vy", "vz"]
        )

        values = df[
            ["px", "py", "pz", "vx", "vy", "vz"]
        ].to_numpy(dtype=np.float64)

        future_times = (
            np.arange(1, OUTPUT_STEPS + 1) * DT
        )

        for start in flight_group["start_index"]:
            start = int(start)
            last = values[start + INPUT_STEPS - 1]

            predicted = (
                last[:3][None, :]
                + future_times[:, None] * last[3:6][None, :]
            )

            actual = values[
                start + INPUT_STEPS:
                start + INPUT_STEPS + OUTPUT_STEPS,
                :3
            ]

            error = predicted - actual

            sum_squared += np.sum(error ** 2)
            sum_absolute += np.sum(np.abs(error))
            total_values += error.size

    rmse = np.sqrt(sum_squared / total_values)
    mae = sum_absolute / total_values

    results.append({
        "condition": condition,
        "sequences": len(group),
        "RMSE_m": rmse,
        "MAE_m": mae
    })

    print(
        f"{condition}: RMSE={rmse:.6f} m, "
        f"MAE={mae:.6f} m"
    )

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(results).to_csv(OUTPUT, index=False)

print("Saved:", OUTPUT)
print("STEP 16.9 COMPLETE")
