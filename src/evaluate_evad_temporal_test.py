
from pathlib import Path
import json

import numpy as np
import pandas as pd

FEATURE_FILE = Path(
    "data/sequences/px4/evad_features.csv"
)

SPLIT_FILE = Path(
    "data/splits/flight_splits.csv"
)

CONFIG_FILE = Path(
    "models/evad_temporal_detector_config.json"
)

OUTPUT_FILE = Path(
    "results/evad_temporal_test_flights.csv"
)

DATA_DIR = Path("data/resampled/px4")


def main():

    print("STEP 19.31: CONTINUOUS EVAD TEST EVALUATION")

    with open(CONFIG_FILE) as f:
        config = json.load(f)

    threshold = config["threshold"]
    required = config["consecutive_alerts"]
    input_steps = config["input_steps"]
    output_steps = config["output_steps"]

    features = pd.read_csv(FEATURE_FILE)
    splits = pd.read_csv(SPLIT_FILE)

    test_features = features[
        features["split"] == "test"
    ]

    test_flights = splits[
        splits["split"] == "test"
    ]

    records = []

    for _, flight in test_flights.iterrows():

        path = flight["file_path"]

        group = test_features[
            test_features["file_path"] == path
        ].sort_values("start_index")

        df = pd.read_csv(
            DATA_DIR / path,
            usecols=["time_usec", "attack_enabled"]
        )

        labels = (
            df["attack_enabled"].to_numpy() > 0
        ).astype(int)

        timestamps = df["time_usec"].to_numpy()

        attack_starts = np.where(
            (labels[1:] == 1)
            & (labels[:-1] == 0)
        )[0] + 1

        alerts = group[
            "mean_position_error"
        ].to_numpy() > threshold

        starts = group[
            "start_index"
        ].to_numpy(dtype=int)

        # An error is only available after all five
        # future observations have arrived.
        available_indices = (
            starts + input_steps + output_steps - 1
        )

        alarm_indices = []

        for i in range(required - 1, len(alerts)):

            recent_alerts = alerts[
                i - required + 1:i + 1
            ]

            recent_starts = starts[
                i - required + 1:i + 1
            ]

            if (
                recent_alerts.all()
                and np.all(np.diff(recent_starts) == 1)
            ):
                alarm_indices.append(
                    int(available_indices[i])
                )

        # First detection after each attack onset
        detection_delays = []

        for attack_start in attack_starts:

            valid_alarms = [
                idx for idx in alarm_indices
                if idx >= attack_start and labels[idx] == 1
            ]

            if valid_alarms:

                first = valid_alarms[0]

                delay = (
                    timestamps[first]
                    - timestamps[attack_start]
                ) / 1_000_000

                detection_delays.append(delay)

        has_attack = bool(labels.any())

        # Count alarms while the ground-truth
        # attack state is normal.
        false_alarms = sum(
            labels[idx] == 0
            for idx in alarm_indices
        )


        # Count distinct false-alarm episodes
        false_alarm_indices = sorted(
            idx for idx in alarm_indices
            if labels[idx] == 0
        )

        false_alarm_episodes = 0
        previous_index = None

        for idx in false_alarm_indices:
            if (
                previous_index is None
                or idx != previous_index + 1
            ):
                false_alarm_episodes += 1

            previous_index = idx


        records.append({
            "file_path": path,
            "trajectory": flight["trajectory"],
            "condition": flight["condition"],
            "has_attack": has_attack,
            "num_attack_onsets": len(attack_starts),
            "total_alarm_windows": len(alarm_indices),
            "false_alarm_windows": false_alarms,
            "false_alarm_episodes": false_alarm_episodes,
            "attack_detected": (
                len(detection_delays) > 0
            ),
            "first_detection_delay_s": (
                min(detection_delays)
                if detection_delays else np.nan
            )
        })

    results = pd.DataFrame(records)

    OUTPUT_FILE.parent.mkdir(
        parents=True, exist_ok=True
    )

    results.to_csv(
        OUTPUT_FILE,
        index=False
    )

    normal = results[
        results["condition"] == "Normal"
    ]

    attacked = results[
        results["condition"] == "Attacked"
    ]

    print("\nTEST RESULTS")

    print("Total test flights:", len(results))
    print("Normal flights:", len(normal))
    print("Attacked flights:", len(attacked))

    print(
        "Normal flights with false alarms:",
        int((normal["false_alarm_windows"] > 0).sum())
    )

    print(
        "Attacked flights detected:",
        int(attacked["attack_detected"].sum())
    )

    print(
        "Median first detection delay (s):",
        attacked["first_detection_delay_s"].median()
    )

    print("\nSaved:", OUTPUT_FILE)
    print("STEP 19.31 COMPLETE")


if __name__ == "__main__":
    main()
