
from pathlib import Path
import pandas as pd

INPUT_FILE = Path(
    "results/evad_temporal_test_flights.csv"
)

OUTPUT_FILE = Path(
    "results/evad_trajectory_summary.csv"
)

df = pd.read_csv(INPUT_FILE)

summary = df.groupby(
    ["trajectory", "condition"]
).agg(
    flights=("file_path", "count"),
    detected=("attack_detected", "sum"),
    false_alarm_episodes=("false_alarm_episodes", "sum"),
    median_delay_s=("first_detection_delay_s", "median")
).reset_index()

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

summary.to_csv(OUTPUT_FILE, index=False)

print("STEP 19.50: GENERATE RESULTS SUMMARY")
print(summary.to_string(index=False))
print("\nSaved:", OUTPUT_FILE)
print("STEP 19.50 COMPLETE")
