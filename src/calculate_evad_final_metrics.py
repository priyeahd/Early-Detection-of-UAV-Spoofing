
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

INPUT_FILE = Path(
    "results/evad_temporal_test_flights.csv"
)

OUTPUT_FILE = Path(
    "results/evad_final_metrics.csv"
)

print("STEP 19.47: FINAL EVAD METRICS")

df = pd.read_csv(INPUT_FILE)

# Flight-level ground truth
y_true = (
    df["condition"] == "Attacked"
).astype(int)

# Whether an attack was detected while active
y_pred = df["attack_detected"].astype(int)

tn, fp, fn, tp = confusion_matrix(
    y_true, y_pred, labels=[0, 1]
).ravel()

metrics = {
    "accuracy": accuracy_score(y_true, y_pred),
    "precision": precision_score(
        y_true, y_pred, zero_division=0
    ),
    "recall": recall_score(
        y_true, y_pred, zero_division=0
    ),
    "f1_score": f1_score(
        y_true, y_pred, zero_division=0
    ),
    "false_positive_rate": (
        fp / (fp + tn) if fp + tn else 0
    ),
    "true_negatives": int(tn),
    "false_positives": int(fp),
    "false_negatives": int(fn),
    "true_positives": int(tp),
    "false_alarm_episodes": int(
        df["false_alarm_episodes"].sum()
    ),
    "median_detection_delay_s": (
        df.loc[
            df["condition"] == "Attacked",
            "first_detection_delay_s"
        ].median()
    )
}

print("\nFLIGHT-LEVEL RESULTS")

for name, value in metrics.items():
    print(f"{name}: {value}")

print("\nCONFUSION MATRIX")
print(confusion_matrix(y_true, y_pred, labels=[0, 1]))

OUTPUT_FILE.parent.mkdir(
    parents=True, exist_ok=True
)

pd.DataFrame([metrics]).to_csv(
    OUTPUT_FILE, index=False
)

print("\nSaved:", OUTPUT_FILE)
print("STEP 19.47 COMPLETE")
