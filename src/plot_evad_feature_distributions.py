
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

DATA_FILE = "data/sequences/px4/evad_features.csv"

df = pd.read_csv(DATA_FILE)
df = df[df["split"] == "test"]

features = [
    "mean_position_error",
    "max_position_error",
    "std_position_error"
]

Path("results").mkdir(exist_ok=True)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, feature in zip(axes, features):

    normal = df.loc[
        df["attack_label"] == 0, feature
    ].dropna()

    attacked = df.loc[
        df["attack_label"] == 1, feature
    ].dropna()

    upper = max(
        normal.quantile(0.99),
        attacked.quantile(0.99)
    )

    ax.hist(
        normal,
        bins=60,
        range=(0, upper),
        alpha=0.5,
        density=True,
        label="Normal"
    )

    ax.hist(
        attacked,
        bins=60,
        range=(0, upper),
        alpha=0.5,
        density=True,
        label="Attacked"
    )

    ax.set_title(feature)
    ax.set_xlabel("Prediction error (m)")
    ax.set_ylabel("Density")
    ax.legend()

plt.tight_layout()

output = "results/evad_feature_distributions.png"

plt.savefig(output, dpi=300)
plt.close()

print("Saved:", output)
