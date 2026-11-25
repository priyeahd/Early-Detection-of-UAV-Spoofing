
import numpy as np
import torch
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

from lstm_model import MSVStackedLSTM

print("STEP 7: EVALUATE MSV-STACKED LSTM")

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Load synthetic sequences
X = np.load("data/sequences/X_sample.npy")
y = np.load("data/sequences/y_sample.npy")

# Use the same chronological split as training
split = int(0.8 * len(X))
X_val = X[split:]
y_val = y[split:]

# Load model and normalization parameters
checkpoint = torch.load(
    "models/msv_stlstm_synthetic.pth",
    map_location=DEVICE,
    weights_only=True
)

model = MSVStackedLSTM(
    input_size=checkpoint["input_size"]
).to(DEVICE)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

# Normalize validation inputs
X_val_scaled = (
    X_val - checkpoint["x_mean"]
) / checkpoint["x_std"]

X_tensor = torch.from_numpy(
    X_val_scaled
).float().to(DEVICE)

# Predict
with torch.no_grad():
    predictions_scaled = model(
        X_tensor
    ).cpu().numpy()

# Convert predicted coordinates to original scale
predictions = (
    predictions_scaled * checkpoint["y_std"]
    + checkpoint["y_mean"]
)

# Calculate metrics
actual_flat = y_val.reshape(-1, 3)
pred_flat = predictions.reshape(-1, 3)

rmse = np.sqrt(
    mean_squared_error(actual_flat, pred_flat)
)

mae = mean_absolute_error(
    actual_flat, pred_flat
)

r2 = r2_score(
    actual_flat, pred_flat
)

print("Device:", DEVICE)
print("Evaluation sequences:", len(X_val))
print("Prediction shape:", predictions.shape)

print(f"RMSE: {rmse:.6f}")
print(f"MAE: {mae:.6f}")
print(f"R2 Score: {r2:.6f}")

# Plot one validation sequence
Path("results").mkdir(exist_ok=True)

sample_index = 0

actual = y_val[sample_index]
predicted = predictions[sample_index]

fig, axes = plt.subplots(
    3, 1,
    figsize=(9, 9),
    sharex=True
)

for i, axis_name in enumerate(["X", "Y", "Z"]):
    axes[i].plot(
        range(1, 6),
        actual[:, i],
        marker="o",
        label="Actual"
    )

    axes[i].plot(
        range(1, 6),
        predicted[:, i],
        marker="x",
        label="Predicted"
    )

    axes[i].set_ylabel(
        f"{axis_name} position"
    )
    axes[i].legend()
    axes[i].grid(True)

axes[-1].set_xlabel(
    "Future prediction step"
)

fig.suptitle(
    "MSV-StLSTM: Actual vs Predicted UAV Positions"
)

plt.tight_layout()

output_file = (
    "results/lstm_trajectory_prediction.png"
)

plt.savefig(
    output_file,
    dpi=300
)

plt.close()

print("Plot saved:", output_file)
print("STEP 7 COMPLETE")
