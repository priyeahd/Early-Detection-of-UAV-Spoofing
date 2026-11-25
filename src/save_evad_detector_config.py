
import json
from pathlib import Path

config = {
    "model": "relative_displacement_lstm",
    "feature": "mean_position_error",
    "threshold": 0.175784,
    "consecutive_alerts": 3,
    "input_steps": 20,
    "output_steps": 5,
    "sampling_interval_seconds": 0.1,
    "threshold_selection": "validation_normal_99.9_percentile",
    "evaluation_status": "validation_selected_test_not_evaluated"
}

output = Path("models/evad_temporal_detector_config.json")

if output.exists():
    raise FileExistsError(
        f"Configuration already exists: {output}"
    )

output.parent.mkdir(parents=True, exist_ok=True)

with open(output, "w") as f:
    json.dump(config, f, indent=4)

print("STEP 19.29: FREEZE EVAD DETECTOR CONFIGURATION")
print(json.dumps(config, indent=4))
print("\nSaved:", output)
print("STEP 19.29 COMPLETE")
