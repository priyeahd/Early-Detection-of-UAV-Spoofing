
import joblib

MODEL_FILE = "models/evad_mlp.pkl"

print("INSPECTING SAVED MLP MODEL")

model = joblib.load(MODEL_FILE)

print("\nModel type:")
print(type(model))

print("\nModel:")
print(model)

if hasattr(model, "named_steps"):
    print("\nPipeline steps:")
    for name, step in model.named_steps.items():
        print(name, ":", type(step).__name__)

print("\nINSPECTION COMPLETE")
