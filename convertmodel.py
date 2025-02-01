import coremltools as ct
import joblib

# Load the trained model
model = joblib.load("eye_glucose_model.pkl")

# Convert it to Core ML format
coreml_model = ct.converters.sklearn.convert(model)

# Save it as a Core ML file
coreml_model.save("eye_glucose_model.mlmodel")

print("Model converted and saved as eye_glucose_model.mlmodel")
