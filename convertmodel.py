import joblib # type: ignore
import coremltools as ct # type: ignore

# Load your trained scikit-learn model (assuming it was saved with joblib)
model = joblib.load("eye_glucose_model.pkl")

# Define the names of the input features. These should match your model’s expected features.
# For example, if your model expects 12 features as shown in your Python code:
input_features = [
    "pupil_size",
    "sclera_redness",
    "vein_prominence",
    "pupil_response_time",
    "ir_intensity",
    "scleral_vein_density",
    "ir_temperature",
    "tear_film_reflectivity",
    "pupil_dilation_rate",
    "sclera_color_balance",
    "vein_pulsation_intensity",
    "birefringence_index"
]

# Define the output feature name.
output_feature = "glucose"  # Adjust this name to match your model’s output if needed.

# Convert the scikit-learn model to a Core ML model.
coreml_model = ct.converters.sklearn.convert(model,
                                             input_features=input_features,
                                             output_feature_names=[output_feature])

# Save the Core ML model to a file.
coreml_model.save("EyeGlucoseModel.mlmodel")
