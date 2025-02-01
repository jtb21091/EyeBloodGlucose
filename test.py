import coremltools as ct

model_file = "eye_glucose_model.mlmodel"

# Load Core ML Model
model = ct.models.MLModel(model_file)

# Print model input structure
print("Model Input Description:")
print(model.input_description)

# Print model output structure
print("\nModel Output Description:")
print(model.output_description)
