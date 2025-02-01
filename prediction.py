import cv2
import pandas as pd
import numpy as np
import os
import joblib

# Load trained model if available
model_file = "eye_glucose_model.pkl"
if os.path.exists(model_file):
    model = joblib.load(model_file)
    trained_features = list(model.feature_names_in_)  # Get trained feature names in correct order
else:
    model = None
    trained_features = []

def detect_pupil(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1.2, 100, param1=50, param2=30, minRadius=10, maxRadius=100)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        return circles[0, 0][2]  # Return pupil radius
    return 0.0  # Default if no pupil detected

def get_pupil_circularity(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1.5, 50, param1=50, param2=30, minRadius=10, maxRadius=100)
    if circles is not None:
        largest_circle = max(circles[0], key=lambda x: x[2])
        area = np.pi * (largest_circle[2] ** 2)
        perimeter = 2 * np.pi * largest_circle[2]
        circularity = (4 * np.pi * area) / (perimeter ** 2)
        return round(circularity, 5)
    return 1.0  # Default if no pupil detected

def get_sclera_redness(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0, 50, 50])
    upper_red = np.array([10, 255, 255])
    mask = cv2.inRange(hsv, lower_red, upper_red)
    return round(np.sum(mask) / (mask.shape[0] * mask.shape[1]), 5)

def get_vein_prominence(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 30, 100)
    return round(np.sum(edges) / (edges.shape[0] * edges.shape[1]), 5)

def get_ir_intensity(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return round(np.mean(gray), 5)

def get_scleral_vein_density(image):
    red_channel = image[:, :, 2]  # Extract red channel
    edges = cv2.Canny(red_channel, 30, 100)
    return round(np.sum(edges) / (edges.shape[0] * edges.shape[1]), 5)

def detect_blink():
    return np.random.randint(0, 3)  # Simulated blink rate for now

def predict_blood_glucose(**kwargs):
    if model is not None:
        # Ensure the order of features matches the training order
        feature_values = [kwargs.get(feature, 0.0) for feature in trained_features]
        features_df = pd.DataFrame([feature_values], columns=trained_features)
        return round(model.predict(features_df)[0], 2)
    return "Model not trained"

def live_eye_analysis():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not capture frame.")
            break
        
        feature_values = {
            "pupil_size": detect_pupil(frame),
            "pupil_circularity": get_pupil_circularity(frame),
            "sclera_redness": get_sclera_redness(frame),
            "vein_prominence": get_vein_prominence(frame),
            "pupil_response_time": 0.2,  # Placeholder
            "ir_intensity": get_ir_intensity(frame),
            "scleral_vein_density": get_scleral_vein_density(frame),
            "blink_rate": detect_blink()
        }
        
        predicted_glucose = predict_blood_glucose(**feature_values)
        
        display_text = f"Pupil: {feature_values['pupil_size']}, Redness: {feature_values['sclera_redness']}, Glucose: {predicted_glucose}"
        cv2.putText(frame, display_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow("Eye Glucose Monitor", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    live_eye_analysis()
