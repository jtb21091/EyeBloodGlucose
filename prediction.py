import cv2
import pandas as pd
import numpy as np
import os
import joblib
import warnings

# Suppress sklearn warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Load trained model if available
model_file = "eye_glucose_model.pkl"
if os.path.exists(model_file):
    model = joblib.load(model_file)
else:
    model = None

def detect_pupil(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1.2, 100, param1=50, param2=30, minRadius=10, maxRadius=100)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        return circles[0, 0][2]  # Return pupil radius
    return 0.0  # Default if no pupil detected

def get_sclera_redness(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0, 50, 50])
    upper_red = np.array([10, 255, 255])
    mask = cv2.inRange(hsv, lower_red, upper_red)
    return round(np.sum(mask) / (mask.shape[0] * mask.shape[1]), 5)

def predict_blood_glucose(pupil_size, sclera_redness, vein_prominence, pupil_response_time):
    if model is not None:
        # Ensure the feature input matches training format
        features = pd.DataFrame([[pupil_size, sclera_redness, vein_prominence, pupil_response_time]],
                                columns=["pupil_size", "sclera_redness", "vein_prominence", "pupil_response_time"])
        return round(model.predict(features)[0], 2)
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
        
        pupil_size = detect_pupil(frame)
        sclera_redness = get_sclera_redness(frame)
        vein_prominence = 0.0  # Placeholder, update if real-time analysis is added
        pupil_response_time = 0.2  # Placeholder, update if real-time tracking is added
        
        predicted_glucose = predict_blood_glucose(pupil_size, sclera_redness, vein_prominence, pupil_response_time)
        
        display_text = f"Pupil: {pupil_size}, Redness: {sclera_redness}, Glucose: {predicted_glucose}"
        cv2.putText(frame, display_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow("Eye Glucose Monitor", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    live_eye_analysis()
