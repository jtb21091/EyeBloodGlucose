import cv2
import pandas as pd
import numpy as np
import os
import joblib

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
        for (x, y, r) in circles[0, :]:
            return r  # Return pupil radius in pixels
    return None

def get_sclera_redness(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0, 50, 50])
    upper_red = np.array([10, 255, 255])
    mask = cv2.inRange(hsv, lower_red, upper_red)
    redness_level = np.sum(mask) / (mask.shape[0] * mask.shape[1])  # Normalize redness intensity
    return round(redness_level, 5)

def predict_blood_glucose(pupil_size, sclera_redness):
    if model is not None and pupil_size is not None:
        features = np.array([[pupil_size, sclera_redness]])
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
        predicted_glucose = predict_blood_glucose(pupil_size, sclera_redness)
        
        display_text = f"Pupil: {pupil_size}, Redness: {sclera_redness}, Glucose: {predicted_glucose}"
        cv2.putText(frame, display_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow("Eye Glucose Monitor", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    live_eye_analysis()
