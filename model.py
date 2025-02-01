import cv2
import os
import pandas as pd
import numpy as np
from datetime import datetime

data_dir = "eye_glucose_data"
image_dir = os.path.join(data_dir, "images")
labels_file = os.path.join(data_dir, "labels.csv")

# Ensure directories exist
os.makedirs(image_dir, exist_ok=True)

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

def capture_eye_image():
    cap = cv2.VideoCapture(0)  # Open webcam (change to 1 if using external webcam)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"eye_{timestamp}.jpg"
        filepath = os.path.join(image_dir, filename)
        cv2.imwrite(filepath, frame)
        print(f"Image saved: {filepath}")
        
        # Extract additional eye measurements
        pupil_size = detect_pupil(frame)
        sclera_redness = get_sclera_redness(frame)
        
        # Log image details without blood glucose level initially
        log_eye_data(filename, frame.shape, pupil_size, sclera_redness)
    else:
        print("Error: Could not capture image.")

def log_eye_data(filename, dimensions, pupil_size, sclera_redness):
    height, width, channels = dimensions
    data = pd.DataFrame([[filename, "", height, width, channels, pupil_size, sclera_redness]], 
                         columns=["filename", "blood_glucose", "height", "width", "channels", "pupil_size", "sclera_redness"])
    if os.path.exists(labels_file):
        data.to_csv(labels_file, mode='a', header=False, index=False)
    else:
        data.to_csv(labels_file, index=False)
    print(f"Logged: {filename} - Dimensions: {height}x{width}x{channels}, Pupil Size: {pupil_size}, Sclera Redness: {sclera_redness}, awaiting blood glucose input.")

if __name__ == "__main__":
    capture_eye_image()
    print("Manually update the CSV file with the blood glucose level.")
