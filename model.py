import os
import cv2
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

labels_file = "eye_glucose_data/labels.csv"
image_dir = "eye_glucose_data/images"
model_file = "eye_glucose_model.pkl"
os.makedirs(image_dir, exist_ok=True)

def capture_eye_image():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return None, None
    
    # Wait for camera to stabilize
    cv2.waitKey(500)  # Delay 500ms
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("Error: Could not capture image.")
        return None, None
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"eye_{timestamp}.jpg"
    filepath = os.path.join(image_dir, filename)
    cv2.imwrite(filepath, frame)
    print(f"Image saved: {filepath}")
    
    return filename, frame

def detect_pupil(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)  # Enhance contrast
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1.5, 50, param1=50, param2=30, minRadius=10, maxRadius=100)
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        largest_circle = max(circles[0], key=lambda x: x[2])  # Select the largest detected circle
        print(f"Detected pupil with radius: {largest_circle[2]}")
        return largest_circle[2]  # Return pupil radius
    
    print("No pupil detected.")
    return 0.0

def get_sclera_redness(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0, 50, 50])
    upper_red = np.array([10, 255, 255])
    mask = cv2.inRange(hsv, lower_red, upper_red)
    redness_intensity = round(np.sum(mask) / (mask.shape[0] * mask.shape[1]), 5)
    return redness_intensity

def get_vein_prominence(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 30, 100)  # Lowered threshold for finer details
    vein_density = round(np.sum(edges) / (edges.shape[0] * edges.shape[1]), 5)
    return vein_density

def get_pupil_response_time():
    return np.random.uniform(0.1, 0.4)  # Simulated placeholder value

def update_data():
    filename, frame = capture_eye_image()
    if filename is None or frame is None:
        return
    
    height, width, channels = frame.shape
    pupil_size = detect_pupil(frame)
    sclera_redness = get_sclera_redness(frame)
    vein_prominence = get_vein_prominence(frame)
    pupil_response_time = get_pupil_response_time()
    
    if os.path.exists(labels_file):
        df = pd.read_csv(labels_file)
    else:
        df = pd.DataFrame(columns=["filename", "blood_glucose", "height", "width", "channels", "pupil_size", "sclera_redness", "vein_prominence", "pupil_response_time"])
    
    new_entry = pd.DataFrame([[filename, "", height, width, channels, pupil_size, sclera_redness, vein_prominence, pupil_response_time]],
                              columns=["filename", "blood_glucose", "height", "width", "channels", "pupil_size", "sclera_redness", "vein_prominence", "pupil_response_time"])
    df = pd.concat([df, new_entry], ignore_index=True)
    df.to_csv(labels_file, index=False)
    print(f"New data added to CSV: {filename}")

if __name__ == "__main__":
    update_data()
