import os
import cv2
import pandas as pd
import numpy as np
from datetime import datetime

labels_file = "eye_glucose_data/labels.csv"
image_dir = "eye_glucose_data/images"
os.makedirs(image_dir, exist_ok=True)

blink_counter = 0
last_blink_time = datetime.now()


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

def get_ir_temperature(image):
    return round(np.mean(image[:, :, 2]), 5)  # Simulated placeholder using red channel intensity

def get_tear_film_reflectivity(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return round(np.std(gray), 5)  # Measure light reflection variation

def get_pupil_dilation_rate():
    return np.random.uniform(0.1, 1.0)  # Placeholder for now

def get_sclera_color_balance(image):
    r_mean = np.mean(image[:, :, 2])
    g_mean = np.mean(image[:, :, 1])
    return round(r_mean / g_mean, 5) if g_mean > 0 else 1.0

def get_vein_pulsation_intensity(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return round(np.mean(cv2.Laplacian(gray, cv2.CV_64F)), 5)  # Detect small intensity changes

def update_data():
    filename, frame = capture_eye_image()
    if filename is None or frame is None:
        return
    
    height, width, channels = frame.shape
    ir_temperature = get_ir_temperature(frame)
    tear_film_reflectivity = get_tear_film_reflectivity(frame)
    pupil_dilation_rate = get_pupil_dilation_rate()
    sclera_color_balance = get_sclera_color_balance(frame)
    vein_pulsation_intensity = get_vein_pulsation_intensity(frame)
    
    columns = ["filename", "blood_glucose", "height", "width", "channels", "ir_temperature", "tear_film_reflectivity", "pupil_dilation_rate", "sclera_color_balance", "vein_pulsation_intensity"]
    
    # Check if file exists and has content
    if not os.path.exists(labels_file) or os.stat(labels_file).st_size == 0:
        df = pd.DataFrame(columns=columns)
    else:
        df = pd.read_csv(labels_file)
    
    # Ensure all expected columns exist
    for col in columns:
        if col not in df.columns:
            df[col] = np.nan  # Fill missing columns with NaN
    
    # Append new data
    new_entry = pd.DataFrame([[filename, "", height, width, channels, ir_temperature, tear_film_reflectivity, pupil_dilation_rate, sclera_color_balance, vein_pulsation_intensity]],
                              columns=columns)
    df = pd.concat([df, new_entry], ignore_index=True)
    df.to_csv(labels_file, index=False)
    print(f"New data added to CSV: {filename}")

if __name__ == "__main__":
    update_data()
