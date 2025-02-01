import os
import cv2
import pandas as pd
import numpy as np
from datetime import datetime

labels_file = "eye_glucose_data/labels.csv"
image_dir = "eye_glucose_data/images"
os.makedirs(image_dir, exist_ok=True)

def capture_eye_image():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return None, None
    
    cv2.waitKey(500)  # Delay 500ms for camera stabilization
    
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

def get_pupil_size(image):
    return np.random.uniform(20, 100)  # Placeholder for actual detection

def get_sclera_redness(image):
    return np.random.uniform(0, 100)  # Placeholder for actual detection

def get_vein_prominence(image):
    return np.random.uniform(0, 10)  # Placeholder for actual detection

def get_ir_temperature(image):
    return round(np.mean(image[:, :, 2]), 5)

def get_tear_film_reflectivity(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return round(np.std(gray), 5)

def get_pupil_dilation_rate():
    return np.random.uniform(0.1, 1.0)

def get_sclera_color_balance(image):
    r_mean = np.mean(image[:, :, 2])
    g_mean = np.mean(image[:, :, 1])
    return round(r_mean / g_mean, 5) if g_mean > 0 else 1.0

def get_vein_pulsation_intensity(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return round(np.mean(cv2.Laplacian(gray, cv2.CV_64F)), 5)

def get_pupil_response_time():
    return np.random.uniform(0.1, 0.5)

def get_ir_intensity(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return round(np.mean(gray), 5)

def get_scleral_vein_density(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)  # Edge detection for veins
    return round(np.sum(edges) / (image.shape[0] * image.shape[1]), 5)

def update_data():
    filename, frame = capture_eye_image()
    if filename is None or frame is None:
        return
    
    height, width, channels = frame.shape
    pupil_size = get_pupil_size(frame)
    sclera_redness = get_sclera_redness(frame)
    vein_prominence = get_vein_prominence(frame)
    ir_temperature = get_ir_temperature(frame)
    tear_film_reflectivity = get_tear_film_reflectivity(frame)
    pupil_dilation_rate = get_pupil_dilation_rate()
    sclera_color_balance = get_sclera_color_balance(frame)
    vein_pulsation_intensity = get_vein_pulsation_intensity(frame)
    pupil_response_time = get_pupil_response_time()
    ir_intensity = get_ir_intensity(frame)
    scleral_vein_density = get_scleral_vein_density(frame)

    columns = [
        "filename", "blood_glucose", "height", "width", "channels", "pupil_size", 
        "sclera_redness", "vein_prominence", "pupil_response_time", "ir_intensity", 
        "scleral_vein_density", "pupil_circularity", "blink_rate", "ir_temperature", 
        "tear_film_reflectivity", "pupil_dilation_rate", "sclera_color_balance", 
        "vein_pulsation_intensity"
    ]

    if not os.path.exists(labels_file) or os.stat(labels_file).st_size == 0:
        df = pd.DataFrame(columns=columns)
    else:
        df = pd.read_csv(labels_file)

    for col in columns:
        if col not in df.columns:
            df[col] = np.nan  # Ensure all expected columns exist

    new_entry = pd.DataFrame([[
        filename, "", height, width, channels, pupil_size, sclera_redness, vein_prominence, 
        pupil_response_time, ir_intensity, scleral_vein_density, np.nan, np.nan, ir_temperature, 
        tear_film_reflectivity, pupil_dilation_rate, sclera_color_balance, vein_pulsation_intensity
    ]], columns=columns)

    df = pd.concat([df, new_entry], ignore_index=True)
    df.to_csv(labels_file, index=False)
    print(f"New data added to CSV: {filename}")

if __name__ == "__main__":
    update_data()
