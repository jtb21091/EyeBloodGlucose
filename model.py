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
    
    cv2.waitKey(500)  # Delay for camera stabilization
    
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

def get_birefringence_index(image):
    """Estimate birefringence using edge density and gradient contrast."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    gradient = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=5)
    return round((np.mean(edges) + np.std(gradient)) / 2, 5)

def update_data():
    filename, frame = capture_eye_image()
    if filename is None or frame is None:
        return
    
    birefringence_index = get_birefringence_index(frame)

    # Define only necessary columns
    columns = [
        "filename", "blood_glucose", "pupil_size", "sclera_redness", "vein_prominence",
        "ir_temperature", "tear_film_reflectivity", "pupil_dilation_rate",
        "sclera_color_balance", "vein_pulsation_intensity", "birefringence_index"
    ]

    if not os.path.exists(labels_file) or os.stat(labels_file).st_size == 0:
        df = pd.DataFrame(columns=columns)
    else:
        df = pd.read_csv(labels_file)

    new_entry = pd.DataFrame([[
        filename, "", np.random.uniform(20, 100), np.random.uniform(0, 100),
        np.random.uniform(0, 10), np.random.uniform(0.1, 0.5), np.random.uniform(10, 255),
        np.random.uniform(0, 1), np.nan, np.random.uniform(10, 40),
        np.random.uniform(0, 255), np.random.uniform(0.1, 1.0),
        np.random.uniform(0.8, 1.2), np.random.uniform(0, 10),
        birefringence_index
    ]], columns=columns)

    df = pd.concat([df, new_entry], ignore_index=True)
    df.to_csv(labels_file, index=False)
    print(f"New data added to CSV: {filename}")

if __name__ == "__main__":
    update_data()
