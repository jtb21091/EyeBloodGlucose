import cv2
import os
import pandas as pd
from datetime import datetime

data_dir = "eye_glucose_data"
image_dir = os.path.join(data_dir, "images")
labels_file = os.path.join(data_dir, "labels.csv")

# Ensure directories exist
os.makedirs(image_dir, exist_ok=True)

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
        
        # Log image details without blood glucose level initially
        log_eye_data(filename, frame.shape)
    else:
        print("Error: Could not capture image.")

def log_eye_data(filename, dimensions):
    height, width, channels = dimensions
    data = pd.DataFrame([[filename, "", height, width, channels]], 
                         columns=["filename", "blood_glucose", "height", "width", "channels"])
    if os.path.exists(labels_file):
        data.to_csv(labels_file, mode='a', header=False, index=False)
    else:
        data.to_csv(labels_file, index=False)
    print(f"Logged: {filename} - Dimensions: {height}x{width}x{channels}, awaiting blood glucose input.")

if __name__ == "__main__":
    capture_eye_image()
    print("Manually update the CSV file with the blood glucose level.")
