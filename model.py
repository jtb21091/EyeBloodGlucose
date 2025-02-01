import os
import cv2
import pandas as pd
import numpy as np
from datetime import datetime

# Define file paths
labels_file = "eye_glucose_data/labels.csv"
image_dir = "eye_glucose_data/images"
os.makedirs(image_dir, exist_ok=True)

# Load Haar cascade for eye detection
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

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

    # Convert to grayscale for detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # If no eyes detected, do not proceed.
    if len(eyes) == 0:
        print("No eyes detected. Data not added.")
        return None, None

    # Compute a bounding box that encloses all detected eyes.
    x_min = min([x for (x, y, w, h) in eyes])
    y_min = min([y for (x, y, w, h) in eyes])
    x_max = max([x + w for (x, y, w, h) in eyes])
    y_max = max([y + h for (x, y, w, h) in eyes])

    # Crop the image to the region containing the eyes.
    roi = frame[y_min:y_max, x_min:x_max]

    # Save the cropped eye region
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"eye_{timestamp}.jpg"
    filepath = os.path.join(image_dir, filename)
    cv2.imwrite(filepath, roi)
    print(f"Eye region saved: {filepath}")

    return filename, roi

def get_birefringence_index(image):
    """Estimate birefringence using edge density and gradient contrast."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    gradient = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=5)
    return round((np.mean(edges) + np.std(gradient)) / 2, 5)

def get_ir_temperature(image):
    """
    Simulate the IR temperature calculation.
    This dummy calculation converts the grayscale average intensity to a temperature range.
    For a real application, replace this with a proper IR sensor measurement and calibration.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    avg_intensity = np.mean(gray)
    # Map average intensity (0-255) to a temperature range (e.g., 20°C to 60°C)
    temperature = (avg_intensity / 255.0) * 40 + 20
    return round(temperature, 2)

def update_data():
    filename, roi = capture_eye_image()
    if filename is None or roi is None:
        # If no eyes are detected, exit without adding data.
        return

    birefringence_index = get_birefringence_index(roi)
    ir_temperature = get_ir_temperature(roi)  # Calculate IR temperature from the ROI

    # Define the columns for our CSV
    columns = [
        'filename', 'blood_glucose', 'pupil_size', 'sclera_redness', 
        'vein_prominence', 'pupil_response_time', 'ir_intensity', 
        'scleral_vein_density', 'ir_temperature', 'tear_film_reflectivity', 
        'pupil_dilation_rate', 'sclera_color_balance', 'vein_pulsation_intensity', 
        'birefringence_index'
    ]

    # If the CSV file does not exist or is empty, create a new DataFrame with the columns.
    if not os.path.exists(labels_file) or os.stat(labels_file).st_size == 0:
        df = pd.DataFrame(columns=columns)
    else:
        df = pd.read_csv(labels_file)
        # Use the columns already present in the file
        columns = df.columns.tolist()

    # Create a new row of data. Ensure the number of values matches the number of columns.
    new_entry = pd.DataFrame([[
        filename,                          # filename
        "",                                # blood_glucose (placeholder)
        np.random.uniform(20, 100),         # pupil_size
        np.random.uniform(0, 100),          # sclera_redness
        np.random.uniform(0, 10),           # vein_prominence
        np.random.uniform(0.1, 0.5),          # pupil_response_time
        np.random.uniform(10, 255),         # ir_intensity
        np.random.uniform(0, 1),            # scleral_vein_density
        ir_temperature,                     # ir_temperature calculated from ROI
        np.random.uniform(10, 40),          # tear_film_reflectivity
        np.random.uniform(0, 255),          # pupil_dilation_rate
        np.random.uniform(0.1, 1.0),          # sclera_color_balance
        np.random.uniform(0.8, 1.2),          # vein_pulsation_intensity
        birefringence_index                # birefringence_index
    ]], columns=columns)

    df = pd.concat([df, new_entry], ignore_index=True)
    df.to_csv(labels_file, index=False)
    print(f"New data added to CSV: {filename}")

if __name__ == "__main__":
    update_data()
