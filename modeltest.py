import os
import cv2
import pandas as pd
import numpy as np
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Define file paths and constants
LABELS_FILE = "eye_glucose_data/labels.csv"
IMAGE_DIR = "eye_glucose_data/images"
os.makedirs(IMAGE_DIR, exist_ok=True)

# Haar cascade for eye detection
EYE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_eye.xml"
eye_cascade = cv2.CascadeClassifier(EYE_CASCADE_PATH)
if eye_cascade.empty():
    logging.error("Failed to load Haar cascade. Check the file path: %s", EYE_CASCADE_PATH)
    exit(1)

# Constants for detection tuning
SCALE_FACTOR = 1.05
MIN_NEIGHBORS = 3
OPEN_THRESHOLD = 0.3  # Heuristic ratio threshold

def capture_eye_image():
    """Captures an image from the webcam, pre-processes it for low light, and detects open eyes."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logging.error("Could not open webcam.")
        return None, None

    try:
        # Allow camera to stabilize
        cv2.waitKey(500)
        ret, frame = cap.read()
        if not ret:
            logging.error("Could not capture image from webcam.")
            return None, None
    finally:
        cap.release()

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Enhance contrast using CLAHE for low-light conditions
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)

    # Optionally, reduce noise (e.g., using a Gaussian blur)
    enhanced_gray = cv2.GaussianBlur(enhanced_gray, (5, 5), 0)

    # Detect eyes with tuned parameters
    eyes = eye_cascade.detectMultiScale(enhanced_gray, scaleFactor=SCALE_FACTOR, minNeighbors=MIN_NEIGHBORS)
    if len(eyes) == 0:
        logging.info("No eyes detected. Data not added.")
        return None, None

    # Filter for open eyes based on height-to-width ratio
    open_eyes = [(x, y, w, h) for (x, y, w, h) in eyes if (h / w) > OPEN_THRESHOLD]
    if not open_eyes:
        logging.info("Eyes appear to be closed. Data not added.")
        return None, None

    # Determine bounding box that encloses all detected open eyes
    x_min = min(x for (x, y, w, h) in open_eyes)
    y_min = min(y for (x, y, w, h) in open_eyes)
    x_max = max(x + w for (x, y, w, h) in open_eyes)
    y_max = max(y + h for (x, y, w, h) in open_eyes)

    # Crop the ROI from the original colored frame
    roi = frame[y_min:y_max, x_min:x_max]

    # Save the ROI with a timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"eye_{timestamp}.jpg"
    filepath = os.path.join(IMAGE_DIR, filename)
    cv2.imwrite(filepath, roi)
    logging.info("Eye region saved: %s", filepath)
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
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    avg_intensity = np.mean(gray)
    temperature = (avg_intensity / 255.0) * 40 + 20  # Map to 20°C to 60°C
    return round(temperature, 14)

def update_data():
    """Capture an eye image, compute indices, and update the CSV dataset."""
    filename, roi = capture_eye_image()
    if filename is None or roi is None:
        return

    birefringence_index = get_birefringence_index(roi)
    ir_temperature = get_ir_temperature(roi)

    # Define CSV columns
    columns = [
        'filename', 'blood_glucose', 'pupil_size', 'sclera_redness',
        'vein_prominence', 'pupil_response_time', 'ir_intensity',
        'scleral_vein_density', 'ir_temperature', 'tear_film_reflectivity',
        'pupil_dilation_rate', 'sclera_color_balance', 'vein_pulsation_intensity',
        'birefringence_index'
    ]

    # Load or initialize the DataFrame
    if not os.path.exists(LABELS_FILE) or os.stat(LABELS_FILE).st_size == 0:
        df = pd.DataFrame(columns=columns)
    else:
        df = pd.read_csv(LABELS_FILE)
        columns = df.columns.tolist()

    # Create a new data entry (using placeholder/random values where appropriate)
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
        birefringence_index                 # birefringence_index
    ]], columns=columns)

    df = pd.concat([df, new_entry], ignore_index=True)
    df.to_csv(LABELS_FILE, index=False)
    logging.info("New data added to CSV: %s", filename)

if __name__ == "__main__":
    update_data()
