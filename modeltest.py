import os
import cv2
import dlib
import numpy as np
import pandas as pd
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Define file paths and constants
LABELS_FILE = "eye_glucose_data/labels.csv"
IMAGE_DIR = "eye_glucose_data/images"
os.makedirs(IMAGE_DIR, exist_ok=True)

# dlib face detector and shape predictor paths
FACE_DETECTOR = dlib.get_frontal_face_detector()
SHAPE_PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"  # Update with your path
shape_predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)

def capture_eye_image_refined():
    """Captures an image from the webcam and refines the capture to just the eye region using facial landmarks."""
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

    # Convert frame to grayscale for detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = FACE_DETECTOR(gray)
    if len(faces) == 0:
        logging.info("No face detected.")
        return None, None

    # For simplicity, work with the first detected face
    face = faces[0]

    # Get the 68 facial landmarks
    landmarks = shape_predictor(gray, face)
    # Convert landmarks to a NumPy array
    landmarks_np = np.array([[landmarks.part(i).x, landmarks.part(i).y] for i in range(68)])

    # Choose which eye to extract (e.g., left eye). dlib's 68-point model uses:
    # - Points 36-41 for the left eye.
    left_eye_points = landmarks_np[36:42]

    # Create a convex hull around the eye landmarks to tightly fit the eye shape.
    hull = cv2.convexHull(left_eye_points)
    # Get bounding rectangle from the convex hull.
    x, y, w, h = cv2.boundingRect(hull)

    # Optionally, you could add some padding to the ROI.
    padding = 5
    x = max(x - padding, 0)
    y = max(y - padding, 0)
    w = w + 2 * padding
    h = h + 2 * padding

    # Crop the eye ROI from the original colored frame.
    eye_roi = frame[y:y+h, x:x+w]

    # Save the ROI with a timestamped filename.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"eye_{timestamp}.jpg"
    filepath = os.path.join(IMAGE_DIR, filename)
    cv2.imwrite(filepath, eye_roi)
    logging.info("Refined eye region saved: %s", filepath)
    return filename, eye_roi

def get_birefringence_index(image):
    """Estimate birefringence using edge density and gradient contrast."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    gradient = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=5)
    return round((np.mean(edges) + np.std(gradient)) / 2, 14)

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
    """Capture a refined eye image, compute indices, and update the CSV dataset."""
    filename, eye_roi = capture_eye_image_refined()
    if filename is None or eye_roi is None:
        return

    birefringence_index = get_birefringence_index(eye_roi)
    ir_temperature = get_ir_temperature(eye_roi)

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
