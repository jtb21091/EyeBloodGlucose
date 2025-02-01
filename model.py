import os
import cv2
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import Optional, Tuple

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# Constants for file paths
LABELS_FILE = "eye_glucose_data/labels.csv"
IMAGE_DIR = "eye_glucose_data/images"
os.makedirs(IMAGE_DIR, exist_ok=True)

def capture_eye_image(camera_index: int = 0, delay: int = 500) -> Tuple[Optional[str], Optional[np.ndarray]]:
    """
    Capture an image from the webcam and save it to the image directory.

    Parameters:
        camera_index (int): The index of the webcam to use.
        delay (int): Delay in milliseconds to allow the camera to stabilize.

    Returns:
        Tuple[Optional[str], Optional[np.ndarray]]:
            - filename: The filename of the saved image if capture is successful, else None.
            - frame: The captured image as a NumPy array if successful, else None.
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        logging.error("Error: Could not open webcam.")
        return None, None

    # Wait for camera stabilization
    cv2.waitKey(delay)
    
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        logging.error("Error: Could not capture image.")
        return None, None
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"eye_{timestamp}.jpg"
    filepath = os.path.join(IMAGE_DIR, filename)
    
    if cv2.imwrite(filepath, frame):
        logging.info(f"Image saved: {filepath}")
    else:
        logging.error(f"Failed to save image: {filepath}")
        return None, None

    return filename, frame

def get_birefringence_index(image: np.ndarray) -> float:
    """
    Estimate the birefringence index of the given image using edge density and gradient contrast.

    Parameters:
        image (np.ndarray): The input image in BGR format.

    Returns:
        float: The calculated birefringence index rounded to 5 decimal places.
    """
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Detect edges using the Canny algorithm
    edges = cv2.Canny(gray, 50, 150)
    # Calculate the gradient using the Sobel operator
    gradient = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=5)
    # Calculate the birefringence index as the average of mean edge intensity and the standard deviation of the gradient
    index_value = (np.mean(edges) + np.std(gradient)) / 2
    return round(index_value, 5)

def update_data() -> None:
    """
    Capture an eye image, compute the birefringence index, and update the CSV file with new measurements.
    """
    filename, frame = capture_eye_image()
    if filename is None or frame is None:
        logging.error("Image capture failed. Data not updated.")
        return
    
    birefringence_index = get_birefringence_index(frame)

    # Define the CSV columns
    columns = [
        "filename", "blood_glucose", "pupil_size", "sclera_redness", "vein_prominence",
        "pupil_response_time", "ir_intensity", "scleral_vein_density", "pupil_circularity",
        "ir_temperature", "tear_film_reflectivity", "pupil_dilation_rate",
        "sclera_color_balance", "vein_pulsation_intensity", "birefringence_index"
    ]

    # Create a DataFrame: if the CSV doesn't exist or is empty, create a new one
    if not os.path.exists(LABELS_FILE) or os.stat(LABELS_FILE).st_size == 0:
        df = pd.DataFrame(columns=columns)
    else:
        try:
            df = pd.read_csv(LABELS_FILE)
        except Exception as e:
            logging.error(f"Error reading CSV file: {e}")
            df = pd.DataFrame(columns=columns)

    # Build a new data entry (adjust the measurement values as needed)
    new_data = {
        "filename": filename,
        "blood_glucose": np.nan,  # Using NaN for missing measurement
        "pupil_size": np.random.uniform(20, 100),
        "sclera_redness": np.random.uniform(0, 100),
        "vein_prominence": np.random.uniform(0, 10),
        "pupil_response_time": np.random.uniform(0.1, 0.5),
        "ir_intensity": np.random.uniform(10, 255),
        "scleral_vein_density": np.random.uniform(0, 1),
        "pupil_circularity": np.nan,  # Measurement not provided
        "ir_temperature": np.random.uniform(10, 40),
        "tear_film_reflectivity": np.random.uniform(0, 255),
        "pupil_dilation_rate": np.random.uniform(0.1, 1.0),
        "sclera_color_balance": np.random.uniform(0.8, 1.2),
        "vein_pulsation_intensity": np.random.uniform(0, 10),
        "birefringence_index": birefringence_index
    }

    new_entry = pd.DataFrame([new_data], columns=columns)
    df = pd.concat([df, new_entry], ignore_index=True)

    try:
        df.to_csv(LABELS_FILE, index=False)
        logging.info(f"New data added to CSV: {filename}")
    except Exception as e:
        logging.error(f"Failed to update CSV file: {e}")

if __name__ == "__main__":
    update_data()
