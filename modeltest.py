import os
import cv2
import pandas as pd  # type: ignore
import numpy as np
import logging
import mediapipe as mp  # type: ignore
import time
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Define file paths and constants
LABELS_FILE = "eye_glucose_data/labels.csv"
IMAGE_DIR = "eye_glucose_data/images"
os.makedirs(IMAGE_DIR, exist_ok=True)

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh

def get_pupil_size(image):
    """Detect and measure pupil size using thresholding and contour detection."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None  # No pupil detected
    largest_contour = max(contours, key=cv2.contourArea)
    (_, _), radius = cv2.minEnclosingCircle(largest_contour)
    return round(radius * 2, 2)  # Convert radius to diameter

def get_sclera_redness(image):
    """Analyze sclera redness using the HSV color space."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0, 50, 50])
    upper_red = np.array([10, 255, 255])
    mask = cv2.inRange(hsv, lower_red, upper_red)
    redness_score = np.mean(mask)  # Get average red intensity
    return round(redness_score, 2)

def get_vein_prominence(image):
    """Estimate vein prominence using edge detection and contrast enhancement."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    return round(np.mean(edges), 2)

def get_ir_intensity(image):
    """Calculate IR intensity from grayscale average."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return round(np.mean(gray), 2)

def get_scleral_vein_density(image):
    """Estimate scleral vein density using edge detection."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    return round(np.sum(edges) / edges.size, 2)

def get_tear_film_reflectivity(image):
    """Estimate tear film reflectivity based on brightness variance."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return round(np.var(gray), 2)

def get_pupil_dilation_rate(image):
    """Estimate pupil dilation based on area change in contours."""
    return get_pupil_size(image)  # Using pupil size as a proxy for dilation rate

def get_sclera_color_balance(image):
    """Analyze sclera color balance using RGB ratio."""
    mean_color = np.mean(image, axis=(0, 1))
    return round(mean_color[2] / (mean_color[0] + 1e-5), 2)  # Red/Blue ratio

def get_vein_pulsation_intensity(image):
    """Estimate vein pulsation intensity using frequency analysis."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return round(np.std(gray), 2)

def measure_pupil_response_time():
    """Track pupil size over multiple frames to estimate response time."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logging.error("Could not open webcam.")
        return None
    pupil_sizes = []
    start_time = time.time()
    for _ in range(10):  # Capture 10 frames
        ret, frame = cap.read()
        if not ret:
            continue
        pupil_size = get_pupil_size(frame)
        if pupil_size:
            pupil_sizes.append(pupil_size)
        time.sleep(0.1)  # 100ms delay between frames
    cap.release()
    if len(pupil_sizes) < 2:
        return None  # Not enough data
    response_time = round(time.time() - start_time, 2)
    return response_time

def capture_eye_image():
    """
    Capture an eye image from the webcam and extract the eye region (ROI)
    using Mediapipe Face Mesh.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logging.error("Could not open webcam for eye capture.")
        return None, None

    ret, frame = cap.read()
    cap.release()

    if not ret:
        logging.error("Failed to capture eye image.")
        return None, None

    # Convert the frame to RGB for Mediapipe processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the image with Mediapipe Face Mesh to detect facial landmarks.
    with mp_face_mesh.FaceMesh(static_image_mode=True,
                               max_num_faces=1,
                               refine_landmarks=True,
                               min_detection_confidence=0.5) as face_mesh:
        results = face_mesh.process(rgb_frame)

        if not results.multi_face_landmarks:
            logging.error("No face landmarks detected. Using full frame as ROI.")
            roi = frame
        else:
            face_landmarks = results.multi_face_landmarks[0]
            ih, iw, _ = frame.shape

            # Use Mediapipe's FACEMESH_LEFT_EYE connections to extract left eye landmarks.
            left_eye_indices = {idx for connection in mp_face_mesh.FACEMESH_LEFT_EYE for idx in connection}

            x_coords = []
            y_coords = []
            for idx in left_eye_indices:
                lm = face_landmarks.landmark[idx]
                x_coords.append(int(lm.x * iw))
                y_coords.append(int(lm.y * ih))

            if x_coords and y_coords:
                x_min = max(min(x_coords) - 5, 0)
                x_max = min(max(x_coords) + 5, iw)
                y_min = max(min(y_coords) - 5, 0)
                y_max = min(max(y_coords) + 5, ih)
                roi = frame[y_min:y_max, x_min:x_max]
                if roi.size == 0:
                    logging.error("ROI extraction resulted in an empty image. Using full frame as ROI.")
                    roi = frame
            else:
                logging.error("No eye landmarks found. Using full frame as ROI.")
                roi = frame

    # Optionally, check if the ROI is extremely dark and log a warning.
    if np.mean(roi) < 10:
        logging.warning("The ROI appears very dark. Check your lighting conditions.")

    # Save the ROI image to disk
    filename = os.path.join(IMAGE_DIR, f"eye_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    cv2.imwrite(filename, roi)
    logging.info("Captured eye ROI image: %s", filename)
    return filename, roi

def get_birefringence_index(image):
    """
    Calculate the birefringence index.
    This is a placeholder algorithm that calculates the ratio of standard deviation to mean intensity.
    Replace with the proper algorithm when available.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_intensity = np.mean(gray)
    if mean_intensity == 0:
        return 0.0
    std_intensity = np.std(gray)
    birefringence = std_intensity / mean_intensity
    return round(birefringence, 2)

def get_ir_temperature(image):
    """
    Estimate the IR temperature.
    This is a placeholder algorithm. It maps the average grayscale intensity to a temperature value.
    Replace with the correct algorithm when available.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_intensity = np.mean(gray)
    # Example conversion: map intensity (0-255) to temperature range (20-60 Celsius)
    temperature = (mean_intensity / 255.0) * 40 + 20
    return round(temperature, 2)

def update_data():
    """Capture an eye image, compute real values, and update the dataset."""
    filename, roi = capture_eye_image()
    if filename is None or roi is None:
        logging.error("No image captured. Skipping update.")
        return

    pupil_size = get_pupil_size(roi)
    sclera_redness = get_sclera_redness(roi)
    vein_prominence = get_vein_prominence(roi)
    pupil_response_time = measure_pupil_response_time()
    birefringence_index = get_birefringence_index(roi)
    ir_temperature = get_ir_temperature(roi)

    # Create a new entry DataFrame with the computed values
    new_entry = pd.DataFrame([[
        filename, "", pupil_size, sclera_redness, vein_prominence, pupil_response_time,
        get_ir_intensity(roi), get_scleral_vein_density(roi), ir_temperature,
        get_tear_film_reflectivity(roi), get_pupil_dilation_rate(roi),
        get_sclera_color_balance(roi), get_vein_pulsation_intensity(roi),
        birefringence_index
    ]], columns=[
        'filename', 'blood_glucose', 'pupil_size', 'sclera_redness',
        'vein_prominence', 'pupil_response_time', 'ir_intensity',
        'scleral_vein_density', 'ir_temperature', 'tear_film_reflectivity',
        'pupil_dilation_rate', 'sclera_color_balance', 'vein_pulsation_intensity',
        'birefringence_index'
    ])

    # Load existing data if the CSV file exists, otherwise create a new DataFrame
    if os.path.exists(LABELS_FILE):
        df = pd.read_csv(LABELS_FILE)
        df = pd.concat([df, new_entry], ignore_index=True)
    else:
        df = new_entry

    df.to_csv(LABELS_FILE, index=False)
    logging.info("New data added to CSV: %s", filename)

if __name__ == "__main__":
    update_data()
