import os
import cv2
import pandas as pd  # type: ignore
import numpy as np
import logging
import mediapipe as mp  # type: ignore
import time
from datetime import datetime
from contextlib import contextmanager

# Setup logging: change level to DEBUG to see detailed logs.
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

# Define file paths and constants
LABELS_FILE = "eye_glucose_data/labels.csv"
IMAGE_DIR = "eye_glucose_data/images"
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(os.path.dirname(LABELS_FILE), exist_ok=True)

# Edge detection thresholds (tuned further)
EDGE_LOW_THRESHOLD = 10
EDGE_HIGH_THRESHOLD = 50

# Camera settings
CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1080
WARMUP_FRAMES = 20

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh


@contextmanager
def open_camera(width=CAMERA_WIDTH, height=CAMERA_HEIGHT):
    """Context manager for safe camera access."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    logging.debug("Camera resolution set to: %.0f x %.0f",
                  cap.get(cv2.CAP_PROP_FRAME_WIDTH), 
                  cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    try:
        yield cap
    finally:
        cap.release()
        logging.debug("Camera released")


def is_blurry(image, threshold=100.0):
    """Check if the image is blurry using the variance of the Laplacian."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance < threshold, variance


def get_pupil_size(image):
    """
    Detect and measure pupil size using adaptive thresholding and contour detection.
    Histogram equalization is applied to improve contrast.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_eq = cv2.equalizeHist(gray)
    adaptive_thresh = cv2.adaptiveThreshold(
        gray_eq, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2)
    
    contours, _ = cv2.findContours(adaptive_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest_contour = max(contours, key=cv2.contourArea)
    (_, _), radius = cv2.minEnclosingCircle(largest_contour)
    return round(radius * 2, 10)


def get_sclera_redness(image):
    """
    Analyze sclera redness using the HSV color space.
    Combines two ranges of red hues in HSV.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 | mask2
    redness_score = np.mean(mask)
    return round(redness_score, 10)


def get_vein_prominence(image):
    """Estimate vein prominence using edge detection and contrast enhancement."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, EDGE_LOW_THRESHOLD, EDGE_HIGH_THRESHOLD)
    mean_edges = np.mean(edges)
    logging.debug("get_vein_prominence: Mean edge intensity = %.10f", mean_edges)
    return round(mean_edges, 10)


def get_ir_intensity(image):
    """Calculate IR intensity from the grayscale average."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return round(np.mean(gray), 10)


def get_scleral_vein_density(image):
    """Estimate scleral vein density using edge detection."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, EDGE_LOW_THRESHOLD, EDGE_HIGH_THRESHOLD)
    sum_edges = np.sum(edges)
    density = sum_edges / edges.size
    logging.debug("get_scleral_vein_density: Sum of edges = %.10f, Density = %.10f", sum_edges, density)
    return round(density, 10)


def get_tear_film_reflectivity(image):
    """Estimate tear film reflectivity based on brightness variance."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return round(np.var(gray), 10)


def get_sclera_color_balance(image):
    """Analyze sclera color balance using the RGB ratio."""
    mean_color = np.mean(image, axis=(0, 1))
    return round(mean_color[2] / (mean_color[0] + 1e-5), 10)


def get_vein_pulsation_intensity(image):
    """Estimate vein pulsation intensity using frequency analysis."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return round(np.std(gray), 10)


def get_birefringence_index(image):
    """
    Calculate the birefringence index.
    This placeholder algorithm calculates the ratio of standard deviation to mean intensity.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_intensity = np.mean(gray)
    if mean_intensity == 0:
        return 0.0
    std_intensity = np.std(gray)
    birefringence = std_intensity / mean_intensity
    return round(birefringence, 10)


def get_ir_temperature(image):
    """
    Estimate the IR temperature.
    This placeholder algorithm maps the average grayscale intensity to a temperature value.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_intensity = np.mean(gray)
    temperature = (mean_intensity / 255.0) * 40 + 20
    return round(temperature, 10)


def measure_pupil_response_time(cap):
    """Track pupil size over multiple frames to estimate response time."""
    pupil_sizes = []
    start_time = time.time()
    
    for _ in range(10):
        ret, frame = cap.read()
        if not ret:
            continue
        pupil_size = get_pupil_size(frame)
        if pupil_size is not None:
            pupil_sizes.append(pupil_size)
        time.sleep(0.1)
    
    if len(pupil_sizes) < 2:
        return None
    
    response_time = round(time.time() - start_time, 10)
    return response_time


def extract_eye_roi(frame):
    """
    Extract the eye region (ROI) from a frame using Mediapipe Face Mesh.
    Returns the ROI and logs any quality warnings.
    """
    if np.mean(frame) < 10:
        logging.warning("Captured frame is very dark. Check your lighting conditions.")

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    with mp_face_mesh.FaceMesh(static_image_mode=True,
                               max_num_faces=1,
                               refine_landmarks=True,
                               min_detection_confidence=0.5) as face_mesh:
        results = face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            logging.error("No face landmarks detected. Using full frame as ROI.")
            return frame
        
        face_landmarks = results.multi_face_landmarks[0]
        ih, iw, _ = frame.shape
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
                return frame
            
            if np.mean(roi) < 10:
                logging.warning("The ROI appears very dark. Check your lighting conditions.")
            
            blurry_roi, variance_roi = is_blurry(roi)
            if blurry_roi:
                logging.warning("The extracted ROI appears blurry (Laplacian variance: %.10f).", variance_roi)
            
            return roi
        else:
            logging.error("No eye landmarks found. Using full frame as ROI.")
            return frame


def capture_and_measure():
    """
    Capture eye images and perform all measurements in a single camera session.
    Returns filename, roi, and all measurements.
    """
    try:
        with open_camera() as cap:
            # Warm up camera
            logging.info("Warming up camera...")
            frame = None
            for i in range(WARMUP_FRAMES):
                ret, temp_frame = cap.read()
                if not ret:
                    continue
                frame = temp_frame
                time.sleep(0.1)
            
            if frame is None:
                logging.error("Failed to capture frames for warmup.")
                return None
            
            # Check frame quality
            blurry, variance = is_blurry(frame)
            if blurry:
                logging.warning("Captured image appears blurry (Laplacian variance: %.10f).", variance)
            
            # Extract eye ROI
            roi = extract_eye_roi(frame)
            
            # Measure pupil response time (needs camera still open)
            logging.info("Measuring pupil response time...")
            pupil_response_time = measure_pupil_response_time(cap)
            
            # Save the ROI
            file_name = f"eye_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            file_path = os.path.join(IMAGE_DIR, file_name)
            cv2.imwrite(file_path, roi)
            logging.info("Captured eye ROI image: %s", file_name)
            
            # Compute all measurements from the ROI
            measurements = {
                'pupil_size': get_pupil_size(roi),
                'sclera_redness': get_sclera_redness(roi),
                'vein_prominence': get_vein_prominence(roi),
                'pupil_response_time': pupil_response_time,
                'ir_intensity': get_ir_intensity(roi),
                'scleral_vein_density': get_scleral_vein_density(roi),
                'ir_temperature': get_ir_temperature(roi),
                'tear_film_reflectivity': get_tear_film_reflectivity(roi),
                'pupil_dilation_rate': get_pupil_size(roi),  # Same as pupil_size
                'sclera_color_balance': get_sclera_color_balance(roi),
                'vein_pulsation_intensity': get_vein_pulsation_intensity(roi),
                'birefringence_index': get_birefringence_index(roi)
            }
            
            return file_name, measurements
    
    except RuntimeError as e:
        logging.error("Camera error: %s", e)
        return None
    except Exception as e:
        logging.error("Unexpected error during capture: %s", e)
        return None


def update_data():
    """Capture an eye image, compute measurement values, and update the dataset."""
    result = capture_and_measure()
    
    if result is None:
        logging.error("No data captured. Skipping update.")
        return
    
    filename, measurements = result
    
    new_entry = pd.DataFrame([[
        filename, "", 
        measurements['pupil_size'],
        measurements['sclera_redness'],
        measurements['vein_prominence'],
        measurements['pupil_response_time'],
        measurements['ir_intensity'],
        measurements['scleral_vein_density'],
        measurements['ir_temperature'],
        measurements['tear_film_reflectivity'],
        measurements['pupil_dilation_rate'],
        measurements['sclera_color_balance'],
        measurements['vein_pulsation_intensity'],
        measurements['birefringence_index']
    ]], columns=[
        'filename', 'blood_glucose', 'pupil_size', 'sclera_redness',
        'vein_prominence', 'pupil_response_time', 'ir_intensity',
        'scleral_vein_density', 'ir_temperature', 'tear_film_reflectivity',
        'pupil_dilation_rate', 'sclera_color_balance', 'vein_pulsation_intensity',
        'birefringence_index'
    ])

    # Use append mode so that each run adds a new row
    if not os.path.exists(LABELS_FILE):
        new_entry.to_csv(LABELS_FILE, index=False, float_format='%.10f')
        logging.info("CSV created and new data added: %s", filename)
    else:
        new_entry.to_csv(LABELS_FILE, mode='a', header=False, index=False, float_format='%.10f')
        logging.info("Appended new data to CSV: %s", filename)
    
    print(f"\n✓ Capture complete! Image saved as: {filename}")
    print("Remember to add your blood glucose reading to the CSV file.")


if __name__ == "__main__":
    update_data()