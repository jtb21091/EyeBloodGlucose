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
WARMUP_FRAMES = 30  # Increased for better auto-adjustment

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh


def adjust_camera_settings(cap):
    """Optimize camera settings for better image quality."""
    try:
        # Try to set auto exposure (may not work on all cameras)
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Manual mode
        cap.set(cv2.CAP_PROP_EXPOSURE, -5)  # Adjust exposure
    except:
        logging.debug("Could not set exposure settings")
    
    try:
        # Increase brightness
        cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.6)
    except:
        logging.debug("Could not set brightness")
    
    try:
        # Adjust contrast
        cap.set(cv2.CAP_PROP_CONTRAST, 0.5)
    except:
        logging.debug("Could not set contrast")
    
    try:
        # Increase saturation (helps with color features)
        cap.set(cv2.CAP_PROP_SATURATION, 0.6)
    except:
        logging.debug("Could not set saturation")
    
    try:
        # Enable autofocus if supported
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
    except:
        logging.debug("Could not set autofocus")
    
    logging.debug("Camera settings optimization attempted")


@contextmanager
def open_camera(width=CAMERA_WIDTH, height=CAMERA_HEIGHT):
    """Context manager for safe camera access with optimized settings."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    # Apply camera optimizations
    adjust_camera_settings(cap)
    
    logging.debug("Camera resolution set to: %.0f x %.0f",
                  cap.get(cv2.CAP_PROP_FRAME_WIDTH), 
                  cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    try:
        yield cap
    finally:
        cap.release()
        logging.debug("Camera released")


def enhance_roi(roi):
    """Apply image enhancements to improve quality."""
    if roi is None or roi.size == 0:
        return roi
    
    try:
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
        l = clahe.apply(l)
        
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        # Sharpen the image slightly
        kernel = np.array([[-0.5,-0.5,-0.5],
                           [-0.5, 5,-0.5],
                           [-0.5,-0.5,-0.5]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        
        # Denoise while preserving details
        denoised = cv2.fastNlMeansDenoisingColored(sharpened, None, 6, 6, 7, 21)
        
        logging.debug("ROI enhancement applied successfully")
        return denoised
    except Exception as e:
        logging.warning(f"Error enhancing ROI: {e}. Using original.")
        return roi


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


def get_lens_clarity_score(image):
    """
    Measure lens opacity/clarity score.
    Higher values indicate more clarity/less opacity.
    Diabetes can cause lens changes over time.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    # Focus on central region (approximate lens area)
    center_y_start, center_y_end = h // 3, 2 * h // 3
    center_x_start, center_x_end = w // 3, 2 * w // 3
    center = gray[center_y_start:center_y_end, center_x_start:center_x_end]
    
    if center.size == 0:
        return 0.0
    
    clarity = np.std(center) / (np.mean(center) + 1e-5)
    return round(clarity, 10)


def get_sclera_yellowness(image):
    """
    Measure yellowish tint in sclera using LAB color space.
    Can indicate metabolic changes or jaundice.
    The b channel in LAB represents yellow-blue axis.
    """
    try:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        b_channel = lab[:, :, 2]  # Yellow-blue axis (higher = more yellow)
        yellowness = np.mean(b_channel)
        return round(yellowness, 10)
    except Exception as e:
        logging.warning("Error calculating sclera yellowness: %s", e)
        return 0.0


def get_vessel_tortuosity(image):
    """
    Estimate blood vessel tortuosity (twistedness) using edge curvature analysis.
    High glucose levels can increase vessel tortuosity.
    Uses edge detection and analyzes local curvature.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply bilateral filter to reduce noise while preserving edges
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Detect edges with stricter thresholds for vessel detection
    edges = cv2.Canny(filtered, 30, 90)
    
    # Find contours representing vessels
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    
    if not contours:
        return 0.0
    
    # Calculate tortuosity for each contour
    tortuosity_scores = []
    for contour in contours:
        if len(contour) > 10:  # Need sufficient points
            # Calculate arc length
            arc_length = cv2.arcLength(contour, False)
            
            # Calculate chord length (straight line distance)
            if len(contour) >= 2:
                start_point = contour[0][0]
                end_point = contour[-1][0]
                chord_length = np.linalg.norm(start_point - end_point)
                
                # Tortuosity = arc_length / chord_length
                # (1.0 = perfectly straight, higher = more tortuous)
                if chord_length > 0:
                    tortuosity = arc_length / (chord_length + 1e-5)
                    tortuosity_scores.append(tortuosity)
    
    if tortuosity_scores:
        # Return mean tortuosity across all detected vessels
        mean_tortuosity = np.mean(tortuosity_scores)
        return round(mean_tortuosity, 10)
    else:
        return 0.0


def get_image_quality_score(image):
    """
    Calculate composite image quality metric.
    Combines blur, brightness, and contrast into a single score.
    Higher scores indicate better quality images.
    """
    _, blur_var = is_blurry(image)
    brightness = np.mean(image)
    contrast = np.std(image)
    
    # Normalize each component (with reasonable expected ranges)
    blur_score = min(blur_var / 100.0, 1.0)  # 100+ is good
    brightness_score = 1.0 - abs(brightness - 128) / 128.0  # 128 is ideal
    contrast_score = min(contrast / 50.0, 1.0)  # 50+ is good
    
    # Composite quality (0-1 scale, then scaled to 0-100)
    quality = (blur_score + brightness_score + contrast_score) / 3.0 * 100
    
    logging.debug("Image quality - Blur: %.2f, Brightness: %.2f, Contrast: %.2f, Overall: %.2f", 
                  blur_score * 100, brightness_score * 100, contrast_score * 100, quality)
    
    return round(quality, 10)


def measure_capture_duration(cap):
    """
    Measure the time duration to capture 10 frames.
    Renamed from pupil_response_time for accuracy.
    """
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
    
    duration = round(time.time() - start_time, 10)
    return duration


def extract_eye_roi(frame):
    """
    Extract the eye region (ROI) from a frame using Mediapipe Face Mesh.
    Returns the ROI and logs any quality warnings.
    IMPROVED: Larger padding for better context.
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
            # IMPROVED: Adaptive padding based on eye size
            # Calculate eye dimensions
            eye_width = max(x_coords) - min(x_coords)
            eye_height = max(y_coords) - min(y_coords)
            
            # Use 40% of eye size as padding (balances context vs purity)
            # Minimum 15 pixels, maximum 25 pixels
            pad_x = int(np.clip(eye_width * 0.4, 15, 25))
            pad_y = int(np.clip(eye_height * 0.4, 15, 25))
            
            x_min = max(min(x_coords) - pad_x, 0)
            x_max = min(max(x_coords) + pad_x, iw)
            y_min = max(min(y_coords) - pad_y, 0)
            y_max = min(max(y_coords) + pad_y, ih)
            
            logging.debug(f"Eye ROI padding: X={pad_x}px, Y={pad_y}px (adaptive)")
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


def show_preview_and_capture(cap, duration=5):
    """Show preview window so user can position themselves before capture."""
    print(f"\n📸 POSITION YOURSELF - Capturing in {duration} seconds...")
    print("   ✓ Remove glasses")
    print("   ✓ Look directly at camera")
    print("   ✓ Ensure good lighting (face a window)")
    print("   ✓ Distance: 12-18 inches from camera")
    print("   ✓ Press 'q' to cancel\n")
    
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Calculate countdown
        elapsed = time.time() - start_time
        remaining = max(0, duration - elapsed)
        
        # Draw countdown and instructions on frame
        display = frame.copy()
        h, w = display.shape[:2]
        
        # Large countdown
        countdown_text = f"{remaining:.1f}s"
        cv2.putText(display, countdown_text, 
                   (w//2 - 80, h//2), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 255, 0), 5)
        
        # Instructions
        cv2.putText(display, "REMOVE GLASSES!", 
                   (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        cv2.putText(display, "Look at camera", 
                   (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        cv2.putText(display, "Press 'q' to cancel", 
                   (50, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        cv2.imshow("Eye Capture Preview", display)
        
        # Check for quit or timeout
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            cv2.destroyAllWindows()
            return None
        
        if elapsed >= duration:
            cv2.destroyAllWindows()
            return frame
    
    cv2.destroyAllWindows()
    return None


def capture_and_measure():
    """
    Capture eye images and perform all measurements in a single camera session.
    Returns filename, roi, and all measurements.
    IMPROVED: Enhanced image quality processing with preview window.
    """
    try:
        with open_camera() as cap:
            # Quick warmup
            logging.info("Warming up camera...")
            for i in range(10):
                ret, _ = cap.read()
                if not ret:
                    continue
                time.sleep(0.1)
            
            # Show preview and capture
            frame = show_preview_and_capture(cap, duration=5)
            
            if frame is None:
                logging.error("Capture cancelled or failed.")
                return None
            
            # Check frame quality
            blurry, variance = is_blurry(frame)
            if blurry:
                logging.warning("Captured image appears blurry (Laplacian variance: %.10f).", variance)
            
            # Extract eye ROI
            roi = extract_eye_roi(frame)
            
            # IMPROVED: Enhance the ROI before processing
            roi_enhanced = enhance_roi(roi)
            
            # Measure capture duration (needs camera still open)
            logging.info("Measuring capture duration...")
            capture_duration = measure_capture_duration(cap)
            
            # Save the enhanced ROI with maximum quality
            file_name = f"eye_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            file_path = os.path.join(IMAGE_DIR, file_name)
            # IMPROVED: Save with no compression for maximum quality
            cv2.imwrite(file_path, roi_enhanced, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            logging.info("Captured enhanced eye ROI image: %s", file_name)
            
            # Compute all measurements from the enhanced ROI
            measurements = {
                'pupil_size': get_pupil_size(roi_enhanced),
                'sclera_redness': get_sclera_redness(roi_enhanced),
                'vein_prominence': get_vein_prominence(roi_enhanced),
                'capture_duration': capture_duration,
                'ir_intensity': get_ir_intensity(roi_enhanced),
                'scleral_vein_density': get_scleral_vein_density(roi_enhanced),
                'ir_temperature': get_ir_temperature(roi_enhanced),
                'tear_film_reflectivity': get_tear_film_reflectivity(roi_enhanced),
                'sclera_color_balance': get_sclera_color_balance(roi_enhanced),
                'vein_pulsation_intensity': get_vein_pulsation_intensity(roi_enhanced),
                'birefringence_index': get_birefringence_index(roi_enhanced),
                'lens_clarity_score': get_lens_clarity_score(roi_enhanced),
                'sclera_yellowness': get_sclera_yellowness(roi_enhanced),
                'vessel_tortuosity': get_vessel_tortuosity(roi_enhanced),
                'image_quality_score': get_image_quality_score(roi_enhanced)
            }
            
            return file_name, measurements
    
    except RuntimeError as e:
        logging.error("Camera error: %s", e)
        return None
    except Exception as e:
        logging.error("Unexpected error during capture: %s", e)
        return None


def test_focus():
    """Test different focus positions to find optimal setting."""
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    
    print("\n🎯 FOCUS TEST MODE")
    print("=" * 60)
    print("Instructions:")
    print("  • Adjust your distance from camera")
    print("  • Try different lighting")
    print("  • Wait for GREEN (blur score >100)")
    print("  • Press 's' to save a good frame")
    print("  • Press 'q' to quit")
    print("=" * 60 + "\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Calculate blur score
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Show frame with blur score
        display = frame.copy()
        color = (0, 255, 0) if blur_score > 100 else (0, 0, 255)
        status = "GOOD - Press 's' to save" if blur_score > 100 else "TOO BLURRY - Adjust position/lighting"
        
        cv2.putText(display, f"Blur Score: {blur_score:.1f}", 
                   (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
        cv2.putText(display, status, 
                   (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(display, "Target: >100 for good quality", 
                   (50, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display, "Press 'q' to quit, 's' to save", 
                   (50, display.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow("Focus Test", display)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite("focus_test.png", frame)
            print(f"\n✓ Saved focus_test.png! Blur score: {blur_score:.1f}")
            if blur_score > 100:
                print("✓ Good quality! This distance/lighting should work well.")
            else:
                print("⚠ Still blurry. Try adjusting position or lighting.")
    
    cap.release()
    cv2.destroyAllWindows()
    print("\nFocus test complete!")


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
        measurements['capture_duration'],
        measurements['ir_intensity'],
        measurements['scleral_vein_density'],
        measurements['ir_temperature'],
        measurements['tear_film_reflectivity'],
        measurements['sclera_color_balance'],
        measurements['vein_pulsation_intensity'],
        measurements['birefringence_index'],
        measurements['lens_clarity_score'],
        measurements['sclera_yellowness'],
        measurements['vessel_tortuosity'],
        measurements['image_quality_score']
    ]], columns=[
        'filename', 'blood_glucose', 'pupil_size', 'sclera_redness',
        'vein_prominence', 'capture_duration', 'ir_intensity',
        'scleral_vein_density', 'ir_temperature', 'tear_film_reflectivity',
        'sclera_color_balance', 'vein_pulsation_intensity', 'birefringence_index',
        'lens_clarity_score', 'sclera_yellowness', 'vessel_tortuosity',
        'image_quality_score'
    ])

    # Use append mode so that each run adds a new row
    if not os.path.exists(LABELS_FILE):
        new_entry.to_csv(LABELS_FILE, index=False, float_format='%.10f')
        logging.info("CSV created and new data added: %s", filename)
    else:
        new_entry.to_csv(LABELS_FILE, mode='a', header=False, index=False, float_format='%.10f')
        logging.info("Appended new data to CSV: %s", filename)
    
    print(f"\n✓ Capture complete! Image saved as: {filename}")
    print(f"✓ Image quality score: {measurements['image_quality_score']:.1f}/100")
    print("\n💡 Tip: For best results, ensure good lighting and look directly at the camera")
    print("\nRemember to add your blood glucose reading to the CSV file.")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--test-focus":
        test_focus()
    else:
        update_data()