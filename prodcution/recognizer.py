import cv2
import mediapipe as mp
import face_recognition
import os
import numpy as np
import re
import platform
import time

mod=""

if "rpi" in platform.release():
    mod="rpi"
    from picamera2 import Picamera2
    camera=Picamera2()
    camera_config=camera.create_still_configuration(
        main={"size":(640,480),"format":"RGB888"},
        raw={"size":camera.sensor_resolution},
    )
    camera.configure(camera_config)
    camera.start()
    time.sleep(1)
else:
    mod="linux"
    cap = cv2.VideoCapture(0)


# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# Folder to store known faces
imgs_folder = "imgs"
os.makedirs(imgs_folder, exist_ok=True)

# Global known face lists
known_face_encodings = []
known_face_names = []

def load_known_faces():
    """Loads existing faces from the imgs/ folder."""
    global known_face_encodings, known_face_names
    known_face_encodings.clear()
    known_face_names.clear()

    for file_name in os.listdir(imgs_folder):
        img_path = os.path.join(imgs_folder, file_name)
        known_image = face_recognition.load_image_file(img_path)
        encodings = face_recognition.face_encodings(known_image)

        if encodings:
            known_face_encodings.append(encodings[0])
            known_face_names.append(file_name.split('.')[0])  # Use filename as label

def get_next_face_number():
    """Finds the next available new_face_X number."""
    existing_numbers = []
    pattern = re.compile(r"new_face_(\d+)\.jpg")

    for filename in os.listdir(imgs_folder):
        match = pattern.match(filename)
        if match:
            existing_numbers.append(int(match.group(1)))

    return max(existing_numbers, default=0) + 1  # Next number

def detect_faces(show_windows=True):
    """Detects faces, recognizes known ones, and saves new faces if unknown.
    
    Args:
        show_windows (bool): If True, displays OpenCV windows; otherwise, runs silently.
    """
    load_known_faces()  # Load known faces before starting




    while True:

        if mod == "linux":    
            ret, frame = cap.read()
        else:
            frame=camera.capture_array()
            frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        if not ret:
            print("Failed to grab frame")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb_frame)

        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape

                x1 = int(bboxC.xmin * iw)
                y1 = int(bboxC.ymin * ih)
                w = int(bboxC.width * iw)
                h = int(bboxC.height * ih)

                # Adjust bounding box
                center_x = x1 + w // 2
                center_y = y1 + h // 2
                new_w = int(1.4 * w)
                new_h = int(1.6 * h)
                new_x1 = max(0, center_x - new_w // 2)
                new_y1 = max(0, center_y - new_h // 2)
                new_x2 = min(iw, new_x1 + new_w)
                new_y2 = min(ih, new_y1 + new_h)

                # Crop and check validity
                face_crop = frame[new_y1:new_y2, new_x1:new_x2]
                if face_crop.size == 0:
                    continue  # Skip invalid crops

                # Convert to RGB before encoding
                face_crop_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)

                face_encodings = face_recognition.face_encodings(face_crop_rgb)
                if not face_encodings:  
                    continue  # Skip if no face encoding is found

                face_encoding = face_encodings[0]

                # Compare with known faces
                name = "Unknown"
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

                best_match_index = np.argmin(face_distances) if matches else None

                if best_match_index is not None and matches[best_match_index]:
                    name = known_face_names[best_match_index]
                else:
                    # Get the next available new_face_X number
                    new_face_id = get_next_face_number()
                    new_face_path = os.path.join(imgs_folder, f"new_face_{new_face_id}.jpg")
                    cv2.imwrite(new_face_path, face_crop)
                    print(f"New face saved as {new_face_path}")

                    # Add to known faces
                    known_face_encodings.append(face_encoding)
                    known_face_names.append(f"new_face_{new_face_id}")

                    name = f"new_face_{new_face_id}"

                # Draw bounding box and name if windows are enabled
                if show_windows:
                    cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)
                    cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.imshow('Cropped Face', face_crop)

        # Show the frame if windows are enabled
        if show_windows:
            cv2.imshow('Face Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Example usage
detect_faces(show_windows=True)  # Run with OpenCV windows
# detect_faces(show_windows=False)  # Run in background
