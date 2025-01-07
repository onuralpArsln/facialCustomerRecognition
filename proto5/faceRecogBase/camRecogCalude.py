import face_recognition
import cv2
import numpy as np
import sys
import os

# Force OpenCV to use X11 instead of Wayland
os.environ["QT_QPA_PLATFORM"] = "xcb"
# Fallback if xcb doesn't work
if not os.environ.get("DISPLAY"):
    os.environ["QT_QPA_PLATFORM"] = "offscreen"

def load_known_face(image_path):
    """Load and encode a known face with error handling"""
    try:
        image = face_recognition.load_image_file(image_path)
        # Get face landmarks first
        face_landmarks = face_recognition.face_landmarks(image)
        if not face_landmarks:
            print(f"No face landmarks found in {image_path}")
            return None
        
        # Use the landmarks to get face locations
        height, width = image.shape[:2]
        face_location = (0, width, height, 0)  # Format: top, right, bottom, left
        
        # Get face encoding using landmarks
        face_encoding = face_recognition.face_encodings(image, [face_location])
        if len(face_encoding) > 0:
            return face_encoding[0]
        print(f"Could not encode face in {image_path}")
        return None
    except Exception as e:
        print(f"Error loading {image_path}: {e}")
        return None

def process_frame(frame, known_face_encodings, known_face_names):
    """Process a single frame with error handling"""
    try:
        # Convert the frame from BGR (OpenCV) to RGB (face_recognition)
        rgb_frame = frame[:, :, ::-1]
        
        # First get face landmarks
        face_landmarks_list = face_recognition.face_landmarks(rgb_frame)
        
        if not face_landmarks_list:
            return [], []
        
        # Convert landmarks to locations
        face_locations = []
        for landmarks in face_landmarks_list:
            # Get bounding box from landmarks
            x_coords = [point[0] for feature in landmarks.values() for point in feature]
            y_coords = [point[1] for feature in landmarks.values() for point in feature]
            top = min(y_coords)
            right = max(x_coords)
            bottom = max(y_coords)
            left = min(x_coords)
            face_locations.append((top, right, bottom, left))
        
        # Get face encodings
        face_encodings = []
        face_names = []
        
        for face_location in face_locations:
            try:
                top, right, bottom, left = face_location
                face_image = rgb_frame[top:bottom, left:right]
                face_encoding = face_recognition.face_encodings(face_image)
                
                if face_encoding:
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding[0], tolerance=0.6)
                    name = "Unknown"
                    
                    if True in matches:
                        first_match_index = matches.index(True)
                        name = known_face_names[first_match_index]
                    
                    face_encodings.append(face_encoding[0])
                    face_names.append(name)
            except Exception as e:
                print(f"Error processing face: {e}")
                continue
        
        return face_locations, face_names
    except Exception as e:
        print(f"Error in process_frame: {e}")
        return [], []

def main():
    # Initialize video capture
    try:
        video_capture = cv2.VideoCapture(0)
        if not video_capture.isOpened():
            raise Exception("Could not open video capture device")
    except Exception as e:
        print(f"Error initializing video: {e}")
        return

    # Load known faces
    known_face_encodings = []
    known_face_names = []
    
    # Load sample faces
    for name, image_path in [("Barack Obama", "obama.jpg"), ("Joe Biden", "biden.jpg")]:
        encoding = load_known_face(image_path)
        if encoding is not None:
            known_face_encodings.append(encoding)
            known_face_names.append(name)
    
    if not known_face_encodings:
        print("No known faces could be loaded. Exiting.")
        video_capture.release()
        return

    frame_count = 0
    
    try:
        while True:
            ret, frame = video_capture.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            # Process every third frame to reduce CPU usage
            frame_count += 1
            if frame_count % 3 != 0:
                continue
            
            # Process frame
            face_locations, face_names = process_frame(frame, known_face_encodings, known_face_names)
            
            # Display results
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                # Draw box and name
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.6, (255, 255, 255), 1)
            
            # Display the frame
            cv2.imshow('Video', frame)
            
            # Check for 'q' key to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        video_capture.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()