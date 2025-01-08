import cv2
import time
import numpy as np

def initialize_detector():
    # Load the pre-trained face detection cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    # Optional: Add eye detection for better validation
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    return face_cascade, eye_cascade

def detect_faces(frame, face_cascade, eye_cascade, scale_factor=1.1, min_neighbors=5):
    # Convert to grayscale for detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    face_data = []
    for (x, y, w, h) in faces:
        face_roi_gray = gray[y:y+h, x:x+w]
        # Detect eyes within the face region
        eyes = eye_cascade.detectMultiScale(face_roi_gray)
        # Only include faces where we detect at least one eye (reduces false positives)
        if len(eyes) > 0:
            face_data.append({
                'coords': (x, y, w, h),
                'eyes': len(eyes)
            })
    
    return face_data

def main():
    # Initialize camera and detectors
    face_cascade, eye_cascade = initialize_detector()
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Initialize FPS calculation
    fps_start_time = time.time()
    fps = 0
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        # Create a copy for drawing
        display_frame = frame.copy()
        
        # Detect faces
        face_data = detect_faces(frame, face_cascade, eye_cascade)
        
        # Draw results
        for face in face_data:
            x, y, w, h = face['coords']
            # Draw face rectangle
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            # Add text showing number of eyes detected
            cv2.putText(display_frame, f"Eyes: {face['eyes']}", 
                       (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (0, 255, 0), 2)
        
        # Calculate and display FPS
        frame_count += 1
        if frame_count >= 30:
            fps = frame_count / (time.time() - fps_start_time)
            fps_start_time = time.time()
            frame_count = 0
        
        # Add FPS and face count to display
        cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(display_frame, f"Faces: {len(face_data)}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Show the result
        cv2.imshow('Face Detection', display_frame)
        
        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()