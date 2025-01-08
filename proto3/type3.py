import cv2
import numpy as np
from datetime import datetime
import os
import sys

class FaceDirectionTracker:
    def __init__(self):
        self.cap = self.initialize_camera()
        if self.cap is None:
            print("Error: Could not initialize any camera")
            sys.exit(1)
            
        try:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            if self.face_cascade.empty():
                raise Exception("Error loading face cascade classifier")
        except Exception as e:
            print(f"Error loading face classifier: {e}")
            self.cap.release()
            sys.exit(1)
        
        self.incoming_dir = 'incoming_people'
        self.outgoing_dir = 'outgoing_people'
        os.makedirs(self.incoming_dir, exist_ok=True)
        os.makedirs(self.outgoing_dir, exist_ok=True)
        
        self.face_tracks = {}
        self.incoming_count = 0
        self.outgoing_count = 0
        self.movement_threshold = 30
        self.safe_zone = 20  # Pixels around initial position where movement is ignored

    def initialize_camera(self):
        camera_indices = [0, 1, 2, -1]
        for idx in camera_indices:
            print(f"Trying to initialize camera {idx}...")
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                ret, test_frame = cap.read()
                if ret:
                    print(f"Successfully initialized camera {idx}")
                    return cap
                else:
                    cap.release()
        return None

    def detect_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        return faces
    
    def track_direction(self, face_x, face_id):
        if face_id in self.face_tracks:
            prev_x = self.face_tracks[face_id]['prev_x']
            stable_x = self.face_tracks[face_id]['initial_x']
            
            # Ignore minor movements within the "safe zone"
            if abs(face_x - stable_x) <= self.safe_zone:
                return False, None
            
            # Check for significant movement over threshold
            if abs(face_x - prev_x) > self.movement_threshold:
                direction = 'incoming' if face_x > prev_x else 'outgoing'
                if direction != self.face_tracks[face_id]['direction']:
                    self.face_tracks[face_id]['direction'] = direction
                    if direction == 'incoming':
                        self.incoming_count += 1
                    else:
                        self.outgoing_count += 1
                    return True, direction
        else:
            self.face_tracks[face_id] = {
                'prev_x': face_x,
                'initial_x': face_x,  # Store the initial x-coordinate for reference
                'direction': None
            }
        
        # Update previous x-coordinate
        self.face_tracks[face_id]['prev_x'] = face_x
        return False, None
    
    def save_face_image(self, frame, face, direction):
        try:
            x, y, w, h = face
            face_img = frame[y:y+h, x:x+w]
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            directory = self.incoming_dir if direction == 'incoming' else self.outgoing_dir
            filename = f'{direction}_{timestamp}.jpg'
            cv2.imwrite(os.path.join(directory, filename), face_img)
        except Exception as e:
            print(f"Error saving face image: {e}")
    
    def run(self):
        print("Starting face tracking...")
        print("Press 'q' to quit")
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Error reading frame from camera")
                    break
                
                faces = self.detect_faces(frame)
                
                for i, (x, y, w, h) in enumerate(faces):
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    face_center_x = x + w//2
                    direction_changed, direction = self.track_direction(face_center_x, i)
                    
                    if direction_changed:
                        self.save_face_image(frame, (x, y, w, h), direction)
                        status = f"Person {direction}!"
                        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                                  1, (0, 255, 0), 2)
                
                # Display incoming and outgoing counts in the top-left corner
                cv2.putText(frame, f"Incoming: {self.incoming_count}", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, f"Outgoing: {self.outgoing_count}", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Display the frame
                cv2.imshow('Face Direction Tracker', frame)
                
                # Break loop on 'q' press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
        except Exception as e:
            print(f"Error during execution: {e}")
        finally:
            print("Cleaning up...")
            self.cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    tracker = FaceDirectionTracker()
    tracker.run()
