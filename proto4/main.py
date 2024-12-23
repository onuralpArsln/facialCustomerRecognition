import cv2
import numpy as np
from detectionEngine import DetectionEngine
from camStarter import MainCam
from collections import defaultdict

class DetectionTracker:
    def __init__(self, history_frames=3, iou_threshold=0.5):
        self.history_frames = history_frames
        self.iou_threshold = iou_threshold  # Intersection over Union threshold
        self.face_history = defaultdict(int)
        self.body_history = defaultdict(int)
        self.confirmed_faces = set()
        self.confirmed_bodies = set()
        self.last_face_positions = {}  # Track last known positions
        
    def _calculate_iou(self, box1, box2):
        """Calculate Intersection over Union between two boxes"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # Calculate coordinates of intersection rectangle
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        
        if xi2 < xi1 or yi2 < yi1:
            return 0.0
        
        # Calculate areas
        intersection_area = (xi2 - xi1) * (yi2 - yi1)
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - intersection_area
        
        return intersection_area / union_area if union_area > 0 else 0

    def _merge_overlapping_detections(self, detections):
        """Merge detections that significantly overlap"""
        if not detections:
            return []
            
        merged = []
        detections = list(detections)
        while detections:
            current = detections.pop(0)
            
            # Find all overlapping detections
            overlaps = []
            i = 0
            while i < len(detections):
                if self._calculate_iou(current, detections[i]) > self.iou_threshold:
                    overlaps.append(detections.pop(i))
                else:
                    i += 1
            
            if overlaps:
                # Average the overlapping detections
                all_boxes = [current] + overlaps
                x = sum(box[0] for box in all_boxes) / len(all_boxes)
                y = sum(box[1] for box in all_boxes) / len(all_boxes)
                w = sum(box[2] for box in all_boxes) / len(all_boxes)
                h = sum(box[3] for box in all_boxes) / len(all_boxes)
                merged.append((int(x), int(y), int(w), int(h)))
            else:
                merged.append(current)
                
        return merged

    def _get_detection_key(self, detection):
        """Convert detection to a key based on position"""
        x, y, w, h = detection
        # Use larger grid (20 pixels) for more stable tracking
        return (x//20, y//20, w//20, h//20)

    def update(self, faces, bodies):
        # Convert detections to tuples and remove duplicates
        faces = [tuple(map(int, face)) for face in faces]
        bodies = [tuple(map(int, body)) for body in bodies]
        
        # Merge overlapping detections
        faces = self._merge_overlapping_detections(faces)
        bodies = self._merge_overlapping_detections(bodies)
        
        # Update face tracking
        current_faces = set()
        for face in faces:
            key = self._get_detection_key(face)
            if key not in self.face_history:
                # Check if this face overlaps significantly with any confirmed face
                overlapping = False
                for conf_face in self.confirmed_faces:
                    if self._calculate_iou(face, conf_face) > self.iou_threshold:
                        overlapping = True
                        break
                
                if not overlapping:
                    self.face_history[key] = 1
                    current_faces.add(key)
            else:
                self.face_history[key] += 1
                current_faces.add(key)
                
                if self.face_history[key] >= self.history_frames:
                    self.confirmed_faces.add(face)
        
        # Similar update for bodies
        current_bodies = set()
        for body in bodies:
            key = self._get_detection_key(body)
            self.body_history[key] += 1
            current_bodies.add(key)
            
            if self.body_history[key] >= self.history_frames:
                self.confirmed_bodies.add(body)
        
        # Clean up old detections
        for key in list(self.face_history.keys()):
            if key not in current_faces:
                self.face_history[key] -= 1
                if self.face_history[key] <= 0:
                    del self.face_history[key]
        
        for key in list(self.body_history.keys()):
            if key not in current_bodies:
                self.body_history[key] -= 1
                if self.body_history[key] <= 0:
                    del self.body_history[key]
        
        # Update confirmed sets with current detections
        self.confirmed_faces = {face for face in self.confirmed_faces 
                              if any(self._calculate_iou(face, current) > self.iou_threshold 
                                    for current in faces)}
        self.confirmed_bodies = {body for body in self.confirmed_bodies 
                               if any(self._calculate_iou(body, current) > self.iou_threshold 
                                     for current in bodies)}
        
        # Merge overlapping confirmed detections
        self.confirmed_faces = set(self._merge_overlapping_detections(self.confirmed_faces))
        self.confirmed_bodies = set(self._merge_overlapping_detections(self.confirmed_bodies))
        
        return list(self.confirmed_faces), list(self.confirmed_bodies)

def main():
    # Initialize Detection Engine and Camera
    detection_engine = DetectionEngine()
    cam = MainCam()
    tracker = DetectionTracker(history_frames=3, iou_threshold=0.5)

    try:
        while True:
            frame = cam.captureFrame()
            if frame is None:
                continue

            # Get raw detections
            raw_faces = detection_engine.detectFaceLocations(frame, method=0, show=False, verbose=False)
            raw_bodies = detection_engine.detectBodyLocations(frame, method=0, show=False, verbose=False)

            # Get temporally verified detections
            confirmed_faces, confirmed_bodies = tracker.update(raw_faces, raw_bodies)

            # Use confirmed detections for person detection
            persons = []
            if confirmed_faces and confirmed_bodies:
                persons = detection_engine.detectPerson(
                    frame, 
                    show=False,
                    method_face=0,
                    method_body=1,
                    verbose=False
                )

            # Visualization
            display_frame = frame.copy()

            # Draw raw detections in thin lines
            for face in raw_faces:
                face_tuple = tuple(face)
                if face_tuple not in confirmed_faces:
                    x, y, w, h = face
                    cv2.rectangle(display_frame, (x, y), (x+w, y+h), (255, 0, 0), 1)

            for body in raw_bodies:
                body_tuple = tuple(body)
                if body_tuple not in confirmed_bodies:
                    x, y, w, h = body
                    cv2.rectangle(display_frame, (x, y), (x+w, y+h), (255, 191, 0), 1)

            # Draw confirmed detections in thick lines
            for face in confirmed_faces:
                x, y, w, h = face
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            for body in confirmed_bodies:
                x, y, w, h = body
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), (255, 191, 0), 2)

            # Draw persons in green thick lines
            for person in persons:
                x, y, w, h = person
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Add text information
            cv2.putText(display_frame, f"Raw Faces: {len(raw_faces)}, Confirmed: {len(confirmed_faces)}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(display_frame, f"Raw Bodies: {len(raw_bodies)}, Confirmed: {len(confirmed_bodies)}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 191, 0), 2)
            cv2.putText(display_frame, f"Persons: {len(persons)}", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow('Detection Feed', display_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        cam.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()