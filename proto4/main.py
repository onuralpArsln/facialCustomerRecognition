import cv2
import numpy as np
from detectionEngine import DetectionEngine
from camStarter import MainCam
from collections import defaultdict
from detection_tracker import DetectionTracker


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
                    method_body=0,
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