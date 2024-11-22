import cv2
from detectionEngine import DetectionEngine
from camStarter import MainCam

def main():
    # Initialize Detection Engine and Camera
    detection_engine = DetectionEngine()
    cam = MainCam()

    try:
        while True:
            # Capture frame from camera
            frame = cam.captureFrame()

            # Detect faces using Haar Cascade (method 0)
            faces = detection_engine.detectFaceLocations(frame, method=0, show=False, verbose=False)
            
            # Detect bodies using Haar Cascade (method 0)
            bodies = detection_engine.detectBodyLocations(frame, method=0, show=False, verbose=False)

            # Draw rectangles for detected faces
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)  # Green for faces

            # Draw rectangles for detected bodies
            for (x, y, w, h) in bodies:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)  # Blue for bodies

            # Add text information
            cv2.putText(frame, f"Faces: {len(faces)}, Bodies: {len(bodies)}", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Display the frame
            cv2.imshow('Detection Feed', frame)

            # Break loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        # Cleanup
        cam.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()