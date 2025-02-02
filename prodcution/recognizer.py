import cv2
import mediapipe as mp

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.2)

# Initialize OpenCV to capture video from the camera
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to detect faces
    results = face_detection.process(rgb_frame)

    # Draw face detections on the original frame and crop faces
    if results.detections:
        for detection in results.detections:
            # Get the bounding box of the detected face
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape

            # Calculate the bounding box coordinates
            x1 = int(bboxC.xmin * iw)
            y1 = int(bboxC.ymin * ih)
            w = int(bboxC.width * iw)
            h = int(bboxC.height * ih)

            # Calculate the center of the face bounding box
            center_x = x1 + w // 2
            center_y = y1 + h // 2

            # Apply an offset to the bounding box size (1.3 times the detected size)
            offset = 0.3
            new_w = int(1.4 * w)
            new_h = int(1.6 * h)

            # Calculate new bounding box to center it around the original face
            new_x1 = max(0, center_x - new_w // 2)
            new_y1 = max(0, center_y - new_h // 2)
            new_x2 = min(iw, new_x1 + new_w)
            new_y2 = min(ih, new_y1 + new_h)

            # Crop the face from the frame with the new bounding box
            face_crop = frame[new_y1:new_y2, new_x1:new_x2]

            # Draw the face bounding box on the original frame
            cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)

            # Optionally, show the cropped face in a separate window
            cv2.imshow('Cropped Face', face_crop)

    # Display the frame with face bounding boxes
    cv2.imshow('Face Detection', frame)

    # Press 'q' to quit the video stream
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()
