import cv2
import mediapipe as mp

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose

# Function to calculate and draw bounding boxes
def draw_bounding_boxes(image, landmarks, image_width, image_height):
    # Head bounding box
    head_points = [
        landmarks[mp_pose.PoseLandmark.NOSE],
        landmarks[mp_pose.PoseLandmark.LEFT_EYE],
        landmarks[mp_pose.PoseLandmark.RIGHT_EYE],
        landmarks[mp_pose.PoseLandmark.LEFT_EAR],
        landmarks[mp_pose.PoseLandmark.RIGHT_EAR],
    ]
    head_x = [point.x for point in head_points]
    head_y = [point.y for point in head_points]
    x_min_head, x_max_head = int(min(head_x) * image_width), int(max(head_x) * image_width)
    y_min_head, y_max_head = int(min(head_y) * image_height), int(max(head_y) * image_height)
    cv2.rectangle(image, (x_min_head, y_min_head), (x_max_head, y_max_head), (0, 255, 0), 2)  # Green for head

    # Face bounding box
    face_points = [
        landmarks[mp_pose.PoseLandmark.NOSE],
        landmarks[mp_pose.PoseLandmark.MOUTH_LEFT],
        landmarks[mp_pose.PoseLandmark.MOUTH_RIGHT],
    ]
    face_x = [point.x for point in face_points]
    face_y = [point.y for point in face_points]
    x_min_face, x_max_face = int(min(face_x) * image_width), int(max(face_x) * image_width)
    y_min_face, y_max_face = int(min(face_y) * image_height), int(max(face_y) * image_height)
    cv2.rectangle(image, (x_min_face, y_min_face), (x_max_face, y_max_face), (255, 0, 0), 2)  # Blue for face

    # Body bounding box
    body_points = [
        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER],
        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER],
        landmarks[mp_pose.PoseLandmark.LEFT_HIP],
        landmarks[mp_pose.PoseLandmark.RIGHT_HIP],
    ]
    body_x = [point.x for point in body_points]
    body_y = [point.y for point in body_points]
    x_min_body, x_max_body = int(min(body_x) * image_width), int(max(body_x) * image_width)
    y_min_body, y_max_body = int(min(body_y) * image_height), int(max(body_y) * image_height)
    cv2.rectangle(image, (x_min_body, y_min_body), (x_max_body, y_max_body), (0, 0, 255), 2)  # Red for body


# Initialize MediaPipe Pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Open a video source (e.g., webcam)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB (MediaPipe expects RGB images)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False  # Optimize memory
    results = pose.process(image)

    # Convert back to BGR for OpenCV
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    h, w, _ = image.shape  # Get image dimensions

    # Draw bounding boxes
    if results.pose_landmarks:
        draw_bounding_boxes(image, results.pose_landmarks.landmark, w, h)

    # Display the processed image
    cv2.imshow("Head, Face, and Body Detection", image)

    # Exit on pressing ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
