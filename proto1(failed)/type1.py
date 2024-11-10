import cv2
import numpy as np
import os

# Create directories for storing pictures
if not os.path.exists('incoming'):
    os.makedirs('incoming')
if not os.path.exists('outgoing'):
    os.makedirs('outgoing')

# Load pre-trained Haar Cascade face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize webcam
cap = cv2.VideoCapture(0)

# Initialize variables for tracking face direction
incoming_people = []
outgoing_people = []
prev_positions = []

def save_face_picture(direction, face_img, count):
    # Save the image with a unique filename
    filename = f"{direction}/{count}.jpg"
    cv2.imwrite(filename, face_img)

def detect_face_direction(face, prev_position, frame_width):
    """Detect face direction based on horizontal movement."""
    x, y, w, h = face
    face_center_x = x + w // 2

    if prev_position is None:
        return None, face_center_x

    prev_x = prev_position
    if face_center_x < prev_x - 50:
        return 'incoming', face_center_x  # Moving left to right (incoming)
    elif face_center_x > prev_x + 50:
        return 'outgoing', face_center_x  # Moving right to left (outgoing)
    
    return None, face_center_x  # No significant movement detected

# Initialize person count trackers
incoming_count = 0
outgoing_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]
        direction, new_position = detect_face_direction((x, y, w, h), prev_positions.get('last_position', None), frame.shape[1])

        if direction == 'incoming':
            incoming_people.append(f'incoming_{incoming_count}')
            save_face_picture('incoming', face_img, incoming_count)
            incoming_count += 1
        elif direction == 'outgoing':
            outgoing_people.append(f'outgoing_{outgoing_count}')
            save_face_picture('outgoing', face_img, outgoing_count)
            outgoing_count += 1

        prev_positions['last_position'] = new_position  # Update previous position

        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Display the resulting frame
    cv2.imshow('Face Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close the window
cap.release()
cv2.destroyAllWindows()
