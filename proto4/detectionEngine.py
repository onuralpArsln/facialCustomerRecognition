import cv2

def detect_faces(image):
    # Load the pre-trained Haar cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Convert the image to grayscale (Haar cascades work on grayscale images)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Blue rectangles

    # Return the annotated image and the face coordinates
    return image, faces


# Example usage
if __name__ == "__main__":
    # Load the image
    image_path = "people.png"  # Replace with your image file path
    image = cv2.imread(image_path)

    # Check if the image was successfully loaded
    if image is None:
        print("Error: Unable to load the image.")
    else:
        # Detect faces
        result_image, detected_faces = detect_faces(image)

        # Display the results
        print(f"Number of faces detected: {len(detected_faces)}")
        for i, (x, y, w, h) in enumerate(detected_faces):
            print(f"Face {i+1}: Location (x={x}, y={y}, width={w}, height={h})")

        # Show the annotated image with rectangles around faces
        cv2.imshow("Detected Faces", result_image)

        # Wait for a key press and close the OpenCV window
        cv2.waitKey(0)
        cv2.destroyAllWindows()
