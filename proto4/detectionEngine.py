import cv2
import os

def detect_faces(image):
    # Load the Haar cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Convert the image to grayscale (Haar cascades work on grayscale images)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Return the face coordinates
    return faces


# Main program
if __name__ == "__main__":
    # Use the image file name (since it's in the same folder as the script)
    image_path = "people.png"

    # Check if the file exists
    if not os.path.exists(image_path):
        print(f"Error: The file '{image_path}' does not exist in the current directory.")
    else:
        # Load the image
        image = cv2.imread(image_path)

        if image is None:
            print("Error: Unable to load the image. The file may be corrupt or unsupported.")
        else:
            print("Image loaded successfully!")

            # Detect faces
            faces = detect_faces(image)
            print(f"Number of faces detected: {len(faces)}")

            # Display individual faces
            for i, (x, y, w, h) in enumerate(faces):
                face = image[y:y+h, x:x+w]  # Crop the face
                window_name = f"Face {i+1}"
                cv2.imshow(window_name, face)
            
            # Show the original image with rectangles around faces
            for (x, y, w, h) in faces:
                cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Blue rectangle
            
            cv2.imshow("Original Image with Faces", image)

            # Wait for a key press and close all windows
            cv2.waitKey(0)
            cv2.destroyAllWindows()
