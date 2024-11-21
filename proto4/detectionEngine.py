import cv2
import os


class DetectionEngine:
    
    def detectFaceLocations(self,image,show=False):
        # Load the Haar cascade for face detection
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Convert the image to grayscale (Haar cascades work on grayscale images)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if show:
            for (x, y, w, h) in faces:
                    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Blue rectangle
            cv2.imshow("Original Image with Faces", image)

        return faces
    
    def getFaces(self,image,show=False):
                # Load the Haar cascade for face detection
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Convert the image to grayscale (Haar cascades work on grayscale images)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        print(f"Number of faces detected: {len(faces)}")

        # Display individual faces
        if show:
            for i, (x, y, w, h) in enumerate(faces):
                    face = image[y:y+h, x:x+w]  
                    window_name = f"Face {i+1}"
                    cv2.imshow(window_name, face)
            cv2.waitKey(0)
            cv2.destroyAllWindows()



# Main program
if __name__ == "__main__":
    # Use the image file name (since it's in the same folder as the script)
    image_path = "people.png"

    image = cv2.imread(image_path)

    testEngine = DetectionEngine()

    testEngine.getFaces(image, show=True)

