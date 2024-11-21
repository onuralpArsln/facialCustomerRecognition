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

    def getBodies(self,image,show=False,method=1):

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                # Load the Haar cascade for face detection

        match method:
            case 0:
                body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')
                bodies = body_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=3, minSize=(50, 100))
            case 1:
                hog = cv2.HOGDescriptor()
                hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
                bodies, _ = hog.detectMultiScale(image, winStride=(8, 8), padding=(8, 8), scale=1.05)
        
        
        
        print(f"Number of bodies detected: {len(bodies)}")

        # Display individual faces
        if show:
            for i, (x, y, w, h) in enumerate(bodies):
                    body = image[y:y+h, x:x+w]  
                    window_name = f"body {i+1}"
                    cv2.imshow(window_name, body)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


# Main program
if __name__ == "__main__":
    # Use the image file name (since it's in the same folder as the script)
    image_path = "people.png"

    image = cv2.imread(image_path)

    testEngine = DetectionEngine()

    testEngine.getFaces(image, show=True)


    image_path = "body.png"

    image = cv2.imread(image_path)
    testEngine.getBodies(image, show=True,method=1)
