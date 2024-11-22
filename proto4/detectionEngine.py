import cv2
import os


class DetectionEngine:

    def __init__(self,) -> None:
        ###generate Face detectors ,
        #1
        self.frontalFaceHaarCascade=cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
        ### generate body detectors
        #1
        self.body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')
        #2
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    
    def detectFaceLocations(self,image,show=False):

        #haar cascade verimliliği için gray scale al
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # haarcascade kullan
        faces = self.frontalFaceHaarCascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if show:
            for (x, y, w, h) in faces:
                    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Blue rectangle
            cv2.imshow("Original Image with Faces", image)
            for i, (x, y, w, h) in enumerate(faces):
                    face = image[y:y+h, x:x+w]  
                    window_name = f"Face {i+1}"
                    cv2.imshow(window_name, face)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return faces
    
    
    def detectBodyLocations(self,image,show=False,method=1):
        match method:
            case 0:
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                bodies = self.body_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=3, minSize=(50, 100))
            case 1:
                bodies, _ = self.hog.detectMultiScale(image, winStride=(8, 8), padding=(8, 8), scale=1.05)
        
        print(f"Number of bodies detected: {len(bodies)}")
        # görselleri ver eğer show ile istenirse
        if show:
            for i, (x, y, w, h) in enumerate(bodies):
                    body = image[y:y+h, x:x+w]  
                    window_name = f"body {i+1}"
                    cv2.imshow(window_name, body)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return bodies


    def imageDownScale(image, desiredSize=(450,450)):
        return cv2.resize(image, desiredSize, interpolation=cv2.INTER_AREA)



# Main program
if __name__ == "__main__":
    # Use the image file name (since it's in the same folder as the script)
    image_path = "people.png"

    image = cv2.imread(image_path)

    testEngine = DetectionEngine()

    testEngine.detectFaceLocations(image, show=True)


    image_path = "body.png"
    image = cv2.imread(image_path)
    testEngine.detectFaceLocations(image, show=True)

    testEngine.detectBodyLocations(image, show=True,method=1)
