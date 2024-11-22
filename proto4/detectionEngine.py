import cv2
import os

MISSING_IMPORT_dlib = False
MISSING_IMPORT_mediapipe= False
try:
    import mediapipe as mp
except:
    MISSING_IMPORT_mediapipe= True
    print("pip3 install mediapipe")
try:
    import dlib
except:
    MISSING_IMPORT_dlib = True
    print("pip3 install dlib")



class DetectionEngine:

    def __init__(self,defaultImageSize=(450,450)) -> None:
        ## default image size for downsizing if enabled
        self.defaultImageSize = defaultImageSize
        ###generate Face detectors ,
        #1
        self.frontalFaceHaarCascade=cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        #2
        if MISSING_IMPORT_dlib:
             print("pip3 install dlib")
        else:
            self.dlibDetector = dlib.get_frontal_face_detector()
        #3
        if MISSING_IMPORT_mediapipe:
             print("pip3 install mediapipe")
        else:
            self.mp_face_detection = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5)



        ### generate body detectors
        #1
        self.body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')
        #2
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    
    def detectFaceLocations(self,image,method=0,show=False,imageDownSize=False,verbose=True):

        if image is None:
            raise ValueError("Input image is None!")
    
        if imageDownSize:
            image = self.imageDownScale(image)
        

        faces = []
        match method:
            case 0:
                #haar cascade verimliliği için gray scale al
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                # haarcascade kullan
                faces = self.frontalFaceHaarCascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            case 1:  # Dlib
                if MISSING_IMPORT_dlib:
                    print("pip3 install dlib")
                else:
                    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    dlib_faces = self.dlibDetector(gray_image)
                    faces = [(face.left(), face.top(), face.width(), face.height()) for face in dlib_faces]
            case 2:  # MediaPipe
                if MISSING_IMPORT_mediapipe:
                    print("pip3 install mediapipe")
                else:
                    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    results = self.mp_face_detection.process(rgb_image)
                    if results.detections:
                        for detection in results.detections:
                            bboxC = detection.location_data.relative_bounding_box
                            ih, iw, _ = image.shape
                            x = int(bboxC.xmin * iw)
                            y = int(bboxC.ymin * ih)
                            w = int(bboxC.width * iw)
                            h = int(bboxC.height * ih)
                            faces.append((x, y, w, h))
            case _:
                raise ValueError("Invalid method! Use 0 (Haar), 1 (Dlib), or 2 (MediaPipe).")
        


        if verbose:
            print(f"Number of faces detected: {len(faces)}")
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
    
    
    def detectBodyLocations(self,image,show=False,method=1,imageDownSize=False,verbose=True):
        
        if image is None:
            raise ValueError("Input image is None!")
        
        if imageDownSize:
            image = self.imageDownScale(image)

        bodies = []   
        match method:
            case 0: # haar body 
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                bodies = self.body_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=3, minSize=(50, 100))
            case 1: # cv2.HOGDescriptor()
                bodies, _ = self.hog.detectMultiScale(image, winStride=(8, 8), padding=(8, 8), scale=1.05)
            case _:
                raise ValueError("Invalid method!")
        
        if verbose:
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


    def imageDownScale(self,image):
        """This method downsizes images, can be used inside detection modes so if the device running do not have 
        enough processing power, down sizing images might help
        Desired size is direct take from objects it self during parameters giving on initialization or it defaults to 450 by 450

        Args:
            image (nparray): opencv image, nparray format obatined with cv2.imread
            
        Returns:
            image (nparray): opencv image, nparray format obatined with cv2.imread
        """
        return cv2.resize(image, self.defaultImageSize, interpolation=cv2.INTER_AREA)



# Main program
if __name__ == "__main__":

    # face images
    if True:
        image_path = "people.png"
        image = cv2.imread(image_path)
        testEngine = DetectionEngine()
        testEngine.detectFaceLocations(image, show=True,imageDownSize=True,verbose=True, method=0)
        testEngine.detectFaceLocations(image, show=True,imageDownSize=True,verbose=True,method=1)
        testEngine.detectFaceLocations(image, show=True,imageDownSize=True,verbose=True,method=2)
       

    # body test 
    if False:
        image_path = "body.png"
        image = cv2.imread(image_path)
        testEngine.detectFaceLocations(image, show=True)
        testEngine.detectBodyLocations(image, show=True,method=1)
