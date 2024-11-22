import cv2
import os
import numpy as np

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
        self.frontalFaceProfileCascade=cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
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
        #1 haarcascade_fullbody.xml is thrash so upper bod
        self.body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_upperbody.xml')
        #2
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        #3
        if MISSING_IMPORT_mediapipe:
            print("pip3 install mediapipe")
        else:
            self.mp_pose = mp.solutions.pose.Pose(
                min_detection_confidence=0.5,  # Confidence level for detecting the pose
                min_tracking_confidence=0.5   # Confidence level for tracking landmarks
            )
    
    def non_max_suppression(self, boxes, overlapThresh=0.4):
        """
        Non-maximum suppression to filter out overlapping detections
        
        Args:
            boxes (list): List of detection boxes
            overlapThresh (float): Threshold for considering boxes overlapping
        
        Returns:
            list: Filtered list of boxes
        """
        # If no boxes, return empty list
        if len(boxes) == 0:
            return []

        # Convert to numpy array
        if isinstance(boxes, list):
            boxes = np.array(boxes)

        # Compute coordinates
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 0] + boxes[:, 2]
        y2 = boxes[:, 1] + boxes[:, 3]

        # Compute area of each box
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        
        # Sort the indices by bottom-right y-coordinate
        idxs = np.argsort(y2)
        
        pick = []
        while len(idxs) > 0:
            # Take the last index
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            # Find the largest (x, y) coordinates for the start of the bounding box
            # and the smallest coordinates for the end of the bounding box
            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

            # Compute width and height of the bounding box
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            # Compute overlap ratio
            overlap = (w * h) / areas[idxs[:last]]

            # Delete indices of boxes with significant overlap
            idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

        return boxes[pick].tolist()

    def detectFaceLocations(self, image, method=0, show=False, imageDownSize=False, verbose=True):
        if image is None:
            raise ValueError("Input image is None!")
        
        if imageDownSize:
            image = self.imageDownScale(image)

        faces = []
        match method:
            case 0:
                # Use both frontal and profile face cascades
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                frontal_faces = self.frontalFaceHaarCascade.detectMultiScale(
                    gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
                )
                profile_faces = self.frontalFaceProfileCascade.detectMultiScale(
                    gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
                )
                
                # Combine and remove overlapping detections
                faces = self.non_max_suppression(list(frontal_faces) + list(profile_faces))
            
            case 1:  # Dlib
                if MISSING_IMPORT_dlib:
                    print("pip3 install dlib")
                else:
                    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    dlib_faces = self.dlibDetector(gray_image)
                    faces = self.non_max_suppression(
                        [(face.left(), face.top(), face.width(), face.height()) for face in dlib_faces]
                    )
            
            case 2:  # MediaPipe
                if MISSING_IMPORT_mediapipe:
                    print("pip3 install mediapipe")
                else:
                    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    results = self.mp_face_detection.process(rgb_image)
                    if results.detections:
                        faces_temp = []
                        for detection in results.detections:
                            bboxC = detection.location_data.relative_bounding_box
                            ih, iw, _ = image.shape
                            x = int(bboxC.xmin * iw)
                            y = int(bboxC.ymin * ih)
                            w = int(bboxC.width * iw)
                            h = int(bboxC.height * ih)
                            faces_temp.append((x, y, w, h))
                        
                        faces = self.non_max_suppression(faces_temp)
            
            case _:
                raise ValueError("Invalid method! Use 0 (Haar), 1 (Dlib), or 2 (MediaPipe).")
        
        if verbose:
            print(f"With method {method} Number of faces detected: {len(faces)}")
        
        if show:
            #indiv
            for i, (x, y, w, h) in enumerate(faces):
                face = image[y:y+h, x:x+w]  
                window_name = f"body {i+1}"
                cv2.imshow(window_name, face)
            # one big 
            for (x, y, w, h) in faces:
                cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Blue rectangle
            cv2.imshow("Original Image with Faces", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return faces

    def detectBodyLocations(self, image, show=False, method=0, imageDownSize=False, verbose=True):
        if image is None:
            raise ValueError("Input image is None!")
        
        if imageDownSize:
            image = self.imageDownScale(image)

        bodies = []   
        match method:
            case 0: # haar body 
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                bodies = self.body_cascade.detectMultiScale(
                    gray_image, scaleFactor=1.1, minNeighbors=3, minSize=(50, 100)
                )
                bodies = self.non_max_suppression(bodies)
            
            case 1: # cv2.HOGDescriptor()
                bodies, _ = self.hog.detectMultiScale(
                    image, winStride=(8, 8), padding=(8, 8), scale=1.05
                )
                bodies = self.non_max_suppression(bodies)
            
            case 2:  # MediaPipe method for body detection
                if MISSING_IMPORT_mediapipe:
                    print("pip3 install mediapipe")
                else:
                    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    results = self.mp_pose.process(rgb_image)
                    if results.pose_landmarks:
                        bodies_temp = []
                        # Use hip and shoulder landmarks to create body boxes
                        landmarks = results.pose_landmarks.landmark
                        left_shoulder = landmarks[11]
                        right_shoulder = landmarks[12]
                        left_hip = landmarks[23]
                        right_hip = landmarks[24]
                        
                        # Calculate body region
                        x = int(min(left_shoulder.x, right_shoulder.x, left_hip.x, right_hip.x) * image.shape[1])
                        y = int(min(left_shoulder.y, right_shoulder.y, left_hip.y, right_hip.y) * image.shape[0])
                        w = int(abs(max(left_shoulder.x, right_shoulder.x, left_hip.x, right_hip.x) - 
                                    min(left_shoulder.x, right_shoulder.x, left_hip.x, right_hip.x)) * image.shape[1])
                        h = int(abs(max(left_shoulder.y, right_shoulder.y, left_hip.y, right_hip.y) - 
                                    min(left_shoulder.y, right_shoulder.y, left_hip.y, right_hip.y)) * image.shape[0])
                        
                        bodies_temp.append((x, y, w, h))
                        bodies = self.non_max_suppression(bodies_temp)
            
            case _:
                raise ValueError("Invalid method!")
        
        if verbose:
            print(f"With method {method} Number of bodies detected: {len(bodies)}")
        
        if show:
            # one big 
            for (x, y, w, h) in bodies:
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # green rectangle bgr
            #indiv
            for i, (x, y, w, h) in enumerate(bodies):
                body = image[y:y+h, x:x+w]  
                window_name = f"body {i+1}"
                cv2.imshow(window_name, body)
            
            cv2.imshow("Original Image with Faces", image)
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

    testEngine = DetectionEngine()
    
    # face images
    if True:
        image_path = "people.png"
        image = cv2.imread(image_path)
        
        testEngine.detectFaceLocations(image, show=True,imageDownSize=True,verbose=True, method=0)
        testEngine.detectFaceLocations(image, show=True,imageDownSize=True,verbose=True,method=1)
        testEngine.detectFaceLocations(image, show=True,imageDownSize=True,verbose=True,method=2)
       

    # body test 
    if True:
        image_path = "body.png"
        image = cv2.imread(image_path)
        testEngine.detectFaceLocations(image, show=True)
        testEngine.detectBodyLocations(image, show=True,method=0)
        testEngine.detectBodyLocations(image, show=True,method=1)
        testEngine.detectBodyLocations(image, show=True,method=2)
