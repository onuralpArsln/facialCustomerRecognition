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
        if len(boxes) == 0:
            return []

        if isinstance(boxes, list):
            boxes = np.array(boxes)

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 0] + boxes[:, 2]
        y2 = boxes[:, 1] + boxes[:, 3]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)

        pick = []
        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            overlap = (w * h) / areas[idxs[:last]]
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
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                frontal_faces = self.frontalFaceHaarCascade.detectMultiScale(
                    gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
                )
                profile_faces = self.frontalFaceProfileCascade.detectMultiScale(
                    gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
                )
                faces = self.non_max_suppression(list(frontal_faces) + list(profile_faces))

            case 1:
                if MISSING_IMPORT_dlib:
                    print("pip3 install dlib")
                else:
                    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    dlib_faces = self.dlibDetector(gray_image)
                    faces = self.non_max_suppression(
                        [(face.left(), face.top(), face.width(), face.height()) for face in dlib_faces]
                    )

            case 2:
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
            for i, (x, y, w, h) in enumerate(faces):
                face = image[y:y+h, x:x+w]  
                window_name = f"body {i+1}"
                cv2.imshow(window_name, face)
            for (x, y, w, h) in faces:
                cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.imshow("Original Image with Faces", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return faces

    def detectBodyLocations(self, image, show=False, method=0, imageDownSize=False, verbose=True, join_overlaps=False, iou_threshold=0.75):
        if image is None:
            raise ValueError("Input image is None!")

        if imageDownSize:
            image = self.imageDownScale(image)

        bodies = []
        match method:
            case 0: 
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                bodies = self.body_cascade.detectMultiScale(
                    gray_image, scaleFactor=1.1, minNeighbors=3, minSize=(50, 100)
                )
                bodies = self.non_max_suppression(bodies)
            
            case 1:
                bodies, _ = self.hog.detectMultiScale(
                    image, winStride=(8, 8), padding=(8, 8), scale=1.05
                )
                bodies = self.non_max_suppression(bodies)
            
            case 2:
                if MISSING_IMPORT_mediapipe:
                    print("pip3 install mediapipe")
                else:
                    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    results = self.mp_pose.process(rgb_image)
                    if results.pose_landmarks:
                        bodies_temp = []
                        landmarks = results.pose_landmarks.landmark
                        left_shoulder = landmarks[11]
                        right_shoulder = landmarks[12]
                        left_hip = landmarks[23]
                        right_hip = landmarks[24]

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

        if join_overlaps:
            bodies = self.join_overlapping_bodies(bodies, iou_threshold)

        if verbose:
            print(f"With method {method} Number of bodies detected: {len(bodies)}")

        if show:
            for (x, y, w, h) in bodies:
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.imshow("Original Image with Bodies", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return bodies

    def join_overlapping_bodies(self, bodies, iou_threshold):
        joined_bodies = []
        used = set()
        for i in range(len(bodies)):
            if i in used:
                continue
            x1, y1, w1, h1 = bodies[i]
            merged = [x1, y1, x1 + w1, y1 + h1]

            for j in range(i + 1, len(bodies)):
                if j in used:
                    continue

                x2, y2, w2, h2 = bodies[j]
                xx1 = max(x1, x2)
                yy1 = max(y1, y2)
                xx2 = min(x1 + w1, x2 + w2)
                yy2 = min(y1 + h1, y2 + h2)

                if xx2 < xx1 or yy2 < yy1:
                    continue

                intersection = (xx2 - xx1) * (yy2 - yy1)
                area1 = w1 * h1
                area2 = w2 * h2
                iou = intersection / (area1 + area2 - intersection)

                if iou >= iou_threshold:
                    merged[0] = min(merged[0], x2)
                    merged[1] = min(merged[1], y2)
                    merged[2] = max(merged[2], x2 + w2)
                    merged[3] = max(merged[3], y2 + h2)
                    used.add(j)

            joined_bodies.append((merged[0], merged[1], merged[2] - merged[0], merged[3] - merged[1]))
            used.add(i)

        return joined_bodies

    def detectPerson(self, image, show=False, method_body=0, method_face=0, verbose=True):
        if image is None:
            raise ValueError("Input image is None!")

        bodies = self.detectBodyLocations(image, method=method_body, verbose=verbose)
        faces = self.detectFaceLocations(image, method=method_face, verbose=verbose)

        persons = []
        for (bx, by, bw, bh) in bodies:
            for (fx, fy, fw, fh) in faces:
                face_center_x = fx + fw / 2
                face_center_y = fy + fh / 2

                if bx <= face_center_x <= bx + bw and by <= face_center_y <= by + bh:
                    persons.append((bx, by, bw, bh))
                    break

        if verbose:
            print(f"Number of persons detected: {len(persons)}")

        if show:
            for (x, y, w, h) in persons:
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

            cv2.imshow("Detected Persons", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return persons

    def imageDownScale(self, image):
        return cv2.resize(image, self.defaultImageSize, interpolation=cv2.INTER_AREA)

if __name__ == "__main__":
    testEngine = DetectionEngine()
    if True:
        image_path = "body.png"
        image = cv2.imread(image_path)
        testEngine.detectPerson(image, show=True, method_body=1, method_face=0)

    
    # face images
    if False:
        image_path = "people.png"
        image = cv2.imread(image_path)
        testEngine.detectFaceLocations(image, show=True,imageDownSize=True,verbose=True, method=0)
        
        testEngine.detectFaceLocations(image, show=True,imageDownSize=True,verbose=True,method=1)
        testEngine.detectFaceLocations(image, show=True,imageDownSize=True,verbose=True,method=2)
       

    # body test 
    if False:
        image_path = "body.png"
        image = cv2.imread(image_path)
        testEngine.detectBodyLocations(image, show=True,method=0)
        testEngine.detectFaceLocations(image, show=True,imageDownSize=True,verbose=True, method=0)
     
        testEngine.detectBodyLocations(image, show=True,method=1,join_overlaps=True,iou_threshold=0.9)