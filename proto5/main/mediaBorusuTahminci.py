import mediapipe as mp
import cv2

class MediaBorusuTahminci:
    def __init__(self, confidence : int = 0.8):
        mp_face_detection = mp.solutions.face_detection
        if confidence>1 or confidence <0:
            raise("Kanka confidence 0.0 ile 1.0 arası olacak mediaBorusuTaminci.__init__")
        self.face_detection =mp_face_detection.FaceDetection(min_detection_confidence=confidence)


    def tahmin( frame : cv2.MathLike ):
        if frame==None:
            raise("Frame none geldi hayrola MediaBorusuTahminci.tahmin methodu")
        