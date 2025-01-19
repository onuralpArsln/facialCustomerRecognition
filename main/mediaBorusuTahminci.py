import mediapipe as mp
import cv2

class MediaBorusuTahminci:

    lastGuessedFrame=None
    lastGuessedLocations=None

    def __init__(self, confidence : int = 0.8):
        mp_face_detection = mp.solutions.face_detection
        if confidence>1 or confidence <0:
            raise("Kanka confidence 0.0 ile 1.0 arası olacak mediaBorusuTaminci.__init__")
        self.face_detection =mp_face_detection.FaceDetection(min_detection_confidence=confidence)


    def tahmin( self,frame : cv2.typing.MatLike , scaleFactor:int = 1, drawBoundingBox: bool =False):
        if frame.any() == None:
            raise("Frame none geldi hayrola MediaBorusuTahminci.tahmin methodu")
        if scaleFactor>1 or scaleFactor<=0 :
            raise("Scale Factoru naptın kankarino sadece 0-1 aralığında olabilir mediaBorusuTahminci.tahmin.scaleFactor")
        
        # Orijinal görüntü boyutlarını al
        height, width = frame.shape[:2]
        frame_org=frame
        # Görüntüyü küçült
        if(scaleFactor<1):
            frame = cv2.resize(frame, (int(width * scaleFactor), int(height * scaleFactor)))
        
        # OpenCV'nin BGR formatını RGB'ye dönüştür
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # MediaPipe ile yüz tespiti yap
        results = self.face_detection.process(frame_rgb)

        self.lastGuessedLocations= results.detections

        if drawBoundingBox:
            if results.detections:
                for detection in results.detections:
                    # Koordinatları orijinal boyuta geri ölçeklendir
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw = frame.shape[:2]
                    
                    # Koordinatları orijinal boyuta dönüştür
                    bbox = int(bboxC.xmin * iw / scaleFactor), int(bboxC.ymin * ih / scaleFactor), \
                        int(bboxC.width * iw / scaleFactor), int(bboxC.height * ih / scaleFactor)
                    
                    
                 
                    cv2.rectangle(frame_org, 
                                (bbox[0], bbox[1]), 
                                (bbox[0] + bbox[2], bbox[1] + bbox[3]), 
                                (0, 255, 0), 2)
           
            self.lastGuessedFrame=frame_org
        else:

            self.lastGuessedFrame=frame
        
        return results.detections

        


        