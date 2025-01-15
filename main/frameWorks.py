import cv2
import numpy as np

class frameWorks:
    
    lastEditedFrame=None
    lastKnownLocations=None


    def drawBoundingBox(self,frame : cv2.typing.MatLike,detectionsFromMbt,label:str):
        
        if detectionsFromMbt is None:
            return frame
       
        if frame.any():
            self.lastKnownLocations=[]  
            for detection in detectionsFromMbt:
                # Koordinatları orijinal boyuta geri ölçeklendir
                bboxC = detection.location_data.relative_bounding_box
                ih, iw = frame.shape[:2]
                bbox=[int(bboxC.xmin * iw),int(bboxC.ymin *ih),int(bboxC.width * iw ),int(bboxC.height * ih)]  
                self.lastKnownLocations.append(bbox)
                cv2.rectangle(frame, 
                            (bbox[0], bbox[1]), 
                            (bbox[0] + bbox[2], bbox[1] + bbox[3]), 
                            (0, 255, 0), 1)
                                
                # ID ve score'u yaz (kutu üzerinde)
                text = f"ID: {label})"
                text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                text_width, text_height = text_size

                # Metin arka planı çiz
                text_background_topleft = (bbox[0], bbox[1] - text_height - 5)
                text_background_bottomright = (bbox[0] + text_width, bbox[1] - 2)
                cv2.rectangle(frame, text_background_topleft, text_background_bottomright, (0, 255, 0), -1)

                # Metni yaz
                cv2.putText(frame, text, (bbox[0], bbox[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        
        self.lastEditedFrame=frame
        return frame
     
    def splitFaces(self,frame):
        crops=[]
        if self.lastKnownLocations:
            for location in self.lastKnownLocations:
                cropped_image = frame[location[1]:location[1]+location[3], location[0]:location[0]+location[3]]

                crops.append(cropped_image)

        return crops

