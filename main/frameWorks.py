import cv2
import numpy as np
import copy

class frameWorks:
    
    lastEditedFrame=None
    lastKnownLocations=None
    lastGivenFrame=None
    detectedHumans=[]
    trackedHumans=0

    def mbt2list(self,detection,frame: cv2.typing.MatLike):
        # Koordinatları orijinal boyuta geri ölçeklendir
        bboxC = detection.location_data.relative_bounding_box
        ih, iw = frame.shape[:2]
        bbox=[int(bboxC.xmin * iw),int(bboxC.ymin *ih),int(bboxC.width * iw ),int(bboxC.height * ih)]  
        return bbox

    def iou(self,box1, box2):
        """
        Calculate the Intersection over Union (IoU) of two bounding boxes.
        
        Parameters:
        box1: list or tuple [x1, y1, x2, y2] -> Top-left and bottom-right of the first box.
        box2: list or tuple [x1, y1, x2, y2] -> Top-left and bottom-right of the second box.
        
        Returns:
        float: IoU value between 0 and 1.
        """
        # Calculate intersection coordinates
        x_left = max(box1[0], box2[0])
        y_top = max(box1[1], box2[1])
        x_right = min(box1[2], box2[2])
        y_bottom = min(box1[3], box2[3])
        
        # Compute the area of intersection
        if x_right < x_left or y_bottom < y_top:
            return 0.0  # No overlap
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Compute the area of both bounding boxes
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        # Compute the union area
        union_area = box1_area + box2_area - intersection_area
        
        # Compute the IoU
        return intersection_area / union_area

    def drawBoundingBox(self,frame : cv2.typing.MatLike, detectionsFromMbt,label:str = None):
        if detectionsFromMbt is None:
            return frame 
        if frame.any():
            self.lastGivenFrame=frame
            self.lastKnownLocations=[]  
            for detection in detectionsFromMbt:
                bbox=self.mbt2list(detection,frame)
                self.lastKnownLocations.append(bbox)
                cv2.rectangle(frame, 
                            (bbox[0], bbox[1]), 
                            (bbox[0] + bbox[2], bbox[1] + bbox[3]), 
                            (0, 255, 0), 1)
                                
                # ID ve score'u yaz (kutu üzerinde)
                text = f"ID: {detection.label_id})"
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
     
    def splitFaces(self,frame=lastGivenFrame):
        crops=[]
        if self.lastKnownLocations:
            for location in self.lastKnownLocations:
                cropped_image = frame[location[1]:location[1]+location[3], location[0]:location[0]+location[2]]
                crops.append(cropped_image)
        return crops
    
    def updateHumans(self,frame):
        if self.lastKnownLocations is None:
            return -1
        if len(self.lastKnownLocations)==0:
            return -1
        
        # en başta bir defa çalışıp durdu
        if len(self.detectedHumans)==0:
            if self.lastKnownLocations:
                for location in self.lastKnownLocations:
                    cropped_image = frame[location[1]:location[1]+location[3], location[0]:location[0]+location[2]]
                    detectTemp= Detected()
                    detectTemp.image=cropped_image
                    detectTemp.location=[ location[0],location[1], location[0]+location[2],location[1]+location[3]]
                    detectTemp.name=str(len(self.detectedHumans))
                    self.detectedHumans.append(detectTemp)
                    self.trackedHumans+=1
            
        else:
            locationsMem=copy.deepcopy(self.lastKnownLocations)
            for human in self.detectedHumans:
                if human.isDeleted: continue
                for i in range(len(locationsMem)):
                    if locationsMem[i]==None:
                        # bu lokasyonda eşleşme yapılmış atla
                        continue
                    locList=[locationsMem[i][0],locationsMem[i][1], locationsMem[i][0]+locationsMem[i][2],locationsMem[i][1]+locationsMem[i][3]]
                    if self.iou(locList,human.location) > 0.6:
                        # iyi bir match yakalamış 
                        human.location=copy.deepcopy(locList)
                        locationsMem[i]=None # bir detection olan locationları none yap
                        print("vurdum brek")
                        break
                else:
                    # eğer buraya vurduysa locationlar bitti ve o human görünürde yok 
                    # break yerse else çalışmaz
                    print("for lese")
                    human.location=[0,0,0,0]
                    human.isDeleted =True
                    self.trackedHumans-=1
                
            # non null locationlarda yeni tanınan insan var demek
            for location in locationsMem:
                if location is not None:
                    cropped_image = frame[location[1]:location[1]+location[3], location[0]:location[0]+location[2]]
                    detectTemp= Detected()
                    detectTemp.image=cropped_image
                    detectTemp.location=[ location[0],location[1], location[0]+location[2],location[1]+location[3]]
                    detectTemp.name=str(len(self.detectedHumans))
                    self.detectedHumans.append(detectTemp)
                    self.trackedHumans+=1

                    


    def updateLocations(self,frame : cv2.typing.MatLike, detectionsFromMbt):
        if detectionsFromMbt is None:
            # eğer tahmin yoksa boş çizmeden döndür
            return -1
        if frame.any() is None:
            # eğer resim bozuksa
            return -1
    
        self.lastGivenFrame=frame
        self.lastKnownLocations=[]
        for detection in detectionsFromMbt:
                bbox=self.mbt2list(detection,frame)
                self.lastKnownLocations.append(bbox)

    def drawTrackedHumans(self,frame):
        for human in self.detectedHumans:
            cv2.rectangle(frame, 
                            (human.location[0], human.location[1]), 
                            (human.location[2],human.location[3]), 
                            (0, 255, 0), 1)
                                
            # ID ve score'u yaz (kutu üzerinde)
            text = f"ID: {human.name})"
            text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            text_width, text_height = text_size

            # Metin arka planı çiz
            text_background_topleft = (human.location[0], human.location[1] - text_height - 5)
            text_background_bottomright = (human.location[0] + text_width, human.location[1] - 2)
            cv2.rectangle(frame, text_background_topleft, text_background_bottomright, (0, 255, 0), -1)

                # Metni yaz
            cv2.putText(frame, text, (human.location[0], human.location[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        self.lastEditedFrame=frame

               
    def fwFacade(self,frame : cv2.typing.MatLike, detectionsFromMbt):
        if self.updateLocations(frame,detectionsFromMbt) == -1: print("pırta")
        if self.updateHumans(frame) == -1 : print("Pırtonce")
        res=self.drawTrackedHumans(frame)
        return res



class Detected:
    image = None
    location=[0,0,0,0]  # x1 y1 x2 y2 
    name=""
    isDeleted=False