import time
from collections import defaultdict

class DetectionTracker:
    def __init__(self, history_frames=3, iou_threshold=0.5, suspicion_limit=5, forget_time=10, revalidate_iou=0.3):
        self.history_frames = history_frames
        self.iou_threshold = iou_threshold
        self.revalidate_iou = revalidate_iou  # Threshold for revalidation
        self.face_history = defaultdict(int)
        self.body_history = defaultdict(int)
        self.suspected_faces = defaultdict(lambda: {"count": 0, "last_seen": time.time()})
        self.confirmed_faces = set()
        self.confirmed_bodies = set()
        self.suspicion_limit = suspicion_limit
        self.forget_time = forget_time
        self.last_detected_faces = set()  # Track the last detected faces

    def _calculate_iou(self, box1, box2):
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        
        if xi2 < xi1 or yi2 < yi1:
            return 0.0
        
        intersection_area = (xi2 - xi1) * (yi2 - yi1)
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - intersection_area
        
        return intersection_area / union_area if union_area > 0 else 0

    def _merge_overlapping_detections(self, detections):
        if not detections:
            return []
            
        merged = []
        detections = list(detections)
        while detections:
            current = detections.pop(0)
            
            overlaps = []
            i = 0
            while i < len(detections):
                if self._calculate_iou(current, detections[i]) > self.iou_threshold:
                    overlaps.append(detections.pop(i))
                else:
                    i += 1
            
            if overlaps:
                all_boxes = [current] + overlaps
                x = sum(box[0] for box in all_boxes) / len(all_boxes)
                y = sum(box[1] for box in all_boxes) / len(all_boxes)
                w = sum(box[2] for box in all_boxes) / len(all_boxes)
                h = sum(box[3] for box in all_boxes) / len(all_boxes)
                merged.append((int(x), int(y), int(w), int(h)))
            else:
                merged.append(current)
                
        return merged

    def _clean_suspected_faces(self):
        current_time = time.time()
        self.suspected_faces = {key: value for key, value in self.suspected_faces.items() 
                                if current_time - value["last_seen"] < self.forget_time}

    def update(self, faces, bodies):
        faces = [tuple(map(int, face)) for face in faces]
        bodies = [tuple(map(int, body)) for body in bodies]
        
        self._clean_suspected_faces()

        # Merge overlapping detections
        faces = self._merge_overlapping_detections(faces)
        bodies = self._merge_overlapping_detections(bodies)

        # Separate new detections from already confirmed ones
        new_faces = []
        for face in faces:
            is_new = True
            for confirmed_face in self.confirmed_faces:
                if self._calculate_iou(face, confirmed_face) > self.revalidate_iou:
                    is_new = False
                    break
            if is_new:
                new_faces.append(face)

        # Update confirmed faces with newly detected faces
        self.confirmed_faces.update(new_faces)

        # Retain only valid faces in confirmed list
        self.confirmed_faces = {face for face in self.confirmed_faces 
                                if any(self._calculate_iou(face, current) > self.revalidate_iou 
                                       for current in faces)}

        # Update confirmed bodies
        self.confirmed_bodies = {body for body in self.confirmed_bodies 
                                 if any(self._calculate_iou(body, current) > self.iou_threshold 
                                        for current in bodies)}

        self.confirmed_bodies.update(bodies)

        return list(self.confirmed_faces), list(self.confirmed_bodies)
