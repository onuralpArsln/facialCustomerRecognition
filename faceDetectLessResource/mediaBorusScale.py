import cv2
import mediapipe as mp

# MediaPipe bileşenlerini oluştur
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Video kaynağı (Webcam veya video dosyası)
cap = cv2.VideoCapture(0)  # Eğer 0 sorun yaratırsa, video dosyasının yolunu buraya koy

# Ölçeklendirme faktörü (0.5 = yarı boyut)
scale_factor = 1

# MediaPipe yüz tespiti başlat
with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Kamera görüntüsü alınamadı.")
            break
        
        # Orijinal görüntü boyutlarını al
        height, width = frame.shape[:2]
        
        # Görüntüyü küçült
        small_frame = cv2.resize(frame, (int(width * scale_factor), int(height * scale_factor)))
        
        # OpenCV'nin BGR formatını RGB'ye dönüştür
        small_frame_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        # MediaPipe ile yüz tespiti yap
        results = face_detection.process(small_frame_rgb)
        
        # Yüzleri orijinal görüntü üzerine işaretle
        if results.detections:
            for detection in results.detections:
                # Koordinatları orijinal boyuta geri ölçeklendir
                bboxC = detection.location_data.relative_bounding_box
                ih, iw = small_frame.shape[:2]
                
                # Koordinatları orijinal boyuta dönüştür
                bbox = int(bboxC.xmin * iw / scale_factor), int(bboxC.ymin * ih / scale_factor), \
                       int(bboxC.width * iw / scale_factor), int(bboxC.height * ih / scale_factor)
                
                # Orijinal görüntü üzerine çiz
                cv2.rectangle(frame, 
                            (bbox[0], bbox[1]), 
                            (bbox[0] + bbox[2], bbox[1] + bbox[3]), 
                            (0, 255, 0), 2)
        
        # Sonuçları göster
        cv2.imshow("MediaPipe Tespiti", frame)
        
        # Çıkış için 'q' tuşuna bas
        if cv2.waitKey(150) & 0xFF == ord('q'):
            break

# Kaynakları serbest bırak
cap.release()
cv2.destroyAllWindows()