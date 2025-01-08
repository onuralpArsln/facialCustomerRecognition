import cv2
import mediapipe as mp

# MediaPipe bileşenlerini oluştur
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Video kaynağı (Webcam veya video dosyası)
cap = cv2.VideoCapture(0)  # Eğer 0 sorun yaratırsa, video dosyasının yolunu buraya koy

# MediaPipe yüz tespiti başlat
with mp_face_detection.FaceDetection(min_detection_confidence=0.95) as face_detection:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Kamera görüntüsü alınamadı.")
            break
        
        # OpenCV'nin BGR formatını RGB'ye dönüştür
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # MediaPipe ile yüz tespiti yap
        results = face_detection.process(frame_rgb)
        
        # Yüzleri işaretle
        if results.detections:
            for detection in results.detections:
                mp_drawing.draw_detection(frame, detection)
        
        # Sonuçları göster
        cv2.imshow("MediaPipe Yüz Tespiti", frame)
        
        # Çıkış için 'q' tuşuna bas
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

# Kaynakları serbest bırak
cap.release()
cv2.destroyAllWindows()
