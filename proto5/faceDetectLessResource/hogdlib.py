import cv2
import dlib
import time

# Dlib yüz dedektörünü başlat
detector = dlib.get_frontal_face_detector()

# Video akışını başlat
cap = cv2.VideoCapture(0)

while True:
    start_time = time.time()

    ret, frame = cap.read()
    if not ret:
        break

    # Çözünürlüğü düşür
    frame = cv2.resize(frame, (320, 240))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Yüz algılama
    faces = detector(gray)

    # Tespit edilen yüzleri çiz
    for face in faces:
        x, y, w, h = (face.left(), face.top(), face.width(), face.height())
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    end_time = time.time()
    fps = 1 / (end_time - start_time)

    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    cv2.imshow("HOG + Dlib Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
