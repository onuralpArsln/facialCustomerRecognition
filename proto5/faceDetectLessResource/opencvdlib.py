import cv2
import dlib

# Dlib yüz dedektörü
detector = dlib.get_frontal_face_detector()

# Video akışı başlat
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Çözünürlüğü düşür
    small_frame = cv2.resize(frame, (320, 240))

    # Dlib ile yüz algılama
    gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        x, y, w, h = (face.left(), face.top(), face.width(), face.height())
        cv2.rectangle(small_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow("Face Detection", small_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
