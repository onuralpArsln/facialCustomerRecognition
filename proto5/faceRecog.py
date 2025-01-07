import cv2
import face_recognition






if __name__ == "__main__" :
    img = cv2.imread("img_steveHarris.jpg")
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_encoding = face_recognition.face_encodings(rgb_img)[0]

    img2 = cv2.imread("img_steveHarris2.jpg")
    rgb_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    img_encoding2 = face_recognition.face_encodings(rgb_img2)[0]

    img3 = cv2.imread("img_brucedickinson.jpg")
    rgb_img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
    img_encoding3 = face_recognition.face_encodings(rgb_img3)[0]

    img4 = cv2.imread("img_bruce2.jpg")
    rgb_img4 = cv2.cvtColor(img4, cv2.COLOR_BGR2RGB)
    img_encoding4 = face_recognition.face_encodings(rgb_img4)[0]

    result = face_recognition.compare_faces([img_encoding], img_encoding2)
    print("Expected True Result: ", result)

    result = face_recognition.compare_faces([img_encoding3], img_encoding2)
    print("Expected false Result: ", result)

    result = face_recognition.compare_faces([img_encoding3], img_encoding4)
    print("Expected True Result: ", result)