# this test the following :
import cv2
# getImage
class MyClass:
    def __init__(self):
        print("Nesne oluşturuldu.")

    def __del__(self):
        print("Nesne yok ediliyor. Kaynaklar serbest bırakıldı.")



a = MyClass()
while True:
        if cv2.waitKey(1) == ord('q'):
            break