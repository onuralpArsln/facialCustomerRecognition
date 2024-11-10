import cv2
from pathlib import Path

cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)

if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False
    print("not working")

while rval:
    cv2.imshow("preview", frame)
    rval, frame = vc.read()
    key = cv2.waitKey(20)
    print("working")
    if key == 27: # exit on ESC
        break

cv2.destroyWindow("preview")
vc.release()