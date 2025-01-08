import cv2
import time


class MainCam:

    cap =""
    lastFrame= None

    def __init__(self) :
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise Exception("No cam available")
        
        
    def captureFrame(self):
        ret, frame = self.cap.read()

        if not ret:
            raise Exception("Error: Cannot read the frame.")
        else:
            return frame
    
    def showFrame(self,frame) :
       
        while True:
            cv2.imshow('Feed', frame)

            if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty('Feed', cv2.WND_PROP_VISIBLE) < 1:
                break
        # Release the capture and destroy all OpenCV windows
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__" :


    print("Working Test 1 : cam test")
    testCam=MainCam()
    frame=testCam.captureFrame()
    testCam.showFrame(frame)