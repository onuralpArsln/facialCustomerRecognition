from proto5.main.cameraControls import Camera
import time


camera = Camera()

while True:
    
    camera.getImage()
    
    camera.displayFrame(fps=10)