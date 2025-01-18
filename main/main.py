from cameraControls import Camera
from mediaBorusuTahminci import MediaBorusuTahminci
from frameWorks import frameWorks
import time
from deneme import App




'''
camera = Camera()
mbt=MediaBorusuTahminci()
fw=frameWorks()





while True:
    
    
    
    camera.getImage()

    locations=mbt.tahmin(camera.lastFrame)

    frame=fw.drawBoundingBox(detectionsFromMbt=locations,frame=camera.lastFrame,label="salak")
  

    camera.displayFrame(frame,fps=10)
'''