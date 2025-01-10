from cameraControls import Camera
from mediaBorusuTahminci import MediaBorusuTahminci
from frameWorks import frameWorks
import time


camera = Camera()
mbt=MediaBorusuTahminci(confidence=0.7)
fw=frameWorks()



while True:
    
    camera.getImage()

    locations=mbt.tahmin(camera.lastFrame)

    #print(locations[0].location_data.relative_bounding_box)

    frame=fw.drawBoundingBox(detectionsFromMbt=locations,frame=camera.lastFrame,label="s")

    
    camera.displayFrame(frame,fps=10)