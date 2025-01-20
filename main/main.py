from cameraControls import Camera
from mediaBorusuTahminci import MediaBorusuTahminci
from frameWorks import frameWorks
import time
from deneme import App



App()
'''
camera = Camera()
mbt=MediaBorusuTahminci()
fw=frameWorks()





while True:
    
    
    
    camera.getImage()

    locations=mbt.tahmin(camera.lastFrame)

    frame=fw.drawBoundingBox(detectionsFromMbt=locations,frame=camera.lastFrame)

    faces=fw.splitFaces(frame=frame)
  
    fw.fwFacade(camera.lastFrame,locations)


    camera.displayFrame(frame,fps=10)

    frame1=fw.fwFacade(camera.lastFrame,locations)

    #camera.displayFrame(additionalFrames=faces, frame=frame,fps=10)
    camera.displayFrame( frame=frame1,fps=10)
    


'''