import multiprocessing as mp
import time
import os


class FaceRemember:
    _images=[]
    _pathImages=""


    def __init__(self,path="imgs"):
        print("initted gen")
        self._pathImages=path
        self._genProcess = mp.Process(target=self.subprocessController)
        self._genProcess.start()
        print("succesful start")
        


            
    def getImgsFromDir(self):
        directory_path=self._pathImages
        try:
            # Get all files in the directory with .png extension
            png_files = [file for file in os.listdir(directory_path) if file.endswith(".png")]
            return png_files
        except FileNotFoundError:
            print(f"The directory '{directory_path}' does not exist.")
            return []

    def subprocessController(self):
        while True:
            time.sleep(1)
            self.subprocessFlow()

    
    def subprocessFlow(self):
        newImgs=set(self.getImgsFromDir())-set(self._images)
        if newImgs == None:return
        if len(newImgs) == 0 : return
        for i in newImgs:
            # add process hereü-
            time.sleep(1)
            print("img name "+i)
            self._images.append(i)

if __name__=="__main__":
    print("Test Init")
    test=FaceRemember()
    print(test.getImgsFromDir())

