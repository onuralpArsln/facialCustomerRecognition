import cv2

class Camera:
    """Camera Kontrol Sınıfı \n
    getImage() -> foto çek  \n
    displayFrame() -> fotoyu göster \n

    Returns:
        _type_: camera
    """
    lastFrame=None  # getImage tarafından gelen son görüntü
    cam=None        # Camera objesi tarafından seçilen kamera

    def __init__(self):
        self.cam = cv2.VideoCapture(0)


    def getImage(self,frame_width=None,frame_height=None):
        """
        Cameradan görüntü alır ve döndürür bunu direkt gösterebiliyon cv2  ile.
        Args:
            frame_width (int, optional): boş bırakırsan otamatik belirler istersen ver.
            frame_height (int, optional): boş bırakırsan otamatik belirler istersen ver.
        Returns:
            frame
        """
        if frame_width is None or frame_height is None:
            frame_width = int(self.cam.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if self.cam is not None:
            ret, self.lastFrame = self.cam.read()
        
        if not ret:
            raise("cv2.VideoCapture cam.read() pırtladı imdat : getImage.py - Camera Class - getImage method")

        return self.lastFrame

    def displayFrame(self , windowHeader: str = "Camera Feed", frame=None, fps: int= None, wait: int=1 ) ->  None:
        """
        Frame görüntülemek içindir eğer parametre vermezsen son çekilen resmi görüntüler
        Args:
            windowHeader: windowa isim ver vermezsen Camera Feed olur
            frame: görüntülenecek foto parametre vermezsen son foto
            wait: eğer görüntü ekranda beklkesin istersen, 0 sonsuz bekleme int vermek ms süreli en az 1 vermeslisn süreklide
        Returns:
            frame
        """
        if self.lastFrame is None and frame is None:
            raise("nothing to display on camera class display method, ya foto çek yada parametre ver")
        elif frame is None:
            frame=self.lastFrame
        
        cv2.imshow('windowHeader', frame)

        if fps is None and wait >=0 :
        # fps yok ise waite göre karar default wait 1 ms    
            key = cv2.waitKey(wait)
        elif wait<0:
            raise("Kanka displayFrame'de Wait=0 sonsuz bekleme wait=1 ms en hızlı daha azında cırtlıyor sistem öyle bakarsın sabaha kadar ne bozuldu diye ")
        elif fps>0:
            # eğer fps hesabı verdiyse yap hesabı
            calulatedWait = 1000//fps
            key= cv2.waitKey(calulatedWait)
        else:
            raise("Kanka displayFrame'de waiti az vermişin fpsi bozmuşun düzelt la CameraControls.DisplayFrame ")

            
        return


    def __del__(self):
        self.cam.release()
        cv2.destroyAllWindows()
    



if __name__ == "__main__":
    print(" resimi seçili tut ve tuşa bas sonraki ekran için")
    testObject = Camera()
    testObject.getImage()
    testObject.displayFrame()
    print(" ilk test tamamlandı : basit işlev ")
    testFrame=testObject.getImage()
    testObject.displayFrame("Test",testFrame,700)
    print(" ikinci test tamamlandı : dış parametreler ")
    for i in range(30):
        testObject.getImage()
        testObject.displayFrame("sürekli",wait=100)

    cv2.destroyAllWindows()
