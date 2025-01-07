# facial customer Recognition with people logger system

Bu aşamada amaç sadece tespit değil takip etmek ve kişileri tespit etmek

NOT SUPPORTED IN Windows  go unix 
# libs:
```bash 
pip3 install face-recognition 
pip3 install opencv-python
pip3 install git+https://github.com/ageitgey/face_recognition_models
```
or 

```bash
python3.13 -m pip install face-recognition 
python3.13 -m pip install opencv-python 
python3.13 -m pip install git+https://github.com/ageitgey/face_recognition_models
```

python3.13 -m pip install face_recognition
python3.13 -m pip install git+https://github.com/ageitgey/face_recognition_models


# Classlar

### Camera from cameraControls.py

Amacı kameranın kontrolüdür

obje oluşunca kamerayı açar, 
getImage methodu ile bir  görüntü alır bunu hem kendi içinde lastFrame yapısında saklar hem de döndürür

displayImage ile istenilen görüntü gösterilir eğer parametre verilmez default değerlerinde lastFrame gösterir bekleme süresi ms cinsinden