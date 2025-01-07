# facial customer Recognition with people logger system

Bu aşamada amaç sadece tespit değil takip etmek ve kişileri tespit etmek

NOT SUPPORTED IN Windows  go unix 

# virtual env olmadan muhtemelen çalışmayacak  VENV İLE ÇALIŞTI 

sudo apt install python3-venv
python3 -m venv .venv
source .venv/bin/activate
deactivate
bashte kullan
sudo apt update
sudo apt install build-essential python3-dev libatlas-base-dev libopenblas-dev
pip3 install --upgrade pip setuptools wheel

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
sudo apt-get install python3-opencv
sudo apt-get remove python3-opencv


python3.13 -m pip install face_recognition
python3.13 -m pip install git+https://github.com/ageitgey/face_recognition_models

external managed env ist  
sudo apt install pipx
pipx ensurepath
then use pipx to install 
# Classlar

### Camera from cameraControls.py

Amacı kameranın kontrolüdür

obje oluşunca kamerayı açar, 
getImage methodu ile bir  görüntü alır bunu hem kendi içinde lastFrame yapısında saklar hem de döndürür

displayImage ile istenilen görüntü gösterilir eğer parametre verilmez default değerlerinde lastFrame gösterir bekleme süresi ms cinsinden