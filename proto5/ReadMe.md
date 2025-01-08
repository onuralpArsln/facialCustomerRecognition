# facial customer Recognition with people logger system

Bu aşamada amaç sadece tespit değil takip etmek ve kişileri tespit etmek

NOT SUPPORTED IN Windows  go unix 

# virtual env olmadan muhtemelen çalışmayacak  VENV İLE ÇALIŞTI 
# akıl sağlığın için python3.10da çalış

python3.10 -m venv .venv
source .venv/bin/activate

sudo apt install python3-venv
python3 -m venv .venv
source .venv/bin/activate
deactivate
bashte kullan
sudo apt update
sudo apt install build-essential python3-dev libatlas-base-dev libopenblas-dev
pip3 install --upgrade pip setuptools wheel



## to get face match working

sudo apt update
sudo apt install python3-apt python3-apt-dev
cd /usr/lib/python3/dist-packages
cp apt_pkg.cpython-34m-i386-linux-gnu.so apt_pkg.so

sudo apt install build-essential python3-dev libatlas-base-dev libopenblas-dev
pip3 install --upgrade pip setuptools wheel

sudo apt-get install build-essential cmake pkg-config
sudo apt-get install libx11-dev libatlas-base-dev
sudo apt-get install libgtk-3-dev libboost-python-dev

pip3 install --upgrade pip
pip3 install numpy
pip3 install opencv-python
pip3 install dlib
pip3 install face-recognition


pip3 install --upgrade pip
pip3 install dlib==19.24.1
pip3 install face-recognition==1.3.0
pip3 install opencv-python==4.8.1.78
pip3 install numpy==1.24.3
sudo apt-get install python3-qt5 


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


# Classlar

### Camera from cameraControls.py

Amacı kameranın kontrolüdür

obje oluşunca kamerayı açar, 
getImage methodu ile bir  görüntü alır bunu hem kendi içinde lastFrame yapısında saklar hem de döndürür

displayImage ile istenilen görüntü gösterilir eğer parametre verilmez default değerlerinde lastFrame gösterir bekleme süresi ms cinsinden