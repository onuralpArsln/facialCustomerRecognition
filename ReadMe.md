# Facial Customer Recognition FCR

Real time image processing adjusted for low spec hardware to track and learn about your visitors. It can learn people to analyze who visits when to generate data about customer trends.


Düşük donanımlar için gerçek zamanlı görüntü işleme sistemi. Kişilerin yüzlerini tanıyarak müşteri eğlimlerini tespit etmeyi hedefler.

# Önemli özellikler
Google Mediapipe ile yüz tepiti
Adam Geitgey Face_Recognition ile kişileri tespit etme / tanıma 
Kişileri loglama  ve öğrenme




# Setup 
NOT SUPPORTED IN Windows -   go unix 

### Akıl sağlığın için python3.10da çalış
Bol bol kaostan sonra 3.10 sonrası için önermiyorum. 3.11 umut verip beklenmedik anda vurdu, 3.12 üzücü bir deneyim sundu. 3.13 ise saç ekim klinikleri için gizli görevde çalışıyor olabilir.

### virtual env olmadan muhtemelen çalışmayacak -> VENV İLE ÇALIŞTI 
Farklı cihazlarda ve platformlarda bir çok sorun yaşadıktan sonra venv olmadan çalıştırmayı denemeyi önermiyorum. 

``` bash 
sudo apt install python3-venv
python3.10 -m venv .venv
source .venv/bin/activate

deactivate
```


## Temel gereksinimler

``` bash 
sudo apt update
sudo apt install build-essential python3-dev libatlas-base-dev libopenblas-dev
pip3 install --upgrade pip setuptools wheel
sudo apt install python3-apt python3-apt-dev
```

### Eğer garip bir apt package hatası alırsan

``` bash 
cd /usr/lib/python3/dist-packages
cp apt_pkg.cpython-34m-i386-linux-gnu.so apt_pkg.so
```
ikinci satır muhtemelen sende farklı olacak o yüzden biraz tab sihri ile otomatik tamamlat




### Temel gereksinimler devam

```bash 
sudo apt install build-essential python3-dev libatlas-base-dev libopenblas-dev
pip3 install --upgrade pip setuptools wheel
sudo apt-get install build-essential cmake pkg-config
sudo apt-get install libx11-dev libatlas-base-dev
sudo apt-get install libgtk-3-dev libboost-python-dev
```


## Proje Gereksinimleri

```bash 
pip3 install --upgrade pip
pip3 install numpy
pip3 install opencv-python
pip3 install dlib
pip3 install face-recognition
```

bunlar dümdüz çalıştı ama versiyonlar gerekirse

```bash
pip3 install --upgrade pip
pip3 install dlib==19.24.1
pip3 install face-recognition==1.3.0
pip3 install opencv-python==4.8.1.78
pip3 install numpy==1.24.3
sudo apt-get install python3-qt5 
```


## Daha büyük gereksinimler 

Burası biraz şenlikli yaklaşımına göre

```bash 
pip3 install face-recognition 
pip3 install opencv-python
pip3 install git+https://github.com/ageitgey/face_recognition_models
```
or 

```bash
python3.10 -m pip install face-recognition 
python3.10 -m pip install opencv-python 
python3.10 -m pip install git+https://github.com/ageitgey/face_recognition_models
```

Opencv düzgün yüklenmezse sil yükle sil yükle 
```bash
sudo apt-get install python3-opencv
sudo apt-get remove python3-opencv
```



# Projedeki Classlar Dökümantasyon kısmı 

## Camera from cameraControls.py

Amacı kameranın kontrolüdür

obje oluşunca kamerayı açar, 
getImage methodu ile bir  görüntü alır bunu hem kendi içinde lastFrame yapısında saklar hem de döndürür

displayImage ile istenilen görüntü gösterilir eğer parametre verilmez default değerlerinde lastFrame gösterir bekleme süresi ms cinsinden