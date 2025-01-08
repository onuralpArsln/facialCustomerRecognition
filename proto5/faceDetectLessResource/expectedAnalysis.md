| **Yöntem**       | **Hız (FPS)** | **Doğruluk** | 
|-------------------|---------------|--------------|
| opencvdlib        | ??????        | iyi          | 
| haarcascade       | ??????        | kötü         | 
| hogdlib           | ??????        | iyimsikötümtrak| 
| opencvdnn         | ??????        | çok iyi        | 
| mediapipe (borusu)| ??????        | vallah en iyisi | 
 



yakalama başarısı 
opencvdlib > haarcascade
hogdlib ~ opencvdlib

opencvdnn baya iyi ama muhtemeln içinden geçiyor performansın onu bi şaapmak lazım 

mediapipe baya iyi çıktı he 
 ama onu biraz çözemek lazım 
media pipe baya iyi ama test edilkmeli acil işe yarayıosa çünkü süperişko
























| **Yöntem**       | **Hız (FPS)** | **Doğruluk** | **False Positive** | **Donanım Kullanımı** |
|-------------------|---------------|--------------|---------------------|------------------------|
| HOG (OpenCV)      | Yüksek        | Orta         | Yüksek              | Düşük                 |
| HOG + Dlib        | Orta          | Yüksek       | Düşük               | Orta                  |
| OpenCV DNN        | Düşük         | Çok Yüksek   | Çok Düşük           | Orta-Yüksek           |
| HOG (OpenCV)      | Yüksek        | Orta         | Yüksek              | Düşük                 |
| HOG + Dlib        | Orta          | Yüksek       | Düşük               | Orta                  |
| OpenCV DNN        | Düşük         | Çok Yüksek   | Çok Düşük           | Orta-Yüksek           |