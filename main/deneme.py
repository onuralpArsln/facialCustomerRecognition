import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import threading
from cameraControls import Camera  # Camera sınıfını buradan içe aktar
from mediaBorusuTahminci import MediaBorusuTahminci
from frameWorks import frameWorks
import time


class App:
    def __init__(self, root):

        self.camera = Camera()
        self.mbt = MediaBorusuTahminci()
        self.fw = frameWorks()

        self.root = root
        self.root.title("Kamera Uygulaması")
        self.root.attributes("-fullscreen", True)
        self.root.resizable(False, False)

        # Kamera nesnesi
        self.camera = Camera()

        # Görüntü ekranı
        self.video_label = tk.Label(self.root, bg="black")
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Geçmiş butonu
        self.history_button = ttk.Button(self.root, text="Geçmiş", command=self.show_history)
        self.history_button.pack(pady=10)

        # Kamerayı güncellemek için bir thread başlatıyoruz
        self.running = True
        self.update_thread = threading.Thread(target=self.update_frame, daemon=True)
        self.update_thread.start()

    def update_frame(self):
        while self.running:
            try:
                time.sleep(0.1)
                # Kamera görüntüsünü al
                frame = self.camera.getImage()
                # Görüntüyü RGB formatına çevir ve boyutlandır
                locations=self.mbt.tahmin(self.camera.lastFrame)

                frame=self.fw.drawBoundingBox(detectionsFromMbt=locations,frame=self.camera.lastFrame,label="salak")
  
                frame = frame[:, :, ::-1]  # BGR'den RGB'ye çevir
                img = Image.fromarray(frame)
                img = img.resize((1600, 1000))  # Görüntüyü pencereye sığdır
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)
            except Exception as e:
                print(f"Hata: {e}")
            self.root.update_idletasks()

    def show_history(self):
        # "Geçmiş" butonuna basıldığında yapılacak işlem
        print("Geçmiş butonuna tıklandı!")

    def on_close(self):
        # Pencere kapatılırken yapılacak işlemler
        self.running = False
        self.update_thread.join()  # Thread'i sonlandır
        del self.camera  # Kamera nesnesini temizle
        self.root.destroy()  # Tkinter ana penceresini kapat


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)  # Pencere kapatma işlemi
    root.mainloop()
