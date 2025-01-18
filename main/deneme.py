import tkinter as tk
from tkinter import ttk
from tkcalendar import Calendar
from PIL import Image, ImageTk
import threading
from cameraControls import Camera  # Camera sınıfını buradan içe aktar
from mediaBorusuTahminci import MediaBorusuTahminci
from frameWorks import frameWorks
from database import FacialCustomerRecognitionDatabase
import time
from datetime import datetime
import io


class App:
    def __init__(self, root):

        self.camera = Camera()
        self.mbt = MediaBorusuTahminci()
        self.fw = frameWorks()
        self.db = FacialCustomerRecognitionDatabase()

        self.db.create_table()

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
        #9a = 0
        while self.running:
            #a = a +1
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
                '''
                if a == 50:
                    self.add_customer(image=self.camera.lastFrame)
                    a = 0
                '''
            except Exception as e:
                print(f"Hata: {e}")
            self.root.update_idletasks()

    def add_customer(self, image):
        # Bugünün tarihini ve saatini al
        now = datetime.now()
        date_str = now.strftime("%d/%m/%Y")
        time_str = now.strftime("%H:%M:%S")

        # Veritabanına müşteri ekle
        self.db.add_customer(photo=image, count=0, day=date_str, time=time_str)
        print("Kaydettik Kardeşiiiiiiim")

    def show_history(self):
        history_window = tk.Toplevel(self.root)
        history_window.title("Geçmiş")
        history_window.geometry("1400x1000")

        # Sol tarafta bir takvim oluştur
        cal = Calendar(history_window, selectmode='day', date_pattern='dd/MM/yyyy')
        cal.grid(row=0, column=0, padx=20, pady=20, sticky="n")

        # Sağ tarafta bir liste çerçevesi oluştur
        history_frame = ttk.Frame(history_window)
        history_frame.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")

        # Fotoğrafları ve diğer bilgileri göstermek için bir Canvas ve Scrollbar ekleyelim
        canvas = tk.Canvas(history_frame)
        scrollbar = ttk.Scrollbar(history_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self.photo_labels = []  # Fotoğraf widget'larını tutmak için bir liste

        def update_history():
            selected_date = cal.get_date()
            records = self.db.get_customers_by_date(selected_date)

            # Önceki içerikleri temizle
            for widget in scrollable_frame.winfo_children():
                widget.destroy()

            self.photo_labels.clear()

            # Yeni içerikleri ekle
            for record in records:
                photo_data = record[1]  # Fotoğraf verisi (BLOB)
                time_str = record[3]  # Saat bilgisi
                count = record[4]  # Sayı bilgisi

                # Fotoğrafı yükle ve göster
                photo = Image.open(io.BytesIO(photo_data))  # BLOB verisini oku
                photo = photo.resize((300, 300))
                photo_img = ImageTk.PhotoImage(photo)

                photo_label = tk.Label(scrollable_frame, image=photo_img)
                photo_label.image = photo_img  # Referansı sakla, yoksa görüntü kaybolur
                photo_label.pack(padx=10, pady=5, side="top")

                # Diğer bilgileri göster
                info_label = tk.Label(scrollable_frame, text=f"Saat: {time_str} | Sayı: {count}")
                info_label.pack(padx=10, pady=5, side="top")

                self.photo_labels.append(photo_label)

        # Takvimde tarih seçildiğinde güncelleme yap
        cal.bind("<<CalendarSelected>>", lambda e: update_history())

        # Varsayılan olarak bugünün tarihini seç ve listeyi doldur
        cal.selection_set(datetime.now().strftime("%d/%m/%Y"))
        update_history()

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
