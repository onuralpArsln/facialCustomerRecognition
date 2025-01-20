import tkinter as tk
from tkinter import ttk
import requests
from tkcalendar import Calendar
from PIL import Image, ImageTk
import threading
from cameraControls import Camera  # Camera sınıfını buradan içe aktar
from mediaBorusuTahminci import MediaBorusuTahminci
from frameWorks import frameWorks
from database import FirebaseHandler
import time
from datetime import datetime
import io

class App:
    def __init__(self, root):
        
        self.camera = Camera()
        self.mbt = MediaBorusuTahminci()
        self.fw = frameWorks()
        self.db = FirebaseHandler()
        

        self.root = root
        self.root.title("Kamera Uygulaması")
        self.root.state('zoomed')  # Fullscreen but with window controls
        self.root.resizable(False, False)

        # Kamera nesnesi
        #self.camera = Camera()

        # Görüntü ekranı
        self.video_label = tk.Label(self.root, bg="black")
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Geçmiş butonu
        self.history_button = ttk.Button(self.root, text="Geçmiş", command=self.add_customer)
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
    
    def add_customer(self):
        now = datetime.now()
        self.db.upload_image_and_save_data(self.camera.lastFrame, str(now), "images")

    def open_history(self):
        # Yeni bir pencere oluştur
        history_window = tk.Toplevel(self.root)
        history_window.title("Geçmiş")
        history_window.geometry("1500x800")

        # Sol çerçeve (Takvim ve giriş sayısı etiketi)
        left_frame = tk.Frame(history_window, width=360)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=20, pady=20)

        # Sol çerçeveyi ortalamak için bir boşluk çerçevesi ekleyin
        left_frame.pack_propagate(False)
        spacer_top = tk.Frame(left_frame, height=200)
        spacer_top.pack(side=tk.TOP, fill=tk.X)

        # Takvim
        self.calendar = Calendar(left_frame, selectmode='day')
        self.calendar.pack(side=tk.TOP, padx=20, pady=20)

        # Giriş sayısı etiketi
        self.entry_count_var = tk.StringVar()
        self.entry_count_var.set("Giriş sayısı: 0")
        entry_count_label = tk.Label(left_frame, textvariable=self.entry_count_var, font=("Montserrat", 12))
        entry_count_label.pack(side=tk.TOP, padx=20, pady=10)


        # Sağ çerçeve (Kaydırılabilir Canvas)
        right_frame = tk.Frame(history_window, width=840)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        canvas = tk.Canvas(right_frame)
        scrollbar = ttk.Scrollbar(right_frame, orient=tk.VERTICAL, command=canvas.yview)
        self.tree_frame = tk.Frame(canvas)

        self.tree_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=self.tree_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        def update_table(event=None):
            selected_date = self.calendar.selection_get()
            records = self.db.get_data_by_date("images", selected_date)
            self.entry_count_var.set(f"Giriş sayısı: {len(records)}")
            # Tabloyu temizle (Canvas içindeki tüm elemanları yok et)
            for widget in self.tree_frame.winfo_children():
                widget.destroy()

            # Görsel boyutu
            image_width, image_height = 200, 200

            # Satıra 3 görsel yerleştirme
            for idx, record in enumerate(records):
                photo_url = record['image_url']
                time = record['timestamp'].strftime("%H:%M")  # Sadece saat
                count = record.get('count', 0)

                try:
                    # Fotoğrafı indir ve göster
                    response = requests.get(photo_url)
                    img_data = response.content
                    with Image.open(io.BytesIO(img_data)) as img:
                        img = img.resize((image_width, image_height))
                        imgtk = ImageTk.PhotoImage(img)

                    # Yeni bir satır oluştur (Her 3 görselde bir)
                    if idx % 4 == 0:
                        row_frame = tk.Frame(self.tree_frame, bg="white", relief=tk.FLAT)
                        row_frame.pack(fill=tk.X, padx=10, pady=10)

                    # Görsel kutusu
                    image_frame = tk.Frame(row_frame, bg="white", relief=tk.RAISED, borderwidth=1)
                    image_frame.pack(side=tk.LEFT, padx=10, pady=10)

                    # Fotoğraf
                    photo_label = tk.Label(image_frame, image=imgtk, bg="white")
                    photo_label.image = imgtk  # Fotoğraf referansını sakla
                    photo_label.pack()

                    # Metin bilgileri
                    info_label = tk.Label(
                        image_frame, 
                        text=f"Saat: {time}\nSayı: {count}", 
                        bg="white", 
                        justify=tk.CENTER
                    )
                    info_label.pack()

                except Exception as e:
                    print(f"Fotoğraf yüklenirken hata: {e}")
        # Takvimde tarih seçildiğinde tabloyu güncelle
        self.calendar.bind("<<CalendarSelected>>", update_table)
        update_table()


    def on_close(self):
        # Pencere kapatılırken yapılacak işlemler
        self.running = False
        #self.update_thread.join()  # Thread'i sonlandır
        #del self.camera  # Kamera nesnesini temizle
        self.root.destroy()  # Tkinter ana penceresini kapat


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)  # Pencere kapatma işlemi
    root.mainloop()
