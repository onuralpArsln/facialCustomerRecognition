import asyncio
import multiprocessing
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
import multiprocessing as mp
import asyncio
import os
import sqlite3

#db = FirebaseHandler()

class App:
    def __init__(self, root):
        
        self.a = 0
        self.camera = Camera()
        self.mbt = MediaBorusuTahminci()
        self.fw = frameWorks()
        
        self.root = root
        self.root.title("Kamera Uygulaması")
        self.root.state('zoomed')  # Fullscreen but with window controls
        self.root.resizable(False, False)

        # Görüntü ekranı
        self.video_label = tk.Label(self.root, bg="black")
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        self.history_window = None
        # Geçmiş butonu
        self.history_button = ttk.Button(self.root, text="Geçmiş", command=self.open_history)
        self.history_button.pack(pady=10)

        self.add_already_customer()
        # Kamerayı güncellemek için bir thread başlatıyoruz
        self.running = True
        self.update_frame()
        
    def update_frame(self):
        #self.a = self.a +1
        if not self.running:
            return
        try:
            # Kamera görüntüsünü al
            frame = self.camera.getImage()
            if frame is None:
                self.root.after(100, self.update_frame)
                return
            # Görüntüyü RGB formatına çevir ve boyutlandır
            locations = self.mbt.tahmin(self.camera.lastFrame)
            frame = self.fw.drawBoundingBox(detectionsFromMbt=locations, frame=self.camera.lastFrame, label="salak")
            frame = frame[:, :, ::-1]  # BGR'den RGB'ye çevir
            '''
            if self.a == 100:
                self.add_customer(frame)
                print("foto çekildi")
                self.a = 0
            '''
            img = Image.fromarray(frame)
            img = img.resize((1600, 1000))  # Görüntüyü pencereye sığdır
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
        except Exception as e:
            pass
        self.root.after(100, self.update_frame)
    
    

    def open_history(self):

        if (self.history_window and self.history_window.winfo_exists()) :
            if self.history_window.state() == "iconic":
                self.history_window.deiconify()
            self.history_window.lift()  # Pencereyi öne getirir
            self.history_window.focus()  # O pencereye odaklanır
            return



        # Yeni bir pencere oluştur
        self.history_window = tk.Toplevel(self.root)
        self.history_window.title("Geçmiş")
        self.history_window.geometry("1500x800")

        # Sol çerçeve (Takvim ve giriş sayısı etiketi)
        left_frame = tk.Frame(self.history_window, width=360)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=20, pady=20)

        # Sol çerçeveyi ortalamak için bir boşluk çerçevesi ekleyin
        left_frame.pack_propagate(False)
        spacer_top = tk.Frame(left_frame, height=200)
        spacer_top.pack(side=tk.TOP, fill=tk.X)

        # Takvim
        self.calendar = Calendar(left_frame, selectmode='day', locale='tr_TR')
        self.calendar.pack(side=tk.TOP, padx=20, pady=20)

        # Giriş sayısı etiketi
        self.entry_count_var = tk.StringVar()
        self.entry_count_var.set("Giriş sayısı: 0")
        entry_count_label = tk.Label(left_frame, textvariable=self.entry_count_var, font=("Montserrat", 12))
        entry_count_label.pack(side=tk.TOP, padx=20, pady=10)

        # Sağ çerçeve (Kaydırılabilir Canvas)
        right_frame = tk.Frame(self.history_window, width=840)
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
            records = []
            selected_date_str = selected_date.strftime("%Y-%m-%d")

            # Veritabanı bağlantısı oluştur
            conn = sqlite3.connect('images.db')
            cursor = conn.cursor()

            # Seçilen tarihe göre kayıtları getir
            cursor.execute('''
                SELECT img_path, counter, first_seen, last_seen
                FROM images
                WHERE DATE(first_seen) = ? 
            ''', (selected_date_str))

            rows = cursor.fetchall()

            for row in rows:
                img_path, counter, first_seen, last_seen = row
                records.append({
                    'image_url': img_path,
                    'timestamp': datetime.strptime(first_seen, "%Y-%m-%d %H:%M:%S.%f"),
                    'count': counter
                })

            # Veritabanı bağlantısını kapat
            conn.close()

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
                    # Fotoğrafı klasörden oku ve göster
                    with Image.open(photo_url) as img:
                        img = img.resize((image_width, image_height))
                        imgtk = ImageTk.PhotoImage(img)

                    # Yeni bir satır oluştur (Her 4 görselde bir)
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
        '''
        jobs = []
        p = multiprocessing.Process(target=update_table, args=(self,))
        jobs.append(p)
        p.start()
        '''
        
        # Takvimde tarih seçildiğinde tabloyu güncelle
        self.calendar.bind("<<CalendarSelected>>", update_table)
        update_table()
    
    def add_image_to_db(self, img_path, first_seen=datetime.now(), last_seen=datetime.now()): 

        # Veritabanı bağlantısı oluştur
        conn = sqlite3.connect('images.db')
        cursor = conn.cursor()

        # Tabloyu oluştur (eğer yoksa)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS images (
                img_path TEXT PRIMARY KEY,
                counter INTEGER,
                first_seen DATE,
                last_seen DATE
            )
        ''')

        # Görüntünün mevcut olup olmadığını kontrol et
        cursor.execute('SELECT * FROM images WHERE img_path = ?', (img_path,))
        record = cursor.fetchone()
        
        if record:
            last_seen_time = datetime.strptime(record[3], "%Y-%m-%d %H:%M:%S.%f")
            if (datetime.now() - last_seen_time).total_seconds() > 300:
                cursor.execute('''
                    UPDATE images
                    SET counter = counter + 1, last_seen = ?
                    WHERE img_path = ?
                ''', (datetime.now(), img_path))
            else:
                cursor.execute('''
                    UPDATE images
                    SET last_seen = ?
                    WHERE img_path = ?
                ''', (datetime.now(), img_path))
        else:
            # Görüntü mevcut değilse, yeni bir kayıt ekle
            cursor.execute('''
                INSERT INTO images (img_path, counter, first_seen, last_seen)
                VALUES (?, ?, ?, ?)
            ''', (img_path, 1, datetime.now(), datetime.now()))

        # Değişiklikleri kaydet ve bağlantıyı kapat
        conn.commit()
        conn.close()

    def on_close(self):
        # Pencere kapatılırken yapılacak işlemler
        self.running = False
        #self.update_thread.join()  # Thread'i sonlandır
        #del self.camera  # Kamera nesnesini temizle
        self.root.destroy()  # Tkinter ana penceresini kapat

    def add_already_customer(self):
        records = []
        image_folder = "prodcution/imgs"

        conn = sqlite3.connect('images.db')
        cursor = conn.cursor()

        # Tabloyu oluştur (eğer yoksa)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS images (
                img_path TEXT PRIMARY KEY,
                counter INTEGER,
                first_seen DATE,
                last_seen DATE
            )
        ''')

        for filename in os.listdir(image_folder):
            if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
                file_path = os.path.join(image_folder, filename)
                try:
                    with Image.open(file_path) as img:
                        file_stats = os.stat(file_path)
                        creation_time = datetime.fromtimestamp(file_stats.st_mtime)
                        if creation_time:
                            date_str = str(creation_time)
                            print(date_str)
                            if date_str:
                                file_date = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S.%f").date()
                                print("appenddeyim")
                                records.append({
                                        'image_url': file_path,
                                        'timestamp': datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S.%f"),
                                        'count': 1
                                })
                                cursor.execute('''
                                INSERT INTO images (img_path, counter, first_seen, last_seen)
                                VALUES (?, ?, ?, ?)
                            ''', (file_path, 1, datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S.%f"), datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S.%f")))
                except Exception as e:
                    print(f"Error reading EXIF data from {filename}: {e}")



if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)  # Pencere kapatma işlemi
    root.mainloop()
    