import asyncio
import multiprocessing
import tkinter as tk
from tkinter import ttk
import requests
from tkcalendar import Calendar
from PIL import Image, ImageTk
import threading
import time
from datetime import datetime
import io
import multiprocessing as mp
import asyncio
import os
import cv2
import mediapipe as mp
import face_recognition
import os
import numpy as np
import re


#db = FirebaseHandler()
# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

        # Folder to store known faces
imgs_folder = "imgs"
os.makedirs(imgs_folder, exist_ok=True)

        # Global known face lists
known_face_encodings = []
known_face_names = []

class App:
    

    def __init__(self, root):
        
        self.a = 0
        
        self.root = root
        self.root.title("Kamera Uygulaması")
        self.root.state('normal')  # Fullscreen but with window controls
        self.root.resizable(False, False)

        # Görüntü ekranı
        self.video_label = tk.Label(self.root, bg="black")
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        self.history_window = None
        # Geçmiş butonu
        self.history_button = ttk.Button(self.root, text="Geçmiş", command=self.open_history)
        self.history_button.pack(pady=10)


        # Kamerayı güncellemek için bir thread başlatıyoruz
        self.running = True
        self.load_known_faces()
        self.update_frame()
    
    def load_known_faces(self):
        """Loads existing faces from the imgs/ folder."""
        global known_face_encodings, known_face_names
        known_face_encodings.clear()
        known_face_names.clear()

        for file_name in os.listdir(imgs_folder):
            img_path = os.path.join(imgs_folder, file_name)
            known_image = face_recognition.load_image_file(img_path)
            encodings = face_recognition.face_encodings(known_image)

            if encodings:
                known_face_encodings.append(encodings[0])
                known_face_names.append(file_name.split('.')[0])  # Use filename as label

    def get_next_face_number(self):
        """Finds the next available new_face_X number."""
        existing_numbers = []
        pattern = re.compile(r"new_face_(\d+)\.jpg")

        for filename in os.listdir(imgs_folder):
            match = pattern.match(filename)
            if match:
                existing_numbers.append(int(match.group(1)))

        return max(existing_numbers, default=0) + 1  # Next number

    def update_frame(self, show_windows=True):
        """Detects faces, recognizes known ones, and saves new faces if unknown.
        
        Args:
            show_windows (bool): If True, displays OpenCV windows; otherwise, runs silently.
        """
          # Load known faces before starting
        try:
            cap = cv2.VideoCapture(0)
            
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(rgb_frame)

            if results.detections:
                for detection in results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = frame.shape

                    x1 = int(bboxC.xmin * iw)
                    y1 = int(bboxC.ymin * ih)
                    w = int(bboxC.width * iw)
                    h = int(bboxC.height * ih)

                    # Adjust bounding box
                    center_x = x1 + w // 2
                    center_y = y1 + h // 2
                    new_w = int(1.4 * w)
                    new_h = int(1.6 * h)
                    new_x1 = max(0, center_x - new_w // 2)
                    new_y1 = max(0, center_y - new_h // 2)
                    new_x2 = min(iw, new_x1 + new_w)
                    new_y2 = min(ih, new_y1 + new_h)

                    # Crop and check validity
                    face_crop = frame[new_y1:new_y2, new_x1:new_x2]
                    if face_crop.size == 0:
                        continue  # Skip invalid crops

                    # Convert to RGB before encoding
                    face_crop_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)

                    face_encodings = face_recognition.face_encodings(face_crop_rgb)
                    if not face_encodings:  
                        continue  # Skip if no face encoding is found

                    face_encoding = face_encodings[0]

                    # Compare with known faces
                    name = "Unknown"
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

                    best_match_index = np.argmin(face_distances) if matches else None

                    if best_match_index is not None and matches[best_match_index]:
                        name = known_face_names[best_match_index]
                    else:
                        # Get the next available new_face_X number
                        new_face_id = self.get_next_face_number()
                        new_face_path = os.path.join(imgs_folder, f"new_face_{new_face_id}.jpg")
                        cv2.imwrite(new_face_path, face_crop)
                        print(f"New face saved as {new_face_path}")

                        # Add to known faces
                        known_face_encodings.append(face_encoding)
                        known_face_names.append(f"new_face_{new_face_id}")

                        name = f"new_face_{new_face_id}"
                    '''
                    # Draw bounding box and name if windows are enabled
                    if show_windows:
                        cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)
                        cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        cv2.imshow('Cropped Face', face_crop)
                    '''
            # Show the frame if windows are enabled
            if show_windows:
                frame = frame[:, :, ::-1]  # BGR'den RGB'ye çevir
                img = Image.fromarray(frame)
                img = img.resize((1200, 700))  # Görüntüyü pencereye sığdır
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)
        except Exception as e:
            pass
        self.root.after(10, self.update_frame)
    
    
    
    
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

            for filename in os.listdir(imgs_folder):
                if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
                    file_path = os.path.join(imgs_folder, filename)
                    try:
                        with Image.open(file_path) as img:
                            file_stats = os.stat(file_path)
                            creation_time = datetime.fromtimestamp(file_stats.st_mtime)
                            if creation_time:
                                date_str = str(creation_time)
                                print(date_str)
                                if date_str:
                                    file_date = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S.%f").date()
                                    if file_date == selected_date:
                                        print("appenddeyim")
                                        records.append({
                                            'image_url': file_path,
                                            'timestamp': datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S.%f"),
                                            'count': 1
                                        })
                    except Exception as e:
                        print(f"Error reading EXIF data from {filename}: {e}")

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
    