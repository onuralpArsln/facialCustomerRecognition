import firebase_admin
from firebase_admin import credentials, storage, firestore
from datetime import datetime

class FirebaseHandler:
    def __init__(self):

        # Firebase Admin SDK ile uygulamayı başlat
        self.cred = credentials.Certificate("firebase.json")
        firebase_admin.initialize_app(self.cred, {
            "storageBucket": "facerec-24861.firebasestorage.app"  # Proje ID'ni buraya yaz
        })
        self.db = firestore.client()
        self.bucket = storage.bucket()

    def upload_image_and_save_data(self, image, image_name, collection_name):
        
        blob = self.bucket.blob(f"images/{image_name}")
        #blob.upload_from_file(image)
        with open(image, "rb") as image_file:
            blob.upload_from_file(image_file)
        blob.make_public()
        image_url = blob.public_url
        timestamp = datetime.now()
        data = {
            'image_url': image_url,
            'timestamp': timestamp,
            'count': 1
        }
        self.db.collection(collection_name).add(data)
        return image_url

    def get_data_by_date(self, collection_name, date):
        date_start = datetime.combine(date, datetime.min.time())
        date_end = datetime.combine(date, datetime.max.time())
        docs = self.db.collection(collection_name).where('timestamp', '>=', date_start).where('timestamp', '<=', date_end).stream()
        return [doc.to_dict() for doc in docs]

    def count_data_by_date(self, collection_name, date):
        data = self.get_data_by_date(collection_name, date)
        return len(data)

    def sum_count_variable(self, collection_name):
        docs = self.db.collection(collection_name).stream()
        total_count = sum(doc.to_dict().get('count', 0) for doc in docs)
        return total_count