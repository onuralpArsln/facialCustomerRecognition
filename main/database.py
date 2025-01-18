import sqlite3
from datetime import datetime

class FacialCustomerRecognitionDatabase:
    def __init__(self, db_name='facial_customer_recognition.db'):
        self.conn = sqlite3.connect(db_name, check_same_thread=False)
        self.create_table()

    def create_table(self):
        with self.conn:
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS customers (
                    ID INTEGER PRIMARY KEY AUTOINCREMENT,
                    photo BLOB NOT NULL,
                    day TEXT NOT NULL, -- DD-MM-YYYY
                    time TEXT NOT NULL, -- SS:MM:HH
                    count INTEGER NOT NULL,
                    columnA TEXT,
                    columnB INTEGER
                )
            ''')

    def add_customer(self, photo, day, time, count, columnA=None, columnB=None):
        with self.conn:
            self.conn.execute('''
                INSERT INTO customers (photo, day, time, count, columnA, columnB)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (photo, day, time, count, columnA, columnB))

    def delete_customer(self, customer_id):
        with self.conn:
            self.conn.execute('''
                DELETE FROM customers WHERE ID = ?
            ''', (customer_id,))

    def update_customer(self, customer_id, photo=None, day=None, time=None, count=None, columnA=None, columnB=None):
        with self.conn:
            query = 'UPDATE customers SET '
            params = []
            if photo is not None:
                query += 'photo = ?, '
                params.append(photo)
            if day is not None:
                query += 'day = ?, '
                params.append(day)
            if time is not None:
                query += 'time = ?, '
                params.append(time)
            if count is not None:
                query += 'count = ?, '
                params.append(count)
            if columnA is not None:
                query += 'columnA = ?, '
                params.append(columnA)
            if columnB is not None:
                query += 'columnB = ?, '
                params.append(columnB)
            query = query.rstrip(', ')
            query += ' WHERE ID = ?'
            params.append(customer_id)
            with self.conn:
                self.conn.execute(query, params)

    def get_customers_by_date(self, day):
        with self.conn:
            cursor = self.conn.execute('''
                SELECT * FROM customers WHERE day = ?
            ''', (day,))
            return cursor.fetchall()

    def __del__(self):
        self.conn.close()