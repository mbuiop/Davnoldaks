import sqlite3
import json
import pickle
import os
from datetime import datetime
import hashlib

class Database:
    """دیتابیس دائمی برای ذخیره همه چیز"""
    
    def __init__(self, db_path="giant_brain.db"):
        self.db_path = db_path
        self.init_db()
    
    def init_db(self):
        """ایجاد جداول"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Users table
        c.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE,
                password TEXT,
                created_at TIMESTAMP,
                last_login TIMESTAMP
            )
        ''')
        
        # Conversations table
        c.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                session_id TEXT,
                message TEXT,
                response TEXT,
                timestamp TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # Knowledge table (for manually added knowledge)
        c.execute('''
            CREATE TABLE IF NOT EXISTS knowledge (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT,
                answer TEXT,
                category TEXT,
                created_at TIMESTAMP,
                used_count INTEGER DEFAULT 0,
                helpful_count INTEGER DEFAULT 0
            )
        ''')
        
        # Training history
        c.execute('''
            CREATE TABLE IF NOT EXISTS training_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT,
                loss REAL,
                timestamp TIMESTAMP,
                tokens_processed INTEGER
            )
        ''')
        
        # Settings
        c.execute('''
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_conversation(self, user_id, session_id, message, response):
        """ذخیره مکالمه"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''
            INSERT INTO conversations (user_id, session_id, message, response, timestamp)
            VALUES (?, ?, ?, ?, ?)
        ''', (user_id, session_id, message, response, datetime.now()))
        conn.commit()
        conn.close()
    
    def get_conversations(self, session_id, limit=50):
        """گرفتن مکالمات یک session"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''
            SELECT message, response, timestamp FROM conversations
            WHERE session_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (session_id, limit))
        results = c.fetchall()
        conn.close()
        return results
    
    def add_knowledge(self, question, answer, category):
        """افزودن دانش دستی"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''
            INSERT INTO knowledge (question, answer, category, created_at)
            VALUES (?, ?, ?, ?)
        ''', (question, answer, category, datetime.now()))
        conn.commit()
        conn_id = c.lastrowid
        conn.close()
        return conn_id
    
    def get_knowledge(self, category=None):
        """گرفتن لیست دانش‌ها"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        if category:
            c.execute('SELECT * FROM knowledge WHERE category = ? ORDER BY used_count DESC', (category,))
        else:
            c.execute('SELECT * FROM knowledge ORDER BY used_count DESC')
        results = c.fetchall()
        conn.close()
        return results
    
    def update_knowledge_usage(self, knowledge_id):
        """آپدیت آمار استفاده از دانش"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('UPDATE knowledge SET used_count = used_count + 1 WHERE id = ?', (knowledge_id,))
        conn.commit()
        conn.close()
    
    def save_training(self, file_path, loss, tokens):
        """ذخیره تاریخچه آموزش"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''
            INSERT INTO training_history (file_path, loss, timestamp, tokens_processed)
            VALUES (?, ?, ?, ?)
        ''', (file_path, loss, datetime.now(), tokens))
        conn.commit()
        conn.close()
    
    def get_training_history(self, limit=100):
        """گرفتن تاریخچه آموزش"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('SELECT * FROM training_history ORDER BY timestamp DESC LIMIT ?', (limit,))
        results = c.fetchall()
        conn.close()
        return results
    
    def set_setting(self, key, value):
        """ذخیره تنظیمات"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''
            INSERT OR REPLACE INTO settings (key, value, updated_at)
            VALUES (?, ?, ?)
        ''', (key, json.dumps(value), datetime.now()))
        conn.commit()
        conn.close()
    
    def get_setting(self, key):
        """گرفتن تنظیمات"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('SELECT value FROM settings WHERE key = ?', (key,))
        result = c.fetchone()
        conn.close()
        if result:
            return json.loads(result[0])
        return None
    
    def get_stats(self):
        """گرفتن آمار کلی"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        stats = {}
        
        c.execute('SELECT COUNT(*) FROM conversations')
        stats['total_conversations'] = c.fetchone()[0]
        
        c.execute('SELECT COUNT(*) FROM knowledge')
        stats['total_knowledge'] = c.fetchone()[0]
        
        c.execute('SELECT COUNT(*) FROM training_history')
        stats['total_trainings'] = c.fetchone()[0]
        
        c.execute('SELECT AVG(loss) FROM training_history')
        stats['avg_loss'] = c.fetchone()[0] or 0
        
        conn.close()
        return stats
