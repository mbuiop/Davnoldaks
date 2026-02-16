# m1.py - Config
import os
from datetime import timedelta

class Config:
    # امنیت
    SECRET_KEY = os.environ.get('SECRET_KEY', 'super-secret-key-2025-20k-users')
    
    # Google Gemini API - مدل درست
    GEMINI_API_KEY = "AIzaSyBt64suioEmwlczwVr4ZVXrjP6lTQsEbC0"
    GEMINI_MODEL = "gemini-pro"  # <-- تغییر به gemini-pro
    GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"  # <-- v1beta رو اضافه کن
    
    # دیتابیس
    KNOWLEDGE_FILE = 'data/knowledge.json'
    UNANSWERED_FILE = 'data/unanswered.json'
    USER_DATA_FILE = 'data/users.json'
    
    # کش
    REDIS_URL = os.environ.get('REDIS_URL', 'redis://localhost:6379/0')
    CACHE_TTL = 3600  # 1 ساعت
    LOCAL_CACHE_SIZE = 1000
    
    # آپلود
    UPLOAD_FOLDER = 'uploads'
    MAX_CONTENT_LENGTH = 10 * 1024 * 1024 * 1024  # 10GB
    ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx', 'csv', 'json'}
    
    # عملکرد
    MAX_WORKERS = 8  # برای Gunicorn
    RATE_LIMIT = "100/minute"  # محدودیت نرخ
    BATCH_SIZE = 100  # برای پردازش دسته‌ای
    
    # یادگیری
    AUTO_LEARN_THRESHOLD = 0.8  # آستانه یادگیری خودکار
    FEEDBACK_WEIGHT = 0.3  # وزن بازخورد کاربر
    
    # پوشه‌ها
    DATA_FOLDER = 'data'
    LOG_FOLDER = 'logs'
    TEMP_FOLDER = 'temp'
    
    @staticmethod
    def init_dirs():
        for folder in [Config.UPLOAD_FOLDER, Config.DATA_FOLDER, 
                      Config.LOG_FOLDER, Config.TEMP_FOLDER]:
            os.makedirs(folder, exist_ok=True)
