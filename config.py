# config.py
import os
from datetime import timedelta

class Config:
    # امنیت
    SECRET_KEY = os.environ.get('SECRET_KEY', 'super-secret-key-2025-1million-users')
    
    # Google Gemini API
    GEMINI_API_KEY = "AIzaSyBt64suioEmwlczwVr4ZVXrjP6lTQsEbC0"
    GEMINI_MODEL = "gemini-pro"  # یا gemini-1.5-pro-latest
    
    # دیتابیس‌ها
    MONGODB_URI = os.environ.get('MONGODB_URI', 'mongodb://localhost:27017/')
    MONGODB_DB = "history_ai"
    POSTGRES_URI = os.environ.get('POSTGRES_URI', 'postgresql://user:pass@localhost/db')
    REDIS_URL = os.environ.get('REDIS_URL', 'redis://localhost:6379/0')
    RABBITMQ_URL = os.environ.get('RABBITMQ_URL', 'amqp://guest:guest@localhost:5672/')
    
    # Elasticsearch
    ELASTICSEARCH_HOSTS = ['http://localhost:9200']
    
    # آپلود فایل
    UPLOAD_FOLDER = 'uploads'
    MAX_CONTENT_LENGTH = 100 * 1024 * 1024 * 1024  # 100GB
    ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx', 'csv', 'json', 'xlsx'}
    
    # کش
    CACHE_TYPE = 'redis'
    CACHE_REDIS_URL = REDIS_URL
    CACHE_DEFAULT_TIMEOUT = 3600  # 1 ساعت
    CACHE_KEY_PREFIX = 'hist_ai_'
    
    # صف
    QUEUE_MAX_SIZE = 10000
    QUEUE_BATCH_SIZE = 100
    
    # عملکرد
    WORKERS = 32  # برای Gunicorn
    MAX_CONNECTIONS = 10000
    RATE_LIMIT = "100/minute"  # محدودیت نرخ
    
    # یادگیری
    LEARNING_RATE = 0.1
    FEEDBACK_WEIGHT = 0.3
    USER_PERSONALIZATION = True
    
    # پوشه‌ها
    DATA_FOLDER = 'data'
    LOG_FOLDER = 'logs'
    MODEL_FOLDER = 'models/saved'
    
    @staticmethod
    def init_dirs():
        for folder in [Config.UPLOAD_FOLDER, Config.DATA_FOLDER, 
                      Config.LOG_FOLDER, Config.MODEL_FOLDER]:
            os.makedirs(folder, exist_ok=True)
