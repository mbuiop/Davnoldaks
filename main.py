# main.py
from flask import Flask, render_template, jsonify
from flask_cors import CORS
from flask_login import LoginManager
from flask_caching import Cache
import os
import signal
import sys
import logging
from logging.handlers import RotatingFileHandler
import multiprocessing
import threading
import time

from config import Config
from models.brain import AIBrain
from models.learning_engine import LearningEngine
from services.gemini_service import GeminiService
from services.cache_service import CacheService
from services.queue_service import QueueService
from services.file_processor import FileProcessor
from algorithms.search_engine import AdvancedSearchEngine
from api.chat_routes import chat_bp, ChatAPI
from api.admin_routes import admin_bp
from api.analytics_routes import analytics_bp
from utils.logger import setup_logger

# ================ راه‌اندازی اولیه ================
app = Flask(__name__)
app.config.from_object(Config)
CORS(app, supports_credentials=True)

# ایجاد پوشه‌ها
Config.init_dirs()

# ================ راه‌اندازی لاگر ================
logger = setup_logger(app)

# ================ راه‌اندازی سرویس‌ها ================
# کش
cache_service = CacheService(Config)

# صف پیام
queue_service = QueueService(Config)

# سرویس جمینی
gemini_service = GeminiService(Config)

# مغز هوش مصنوعی
brain = AIBrain(Config)

# موتور یادگیری
learning_engine = LearningEngine(Config, brain, gemini_service)

# موتور جستجو
search_engine = AdvancedSearchEngine(Config)

# پردازشگر فایل
file_processor = FileProcessor(Config, brain, queue_service)

# ================ راه‌اندازی API‌ها ================
# چت
chat_api = ChatAPI(brain, gemini_service, cache_service, learning_engine, queue_service)
chat_api.register_routes(chat_bp)
app.register_blueprint(chat_bp)

# مدیریت
app.register_blueprint(admin_bp)

# تحلیل
app.register_blueprint(analytics_bp)

# ================ مدیریت کاربران ================
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'admin.login'

from models.user import User, load_user
login_manager.user_loader(load_user)

# ================ صفحات اصلی ================
@app.route('/')
def index():
    """صفحه اصلی"""
    return render_template('index.html')

@app.route('/health')
def health():
    """بررسی سلامت سیستم"""
    return jsonify({
        'status': 'healthy',
        'timestamp': time.time(),
        'services': {
            'redis': cache_service.redis_client.ping() if hasattr(cache_service, 'redis_client') else False,
            'queue': queue_service.get_queue_length('health') is not None,
            'gemini': gemini_service.model is not None,
            'brain': len(brain.knowledge_base) > 0
        }
    })

@app.route('/stats')
def stats():
    """آمار کلی سیستم"""
    return jsonify({
        'knowledge': len(brain.knowledge_base),
        'users': len(learning_engine.user_patterns),
        'conversations': sum(len(p['questions']) for p in learning_engine.user_patterns.values()),
        'cache': cache_service.get_stats(),
        'queue': queue_service.get_stats(),
        'learning': learning_engine.get_learning_stats()
    })

# ================ مدیریت graceful shutdown ================
def signal_handler(sig, frame):
    logger.info('Shutting down gracefully...')
    
    # ذخیره مدل‌ها
    brain.save_knowledge()
    learning_engine.save_models()
    
    # بستن اتصال‌ها
    if hasattr(cache_service, 'redis_client'):
        cache_service.redis_client.close()
        
    logger.info('Shutdown complete')
    sys.exit(0)
    
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# ================ background tasks ================
def background_learning():
    """یادگیری در پس‌زمینه"""
    while True:
        try:
            time.sleep(300)  # هر ۵ دقیقه
            learning_engine.learn_from_crowd()
            
            # به‌روزرسانی ایندکس جستجو
            if len(brain.knowledge_base) > search_engine.tfidf_matrix.shape[0]:
                search_engine.update_index(brain.knowledge_base)
                
        except Exception as e:
            logger.error(f"Background learning error: {e}")
            
def background_queue_consumer():
    """مصرف‌کننده صف در پس‌زمینه"""
    def process_message(message):
        logger.info(f"Processing message: {message}")
        return True
        
    queue_service.consume('learning_tasks', process_message)
    queue_service.start_consuming()
    
# شروع تسک‌های پس‌زمینه
threading.Thread(target=background_learning, daemon=True).start()
threading.Thread(target=background_queue_consumer, daemon=True).start()

# ================ اجرا ================
if __name__ == '__main__':
    logger.info("""
    ╔════════════════════════════════════════════════════════════════╗
    ║     🤖 تاریخ‌دان هوشمند - نسخه Enterprise با ۱ میلیون کاربر   ║
    ╠════════════════════════════════════════════════════════════════╣
    ║  📚 دانش فعلی: {} مورد                                         ║
    ║  👥 کاربران فعال: {}                                           ║
    ║  💬 مکالمات: {}                                                ║
    ║  ⚡ کش: {} hit / {} miss ({}%)                                 ║
    ║  🎯 API Key: {}...                                             ║
    ╚════════════════════════════════════════════════════════════════╝
    """.format(
        len(brain.knowledge_base),
        len(learning_engine.user_patterns),
        sum(len(p['questions']) for p in learning_engine.user_patterns.values()),
        cache_service.hit_count,
        cache_service.miss_count,
        cache_service.hit_count / (cache_service.hit_count + cache_service.miss_count) * 100 if (cache_service.hit_count + cache_service.miss_count) > 0 else 0,
        Config.GEMINI_API_KEY[:10]
    ))
    
    # اجرا با Gunicorn
    app.run(debug=False, host='0.0.0.0', port=5000)
