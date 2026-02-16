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
import threading
import time

from config import Config

# ================ راه‌اندازی اولیه ================
app = Flask(__name__)
app.config.from_object(Config)
CORS(app, supports_credentials=True)

# ایجاد پوشه‌ها
Config.init_dirs()

# ================ راه‌اندازی لاگر ================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
handler = RotatingFileHandler('logs/app.log', maxBytes=10000000, backupCount=5)
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

# ================ راه‌اندازی سرویس‌ها ================

# 1. کش
class SimpleCache:
    """کش ساده برای مواقعی که Redis نداریم"""
    def __init__(self):
        self.cache = {}
        self.timers = {}
        
    def get(self, key):
        if key in self.cache:
            return self.cache[key]
        return None
        
    def set(self, key, value, ttl=3600):
        self.cache[key] = value
        
    def make_key(self, *args):
        return ':'.join(str(arg) for arg in args)

try:
    from services.cache_service import CacheService
    cache_service = CacheService(Config)
    logger.info("✅ CacheService راه‌اندازی شد")
except Exception as e:
    logger.warning(f"⚠️ خطا در راه‌اندازی CacheService: {e} - استفاده از کش ساده")
    cache_service = SimpleCache()

# 2. سرویس جمینی
try:
    from services.gemini_service import GeminiService
    gemini_service = GeminiService(Config)
    logger.info("✅ GeminiService راه‌اندازی شد")
except Exception as e:
    logger.warning(f"⚠️ خطا در راه‌اندازی GeminiService: {e}")
    gemini_service = None

# 3. مغز هوش مصنوعی
try:
    from models.brain import AIBrain
    brain = AIBrain(Config)
    logger.info("✅ AIBrain راه‌اندازی شد")
except Exception as e:
    logger.warning(f"⚠️ خطا در راه‌اندازی AIBrain: {e}")
    
    # مغز ساده برای مواقع ضروری
    class SimpleBrain:
        def __init__(self):
            self.knowledge_base = []
            self.unanswered = []
            
        def search(self, query, user_id=None):
            return []
            
        def record_unanswered(self, question):
            self.unanswered.append(question)
            
    brain = SimpleBrain()
    logger.info("✅ SimpleBrain جایگزین شد")

# 4. موتور یادگیری
try:
    from models.learning_engine import LearningEngine
    learning_engine = LearningEngine(Config, brain, gemini_service)
    logger.info("✅ LearningEngine راه‌اندازی شد")
except Exception as e:
    logger.warning(f"⚠️ خطا در راه‌اندازی LearningEngine: {e}")
    
    class SimpleLearning:
        def learn_from_interaction(self, *args, **kwargs): pass
        def learn_from_feedback(self, *args, **kwargs): pass
        def get_user_profile(self, *args, **kwargs): return {}
        def get_recommendations(self, *args, **kwargs): return []
        def get_trending_topics(self, *args, **kwargs): return []
        
    learning_engine = SimpleLearning()
    logger.info("✅ SimpleLearning جایگزین شد")

# 5. صف پیام
try:
    from services.queue_service import QueueService
    queue_service = QueueService(Config)
    logger.info("✅ QueueService راه‌اندازی شد")
except Exception as e:
    logger.warning(f"⚠️ خطا در راه‌اندازی QueueService: {e}")
    
    class SimpleQueue:
        def publish_chat_message(self, *args, **kwargs): pass
        def publish_learning_task(self, *args, **kwargs): pass
        def get_stats(self): return {}
        
    queue_service = SimpleQueue()
    logger.info("✅ SimpleQueue جایگزین شد")

# 6. پردازشگر فایل
try:
    from services.file_processor import FileProcessor
    file_processor = FileProcessor(Config, brain, queue_service)
    logger.info("✅ FileProcessor راه‌اندازی شد")
except Exception as e:
    logger.warning(f"⚠️ خطا در راه‌اندازی FileProcessor: {e}")
    file_processor = None

# 7. موتور جستجو
try:
    from algorithms.search_engine import AdvancedSearchEngine
    search_engine = AdvancedSearchEngine(Config)
    logger.info("✅ AdvancedSearchEngine راه‌اندازی شد")
except Exception as e:
    logger.warning(f"⚠️ خطا در راه‌اندازی AdvancedSearchEngine: {e}")
    search_engine = None

# ================ راه‌اندازی API‌ها ================

# 1. چت
try:
    from api.chat_routes import chat_bp, ChatAPI
    chat_api = ChatAPI(brain, gemini_service, cache_service, learning_engine, queue_service)
    chat_api.register_routes(chat_bp)
    app.register_blueprint(chat_bp)
    logger.info("✅ ChatAPI راه‌اندازی شد")
except Exception as e:
    logger.error(f"❌ خطا در راه‌اندازی ChatAPI: {e}")

# 2. مدیریت
try:
    from api.admin_routes import admin_bp
    app.register_blueprint(admin_bp)
    logger.info("✅ AdminAPI راه‌اندازی شد")
except Exception as e:
    logger.warning(f"⚠️ خطا در راه‌اندازی AdminAPI: {e}")

# 3. تحلیل
try:
    from api.analytics_routes import analytics_bp
    app.register_blueprint(analytics_bp)
    logger.info("✅ AnalyticsAPI راه‌اندازی شد")
except Exception as e:
    logger.warning(f"⚠️ خطا در راه‌اندازی AnalyticsAPI: {e}")

# ================ مدیریت کاربران ================
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'admin_login'

# مدل کاربر ساده
class User:
    def __init__(self, id, username, password, role='admin'):
        self.id = id
        self.username = username
        self.password = password
        self.role = role
        
    def is_authenticated(self):
        return True
        
    def is_active(self):
        return True
        
    def is_anonymous(self):
        return False
        
    def get_id(self):
        return str(self.id)

# کاربر پیش‌فرض
users = {
    '1': User('1', 'admin', 'admin123', 'admin')
}

@login_manager.user_loader
def load_user(user_id):
    return users.get(user_id)

# ================ صفحات اصلی ================
@app.route('/')
def index():
    """صفحه اصلی"""
    try:
        return render_template('index.html')
    except:
        return "صفحه اصلی - تاریخ‌دان هوشمند"

@app.route('/admin-login')
def admin_login():
    """صفحه ورود مدیریت"""
    try:
        return render_template('admin_login.html')
    except:
        return "صفحه ورود مدیریت"

@app.route('/health')
def health():
    """بررسی سلامت سیستم"""
    return jsonify({
        'status': 'healthy',
        'timestamp': time.time(),
        'services': {
            'brain': brain is not None,
            'cache': cache_service is not None,
            'gemini': gemini_service is not None,
            'queue': queue_service is not None,
            'learning': learning_engine is not None
        }
    })

@app.route('/stats')
def stats():
    """آمار کلی سیستم"""
    knowledge_count = len(getattr(brain, 'knowledge_base', []))
    return jsonify({
        'knowledge': knowledge_count,
        'status': 'running'
    })

# ================ background tasks ================
def background_task():
    """تسک پس‌زمینه"""
    while True:
        try:
            time.sleep(60)
            logger.info("Background task running...")
        except Exception as e:
            logger.error(f"Background task error: {e}")

# شروع تسک پس‌زمینه
threading.Thread(target=background_task, daemon=True).start()

# ================ مدیریت graceful shutdown ================
def signal_handler(sig, frame):
    logger.info('Shutting down gracefully...')
    sys.exit(0)
    
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# ================ اجرا ================
if __name__ == '__main__':
    print("""
    ╔════════════════════════════════════════════════════════════════╗
    ║     🤖 تاریخ‌دان هوشمند - نسخه Enterprise                     ║
    ╠════════════════════════════════════════════════════════════════╣
    ║  📚 دانش فعلی: {} مورد                                         ║
    ║  🌐 صفحه چت: http://localhost:5000                            ║
    ║  🔐 پنل مدیریت: http://localhost:5000/admin-login             ║
    ║  👤 کاربر: admin / رمز: admin123                              ║
    ║  ⚡ وضعیت سرویس‌ها:                                            ║
    ║     - مغز هوش مصنوعی: {'✅' if brain else '❌'}                ║
    ║     - کش: {'✅' if cache_service else '❌'}                    ║
    ║     - جمینی: {'✅' if gemini_service else '❌'}                ║
    ║     - یادگیری: {'✅' if learning_engine else '❌'}             ║
    ╚════════════════════════════════════════════════════════════════╝
    """.format(
        len(getattr(brain, 'knowledge_base', [])),
        '✅' if brain else '❌',
        '✅' if cache_service else '❌',
        '✅' if gemini_service else '❌',
        '✅' if learning_engine else '❌'
    ))
    
    # اجرا
    app.run(debug=True, host='0.0.0.0', port=5000)
