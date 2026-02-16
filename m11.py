# m11.py - Main Application
from flask import Flask, render_template, request, redirect, url_for, jsonify  # <-- request اینجا اضافه شد
from flask_cors import CORS
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user
import os
import hashlib  # <-- hashlib اینجا اضافه شد
from datetime import datetime
import threading
import time
import logging

# ================ ایمپورت ماژول‌ها ================
from m1 import Config
from m2 import KnowledgeBase, UnansweredManager, UserManager
from m3 import GeminiService
from m4 import SearchEngine
from m5 import CacheService
from m6 import LearningEngine
from m7 import QueueService
from m8 import FileProcessor
from m9 import chat_bp, ChatAPI
from m10 import admin_bp, AdminAPI

# ================ راه‌اندازی ================
app = Flask(__name__)
app.config['SECRET_KEY'] = 'super-secret-key-2025'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024 * 1024

CORS(app, supports_credentials=True)

# ایجاد پوشه‌ها
os.makedirs('uploads', exist_ok=True)
os.makedirs('data', exist_ok=True)
os.makedirs('logs', exist_ok=True)

# ================ لاگر ================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ================ سرویس‌ها ================
config = Config()
knowledge = KnowledgeBase(config)
unanswered = UnansweredManager(config)
user_manager = UserManager(config)
gemini = GeminiService(config)
search = SearchEngine(config)
cache = CacheService(config)
learning = LearningEngine(config, knowledge)
queue = QueueService(config)
file_processor = FileProcessor(config, knowledge)

# به‌روزرسانی ایندکس جستجو
search.update_index(knowledge.data)

# ================ API‌ها ================
chat_api = ChatAPI(knowledge, unanswered, gemini, search, cache, learning, queue)
chat_api.register_routes(chat_bp)
app.register_blueprint(chat_bp)

admin_api = AdminAPI(knowledge, unanswered, gemini, file_processor, learning)
admin_api.register_routes(admin_bp)
app.register_blueprint(admin_bp)

# ================ مدیریت کاربران ================
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'admin.admin_panel'

class User(UserMixin):
    def __init__(self, id, username, password):
        self.id = id
        self.username = username
        self.password = password

admin_pass = hashlib.md5('admin123'.encode()).hexdigest()
users = {'1': User('1', 'admin', admin_pass)}

@login_manager.user_loader
def load_user(user_id):
    return users.get(user_id)

# ================ صفحات ================

@app.route('/')
def index():
    """صفحه اصلی چت"""
    return render_template('index.html', 
                         now=datetime.now(),
                         page='chat',
                         user=None,
                         stats=None,
                         error=None)

@app.route('/admin-login', methods=['GET', 'POST'])
def admin_login():
    """صفحه ورود مجزا"""
    if request.method == 'POST':
        username = request.form.get('username', '')
        password = hashlib.md5(request.form.get('password', '').encode()).hexdigest()
        
        if username == 'admin' and password == admin_pass:
            login_user(users['1'])
            return redirect(url_for('admin.admin_panel'))
        else:
            return render_template('index.html', 
                                 error='اطلاعات ورود اشتباه است',
                                 page='login',
                                 user=None,
                                 stats=None,
                                 now=datetime.now())
    
    return render_template('index.html', 
                         page='login',
                         user=None,
                         stats=None,
                         error=None,
                         now=datetime.now())

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

# ================ health check ================
@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'knowledge': len(knowledge.data),
        'cache': cache.get_stats(),
        'gemini': gemini.get_stats() if hasattr(gemini, 'get_stats') else {},
        'time': datetime.now().isoformat()
    })

# ================ پس‌زمینه ================
def background_index_updater():
    """به‌روزرسانی خودکار ایندکس"""
    while True:
        time.sleep(300)  # هر ۵ دقیقه
        search.update_index(knowledge.data)
        logger.info("Search index updated")

threading.Thread(target=background_index_updater, daemon=True).start()

# ================ اجرا ================
# در m11.py، بخش آخر رو اینطوری اصلاح کن:

if __name__ == '__main__':
    # تست Gemini با مدل‌های مختلف
    print("🔄 تست اتصال به Gemini API...")
    test_result = gemini.ask("سلام، چطوری؟")
    
    if test_result and test_result.get('success'):
        gemini_status = f"✅ فعال (مدل: {test_result.get('model', 'unknown')})"
        print(f"✅ پاسخ تست: {test_result['answer'][:50]}...")
    else:
        gemini_status = "❌ غیرفعال - خطا در اتصال"
    
    cache_stats = cache.get_stats()
    
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║     🤖 تاریخ‌دان هوشمند - پشتیبانی ۲۰ هزار کاربر                ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║  📚 دانش: {} مورد                                                 ║
    ║  🌐 صفحه چت: http://localhost:5000                              ║
    ║  🔐 پنل مدیریت: http://localhost:5000/admin-login               ║
    ║  👤 کاربر: admin / رمز: admin123                                ║
    ║  🤖 Gemini API: {}                                               ║
    ║  ⚡ کش: {} hit/{} miss ({}%)                                     ║
    ╚══════════════════════════════════════════════════════════════════╝
    """.format(
        len(knowledge.data),
        gemini_status,
        cache_stats['hits'],
        cache_stats['misses'],
        cache_stats['hit_rate']
    ))
    
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
