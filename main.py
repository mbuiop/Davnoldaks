# main.py
from flask import Flask, render_template, jsonify, request, redirect, url_for
from flask_cors import CORS
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user
import os
import signal
import sys
import logging
from logging.handlers import RotatingFileHandler
import threading
import time
import hashlib
from datetime import datetime

# ================ تنظیمات ================
app = Flask(__name__)
app.config['SECRET_KEY'] = 'super-secret-key-2025'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024 * 1024  # 100GB

CORS(app, supports_credentials=True)

# ایجاد پوشه‌ها
os.makedirs('uploads', exist_ok=True)
os.makedirs('data', exist_ok=True)
os.makedirs('logs', exist_ok=True)
os.makedirs('models/saved', exist_ok=True)

# ================ راه‌اندازی لاگر ================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
handler = RotatingFileHandler('logs/app.log', maxBytes=10000000, backupCount=5)
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

# ================ کلاس‌های ساده (برای مواقعی که سرویس‌ها در دسترس نیستن) ================

class SimpleBrain:
    """مغز ساده هوش مصنوعی"""
    def __init__(self):
        self.knowledge_base = []
        self.unanswered = []
        self.load_data()
        
    def load_data(self):
        """بارگذاری دانش از فایل"""
        import json
        import os
        
        if os.path.exists('data/knowledge.json'):
            try:
                with open('data/knowledge.json', 'r', encoding='utf-8') as f:
                    self.knowledge_base = json.load(f)
                logger.info(f"📚 {len(self.knowledge_base)} دانش بارگذاری شد")
            except:
                self.knowledge_base = []
                
    def search(self, query, user_id=None):
        """جستجوی ساده"""
        results = []
        for item in self.knowledge_base:
            if query in item.get('question', '') or query in item.get('answer', ''):
                results.append({
                    'id': item.get('id', 0),
                    'question': item.get('question', ''),
                    'answer': item.get('answer', ''),
                    'score': 0.8,
                    'category': item.get('category', 'عمومی')
                })
        return results[:5]
        
    def record_unanswered(self, question):
        """ثبت سوال بی‌پاسخ"""
        self.unanswered.append({
            'question': question,
            'timestamp': datetime.now().isoformat()
        })
        import json
        with open('data/unanswered.json', 'w', encoding='utf-8') as f:
            json.dump(self.unanswered[-100:], f, ensure_ascii=False, indent=2)

class SimpleCache:
    """کش ساده"""
    def __init__(self):
        self.cache = {}
        
    def get(self, key):
        return self.cache.get(key)
        
    def set(self, key, value, ttl=3600):
        self.cache[key] = value
        
    def make_key(self, *args):
        return ':'.join(str(arg) for arg in args)

class SimpleLearning:
    """یادگیری ساده"""
    def learn_from_interaction(self, *args, **kwargs): pass
    def learn_from_feedback(self, *args, **kwargs): pass
    def get_user_profile(self, *args, **kwargs): return {}
    def get_recommendations(self, *args, **kwargs): return []
    def get_trending_topics(self, *args, **kwargs): return []

class SimpleQueue:
    """صف ساده"""
    def publish_chat_message(self, *args, **kwargs): pass
    def publish_learning_task(self, *args, **kwargs): pass
    def get_stats(self): return {}

# ================ نمونه‌سازی سرویس‌ها ================

# 1. مغز هوش مصنوعی
brain = SimpleBrain()
logger.info("✅ SimpleBrain راه‌اندازی شد")

# 2. کش
cache_service = SimpleCache()
logger.info("✅ SimpleCache راه‌اندازی شد")

# 3. سرویس جمینی (اختیاری)
try:
    import google.generativeai as genai
    genai.configure(api_key="AIzaSyBt64suioEmwlczwVr4ZVXrjP6lTQsEbC0")
    gemini_model = genai.GenerativeModel('gemini-pro')
    
    class GeminiService:
        def generate_answer(self, question, context=None):
            try:
                response = gemini_model.generate_content(question)
                return {
                    'answer': response.text,
                    'confidence': 0.9
                }
            except:
                return None
                
    gemini_service = GeminiService()
    logger.info("✅ GeminiService راه‌اندازی شد")
except Exception as e:
    logger.warning(f"⚠️ خطا در راه‌اندازی Gemini: {e}")
    gemini_service = None

# 4. موتور یادگیری
learning_engine = SimpleLearning()
logger.info("✅ SimpleLearning راه‌اندازی شد")

# 5. صف پیام
queue_service = SimpleQueue()
logger.info("✅ SimpleQueue راه‌اندازی شد")

# 6. پردازشگر فایل (اختیاری)
file_processor = None

# 7. موتور جستجو
search_engine = None

# ================ مدیریت کاربران ================
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'admin_login'

class User(UserMixin):
    def __init__(self, id, username, password, role='admin'):
        self.id = id
        self.username = username
        self.password = password
        self.role = role

# کاربران
users = {
    '1': User('1', 'admin', hashlib.md5('admin123'.encode()).hexdigest(), 'admin'),
}

@login_manager.user_loader
def load_user(user_id):
    return users.get(user_id)

# ================ صفحات اصلی ================

@app.route('/')
def index():
    """صفحه اصلی"""
    return render_template('index.html')

@app.route('/admin-login', methods=['GET', 'POST'])
def admin_login():
    """صفحه ورود مدیریت"""
    if request.method == 'POST':
        username = request.form['username']
        password = hashlib.md5(request.form['password'].encode()).hexdigest()
        
        for user in users.values():
            if user.username == username and user.password == password:
                login_user(user)
                return redirect(url_for('admin_panel'))
                
        return "❌ نام کاربری یا رمز عبور اشتباه است"
    
    return render_template('admin_login.html')

@app.route('/admin')
@login_required
def admin_panel():
    """پنل مدیریت"""
    stats = {
        'total': len(brain.knowledge_base),
        'unanswered': len(brain.unanswered)
    }
    return render_template('admin_panel.html', stats=stats)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

# ================ API ================

@app.route('/api/chat', methods=['POST'])
def api_chat():
    """API چت"""
    data = request.json
    question = data.get('message', '').strip()
    user_id = data.get('user_id', 'anonymous')
    
    if not question:
        return jsonify({'error': 'سوال نمی‌تواند خالی باشد'}), 400
        
    # جستجو
    results = brain.search(question)
    
    if results:
        response = {
            'answer': results[0]['answer'],
            'confidence': results[0]['score'],
            'source': 'knowledge_base',
            'found': True
        }
    else:
        # امتحان با Gemini
        if gemini_service:
            gemini_result = gemini_service.generate_answer(question)
            if gemini_result:
                response = {
                    'answer': gemini_result['answer'],
                    'confidence': gemini_result['confidence'],
                    'source': 'gemini',
                    'found': True
                }
            else:
                response = {'answer': None, 'found': False}
                brain.record_unanswered(question)
        else:
            response = {'answer': None, 'found': False}
            brain.record_unanswered(question)
            
    response['timestamp'] = datetime.now().isoformat()
    
    return jsonify(response)

@app.route('/api/chat/history', methods=['GET'])
def chat_history():
    """تاریخچه چت"""
    user_id = request.args.get('user_id', 'anonymous')
    limit = int(request.args.get('limit', 50))
    return jsonify({'history': []})

@app.route('/health')
def health():
    """بررسی سلامت"""
    return jsonify({
        'status': 'healthy',
        'timestamp': time.time(),
        'services': {
            'brain': True,
            'cache': True,
            'gemini': gemini_service is not None,
            'queue': True,
            'learning': True
        }
    })

@app.route('/stats')
def stats():
    """آمار"""
    return jsonify({
        'knowledge': len(brain.knowledge_base),
        'unanswered': len(brain.unanswered),
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

threading.Thread(target=background_task, daemon=True).start()

# ================ handler سیگنال ================
def signal_handler(sig, frame):
    logger.info('Shutting down gracefully...')
    sys.exit(0)
    
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# ================ اجرا ================
if __name__ == '__main__':
    print("""
    ╔════════════════════════════════════════════════════════════════╗
    ║     🤖 تاریخ‌دان هوشمند - نسخه ساده                         ║
    ╠════════════════════════════════════════════════════════════════╣
    ║  📚 دانش فعلی: {} مورد                                         ║
    ║  🌐 صفحه چت: http://localhost:5000                            ║
    ║  🔐 پنل مدیریت: http://localhost:5000/admin-login             ║
    ║  👤 کاربر: admin / رمز: admin123                              ║
    ║  ⚡ وضعیت سرویس‌ها:                                            ║
    ║     - مغز هوش مصنوعی: ✅                                      ║
    ║     - کش: ✅                                                  ║
    ║     - جمینی: {}                                                ║
    ║     - یادگیری: ✅                                             ║
    ╚════════════════════════════════════════════════════════════════╝
    """.format(
        len(brain.knowledge_base),
        '✅' if gemini_service else '❌'
    ))
    
    # اجرا
    app.run(debug=True, host='0.0.0.0', port=5000)
