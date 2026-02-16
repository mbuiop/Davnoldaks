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
import json
import requests  # برای API calls

# ================ تنظیمات ================
app = Flask(__name__)
app.config['SECRET_KEY'] = 'super-secret-key-2025'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024 * 1024  # 100GB

# Google Gemini API Key
GEMINI_API_KEY = "AIzaSyBt64suioEmwlczwVr4ZVXrjP6lTQsEbC0"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"

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

# ================ کلاس مغز هوش مصنوعی با پشتیبانی از Gemini ================

class AIBrain:
    """مغز هوش مصنوعی با قابلیت استفاده از Gemini"""
    
    def __init__(self):
        self.knowledge_base = []
        self.unanswered = []
        self.load_data()
        
    def load_data(self):
        """بارگذاری دانش از فایل"""
        if os.path.exists('data/knowledge.json'):
            try:
                with open('data/knowledge.json', 'r', encoding='utf-8') as f:
                    self.knowledge_base = json.load(f)
                logger.info(f"📚 {len(self.knowledge_base)} دانش بارگذاری شد")
            except:
                self.knowledge_base = []
                
    def save_data(self):
        """ذخیره دانش"""
        with open('data/knowledge.json', 'w', encoding='utf-8') as f:
            json.dump(self.knowledge_base, f, ensure_ascii=False, indent=2)
            
    def search(self, query, user_id=None):
        """جستجوی محلی در دانش"""
        results = []
        query = query.lower()
        
        for item in self.knowledge_base:
            question = item.get('question', '').lower()
            answer = item.get('answer', '').lower()
            
            if query in question or query in answer:
                # محاسبه امتیاز شباهت
                score = 0.8
                if query == question:
                    score = 1.0
                elif query in question:
                    score = 0.9
                    
                results.append({
                    'id': item.get('id', 0),
                    'question': item.get('question', ''),
                    'answer': item.get('answer', ''),
                    'score': score,
                    'category': item.get('category', 'عمومی')
                })
                
        # مرتب‌سازی بر اساس امتیاز
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:5]
        
    def ask_gemini(self, question):
        """پرسش از Google Gemini API"""
        try:
            headers = {
                'Content-Type': 'application/json'
            }
            
            data = {
                "contents": [{
                    "parts": [{
                        "text": f"تو یک تاریخ‌دان حرفه‌ای هستی. به سوال زیر به زبان فارسی پاسخ بده:\n\n{question}"
                    }]
                }]
            }
            
            response = requests.post(
                f"{GEMINI_API_URL}?key={GEMINI_API_KEY}",
                headers=headers,
                json=data,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                answer = result['candidates'][0]['content']['parts'][0]['text']
                return {
                    'answer': answer,
                    'confidence': 0.9,
                    'source': 'gemini'
                }
            else:
                logger.error(f"Gemini API error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Error calling Gemini API: {e}")
            return None
            
    def add_knowledge(self, question, answer, category='عمومی'):
        """اضافه کردن دانش جدید"""
        new_item = {
            'id': len(self.knowledge_base) + 1,
            'question': question,
            'answer': answer,
            'category': category,
            'times_used': 0,
            'created_at': datetime.now().isoformat()
        }
        
        self.knowledge_base.append(new_item)
        self.save_data()
        return True
        
    def record_unanswered(self, question):
        """ثبت سوال بی‌پاسخ"""
        self.unanswered.append({
            'question': question,
            'timestamp': datetime.now().isoformat()
        })
        
        with open('data/unanswered.json', 'w', encoding='utf-8') as f:
            json.dump(self.unanswered[-100:], f, ensure_ascii=False, indent=2)

# ================ نمونه مغز هوش مصنوعی ================
brain = AIBrain()
logger.info("✅ AIBrain با پشتیبانی از Gemini راه‌اندازی شد")

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
    return render_template('index.html', now=datetime.now())

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
        'unanswered': len(brain.unanswered),
        'categories': {}
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
    """API چت با پشتیبانی از Gemini"""
    data = request.json
    question = data.get('message', '').strip()
    use_gemini = data.get('use_gemini', True)  # پیش‌فرض استفاده از Gemini
    
    if not question:
        return jsonify({'error': 'سوال نمی‌تواند خالی باشد'}), 400
    
    start_time = time.time()
    
    # مرحله ۱: جستجو در دانش محلی
    local_results = brain.search(question)
    
    # اگر نتیجه خوبی پیدا شد (امتیاز بالا)
    if local_results and local_results[0]['score'] > 0.8:
        response = {
            'answer': local_results[0]['answer'],
            'confidence': local_results[0]['score'],
            'source': 'knowledge_base',
            'found': True,
            'response_time': time.time() - start_time
        }
        
    # اگر نتیجه متوسط بود
    elif local_results and local_results[0]['score'] > 0.5:
        # سعی می‌کنیم با Gemini تکمیل کنیم
        if use_gemini:
            gemini_result = brain.ask_gemini(question)
            if gemini_result:
                response = {
                    'answer': gemini_result['answer'],
                    'confidence': 0.85,
                    'source': 'gemini_enhanced',
                    'local_match': local_results[0]['answer'],
                    'found': True,
                    'response_time': time.time() - start_time
                }
            else:
                response = {
                    'answer': local_results[0]['answer'],
                    'confidence': local_results[0]['score'],
                    'source': 'knowledge_base',
                    'found': True,
                    'response_time': time.time() - start_time
                }
        else:
            response = {
                'answer': local_results[0]['answer'],
                'confidence': local_results[0]['score'],
                'source': 'knowledge_base',
                'found': True,
                'response_time': time.time() - start_time
            }
    
    # اگر هیچ نتیجه‌ای پیدا نشد
    else:
        if use_gemini:
            gemini_result = brain.ask_gemini(question)
            if gemini_result:
                response = {
                    'answer': gemini_result['answer'],
                    'confidence': gemini_result['confidence'],
                    'source': 'gemini',
                    'found': True,
                    'response_time': time.time() - start_time
                }
                
                # ذخیره خودکار پاسخ خوب Gemini
                if len(question) > 10 and len(gemini_result['answer']) > 50:
                    brain.add_knowledge(question, gemini_result['answer'], 'gemini_auto')
                    
            else:
                response = {
                    'answer': None,
                    'found': False,
                    'message': 'سوال شما ثبت شد',
                    'response_time': time.time() - start_time
                }
                brain.record_unanswered(question)
        else:
            response = {
                'answer': None,
                'found': False,
                'message': 'سوال شما ثبت شد',
                'response_time': time.time() - start_time
            }
            brain.record_unanswered(question)
    
    response['timestamp'] = datetime.now().isoformat()
    
    return jsonify(response)

@app.route('/api/chat/gemini-only', methods=['POST'])
def api_chat_gemini_only():
    """API فقط با Gemini"""
    data = request.json
    question = data.get('message', '').strip()
    
    if not question:
        return jsonify({'error': 'سوال نمی‌تواند خالی باشد'}), 400
    
    start_time = time.time()
    gemini_result = brain.ask_gemini(question)
    
    if gemini_result:
        response = {
            'answer': gemini_result['answer'],
            'confidence': gemini_result['confidence'],
            'source': 'gemini',
            'found': True,
            'response_time': time.time() - start_time
        }
    else:
        response = {
            'answer': None,
            'found': False,
            'response_time': time.time() - start_time
        }
    
    return jsonify(response)

@app.route('/api/knowledge/list')
def api_knowledge_list():
    """لیست دانش"""
    return jsonify(brain.knowledge_base)

@app.route('/api/unanswered/list')
def api_unanswered_list():
    """لیست سوالات بی‌پاسخ"""
    if os.path.exists('data/unanswered.json'):
        with open('data/unanswered.json', 'r', encoding='utf-8') as f:
            return jsonify(json.load(f))
    return jsonify([])

@app.route('/admin/add', methods=['POST'])
@login_required
def admin_add():
    """اضافه کردن دانش"""
    question = request.form['question']
    answer = request.form['answer']
    category = request.form.get('category', 'عمومی')
    
    brain.add_knowledge(question, answer, category)
    
    return redirect(url_for('admin_panel'))

@app.route('/admin/upload', methods=['POST'])
@login_required
def admin_upload():
    """آپلود فایل"""
    if 'file' not in request.files:
        return "❌ فایلی انتخاب نشده است"
    
    files = request.files.getlist('file')
    count = 0
    
    for file in files:
        if file and file.filename:
            filename = file.filename
            content = file.read().decode('utf-8', errors='ignore')
            
            # پردازش فایل متنی
            lines = content.split('\n')
            for line in lines:
                if '|' in line:
                    parts = line.split('|', 1)
                    if len(parts) == 2:
                        q, a = parts
                        brain.add_knowledge(q.strip(), a.strip(), 'file_upload')
                        count += 1
    
    return f"✅ {count} مورد با موفقیت اضافه شد <a href='/admin'>بازگشت</a>"

@app.route('/health')
def health():
    """بررسی سلامت"""
    # تست Gemini API
    gemini_status = False
    try:
        test = brain.ask_gemini("سلام")
        if test:
            gemini_status = True
    except:
        pass
    
    return jsonify({
        'status': 'healthy',
        'timestamp': time.time(),
        'gemini_api': gemini_status,
        'knowledge_count': len(brain.knowledge_base),
        'unanswered_count': len(brain.unanswered)
    })

@app.route('/stats')
def stats():
    """آمار"""
    return jsonify({
        'knowledge': len(brain.knowledge_base),
        'unanswered': len(brain.unanswered),
        'status': 'running',
        'gemini_available': True
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
    brain.save_data()
    sys.exit(0)
    
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# ================ اجرا ================
if __name__ == '__main__':
    print("""
    ╔════════════════════════════════════════════════════════════════╗
    ║     🤖 تاریخ‌دان هوشمند - با پشتیبانی Google Gemini          ║
    ╠════════════════════════════════════════════════════════════════╣
    ║  📚 دانش فعلی: {} مورد                                         ║
    ║  🌐 صفحه چت: http://localhost:5000                            ║
    ║  🔐 پنل مدیریت: http://localhost:5000/admin-login             ║
    ║  👤 کاربر: admin / رمز: admin123                              ║
    ║  🤖 وضعیت Gemini: فعال ✅                                     ║
    ║  ⚡ API Key: {}...                                             ║
    ╚════════════════════════════════════════════════════════════════╝
    """.format(
        len(brain.knowledge_base),
        GEMINI_API_KEY[:15]
    ))
    
    # اجرا
    app.run(debug=True, host='0.0.0.0', port=5000)
