# advanced_history_bot.py
from flask import Flask, render_template, request, jsonify, session, render_template_string, redirect, url_for
from flask_cors import CORS
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import json
import os
import hashlib
from datetime import datetime
from werkzeug.utils import secure_filename
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import re

app = Flask(__name__)
app.config['SECRET_KEY'] = 'super-secret-history-bot-key-2024'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB
CORS(app)

# ایجاد پوشه‌ها
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('data', exist_ok=True)

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

# کاربران پیش‌فرض
users = {
    '1': User('1', 'admin', hashlib.md5('admin123'.encode()).hexdigest(), 'admin'),
}

@login_manager.user_loader
def load_user(user_id):
    return users.get(user_id)

# ================ الگوریتم هوشمند پیشرفته ================
class AdvancedHistoryBrain:
    def __init__(self, data_file='data/history_knowledge.json'):
        self.data_file = data_file
        self.knowledge_base = []
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.question_vectors = None
        self.unanswered_questions = []
        self.load_knowledge()
        self.update_vectors()
        
    def load_knowledge(self):
        """بارگذاری دانش"""
        if os.path.exists(self.data_file):
            with open(self.data_file, 'r', encoding='utf-8') as f:
                self.knowledge_base = json.load(f)
            print(f"📚 {len(self.knowledge_base)} دانش بارگذاری شد")
        else:
            self.knowledge_base = []
            # نمونه اولیه
            sample_data = [
                {"id": 1, "question": "کوروش کبیر که بود", "answer": "کوروش بزرگ بنیانگذار شاهنشاهی هخامنشی بود", "category": "ایران باستان", "times_used": 0},
                {"id": 2, "question": "داریوش چه کرد", "answer": "داریوش بزرگ جاده شاهی را ساخت و امپراتوری را به ساتراپی‌ها تقسیم کرد", "category": "ایران باستان", "times_used": 0},
                {"id": 3, "question": "خشایارشا که بود", "answer": "خشایارشا پسر داریوش بزرگ بود که به یونان لشکر کشید", "category": "ایران باستان", "times_used": 0}
            ]
            self.knowledge_base = sample_data
            self.save_knowledge()
            
    def save_knowledge(self):
        """ذخیره دانش"""
        with open(self.data_file, 'w', encoding='utf-8') as f:
            json.dump(self.knowledge_base, f, ensure_ascii=False, indent=2)
            
    def update_vectors(self):
        """به‌روزرسانی بردارهای سوالات"""
        if self.knowledge_base:
            questions = [item['question'] for item in self.knowledge_base]
            try:
                self.question_vectors = self.vectorizer.fit_transform(questions)
            except:
                self.question_vectors = None
                
    def preprocess_text(self, text):
        """پیش‌پردازش متن"""
        text = re.sub(r'[^\w\s]', ' ', text)
        text = text.lower()
        text = ' '.join([word for word in text.split() if len(word) > 1])
        return text
    
    def search_smart(self, query):
        """جستجوی هوشمند با TF-IDF"""
        if not self.knowledge_base:
            return []
            
        query = self.preprocess_text(query)
        
        # جستجوی ساده کلمات کلیدی
        keyword_results = []
        query_words = set(query.split())
        
        for item in self.knowledge_base:
            question_words = set(item['question'].split())
            common_words = query_words.intersection(question_words)
            
            if common_words:
                score = len(common_words) / max(len(question_words), 1)
                if query == item['question']:
                    score = 1.0
                    
                keyword_results.append({
                    'id': item['id'],
                    'answer': item['answer'],
                    'score': score,
                    'category': item.get('category', 'عمومی')
                })
        
        # جستجوی برداری
        vector_results = []
        if self.question_vectors is not None and len(self.knowledge_base) > 0:
            try:
                query_vector = self.vectorizer.transform([query])
                similarities = cosine_similarity(query_vector, self.question_vectors)[0]
                
                for i, score in enumerate(similarities):
                    if score > 0.1:
                        item = self.knowledge_base[i]
                        vector_results.append({
                            'id': item['id'],
                            'answer': item['answer'],
                            'score': float(score),
                            'category': item.get('category', 'عمومی')
                        })
            except:
                pass
        
        # ترکیب نتایج
        combined = {}
        for result in keyword_results + vector_results:
            rid = result['id']
            if rid not in combined or result['score'] > combined[rid]['score']:
                combined[rid] = result
                
        results = sorted(combined.values(), key=lambda x: x['score'], reverse=True)
        
        # به‌روزرسانی آمار استفاده
        for result in results[:3]:
            for item in self.knowledge_base:
                if item['id'] == result['id']:
                    item['times_used'] = item.get('times_used', 0) + 1
                    item['last_used'] = datetime.now().isoformat()
                    break
                    
        self.save_knowledge()
        return results[:3]
    
    def add_knowledge(self, question, answer, category='عمومی'):
        """اضافه کردن دانش جدید"""
        # بررسی تکراری نبودن
        for item in self.knowledge_base:
            if item['question'].lower() == question.lower():
                return False, "این سوال قبلاً ثبت شده است"
                
        new_item = {
            'id': len(self.knowledge_base) + 1,
            'question': self.preprocess_text(question),
            'answer': answer,
            'category': category,
            'date_added': datetime.now().isoformat(),
            'times_used': 0,
            'last_used': None
        }
        
        self.knowledge_base.append(new_item)
        self.save_knowledge()
        self.update_vectors()
        return True, "دانش با موفقیت اضافه شد"
    
    def add_bulk_from_text(self, text, category='عمومی'):
        """اضافه کردن گروهی از متن"""
        lines = text.strip().split('\n')
        count = 0
        errors = []
        
        for line in lines:
            if '|' in line:
                parts = line.split('|', 1)
                if len(parts) == 2:
                    q, a = parts
                    success, msg = self.add_knowledge(q.strip(), a.strip(), category)
                    if success:
                        count += 1
                    else:
                        errors.append(f"خطا در {q}: {msg}")
                        
        return count, errors
    
    def add_from_file(self, filename):
        """اضافه کردن دانش از فایل"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                content = f.read()
            return self.add_bulk_from_text(content, 'file_upload')
        except Exception as e:
            return 0, [str(e)]
    
    def record_unanswered(self, question):
        """ثبت سوالات بی‌پاسخ"""
        self.unanswered_questions.append({
            'question': question,
            'timestamp': datetime.now().isoformat()
        })
        
        # ذخیره سوالات بی‌پاسخ
        with open('data/unanswered.json', 'w', encoding='utf-8') as f:
            json.dump(self.unanswered_questions[-100:], f, ensure_ascii=False, indent=2)
    
    def get_stats(self):
        """گرفتن آمار"""
        total = len(self.knowledge_base)
        if total == 0:
            return {}
            
        categories = Counter([item.get('category', 'عمومی') for item in self.knowledge_base])
        most_used = sorted(self.knowledge_base, key=lambda x: x.get('times_used', 0), reverse=True)[:5]
        never_used = len([item for item in self.knowledge_base if item.get('times_used', 0) == 0])
        
        return {
            'total': total,
            'categories': dict(categories),
            'most_used': most_used,
            'never_used_count': never_used,
            'unanswered_count': len(self.unanswered_questions)
        }

# ================ نمونه اصلی ================
brain = AdvancedHistoryBrain()

# ================ صفحات اصلی ================
@app.route('/')
def index():
    """صفحه اصلی چت - تمام صفحه"""
    stats = brain.get_stats()
    return render_template_string('''
    <!DOCTYPE html>
    <html lang="fa" dir="rtl">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>تاریخ‌دان هوشمند</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body {
                font-family: 'Tahoma', sans-serif;
                background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
                height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
            }
            .chat-container {
                width: 95%;
                max-width: 1200px;
                height: 90vh;
                background: white;
                border-radius: 20px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                display: flex;
                flex-direction: column;
                overflow: hidden;
            }
            .chat-header {
                background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
                color: white;
                padding: 20px 30px;
                display: flex;
                align-items: center;
                justify-content: space-between;
            }
            .chat-header h1 { font-size: 1.8em; display: flex; align-items: center; gap: 10px; }
            .header-stats {
                background: rgba(255,255,255,0.2);
                padding: 8px 15px;
                border-radius: 20px;
                font-size: 0.9em;
            }
            .admin-link {
                color: white;
                text-decoration: none;
                padding: 8px 15px;
                border-radius: 20px;
                background: rgba(255,255,255,0.2);
                transition: all 0.3s;
            }
            .admin-link:hover { background: rgba(255,255,255,0.3); }
            .chat-messages {
                flex: 1;
                overflow-y: auto;
                padding: 30px;
                background: #f8f9fa;
            }
            .message {
                display: flex;
                margin-bottom: 25px;
                animation: fadeIn 0.3s ease;
            }
            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(10px); }
                to { opacity: 1; transform: translateY(0); }
            }
            .message.user { justify-content: flex-end; }
            .message.bot { justify-content: flex-start; }
            .message-content {
                max-width: 70%;
                padding: 15px 20px;
                border-radius: 20px;
                position: relative;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                line-height: 1.6;
            }
            .user .message-content {
                background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
                color: white;
                border-bottom-right-radius: 5px;
            }
            .bot .message-content {
                background: white;
                border-bottom-left-radius: 5px;
            }
            .message-time {
                font-size: 0.7em;
                opacity: 0.7;
                margin-top: 5px;
                text-align: left;
            }
            .chat-input-container {
                padding: 20px 30px;
                background: white;
                border-top: 1px solid #eee;
                display: flex;
                gap: 15px;
            }
            .chat-input {
                flex: 1;
                padding: 15px 20px;
                border: 2px solid #e0e0e0;
                border-radius: 30px;
                font-size: 1em;
                outline: none;
                transition: all 0.3s;
                font-family: 'Tahoma', sans-serif;
            }
            .chat-input:focus {
                border-color: #1e3c72;
                box-shadow: 0 0 0 3px rgba(30,60,114,0.1);
            }
            .send-btn {
                width: 60px;
                height: 60px;
                border-radius: 50%;
                background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
                color: white;
                border: none;
                cursor: pointer;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 1.5em;
                transition: all 0.3s;
            }
            .send-btn:hover {
                transform: scale(1.1);
                box-shadow: 0 5px 20px rgba(30,60,114,0.4);
            }
            .typing-indicator {
                padding: 15px 25px;
                background: white;
                border-radius: 20px;
                display: inline-block;
            }
            .typing-indicator span {
                display: inline-block;
                width: 8px;
                height: 8px;
                border-radius: 50%;
                background: #1e3c72;
                margin: 0 3px;
                animation: typing 1.4s infinite;
            }
            .typing-indicator span:nth-child(2) { animation-delay: 0.2s; }
            .typing-indicator span:nth-child(3) { animation-delay: 0.4s; }
            @keyframes typing {
                0%, 60%, 100% { transform: translateY(0); }
                30% { transform: translateY(-10px); }
            }
        </style>
    </head>
    <body>
        <div class="chat-container">
            <div class="chat-header">
                <h1>
                    <span>🤖 تاریخ‌دان هوشمند</span>
                    <span class="header-stats">📚 {{ stats.total }} دانش</span>
                </h1>
                <a href="/admin-login" class="admin-link">⚙️ پنل مدیریت</a>
            </div>
            
            <div class="chat-messages" id="chat-messages">
                <div class="message bot">
                    <div class="message-content">
                        سلام! من تاریخ‌دان هوشمند هستم. هر سوال تاریخی داری بپرس!
                        <div class="message-time">{{ now.strftime('%H:%M') }}</div>
                    </div>
                </div>
            </div>
            
            <div class="chat-input-container">
                <input type="text" class="chat-input" id="message-input" 
                       placeholder="سوال تاریخی خود را بپرسید..." 
                       onkeypress="if(event.key==='Enter') sendMessage()">
                <button class="send-btn" onclick="sendMessage()">
                    <span>➤</span>
                </button>
            </div>
        </div>
        
        <script>
            const messagesContainer = document.getElementById('chat-messages');
            const messageInput = document.getElementById('message-input');
            
            function addMessage(text, isUser = false) {
                const messageDiv = document.createElement('div');
                messageDiv.className = `message ${isUser ? 'user' : 'bot'}`;
                
                const time = new Date().toLocaleTimeString('fa-IR', { hour: '2-digit', minute: '2-digit' });
                
                messageDiv.innerHTML = `
                    <div class="message-content">
                        ${text}
                        <div class="message-time">${time}</div>
                    </div>
                `;
                
                messagesContainer.appendChild(messageDiv);
                messagesContainer.scrollTop = messagesContainer.scrollHeight;
            }
            
            function showTyping() {
                const typingDiv = document.createElement('div');
                typingDiv.className = 'message bot';
                typingDiv.id = 'typing-indicator';
                typingDiv.innerHTML = `
                    <div class="typing-indicator">
                        <span></span>
                        <span></span>
                        <span></span>
                    </div>
                `;
                messagesContainer.appendChild(typingDiv);
                messagesContainer.scrollTop = messagesContainer.scrollHeight;
            }
            
            function hideTyping() {
                const typing = document.getElementById('typing-indicator');
                if (typing) typing.remove();
            }
            
            async function sendMessage() {
                const message = messageInput.value.trim();
                if (!message) return;
                
                addMessage(message, true);
                messageInput.value = '';
                showTyping();
                
                try {
                    const response = await fetch('/api/chat', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({message: message})
                    });
                    
                    const data = await response.json();
                    hideTyping();
                    
                    if (data.answer) {
                        addMessage(data.answer);
                    } else {
                        addMessage('متأسفم! نتونستم جوابی پیدا کنم. این سوال برای مدیر ارسال شد.');
                    }
                    
                } catch (error) {
                    hideTyping();
                    addMessage('خطا در ارتباط با سرور');
                }
            }
        </script>
    </body>
    </html>
    ''', stats=stats, now=datetime.now())

@app.route('/api/chat', methods=['POST'])
def api_chat():
    """API چت"""
    data = request.json
    question = data.get('message', '').strip()
    
    if not question:
        return jsonify({'error': 'سوال نمی‌تواند خالی باشد'})
    
    results = brain.search_smart(question)
    
    if results:
        return jsonify({
            'answer': results[0]['answer'],
            'confidence': results[0]['score'],
            'found': True
        })
    else:
        brain.record_unanswered(question)
        return jsonify({
            'answer': None,
            'found': False
        })

# ================ پنل مدیریت ================
@app.route('/admin-login', methods=['GET', 'POST'])
def admin_login():
    """صفحه لاگین مدیریت"""
    if request.method == 'POST':
        username = request.form['username']
        password = hashlib.md5(request.form['password'].encode()).hexdigest()
        
        for user in users.values():
            if user.username == username and user.password == password:
                login_user(user)
                return redirect(url_for('admin_panel'))
                
        return "❌ نام کاربری یا رمز عبور اشتباه است"
    
    return render_template_string('''
    <!DOCTYPE html>
    <html>
    <head>
        <title>ورود به پنل مدیریت</title>
        <style>
            body {
                font-family: Tahoma;
                background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
                height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
            }
            .login-box {
                background: white;
                padding: 40px;
                border-radius: 20px;
                box-shadow: 0 10px 40px rgba(0,0,0,0.2);
                width: 350px;
            }
            h2 { text-align: center; color: #333; margin-bottom: 30px; }
            input {
                width: 100%;
                padding: 12px;
                margin: 10px 0;
                border: 2px solid #e0e0e0;
                border-radius: 8px;
                font-family: Tahoma;
            }
            button {
                width: 100%;
                padding: 12px;
                background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
                color: white;
                border: none;
                border-radius: 8px;
                cursor: pointer;
                font-size: 1.1em;
            }
        </style>
    </head>
    <body>
        <div class="login-box">
            <h2>🔐 ورود به پنل مدیریت</h2>
            <form method="POST">
                <input type="text" name="username" placeholder="نام کاربری" value="admin" required>
                <input type="password" name="password" placeholder="رمز عبور" value="admin123" required>
                <button type="submit">ورود</button>
            </form>
        </div>
    </body>
    </html>
    ''')

@app.route('/admin')
@login_required
def admin_panel():
    """پنل مدیریت اصلی"""
    stats = brain.get_stats()
    
    return render_template_string('''
    <!DOCTYPE html>
    <html lang="fa" dir="rtl">
    <head>
        <meta charset="UTF-8">
        <title>پنل مدیریت - تاریخ‌دان هوشمند</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body {
                font-family: Tahoma;
                background: #f5f5f5;
                padding: 20px;
            }
            .header {
                background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
                color: white;
                padding: 20px;
                border-radius: 10px;
                margin-bottom: 20px;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            .stats-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin-bottom: 30px;
            }
            .stat-card {
                background: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                text-align: center;
            }
            .stat-number {
                font-size: 2em;
                color: #1e3c72;
                font-weight: bold;
            }
            .card {
                background: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                margin-bottom: 20px;
            }
            textarea, input[type=text], select {
                width: 100%;
                padding: 10px;
                margin: 10px 0;
                border: 2px solid #e0e0e0;
                border-radius: 5px;
                font-family: Tahoma;
            }
            button {
                background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
                color: white;
                padding: 12px 25px;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                font-size: 1em;
            }
            .file-upload {
                border: 2px dashed #1e3c72;
                padding: 30px;
                text-align: center;
                border-radius: 10px;
                cursor: pointer;
                margin: 20px 0;
            }
            .grid-2 {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 20px;
            }
            .nav-links {
                display: flex;
                gap: 10px;
            }
            .nav-links a {
                color: white;
                text-decoration: none;
                padding: 8px 15px;
                border-radius: 20px;
                background: rgba(255,255,255,0.2);
            }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>⚙️ پنل مدیریت تاریخ‌دان</h1>
            <div class="nav-links">
                <a href="/" target="_blank">🌐 صفحه چت</a>
                <a href="/logout">🚪 خروج</a>
            </div>
        </div>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-number">{{ stats.total }}</div>
                <div>کل دانش</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{{ stats.never_used_count }}</div>
                <div>استفاده نشده</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{{ stats.unanswered_count }}</div>
                <div>سوال بی‌پاسخ</div>
            </div>
        </div>
        
        <div class="grid-2">
            <div class="card">
                <h2>📝 آموزش تکی</h2>
                <form action="/admin/add" method="POST">
                    <input type="text" name="question" placeholder="سوال" required>
                    <textarea name="answer" rows="4" placeholder="جواب" required></textarea>
                    <select name="category">
                        <option>ایران باستان</option>
                        <option>اسلامی</option>
                        <option>معاصر</option>
                        <option>جهان</option>
                    </select>
                    <button type="submit">➕ اضافه کن</button>
                </form>
            </div>
            
            <div class="card">
                <h2>📁 آپلود فایل آموزشی</h2>
                <form action="/admin/upload" method="POST" enctype="multipart/form-data">
                    <div class="file-upload" onclick="document.getElementById('file').click()">
                        <p>📤 برای آپلود کلیک کنید</p>
                        <p style="font-size:0.9em; color:#666;">فرمت: هر خط: سوال | جواب</p>
                    </div>
                    <input type="file" id="file" name="file" style="display:none;" accept=".txt">
                    <button type="submit">📥 آپلود و آموزش</button>
                </form>
            </div>
        </div>
        
        <div class="card">
            <h2>📊 آمار دقیق</h2>
            <h3>دسته‌بندی‌ها:</h3>
            <ul>
            {% for cat, count in stats.categories.items() %}
                <li>{{ cat }}: {{ count }} مورد</li>
            {% endfor %}
            </ul>
            
            <h3>پراستفاده‌ترین‌ها:</h3>
            {% for item in stats.most_used %}
            <div style="background:#f0f0f0; padding:10px; margin:5px 0; border-radius:5px;">
                {{ item.question }} - {{ item.times_used }} بار
            </div>
            {% endfor %}
        </div>
        
        <script>
            function showTab(tab) {
                document.querySelectorAll('.tab-content').forEach(el => el.style.display = 'none');
                document.getElementById(tab + '-tab').style.display = 'block';
            }
        </script>
    </body>
    </html>
    ''', stats=stats)

@app.route('/admin/add', methods=['POST'])
@login_required
def admin_add():
    """اضافه کردن دانش تکی"""
    question = request.form['question']
    answer = request.form['answer']
    category = request.form.get('category', 'عمومی')
    
    success, msg = brain.add_knowledge(question, answer, category)
    
    if success:
        return redirect(url_for('admin_panel'))
    else:
        return f"❌ {msg} <a href='/admin'>بازگشت</a>"

@app.route('/admin/upload', methods=['POST'])
@login_required
def admin_upload():
    """آپلود فایل آموزشی"""
    if 'file' not in request.files:
        return "❌ فایلی انتخاب نشده است"
        
    file = request.files['file']
    if file.filename == '':
        return "❌ نام فایل معتبر نیست"
        
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        count, errors = brain.add_from_file(filepath)
        
        if errors:
            return f"✅ {count} مورد اضافه شد<br>❌ خطاها: {errors} <a href='/admin'>بازگشت</a>"
        else:
            return f"✅ {count} مورد با موفقیت اضافه شد <a href='/admin'>بازگشت</a>"

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

# ================ اجرای برنامه ================
if __name__ == '__main__':
    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║     🤖 ربات تاریخ‌دان هوشمند - نسخه نهایی                 ║
    ╠════════════════════════════════════════════════════════════╣
    ║  📚 دانش فعلی: {} مورد                                     ║
    ║  🌐 صفحه چت: http://localhost:5000                        ║
    ║  🔐 پنل مدیریت: http://localhost:5000/admin-login         ║
    ║  👤 کاربر: admin / رمز: admin123                          ║
    ╚════════════════════════════════════════════════════════════╝
    """.format(len(brain.knowledge_base)))
    
    app.run(debug=True, host='0.0.0.0', port=5000)
