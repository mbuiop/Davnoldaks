# professional_ai.py
from flask import Flask, render_template, request, jsonify, session, render_template_string, redirect, url_for
from flask_cors import CORS
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user
import hashlib
import os
import json
import re
import time
from datetime import datetime
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
app.config['SECRET_KEY'] = 'super-secret-key'
app.config['UPLOAD_FOLDER'] = 'uploads'
CORS(app)

os.makedirs('data', exist_ok=True)
os.makedirs('uploads', exist_ok=True)

# ================ پایگاه داده ================
class Database:
    def __init__(self):
        self.file = 'data/knowledge.json'
        self.load()
    
    def load(self):
        if os.path.exists(self.file):
            with open(self.file, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
        else:
            self.data = {
                'knowledge': [],
                'questions': [],
                'stats': {'learned': 0, 'asked': 0}
            }
            self.save()
    
    def save(self):
        with open(self.file, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)

# ================ هسته هوش مصنوعی ================
class PersianAI:
    def __init__(self):
        self.db = Database()
        self.knowledge = self.db.data['knowledge']
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 3))
        self.vectors = None
        self.update_vectors()
    
    def normalize(self, text):
        text = text.replace('ي', 'ی').replace('ك', 'ک')
        text = re.sub(r'[^\w\s\u0600-\u06FF]', ' ', text)
        return re.sub(r'\s+', ' ', text).strip()
    
    def update_vectors(self):
        if self.knowledge:
            questions = [item['q'] for item in self.knowledge]
            self.vectors = self.vectorizer.fit_transform(questions)
    
    def learn(self, question, answer):
        q_norm = self.normalize(question)
        
        # بررسی تکراری
        for item in self.knowledge:
            if item['q'] == q_norm:
                item['a'] = answer
                item['count'] += 1
                self.db.save()
                return True, "به‌روزرسانی شد"
        
        # اضافه کردن جدید
        self.knowledge.append({
            'id': len(self.knowledge) + 1,
            'q': q_norm,
            'a': answer,
            'count': 1,
            'time': datetime.now().isoformat()
        })
        
        self.update_vectors()
        self.db.data['stats']['learned'] += 1
        self.db.save()
        return True, "یاد گرفته شد"
    
    def ask(self, question):
        self.db.data['stats']['asked'] += 1
        q_norm = self.normalize(question)
        
        # 1. جستجوی دقیق
        for item in self.knowledge:
            if item['q'] == q_norm:
                return {'answer': item['a'], 'method': 'دقیق'}
        
        # 2. جستجوی کلمات کلیدی
        q_words = set(q_norm.split())
        best = None
        best_score = 0
        
        for item in self.knowledge:
            item_words = set(item['q'].split())
            common = q_words & item_words
            if common:
                score = len(common) / max(len(q_words), len(item_words))
                if score > best_score and score > 0.3:
                    best_score = score
                    best = item['a']
        
        if best:
            return {'answer': best, 'method': 'کلیدی'}
        
        # 3. جستجوی برداری
        if self.vectors is not None and len(self.knowledge) > 0:
            try:
                q_vec = self.vectorizer.transform([q_norm])
                similarities = cosine_similarity(q_vec, self.vectors)[0]
                best_idx = np.argmax(similarities)
                if similarities[best_idx] > 0.2:
                    return {
                        'answer': self.knowledge[best_idx]['a'],
                        'method': 'برداری'
                    }
            except:
                pass
        
        # ثبت سوال بی‌پاسخ
        self.db.data['questions'].append({
            'q': question,
            'time': datetime.now().isoformat()
        })
        self.db.save()
        
        return {'answer': None, 'method': 'پیدا نشد'}
    
    def get_stats(self):
        return {
            'knowledge': len(self.knowledge),
            'asked': self.db.data['stats']['asked'],
            'learned': self.db.data['stats']['learned']
        }

ai = PersianAI()

# ================ مدیریت کاربران ================
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'admin_login'

class User(UserMixin):
    def __init__(self, id, username, password):
        self.id = id
        self.username = username
        self.password = password

users = {'1': User('1', 'admin', hashlib.md5('admin123'.encode()).hexdigest())}

@login_manager.user_loader
def load_user(user_id):
    return users.get(user_id)

# ================ صفحه اصلی چت ================
@app.route('/')
def index():
    return render_template_string('''
    <!DOCTYPE html>
    <html lang="fa" dir="rtl">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
        <title>هوش مصنوعی ایرانی</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
                -webkit-tap-highlight-color: transparent;
                -webkit-touch-callout: none;
                -webkit-user-select: none;
                user-select: none;
            }
            
            html, body {
                height: 100%;
                overflow: hidden;
                position: fixed;
                width: 100%;
                touch-action: pan-y pinch-zoom;
            }
            
            body {
                font-family: 'Vazir', 'Tahoma', sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                display: flex;
                align-items: center;
                justify-content: center;
                padding: 8px;
            }
            
            .chat-container {
                width: 100%;
                max-width: 500px;
                height: 100%;
                max-height: 800px;
                background: white;
                border-radius: 30px;
                box-shadow: 0 25px 50px rgba(0,0,0,0.3);
                display: flex;
                flex-direction: column;
                overflow: hidden;
                position: relative;
            }
            
            .chat-header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 15px 20px;
                display: flex;
                align-items: center;
                justify-content: space-between;
                flex-shrink: 0;
            }
            
            .menu-btn {
                background: none;
                border: none;
                color: white;
                font-size: 28px;
                cursor: pointer;
                width: 44px;
                height: 44px;
                display: flex;
                align-items: center;
                justify-content: center;
                border-radius: 50%;
                transition: all 0.3s;
            }
            
            .menu-btn:active {
                background: rgba(255,255,255,0.2);
                transform: scale(0.95);
            }
            
            .header-title {
                font-size: 1.3em;
                font-weight: bold;
                display: flex;
                align-items: center;
                gap: 8px;
            }
            
            .chat-messages {
                flex: 1;
                overflow-y: auto;
                padding: 20px;
                background: #f8fafc;
                display: flex;
                flex-direction: column;
                gap: 15px;
                -webkit-overflow-scrolling: touch;
                scroll-behavior: smooth;
            }
            
            .chat-messages::-webkit-scrollbar {
                width: 5px;
            }
            
            .chat-messages::-webkit-scrollbar-track {
                background: #f1f1f1;
            }
            
            .chat-messages::-webkit-scrollbar-thumb {
                background: #667eea;
                border-radius: 10px;
            }
            
            .message {
                display: flex;
                animation: slideIn 0.3s ease;
                width: 100%;
            }
            
            @keyframes slideIn {
                from { opacity: 0; transform: translateY(10px); }
                to { opacity: 1; transform: translateY(0); }
            }
            
            .message.user { justify-content: flex-end; }
            .message.bot { justify-content: flex-start; }
            
            .message-content {
                max-width: 85%;
                padding: 14px 18px;
                border-radius: 25px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                line-height: 1.6;
                font-size: 1rem;
                word-wrap: break-word;
            }
            
            .user .message-content {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
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
            
            .typing-indicator {
                padding: 14px 20px;
                background: white;
                border-radius: 25px;
                display: inline-block;
            }
            
            .typing-indicator span {
                display: inline-block;
                width: 8px;
                height: 8px;
                border-radius: 50%;
                background: #667eea;
                margin: 0 3px;
                animation: typing 1.4s infinite;
            }
            
            .typing-indicator span:nth-child(2) { animation-delay: 0.2s; }
            .typing-indicator span:nth-child(3) { animation-delay: 0.4s; }
            
            @keyframes typing {
                0%, 60%, 100% { transform: translateY(0); }
                30% { transform: translateY(-10px); }
            }
            
            .chat-input-container {
                padding: 15px 20px;
                background: white;
                border-top: 1px solid #eee;
                display: flex;
                gap: 10px;
                align-items: center;
                flex-shrink: 0;
            }
            
            .chat-input {
                flex: 1;
                padding: 14px 18px;
                border: 2px solid #e0e0e0;
                border-radius: 30px;
                font-size: 1rem;
                outline: none;
                font-family: inherit;
                background: #f8fafc;
                -webkit-appearance: none;
                touch-action: manipulation;
            }
            
            .chat-input:focus {
                border-color: #667eea;
                box-shadow: 0 0 0 3px rgba(102,126,234,0.1);
                background: white;
            }
            
            .send-btn {
                width: 52px;
                height: 52px;
                border-radius: 50%;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                cursor: pointer;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 1.4em;
                transition: all 0.3s;
                flex-shrink: 0;
                touch-action: manipulation;
            }
            
            .send-btn:active {
                transform: scale(0.95);
                opacity: 0.9;
            }
            
            .menu-overlay {
                position: fixed;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: rgba(0,0,0,0.5);
                z-index: 1000;
                display: none;
            }
            
            .menu-panel {
                position: fixed;
                top: 0;
                right: -300px;
                width: 280px;
                height: 100%;
                background: white;
                z-index: 1001;
                transition: right 0.3s ease;
                box-shadow: -5px 0 30px rgba(0,0,0,0.2);
                padding: 20px;
                overflow-y: auto;
            }
            
            .menu-panel.open { right: 0; }
            
            .menu-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 30px;
                padding-bottom: 15px;
                border-bottom: 2px solid #eee;
            }
            
            .menu-item {
                padding: 15px;
                margin: 5px 0;
                border-radius: 15px;
                cursor: pointer;
                display: flex;
                align-items: center;
                gap: 15px;
                text-decoration: none;
                color: #333;
                transition: all 0.3s;
                touch-action: manipulation;
            }
            
            .menu-item:active {
                background: #f0f2f5;
                transform: translateX(-5px);
            }
            
            .menu-item.admin {
                background: linear-gradient(135deg, #667eea10 0%, #764ba210 100%);
                border: 1px solid #667eea;
            }
            
            .welcome-message {
                text-align: center;
                padding: 20px;
                background: linear-gradient(135deg, #667eea08 0%, #764ba208 100%);
                border-radius: 20px;
                margin-bottom: 10px;
            }
            
            .welcome-message h3 {
                color: #667eea;
                margin-bottom: 8px;
                font-size: 1.3em;
            }
            
            .welcome-message p {
                color: #666;
                font-size: 0.95em;
            }
            
            @media (max-width: 480px) {
                body { padding: 0; }
                .chat-container {
                    border-radius: 0;
                    max-height: 100%;
                }
                .message-content {
                    max-width: 90%;
                    font-size: 0.95rem;
                }
            }
        </style>
    </head>
    <body>
        <div class="chat-container">
            <div class="chat-header">
                <button class="menu-btn" onclick="toggleMenu()">☰</button>
                <div class="header-title">
                    <span>🤖</span> هوش ایرانی
                </div>
                <div style="width: 44px;"></div>
            </div>
            
            <div class="chat-messages" id="chat-messages">
                <div class="welcome-message">
                    <h3>🌟 به هوش مصنوعی خوش آمدید</h3>
                    <p>هر سوالی دارید بپرسید. من یاد می‌گیرم و بهتر جواب می‌دم!</p>
                </div>
            </div>
            
            <div class="chat-input-container">
                <input type="text" class="chat-input" id="message-input" 
                       placeholder="سوال خود را بپرسید..." 
                       onkeypress="if(event.key==='Enter') sendMessage()"
                       autocomplete="off">
                <button class="send-btn" onclick="sendMessage()">➤</button>
            </div>
        </div>
        
        <div class="menu-overlay" id="menuOverlay" onclick="closeMenu()"></div>
        <div class="menu-panel" id="menuPanel">
            <div class="menu-header">
                <h3>منو</h3>
                <button class="close-menu" onclick="closeMenu()" style="background:none; border:none; font-size:24px;">✕</button>
            </div>
            
            <a href="/m.html" class="menu-item">
                <span style="font-size:1.5em;">📄</span> صفحه M
            </a>
            
            <a href="/admin-login" class="menu-item admin">
                <span style="font-size:1.5em;">⚙️</span> پنل مدیریت
            </a>
            
            <div class="menu-item" onclick="clearHistory()">
                <span style="font-size:1.5em;">🗑️</span> پاک کردن تاریخچه
            </div>
        </div>
        
        <script>
            let chatHistory = JSON.parse(localStorage.getItem('chat_history')) || [];
            let isProcessing = false;
            
            // نمایش تاریخچه
            chatHistory.forEach(msg => {
                addMessage(msg.text, msg.isUser, msg.time, false);
            });
            
            function toggleMenu() {
                document.getElementById('menuOverlay').style.display = 'block';
                document.getElementById('menuPanel').classList.add('open');
            }
            
            function closeMenu() {
                document.getElementById('menuOverlay').style.display = 'none';
                document.getElementById('menuPanel').classList.remove('open');
            }
            
            function addMessage(text, isUser = false, time = null, save = true) {
                const messagesDiv = document.getElementById('chat-messages');
                const messageDiv = document.createElement('div');
                messageDiv.className = `message ${isUser ? 'user' : 'bot'}`;
                
                const now = new Date();
                const messageTime = time || now.toLocaleTimeString('fa-IR', { 
                    hour: '2-digit', 
                    minute: '2-digit'
                });
                
                messageDiv.innerHTML = `
                    <div class="message-content">
                        ${text.replace(/\\n/g, '<br>')}
                        <div class="message-time">${messageTime}</div>
                    </div>
                `;
                
                messagesDiv.appendChild(messageDiv);
                messagesDiv.scrollTo({
                    top: messagesDiv.scrollHeight,
                    behavior: 'smooth'
                });
                
                if (save) {
                    chatHistory.push({ text, isUser, time: messageTime });
                    if (chatHistory.length > 50) {
                        chatHistory = chatHistory.slice(-50);
                    }
                    localStorage.setItem('chat_history', JSON.stringify(chatHistory));
                }
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
                document.getElementById('chat-messages').appendChild(typingDiv);
                document.getElementById('chat-messages').scrollTo({
                    top: document.getElementById('chat-messages').scrollHeight,
                    behavior: 'smooth'
                });
            }
            
            function hideTyping() {
                const typing = document.getElementById('typing-indicator');
                if (typing) typing.remove();
            }
            
            async function sendMessage() {
                if (isProcessing) return;
                
                const input = document.getElementById('message-input');
                const message = input.value.trim();
                if (!message) return;
                
                isProcessing = true;
                addMessage(message, true);
                input.value = '';
                showTyping();
                
                try {
                    const response = await fetch('/api/chat', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({message})
                    });
                    
                    const data = await response.json();
                    hideTyping();
                    
                    if (data.answer) {
                        addMessage(data.answer);
                    } else {
                        addMessage('🤔 متأسفم! نتونستم پاسخ رو پیدا کنم.');
                    }
                    
                } catch (error) {
                    hideTyping();
                    addMessage('⚠️ خطا در ارتباط با سرور');
                } finally {
                    isProcessing = false;
                }
            }
            
            function clearHistory() {
                if (confirm('آیا تاریخچه چت پاک شود؟')) {
                    localStorage.removeItem('chat_history');
                    chatHistory = [];
                    document.getElementById('chat-messages').innerHTML = `
                        <div class="welcome-message">
                            <h3>🌟 به هوش مصنوعی خوش آمدید</h3>
                            <p>هر سوالی دارید بپرسید. من یاد می‌گیرم و بهتر جواب می‌دم!</p>
                        </div>
                    `;
                    closeMenu();
                }
            }
            
            // جلوگیری از حرکت صفحه با دو انگشت
            document.addEventListener('touchmove', function(e) {
                if (e.target.classList.contains('chat-messages')) return;
                e.preventDefault();
            }, { passive: false });
        </script>
    </body>
    </html>
    ''')

# ================ صفحه M ================
@app.route('/m.html')
def m_page():
    return render_template_string('''
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>صفحه M</title>
        <style>
            body {
                font-family: Tahoma;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
                padding: 20px;
                margin: 0;
            }
            .container {
                background: white;
                border-radius: 30px;
                padding: 40px;
                max-width: 600px;
                text-align: center;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            }
            h1 { color: #333; margin-bottom: 20px; }
            p { color: #666; line-height: 1.8; margin-bottom: 30px; }
            .btn {
                display: inline-block;
                padding: 15px 40px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                text-decoration: none;
                border-radius: 30px;
                font-size: 1.2em;
                transition: all 0.3s;
                border: none;
                cursor: pointer;
            }
            .btn:active {
                transform: scale(0.95);
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📄 صفحه M</h1>
            <p>این صفحه مخصوص منوی کشویی است</p>
            <a href="/" class="btn">بازگشت به چت</a>
        </div>
    </body>
    </html>
    ''')

# ================ API چت ================
@app.route('/api/chat', methods=['POST'])
def api_chat():
    try:
        data = request.json
        question = data.get('message', '').strip()
        
        if not question:
            return jsonify({'error': 'سوال نمی‌تواند خالی باشد'})
        
        result = ai.ask(question)
        
        if result['answer']:
            return jsonify({
                'answer': result['answer'],
                'method': result['method'],
                'found': True
            })
        else:
            return jsonify({
                'answer': None,
                'found': False
            })
            
    except Exception as e:
        return jsonify({'error': str(e)})

# ================ پنل مدیریت ================
@app.route('/admin-login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form['username']
        password = hashlib.md5(request.form['password'].encode()).hexdigest()
        
        if username == 'admin' and password == hashlib.md5('admin123'.encode()).hexdigest():
            login_user(users['1'])
            return redirect(url_for('admin_panel'))
        
        return "❌ رمز اشتباه است"
    
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>ورود</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body {
                font-family: Tahoma;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
                padding: 20px;
                margin: 0;
            }
            .login-box {
                background: white;
                padding: 40px;
                border-radius: 30px;
                width: 100%;
                max-width: 400px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            }
            h2 { text-align: center; margin-bottom: 30px; color: #333; }
            input {
                width: 100%;
                padding: 15px;
                margin: 10px 0;
                border: 2px solid #e0e0e0;
                border-radius: 15px;
                font-family: Tahoma;
                font-size: 1rem;
            }
            input:focus {
                border-color: #667eea;
                outline: none;
            }
            button {
                width: 100%;
                padding: 15px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                border-radius: 15px;
                cursor: pointer;
                font-size: 1.1em;
                margin-top: 20px;
            }
            button:active {
                transform: scale(0.98);
            }
        </style>
    </head>
    <body>
        <div class="login-box">
            <h2>🔐 پنل مدیریت</h2>
            <form method="POST">
                <input type="text" name="username" placeholder="نام کاربری" value="admin">
                <input type="password" name="password" placeholder="رمز عبور" value="admin123">
                <button type="submit">ورود</button>
            </form>
        </div>
    </body>
    </html>
    '''

@app.route('/admin')
@login_required
def admin_panel():
    stats = ai.get_stats()
    return f'''
    <!DOCTYPE html>
    <html>
    <head>
        <title>پنل مدیریت</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            body {{
                font-family: Tahoma;
                background: #f5f5f5;
                padding: 20px;
            }}
            .header {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 20px;
                border-radius: 15px;
                margin-bottom: 20px;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }}
            .stats-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                gap: 15px;
                margin-bottom: 20px;
            }}
            .stat-card {{
                background: white;
                padding: 20px;
                border-radius: 15px;
                text-align: center;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }}
            .stat-number {{
                font-size: 2.5em;
                color: #667eea;
                font-weight: bold;
            }}
            .card {{
                background: white;
                padding: 20px;
                border-radius: 15px;
                margin-bottom: 20px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }}
            .card h3 {{ margin-bottom: 15px; color: #333; }}
            textarea, input, select {{
                width: 100%;
                padding: 12px;
                margin: 10px 0;
                border: 2px solid #e0e0e0;
                border-radius: 10px;
                font-family: Tahoma;
            }}
            button {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 12px 25px;
                border: none;
                border-radius: 10px;
                cursor: pointer;
                font-size: 1rem;
            }}
            button:active {{
                transform: scale(0.98);
            }}
            .file-upload {{
                border: 2px dashed #667eea;
                padding: 30px;
                text-align: center;
                border-radius: 10px;
                cursor: pointer;
                margin: 20px 0;
                background: #f8fafc;
            }}
            .grid-2 {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 20px;
            }}
            @media (max-width: 600px) {{
                .grid-2 {{ grid-template-columns: 1fr; }}
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h2>⚙️ پنل مدیریت</h2>
            <div>
                <a href="/" style="color: white; margin-right: 15px;">🏠 چت</a>
                <a href="/logout" style="color: white;">🚪 خروج</a>
            </div>
        </div>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-number">{stats['knowledge']}</div>
                <div>کل دانش</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{stats['learned']}</div>
                <div>یادگیری‌ها</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{stats['asked']}</div>
                <div>سوالات</div>
            </div>
        </div>
        
        <div class="grid-2">
            <div class="card">
                <h3>➕ آموزش تکی</h3>
                <form action="/admin/learn" method="POST">
                    <input type="text" name="question" placeholder="سوال" required>
                    <textarea name="answer" rows="4" placeholder="پاسخ" required></textarea>
                    <button type="submit">📚 یاد بگیر</button>
                </form>
            </div>
            
            <div class="card">
                <h3>📁 آپلود فایل</h3>
                <form action="/admin/learn/file" method="POST" enctype="multipart/form-data">
                    <div class="file-upload" onclick="document.getElementById('file').click()">
                        <p style="font-size: 2em;">📤</p>
                        <p>کلیک برای آپلود فایل</p>
                        <p style="color: #666; margin-top: 10px;">فرمت: سوال | جواب (هر خط)</p>
                    </div>
                    <input type="file" id="file" name="file" style="display:none;" accept=".txt">
                    <button type="submit">📚 یادگیری از فایل</button>
                </form>
            </div>
        </div>
        
        <div class="card">
            <h3>✨ نمونه فرمت فایل</h3>
            <textarea rows="5" readonly style="background: #f8fafc; font-family: monospace;">سلام | سلام! چطور میتونم کمک کنم؟
کوروش کبیر که بود | کوروش بزرگ بنیانگذار هخامنشی بود
حرف ب چیست | حرف ب چهارمین حرف الفبای فارسی است</textarea>
            <button onclick="copyCode()" style="margin-top: 10px;">📋 کپی کن</button>
        </div>
        
        <script>
            function copyCode() {{
                const textarea = document.querySelector('textarea');
                textarea.select();
                document.execCommand('copy');
                alert('✅ کپی شد!');
            }}
        </script>
    </body>
    </html>
    '''

# ================ یادگیری ================
@app.route('/admin/learn', methods=['POST'])
@login_required
def learn():
    question = request.form['question']
    answer = request.form['answer']
    
    success, msg = ai.learn(question, answer)
    return redirect(url_for('admin_panel'))

@app.route('/admin/learn/file', methods=['POST'])
@login_required
def learn_file():
    try:
        if 'file' not in request.files:
            return "❌ فایلی انتخاب نشده"
        
        file = request.files['file']
        if file.filename == '':
            return "❌ نام فایل معتبر نیست"
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        lines = content.strip().split('\n')
        count = 0
        
        for line in lines:
            if '|' in line:
                q, a = line.split('|', 1)
                ai.learn(q.strip(), a.strip())
                count += 1
        
        os.remove(filepath)
        
        return f"✅ {count} مورد یاد گرفته شد<br><a href='/admin'>بازگشت</a>"
        
    except Exception as e:
        return f"❌ خطا: {str(e)}<br><a href='/admin'>بازگشت</a>"

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

if __name__ == '__main__':
    print("""
    ╔════════════════════════════════════════════════════╗
    ║     🤖 هوش مصنوعی ایرانی - نسخه نهایی             ║
    ╠════════════════════════════════════════════════════╣
    ║  📚 دانش: {} مورد                                   ║
    ║  🌐 چت: http://localhost:5000                     ║
    ║  🔐 پنل: http://localhost:5000/admin-login        ║
    ║  👤 کاربر: admin / admin123                       ║
    ║  📱 موبایل: کاملاً ثابت - بدون حرکت با دو انگشت  ║
    ╚════════════════════════════════════════════════════╝
    """.format(len(ai.knowledge)))
    
    app.run(debug=True, host='0.0.0.0', port=5000)
