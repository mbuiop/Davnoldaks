# ultimate_ai_final.py
from flask import Flask, render_template, request, jsonify, session, render_template_string, redirect, url_for, make_response
from flask_cors import CORS
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user
from werkzeug.utils import secure_filename  # این خط رو اضافه کن
import hashlib
import os
import json
import re
import time
import uuid
from datetime import datetime, timedelta
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ================ کتابخانه تشخیص متن ================
import langid  # تشخیص زبان
import textblob  # تحلیل متن
import nltk  # پردازش زبان طبیعی
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# دانلود منابع مورد نیاز nltk (یک بار)
try:
    nltk.data.find('tokenizers/punkt')
except:
    nltk.download('punkt')
    nltk.download('stopwords')

app = Flask(__name__)
app.config['SECRET_KEY'] = 'super-secret-key-for-ai'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024
CORS(app)

# ایجاد پوشه‌ها
os.makedirs('data', exist_ok=True)
os.makedirs('uploads', exist_ok=True)
os.makedirs('code_templates', exist_ok=True)

# ================ تشخیص دهنده متن ================
class TextDetector:
    """تشخیص نوع سوال و تحلیل متن"""
    
    def __init__(self):
        self.question_patterns = {
            'person': r'(کیست|که بود|چه کسی|بیوگرافی|زندگینامه|افراد|شخص)',
            'place': r'(کجاست|کجا|مکان|شهر|کشور|استان|موقعیت|محل)',
            'time': r'(کی|چه زمانی|تاریخ|سال|قرن|دوره|میلادی|شمسی|هجری)',
            'reason': r'(چرا|دلیل|علت|چگونه|چطور|به چه دلیل|به خاطر)',
            'definition': r'(چیست|چه بود|تعریف|توضیح|معنی|مفهوم|یعنی چه)',
            'quantity': r'(چند|تعداد|مقدار|چه قدر|چه اندازه)',
            'comparison': r'(فرق|تفاوت|شباهت|مقایسه|بهتر|بدتر)',
            'code': r'(کد|برنامه|نویسی|پایتون|جاوا|php|html|css|javascript|الگوریتم|تابع)',
            'alphabet': r'(حرف|الفبا|نوشتن|املا|خواندن|صدا|کلمه)'
        }
        
    def detect_language(self, text):
        """تشخیص زبان متن"""
        try:
            lang, confidence = langid.classify(text)
            return lang, confidence
        except:
            return 'fa', 0
    
    def detect_question_type(self, text):
        """تشخیص نوع سوال"""
        text = text.lower()
        for q_type, pattern in self.question_patterns.items():
            if re.search(pattern, text):
                return q_type
        return 'general'
    
    def extract_keywords(self, text):
        """استخراج کلمات کلیدی"""
        # توکنایز
        tokens = word_tokenize(text)
        
        # حذف کلمات ایست
        stop_words = set(stopwords.words('persian') + stopwords.words('english'))
        keywords = [word for word in tokens if word.lower() not in stop_words and len(word) > 2]
        
        return keywords
    
    def analyze_sentiment(self, text):
        """تحلیل احساسات متن"""
        try:
            blob = textblob.TextBlob(text)
            return {
                'polarity': blob.sentiment.polarity,
                'subjectivity': blob.sentiment.subjectivity
            }
        except:
            return {'polarity': 0, 'subjectivity': 0}
    
    def analyze(self, text):
        """تحلیل کامل متن"""
        lang, conf = self.detect_language(text)
        return {
            'text': text,
            'language': lang,
            'lang_confidence': conf,
            'type': self.detect_question_type(text),
            'keywords': self.extract_keywords(text),
            'sentiment': self.analyze_sentiment(text),
            'length': len(text),
            'word_count': len(text.split())
        }

# ================ موتور جستجوی پیشرفته ================
class SearchEngine:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 3))
        self.documents = []
        self.vectors = None
        self.detector = TextDetector()
        
    def add_document(self, question, answer, category):
        """افزودن سند"""
        self.documents.append({
            'q': question,
            'a': answer,
            'cat': category,
            'analysis': self.detector.analyze(question)
        })
        
    def update_vectors(self):
        """به‌روزرسانی بردارها"""
        if self.documents:
            questions = [d['q'] for d in self.documents]
            self.vectors = self.vectorizer.fit_transform(questions)
    
    def search(self, query, top_k=5):
        """جستجوی هوشمند"""
        analysis = self.detector.analyze(query)
        results = []
        
        if not self.documents:
            return results, analysis
        
        # 1. جستجوی برداری
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.vectors)[0]
        
        # 2. تطابق نوع سوال
        for i, doc in enumerate(self.documents):
            score = similarities[i]
            
            # افزایش امتیاز اگر نوع سوال یکی باشه
            if doc['analysis']['type'] == analysis['type']:
                score *= 1.2
            
            # افزایش امتیاز اگر کلمات کلیدی مشترک داشته باشن
            common_keywords = set(doc['analysis']['keywords']) & set(analysis['keywords'])
            if common_keywords:
                score *= (1 + len(common_keywords) * 0.1)
            
            if score > 0.1:
                results.append({
                    'answer': doc['a'],
                    'score': float(score),
                    'category': doc['cat']
                })
        
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k], analysis

# ================ پایگاه داده ================
class Database:
    def __init__(self):
        self.file = 'data/knowledge.json'
        self.code_templates = 'data/code_templates.json'
        self.load()
    
    def load(self):
        if os.path.exists(self.file):
            with open(self.file, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
        else:
            self.data = {
                'knowledge': [],
                'code_templates': [],
                'questions': [],
                'stats': {
                    'learned': 0,
                    'asked': 0,
                    'code_saved': 0
                }
            }
            self.save()
    
    def save(self):
        with open(self.file, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)

# ================ هوش مصنوعی اصلی ================
class PersianAI:
    def __init__(self):
        self.db = Database()
        self.search = SearchEngine()
        self.detector = TextDetector()
        
        # بارگذاری دانش
        for item in self.db.data['knowledge']:
            self.search.add_document(item['q'], item['a'], item.get('cat', 'general'))
        
        self.search.update_vectors()
        print(f"✅ {len(self.db.data['knowledge'])} دانش بارگذاری شد")
    
    def learn(self, question, answer, category='general'):
        """یادگیری دانش جدید"""
        q_norm = self._normalize(question)
        
        # بررسی تکراری
        for item in self.db.data['knowledge']:
            if item['q'] == q_norm:
                item['a'] = answer
                item['count'] = item.get('count', 1) + 1
                self.db.save()
                return True, "به‌روزرسانی شد"
        
        # اضافه کردن جدید
        new_item = {
            'id': str(uuid.uuid4())[:8],
            'q': q_norm,
            'a': answer,
            'cat': category,
            'count': 1,
            'time': datetime.now().isoformat()
        }
        
        self.db.data['knowledge'].append(new_item)
        self.db.data['stats']['learned'] += 1
        self.db.save()
        
        # به‌روزرسانی موتور جستجو
        self.search.add_document(q_norm, answer, category)
        self.search.update_vectors()
        
        return True, "یاد گرفته شد"
    
    def learn_code(self, title, code, description, language='python'):
        """یادگیری کد نویسی"""
        code_item = {
            'id': str(uuid.uuid4())[:8],
            'title': title,
            'code': code,
            'description': description,
            'language': language,
            'time': datetime.now().isoformat()
        }
        
        if 'code_templates' not in self.db.data:
            self.db.data['code_templates'] = []
        
        self.db.data['code_templates'].append(code_item)
        self.db.data['stats']['code_saved'] += 1
        self.db.save()
        
        return True
    
    def ask(self, question):
        """پرسش و پاسخ"""
        self.db.data['stats']['asked'] += 1
        self.db.save()
        
        results, analysis = self.search.search(question)
        
        if results:
            best = results[0]
            return {
                'answer': best['answer'],
                'confidence': f"{best['score']*100:.0f}%",
                'type': analysis['type'],
                'found': True
            }
        
        # ثبت سوال بی‌پاسخ
        self.db.data['questions'].append({
            'q': question,
            'analysis': analysis,
            'time': datetime.now().isoformat()
        })
        self.db.save()
        
        return {
            'answer': None,
            'found': False
        }
    
    def get_code_templates(self, language=None):
        """گرفتن قالب‌های کد"""
        templates = self.db.data.get('code_templates', [])
        if language:
            templates = [t for t in templates if t['language'] == language]
        return templates
    
    def _normalize(self, text):
        text = text.replace('ي', 'ی').replace('ك', 'ک')
        text = re.sub(r'[^\w\s\u0600-\u06FF]', ' ', text)
        return re.sub(r'\s+', ' ', text).strip()
    
    def get_stats(self):
        return {
            'knowledge': len(self.db.data['knowledge']),
            'codes': len(self.db.data.get('code_templates', [])),
            'asked': self.db.data['stats']['asked'],
            'learned': self.db.data['stats']['learned']
        }

# ================ نمونه اصلی ================
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
    user_id = request.cookies.get('user_id', str(uuid.uuid4()))
    resp = make_response(render_template_string('''
    <!DOCTYPE html>
    <html lang="fa" dir="rtl">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>هوش مصنوعی پیشرفته</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body {
                font-family: 'Vazir', 'Tahoma', sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
                padding: 10px;
            }
            .chat-container {
                width: 100%;
                max-width: 500px;
                height: 95vh;
                background: white;
                border-radius: 30px;
                box-shadow: 0 25px 50px rgba(0,0,0,0.3);
                display: flex;
                flex-direction: column;
                overflow: hidden;
            }
            .chat-header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 15px 20px;
                display: flex;
                align-items: center;
                justify-content: space-between;
            }
            .menu-btn {
                background: none;
                border: none;
                color: white;
                font-size: 28px;
                cursor: pointer;
                width: 44px;
                height: 44px;
            }
            .chat-messages {
                flex: 1;
                overflow-y: auto;
                padding: 20px;
                background: #f8fafc;
                display: flex;
                flex-direction: column;
                gap: 15px;
            }
            .message {
                display: flex;
                animation: slideIn 0.3s ease;
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
            }
            .chat-input {
                flex: 1;
                padding: 14px 18px;
                border: 2px solid #e0e0e0;
                border-radius: 30px;
                font-size: 1rem;
                outline: none;
                font-family: inherit;
            }
            .chat-input:focus {
                border-color: #667eea;
            }
            .send-btn {
                width: 52px;
                height: 52px;
                border-radius: 50%;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                cursor: pointer;
                font-size: 1.4em;
            }
            .menu-panel {
                position: fixed;
                top: 0;
                right: -300px;
                width: 280px;
                height: 100%;
                background: white;
                transition: right 0.3s ease;
                box-shadow: -5px 0 30px rgba(0,0,0,0.2);
                padding: 20px;
                z-index: 1001;
            }
            .menu-panel.open { right: 0; }
            .menu-overlay {
                position: fixed;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: rgba(0,0,0,0.5);
                display: none;
                z-index: 1000;
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
            }
            .menu-item:hover { background: #f0f2f5; }
        </style>
    </head>
    <body>
        <div class="chat-container">
            <div class="chat-header">
                <button class="menu-btn" onclick="toggleMenu()">☰</button>
                <div>🤖 هوش پیشرفته</div>
                <div style="width:44px;"></div>
            </div>
            
            <div class="chat-messages" id="chat-messages">
                <div class="message bot">
                    <div class="message-content">
                        سلام! هر سوالی دارید بپرسید. من تشخیص می‌دم چی می‌خوای!
                        <div class="message-time">{{ now }}</div>
                    </div>
                </div>
            </div>
            
            <div class="chat-input-container">
                <input type="text" class="chat-input" id="message-input" 
                       placeholder="سوال خود را بپرسید..." 
                       onkeypress="if(event.key==='Enter') sendMessage()">
                <button class="send-btn" onclick="sendMessage()">➤</button>
            </div>
        </div>
        
        <div class="menu-overlay" id="menuOverlay" onclick="closeMenu()"></div>
        <div class="menu-panel" id="menuPanel">
            <h3 style="margin-bottom:20px;">منو</h3>
            <a href="/m.html" class="menu-item">📄 صفحه M</a>
            <a href="/admin-login" class="menu-item">⚙️ پنل مدیریت</a>
            <div class="menu-item" onclick="clearHistory()">🗑️ پاک کردن</div>
        </div>
        
        <script>
            let chatHistory = JSON.parse(localStorage.getItem('chat_history')) || [];
            
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
                const div = document.createElement('div');
                div.className = `message ${isUser ? 'user' : 'bot'}`;
                
                const msgTime = time || new Date().toLocaleTimeString('fa-IR');
                
                div.innerHTML = `
                    <div class="message-content">
                        ${text}
                        <div class="message-time">${msgTime}</div>
                    </div>
                `;
                
                document.getElementById('chat-messages').appendChild(div);
                div.scrollIntoView({ behavior: 'smooth' });
                
                if (save) {
                    chatHistory.push({ text, isUser, time: msgTime });
                    if (chatHistory.length > 50) chatHistory = chatHistory.slice(-50);
                    localStorage.setItem('chat_history', JSON.stringify(chatHistory));
                }
            }
            
            function showTyping() {
                const div = document.createElement('div');
                div.className = 'message bot';
                div.id = 'typing';
                div.innerHTML = '<div class="typing-indicator"><span></span><span></span><span></span></div>';
                document.getElementById('chat-messages').appendChild(div);
                div.scrollIntoView({ behavior: 'smooth' });
            }
            
            function hideTyping() {
                const typing = document.getElementById('typing');
                if (typing) typing.remove();
            }
            
            async function sendMessage() {
                const input = document.getElementById('message-input');
                const message = input.value.trim();
                if (!message) return;
                
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
                        let answer = data.answer;
                        if (data.confidence) {
                            answer += `\\n\\n✨ اعتماد: ${data.confidence}`;
                        }
                        addMessage(answer);
                    } else {
                        addMessage('🤔 متأسفم! نتونستم پیدا کنم.');
                    }
                    
                } catch (error) {
                    hideTyping();
                    addMessage('⚠️ خطا');
                }
            }
            
            function clearHistory() {
                if (confirm('پاک شود؟')) {
                    localStorage.removeItem('chat_history');
                    chatHistory = [];
                    location.reload();
                }
            }
        </script>
    </body>
    </html>
    ''', now=datetime.now().strftime('%H:%M')))
    
    resp.set_cookie('user_id', user_id, max_age=365*24*60*60)
    return resp

@app.route('/m.html')
def m_page():
    return '''
    <!DOCTYPE html>
    <html>
    <head><title>صفحه M</title>
    <style>body{font-family:Tahoma;background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);min-height:100vh;display:flex;align-items:center;justify-content:center;padding:20px;}.container{background:white;border-radius:30px;padding:40px;max-width:600px;text-align:center;}.btn{display:inline-block;padding:15px 40px;background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);color:white;text-decoration:none;border-radius:30px;margin-top:20px;}</style>
    </head>
    <body><div class="container"><h1>📄 صفحه M</h1><p>صفحه مخصوص منو</p><a href="/" class="btn">بازگشت</a></div></body>
    </html>
    '''

@app.route('/api/chat', methods=['POST'])
def api_chat():
    try:
        data = request.json
        question = data.get('message', '').strip()
        
        if not question:
            return jsonify({'error': 'سوال خالی است'})
        
        result = ai.ask(question)
        
        if result['found']:
            return jsonify({
                'answer': result['answer'],
                'confidence': result.get('confidence', ''),
                'found': True
            })
        else:
            return jsonify({'answer': None, 'found': False})
            
    except Exception as e:
        return jsonify({'error': str(e)})

# ================ پنل مدیریت با ۵ بخش کدنویسی ================
@app.route('/admin-login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form['username']
        password = hashlib.md5(request.form['password'].encode()).hexdigest()
        
        if username == 'admin' and password == hashlib.md5('admin123'.encode()).hexdigest():
            login_user(users['1'])
            return redirect(url_for('admin_panel'))
        
        return "❌ رمز اشتباه"
    
    return '''
    <!DOCTYPE html>
    <html>
    <head><title>ورود</title>
    <style>body{font-family:Tahoma;background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);height:100vh;display:flex;align-items:center;justify-content:center;}.login-box{background:white;padding:40px;border-radius:30px;width:400px;}input,button{width:100%;padding:15px;margin:10px 0;border-radius:15px;border:2px solid #e0e0e0;}button{background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);color:white;border:none;cursor:pointer;}</style>
    </head>
    <body><div class="login-box"><h2>🔐 ورود</h2><form method="POST"><input name="username" value="admin"><input name="password" type="password" value="admin123"><button type="submit">ورود</button></form></div></body>
    </html>
    '''

@app.route('/admin')
@login_required
def admin_panel():
    stats = ai.get_stats()
    codes = ai.get_code_templates()
    
    return f'''
    <!DOCTYPE html>
    <html>
    <head><title>پنل مدیریت</title>
    <style>
        *{{margin:0;padding:0;box-sizing:border-box;}}
        body{{font-family:Tahoma;background:#f5f5f5;padding:20px;}}
        .header{{background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);color:white;padding:20px;border-radius:15px;margin-bottom:20px;display:flex;justify-content:space-between;}}
        .stats-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:15px;margin-bottom:20px;}}
        .stat-card{{background:white;padding:20px;border-radius:15px;text-align:center;}}
        .stat-number{{font-size:2.5em;color:#667eea;font-weight:bold;}}
        .card{{background:white;padding:20px;border-radius:15px;margin-bottom:20px;}}
        .card h3{{margin-bottom:15px;color:#333;border-bottom:2px solid #667eea;padding-bottom:5px;}}
        textarea,input,select{{width:100%;padding:12px;margin:10px 0;border:2px solid #e0e0e0;border-radius:10px;font-family:monospace;}}
        button{{background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);color:white;padding:12px 25px;border:none;border-radius:10px;cursor:pointer;margin:5px;}}
        .grid-5{{
            display:grid;
            grid-template-columns:repeat(5,1fr);
            gap:15px;
            margin-bottom:20px;
        }}
        .code-box{{
            background:#1e1e1e;
            color:#fff;
            padding:15px;
            border-radius:10px;
            font-family:monospace;
            white-space:pre-wrap;
            margin:10px 0;
            max-height:200px;
            overflow-y:auto;
        }}
        .code-item{{
            background:#f8fafc;
            padding:15px;
            margin:10px 0;
            border-radius:10px;
            border-right:4px solid #667eea;
        }}
        .copy-btn{{
            background:#28a745;
            color:white;
            border:none;
            padding:8px 15px;
            border-radius:5px;
            cursor:pointer;
        }}
        @media (max-width:1000px){{.grid-5{{grid-template-columns:repeat(2,1fr);}}}}
        @media (max-width:600px){{.grid-5{{grid-template-columns:1fr;}}}}
    </style>
    </head>
    <body>
        <div class="header">
            <h2>⚙️ پنل مدیریت هوش مصنوعی</h2>
            <div>
                <a href="/" style="color:white;margin-right:15px;">🏠 چت</a>
                <a href="/logout" style="color:white;">🚪 خروج</a>
            </div>
        </div>
        
        <div class="stats-grid">
            <div class="stat-card"><div class="stat-number">{stats['knowledge']}</div><div>دانش</div></div>
            <div class="stat-card"><div class="stat-number">{stats['codes']}</div><div>کدها</div></div>
            <div class="stat-card"><div class="stat-number">{stats['asked']}</div><div>سوالات</div></div>
        </div>
        
        <!-- ۵ بخش کدنویسی -->
        <h2 style="margin:20px 0;">📚 ۵ بخش آموزش کدنویسی</h2>
        <div class="grid-5">
            <div class="card">
                <h3>🐍 پایتون</h3>
                <form action="/admin/learn/code" method="POST">
                    <input type="hidden" name="language" value="python">
                    <input type="text" name="title" placeholder="عنوان کد" required>
                    <textarea name="code" rows="4" placeholder="کد پایتون" required></textarea>
                    <textarea name="description" rows="2" placeholder="توضیحات"></textarea>
                    <button type="submit">➕ ذخیره</button>
                </form>
            </div>
            
            <div class="card">
                <h3>☕ جاوا</h3>
                <form action="/admin/learn/code" method="POST">
                    <input type="hidden" name="language" value="java">
                    <input type="text" name="title" placeholder="عنوان کد" required>
                    <textarea name="code" rows="4" placeholder="کد جاوا" required></textarea>
                    <textarea name="description" rows="2" placeholder="توضیحات"></textarea>
                    <button type="submit">➕ ذخیره</button>
                </form>
            </div>
            
            <div class="card">
                <h3>🌐 PHP</h3>
                <form action="/admin/learn/code" method="POST">
                    <input type="hidden" name="language" value="php">
                    <input type="text" name="title" placeholder="عنوان کد" required>
                    <textarea name="code" rows="4" placeholder="کد PHP" required></textarea>
                    <textarea name="description" rows="2" placeholder="توضیحات"></textarea>
                    <button type="submit">➕ ذخیره</button>
                </form>
            </div>
            
            <div class="card">
                <h3>🎨 HTML/CSS</h3>
                <form action="/admin/learn/code" method="POST">
                    <input type="hidden" name="language" value="html">
                    <input type="text" name="title" placeholder="عنوان کد" required>
                    <textarea name="code" rows="4" placeholder="کد HTML" required></textarea>
                    <textarea name="description" rows="2" placeholder="توضیحات"></textarea>
                    <button type="submit">➕ ذخیره</button>
                </form>
            </div>
            
            <div class="card">
                <h3>⚡ JavaScript</h3>
                <form action="/admin/learn/code" method="POST">
                    <input type="hidden" name="language" value="javascript">
                    <input type="text" name="title" placeholder="عنوان کد" required>
                    <textarea name="code" rows="4" placeholder="کد JavaScript" required></textarea>
                    <textarea name="description" rows="2" placeholder="توضیحات"></textarea>
                    <button type="submit">➕ ذخیره</button>
                </form>
            </div>
        </div>
        
        <!-- آموزش معمولی -->
        <div class="card">
            <h3>📝 آموزش معمولی</h3>
            <form action="/admin/learn" method="POST">
                <input type="text" name="question" placeholder="سوال" required>
                <textarea name="answer" rows="3" placeholder="پاسخ" required></textarea>
                <select name="category">
                    <option value="general">عمومی</option>
                    <option value="history">تاریخ</option>
                    <option value="science">علمی</option>
                    <option value="code">برنامه‌نویسی</option>
                </select>
                <button type="submit">📚 یاد بگیر</button>
            </form>
        </div>
        
        <!-- آپلود فایل -->
        <div class="card">
            <h3>📁 آپلود فایل</h3>
            <form action="/admin/learn/file" method="POST" enctype="multipart/form-data">
                <input type="file" name="file" accept=".txt" required>
                <button type="submit">📤 آپلود و یادگیری</button>
            </form>
        </div>
        
        <!-- کدهای ذخیره شده -->
        <div class="card">
            <h3>📋 کدهای ذخیره شده (برای کپی)</h3>
            <div style="max-height:400px; overflow-y:auto;">
                {''.join([f'''
                <div class="code-item">
                    <strong>{c['title']}</strong> <span style="color:#667eea;">({c['language']})</span>
                    <p style="color:#666; margin:5px 0;">{c.get('description', '')}</p>
                    <div class="code-box">{c['code']}</div>
                    <button class="copy-btn" onclick="copyCode(`{c['code'].replace('`', '\\`')}`)">📋 کپی کد</button>
                </div>
                ''' for c in codes[-10:]])}
            </div>
        </div>
        
        <script>
            function copyCode(code) {
                navigator.clipboard.writeText(code).then(() => {{
                    alert('✅ کد کپی شد!');
                }});
            }
        </script>
    </body>
    </html>
    '''

# ================ آموزش ================
@app.route('/admin/learn', methods=['POST'])
@login_required
def learn():
    q = request.form['question']
    a = request.form['answer']
    cat = request.form.get('category', 'general')
    ai.learn(q, a, cat)
    return redirect(url_for('admin_panel'))

@app.route('/admin/learn/code', methods=['POST'])
@login_required
def learn_code():
    title = request.form['title']
    code = request.form['code']
    desc = request.form.get('description', '')
    lang = request.form['language']
    
    ai.learn_code(title, code, desc, lang)
    return redirect(url_for('admin_panel'))

@app.route('/admin/learn/file', methods=['POST'])
@login_required
def learn_file():
    try:
        file = request.files['file']
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        count = 0
        for line in content.split('\n'):
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
    ╔══════════════════════════════════════════════════════════╗
    ║     🤖 هوش مصنوعی پیشرفته با تشخیص متن                  ║
    ╠══════════════════════════════════════════════════════════╣
    ║  📚 دانش: {}                                               ║
    ║  📝 کدها: {}                                                ║
    ║  🔍 تشخیص: زبان + نوع سوال + کلمات کلیدی                ║
    ║  🌐 چت: http://localhost:5000                             ║
    ║  🔐 پنل: http://localhost:5000/admin-login                ║
    ║  👤 کاربر: admin / admin123                                ║
    ║  📱 ۵ بخش کدنویسی: پایتون، جاوا، PHP، HTML، JavaScript   ║
    ╚══════════════════════════════════════════════════════════╝
    """.format(ai.get_stats()['knowledge'], ai.get_stats()['codes']))
    
    app.run(debug=True, host='0.0.0.0', port=5000)
