# ultimate_ai_learning.py
from flask import Flask, render_template, request, jsonify, session, render_template_string, redirect, url_for, make_response, flash
from flask_cors import CORS
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user
from werkzeug.utils import secure_filename
import hashlib
import os
import json
import re
import time
import uuid
import pickle
import threading
from datetime import datetime, timedelta
from collections import Counter, defaultdict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import HashingVectorizer

# ================ کتابخانه‌های تحلیل متن ================
import langid
from textblob import TextBlob
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import hazm

# دانلود منابع
try:
    nltk.data.find('tokenizers/punkt')
except:
    nltk.download('punkt')
    nltk.download('stopwords')

app = Flask(__name__)
app.config['SECRET_KEY'] = 'super-secret-key-for-ai-learning'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=365)
CORS(app)

# ایجاد پوشه‌ها
os.makedirs('data', exist_ok=True)
os.makedirs('uploads', exist_ok=True)
os.makedirs('memory', exist_ok=True)
os.makedirs('knowledge_base', exist_ok=True)

# ================ تحلیلگر متن ================
class TextAnalyzer:
    def __init__(self):
        self.normalizer = hazm.Normalizer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set()
        try:
            self.stop_words.update(stopwords.words('persian'))
        except:
            pass
        try:
            self.stop_words.update(stopwords.words('english'))
        except:
            pass
    
    def normalize(self, text):
        text = self.normalizer.normalize(text)
        text = text.replace('ي', 'ی').replace('ك', 'ک')
        text = re.sub(r'[^\w\s\u0600-\u06FF]', ' ', text)
        return re.sub(r'\s+', ' ', text).strip()
    
    def extract_keywords(self, text, top_k=5):
        words = text.split()
        keywords = []
        for word in words:
            if len(word) > 2 and word not in self.stop_words:
                keywords.append(word)
        return keywords[:top_k]

# ================ پردازشگر فایل ================
class FileProcessor:
    """پردازشگر فوق پیشرفته فایل با ۵ روش مختلف"""
    
    def __init__(self):
        self.supported_formats = ['.txt', '.csv', '.json', '.md']
        self.analyzer = TextAnalyzer()
    
    def process_file(self, filepath, filename):
        """پردازش فایل با تشخیص خودکار فرمت"""
        ext = os.path.splitext(filename)[1].lower()
        
        if ext == '.txt':
            return self.process_txt(filepath)
        elif ext == '.csv':
            return self.process_csv(filepath)
        elif ext == '.json':
            return self.process_json(filepath)
        elif ext == '.md':
            return self.process_md(filepath)
        else:
            return self.process_generic(filepath)
    
    def process_txt(self, filepath):
        """پردازش فایل متنی با ۳ فرمت مختلف"""
        items = []
        encodings = ['utf-8', 'cp1256', 'iso-8859-6']
        
        for enc in encodings:
            try:
                with open(filepath, 'r', encoding=enc) as f:
                    content = f.read()
                break
            except:
                continue
        else:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        
        # تشخیص فرمت
        lines = content.split('\n')
        
        # فرمت ۱: سوال | جواب
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            if '|' in line:
                parts = line.split('|', 1)
                if len(parts) == 2:
                    q = parts[0].strip()
                    a = parts[1].strip()
                    if q and a:
                        items.append({
                            'type': 'qa',
                            'question': q,
                            'answer': a,
                            'category': 'general'
                        })
            
            # فرمت ۲: سوال: جواب
            elif ':' in line and len(line.split(':', 1)) == 2:
                q, a = line.split(':', 1)
                items.append({
                    'type': 'qa',
                    'question': q.strip(),
                    'answer': a.strip(),
                    'category': 'general'
                })
            
            # فرمت ۳: متن ساده (هر خط یک دانش)
            elif len(line) > 20:
                items.append({
                    'type': 'info',
                    'content': line,
                    'category': 'general'
                })
        
        return items
    
    def process_csv(self, filepath):
        """پردازش فایل CSV"""
        items = []
        try:
            import csv
            with open(filepath, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) >= 2:
                        items.append({
                            'type': 'qa',
                            'question': row[0].strip(),
                            'answer': row[1].strip(),
                            'category': row[2] if len(row) > 2 else 'general'
                        })
        except:
            pass
        return items
    
    def process_json(self, filepath):
        """پردازش فایل JSON"""
        items = []
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        if 'question' in item and 'answer' in item:
                            items.append({
                                'type': 'qa',
                                'question': item['question'],
                                'answer': item['answer'],
                                'category': item.get('category', 'general')
                            })
            elif isinstance(data, dict):
                if 'data' in data and isinstance(data['data'], list):
                    for item in data['data']:
                        if 'q' in item and 'a' in item:
                            items.append({
                                'type': 'qa',
                                'question': item['q'],
                                'answer': item['a'],
                                'category': item.get('cat', 'general')
                            })
        except:
            pass
        return items
    
    def process_md(self, filepath):
        """پردازش فایل Markdown"""
        items = []
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # استخراج سوال و جواب از Markdown
            lines = content.split('\n')
            current_q = None
            current_a = []
            
            for line in lines:
                if line.startswith('##') or line.startswith('#'):
                    if current_q and current_a:
                        items.append({
                            'type': 'qa',
                            'question': current_q,
                            'answer': '\n'.join(current_a).strip(),
                            'category': 'general'
                        })
                    current_q = line.replace('#', '').strip()
                    current_a = []
                elif current_q:
                    current_a.append(line)
            
            if current_q and current_a:
                items.append({
                    'type': 'qa',
                    'question': current_q,
                    'answer': '\n'.join(current_a).strip(),
                    'category': 'general'
                })
        except:
            pass
        return items
    
    def process_generic(self, filepath):
        """پردازش عمومی"""
        items = []
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # تقسیم به پاراگراف‌ها
            paragraphs = content.split('\n\n')
            for para in paragraphs:
                if len(para) > 50:  # پاراگراف‌های بلند
                    items.append({
                        'type': 'info',
                        'content': para.strip(),
                        'category': 'general'
                    })
        except:
            pass
        return items

# ================ موتور جستجو ================
class SearchEngine:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 3))
        self.knowledge = []
        self.vectors = None
        self.analyzer = TextAnalyzer()
    
    def add_knowledge(self, question, answer, category='general', source='manual'):
        """افزودن دانش"""
        q_norm = self.analyzer.normalize(question)
        
        item = {
            'id': str(uuid.uuid4())[:8],
            'question': q_norm,
            'original_question': question,
            'answer': answer,
            'category': category,
            'source': source,
            'keywords': self.analyzer.extract_keywords(q_norm),
            'created': datetime.now().isoformat(),
            'use_count': 0
        }
        
        self.knowledge.append(item)
        return item
    
    def add_info(self, content, category='general', source='file'):
        """افزودن اطلاعات عمومی"""
        # تبدیل اطلاعات به سوال و جواب فرضی
        sentences = content.split('.')
        for i, sent in enumerate(sentences):
            if len(sent.strip()) > 20:
                item = {
                    'id': str(uuid.uuid4())[:8],
                    'question': f"درباره {sent[:30]}...",
                    'answer': sent.strip(),
                    'category': category,
                    'source': source,
                    'keywords': self.analyzer.extract_keywords(sent),
                    'created': datetime.now().isoformat(),
                    'use_count': 0
                }
                self.knowledge.append(item)
    
    def update_vectors(self):
        """به‌روزرسانی بردارها"""
        if self.knowledge:
            questions = [item['question'] for item in self.knowledge]
            self.vectors = self.vectorizer.fit_transform(questions)
    
    def search(self, query, top_k=3):
        """جستجو در دانش"""
        if not self.knowledge or self.vectors is None:
            return []
        
        q_norm = self.analyzer.normalize(query)
        q_vec = self.vectorizer.transform([q_norm])
        similarities = cosine_similarity(q_vec, self.vectors)[0]
        
        results = []
        for i, score in enumerate(similarities):
            if score > 0.1:
                item = self.knowledge[i]
                results.append({
                    'answer': item['answer'],
                    'score': float(score),
                    'category': item['category']
                })
                item['use_count'] += 1
        
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]
    
    def get_stats(self):
        return {
            'total': len(self.knowledge),
            'used': sum(item.get('use_count', 0) for item in self.knowledge)
        }

# ================ هوش مصنوعی اصلی ================
class UltimateAI:
    def __init__(self):
        self.search = SearchEngine()
        self.analyzer = TextAnalyzer()
        self.file_processor = FileProcessor()
        self.stats = {
            'learned': 0,
            'asked': 0,
            'files_processed': 0
        }
        self.load_knowledge()
    
    def load_knowledge(self):
        """بارگذاری دانش ذخیره شده"""
        kb_file = 'knowledge_base/knowledge.json'
        if os.path.exists(kb_file):
            with open(kb_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for item in data:
                    self.search.knowledge.append(item)
        
        self.search.update_vectors()
        print(f"✅ {len(self.search.knowledge)} دانش بارگذاری شد")
    
    def save_knowledge(self):
        """ذخیره دانش"""
        kb_file = 'knowledge_base/knowledge.json'
        with open(kb_file, 'w', encoding='utf-8') as f:
            json.dump(self.search.knowledge[-10000:], f, ensure_ascii=False, indent=2)
    
    def learn(self, question, answer, category='general'):
        """یادگیری تکی"""
        self.search.add_knowledge(question, answer, category, 'manual')
        self.search.update_vectors()
        self.stats['learned'] += 1
        self.save_knowledge()
        return True
    
    def learn_from_file(self, filepath, filename):
        """یادگیری از فایل"""
        items = self.file_processor.process_file(filepath, filename)
        
        count = 0
        categories = Counter()
        
        for item in items:
            if item['type'] == 'qa':
                self.search.add_knowledge(
                    item['question'],
                    item['answer'],
                    item['category'],
                    'file'
                )
                count += 1
                categories[item['category']] += 1
            
            elif item['type'] == 'info':
                self.search.add_info(
                    item['content'],
                    item['category'],
                    'file'
                )
                count += 1
        
        self.search.update_vectors()
        self.stats['learned'] += count
        self.stats['files_processed'] += 1
        self.save_knowledge()
        
        return {
            'total': count,
            'categories': dict(categories),
            'success': True
        }
    
    def ask(self, question):
        """پرسش و پاسخ"""
        self.stats['asked'] += 1
        results = self.search.search(question)
        
        if results:
            best = results[0]
            return {
                'answer': best['answer'],
                'confidence': best['score'],
                'found': True
            }
        
        return {'answer': None, 'found': False}
    
    def get_stats(self):
        return {
            'knowledge': len(self.search.knowledge),
            'learned': self.stats['learned'],
            'asked': self.stats['asked'],
            'files': self.stats['files_processed']
        }

# ================ نمونه اصلی ================
ai = UltimateAI()

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
    stats = ai.get_stats()
    
    resp = make_response(render_template_string('''
    <!DOCTYPE html>
    <html lang="fa" dir="rtl">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>هوش مصنوعی یادگیرنده</title>
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
            .stats-badge {
                background: rgba(255,255,255,0.2);
                padding: 5px 10px;
                border-radius: 20px;
                font-size: 0.8em;
            }
        </style>
    </head>
    <body>
        <div class="chat-container">
            <div class="chat-header">
                <button class="menu-btn" onclick="toggleMenu()">☰</button>
                <div class="stats-badge">{{ stats.knowledge }} دانش</div>
            </div>
            
            <div class="chat-messages" id="chat-messages">
                <div class="message bot">
                    <div class="message-content">
                        سلام! من هر روز یاد می‌گیرم. هر سوالی داری بپرس!
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
            <h3 style="margin-bottom:20px;">📋 منو</h3>
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
                        ${text.replace(/\\n/g, '<br>')}
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
                        addMessage(data.answer);
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
    ''', stats=stats, now=datetime.now().strftime('%H:%M')))
    
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
                'found': True
            })
        else:
            return jsonify({'answer': None, 'found': False})
            
    except Exception as e:
        return jsonify({'error': str(e)})

# ================ پنل مدیریت با آپلود پیشرفته ================
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
    
    return f'''
    <!DOCTYPE html>
    <html>
    <head>
        <title>پنل مدیریت</title>
        <style>
            *{{margin:0;padding:0;box-sizing:border-box;}}
            body{{font-family:Tahoma;background:#f5f5f5;padding:20px;}}
            .header{{background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);color:white;padding:20px;border-radius:15px;margin-bottom:20px;display:flex;justify-content:space-between;}}
            .stats-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:15px;margin-bottom:20px;}}
            .stat-card{{background:white;padding:20px;border-radius:15px;text-align:center;}}
            .stat-number{{font-size:2.5em;color:#667eea;font-weight:bold;}}
            .card{{background:white;padding:20px;border-radius:15px;margin-bottom:20px;}}
            textarea,input,select{{width:100%;padding:12px;margin:10px 0;border:2px solid #e0e0e0;border-radius:10px;}}
            button{{background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);color:white;padding:12px 25px;border:none;border-radius:10px;cursor:pointer;}}
            .file-upload{{
                border:2px dashed #667eea;
                padding:30px;
                text-align:center;
                border-radius:10px;
                cursor:pointer;
                margin:20px 0;
                background:#f8fafc;
            }}
            .alert-success{{
                background:#d4edda;
                color:#155724;
                padding:15px;
                border-radius:10px;
                margin:10px 0;
            }}
            .grid-2{{
                display:grid;
                grid-template-columns:1fr 1fr;
                gap:20px;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h2>⚙️ پنل مدیریت</h2>
            <div>
                <a href="/" style="color:white;margin-right:15px;">🏠 چت</a>
                <a href="/logout" style="color:white;">🚪 خروج</a>
            </div>
        </div>
        
        <div class="stats-grid">
            <div class="stat-card"><div class="stat-number">{stats['knowledge']}</div><div>کل دانش</div></div>
            <div class="stat-card"><div class="stat-number">{stats['learned']}</div><div>یادگیری‌ها</div></div>
            <div class="stat-card"><div class="stat-number">{stats['asked']}</div><div>سوالات</div></div>
            <div class="stat-card"><div class="stat-number">{stats['files']}</div><div>فایل‌ها</div></div>
        </div>
        
        <div class="grid-2">
            <div class="card">
                <h3>📝 آموزش تکی</h3>
                <form action="/admin/learn" method="POST">
                    <input type="text" name="question" placeholder="سوال" required>
                    <textarea name="answer" rows="4" placeholder="پاسخ" required></textarea>
                    <select name="category">
                        <option value="general">عمومی</option>
                        <option value="history">تاریخ</option>
                        <option value="science">علمی</option>
                        <option value="code">برنامه‌نویسی</option>
                    </select>
                    <button type="submit">📚 یاد بگیر</button>
                </form>
            </div>
            
            <div class="card">
                <h3>📁 آپلود فایل آموزشی</h3>
                <p style="color:#666; margin-bottom:10px;">فرمت‌های پشتیبانی: .txt, .csv, .json, .md</p>
                <form action="/admin/learn/file" method="POST" enctype="multipart/form-data">
                    <div class="file-upload" onclick="document.getElementById('file').click()">
                        <p style="font-size:2em;">📤</p>
                        <p>کلیک برای انتخاب فایل</p>
                        <p style="color:#666; font-size:0.9em; margin-top:10px;">
                            فرمت‌های قابل قبول:<br>
                            • txt: سوال | جواب (هر خط)<br>
                            • csv: سوال,جواب,دسته<br>
                            • json: [{{"question":"...","answer":"..."}}]<br>
                            • md: ## عنوان (متن زیر)
                        </p>
                    </div>
                    <input type="file" id="file" name="file" style="display:none;" accept=".txt,.csv,.json,.md">
                    <button type="submit" style="width:100%;">📥 آپلود و یادگیری</button>
                </form>
            </div>
        </div>
        
        <div class="card">
            <h3>✨ نمونه فرمت فایل</h3>
            <textarea rows="6" readonly style="background:#f8fafc; font-family:monospace;">سلام | سلام! چطور میتونم کمک کنم؟
کوروش کبیر که بود | کوروش بزرگ بنیانگذار هخامنشی بود
حرف ب چیست | حرف ب چهارمین حرف الفبای فارسی است
پایتخت ایران کجاست | تهران پایتخت ایران است

# یا فرمت CSV:
سوال,جواب,دسته
"تاریخ تخت جمشید","۵۱۸ پیش از میلاد","history"</textarea>
            <button onclick="copySample()" style="margin-top:10px;">📋 کپی نمونه</button>
        </div>
        
        <script>
            function copySample() {{
                const textarea = document.querySelector('textarea');
                textarea.select();
                document.execCommand('copy');
                alert('✅ نمونه کپی شد!');
            }}
        </script>
    </body>
    </html>
    '''

@app.route('/admin/learn', methods=['POST'])
@login_required
def learn():
    q = request.form['question']
    a = request.form['answer']
    cat = request.form.get('category', 'general')
    ai.learn(q, a, cat)
    return redirect(url_for('admin_panel'))

@app.route('/admin/learn/file', methods=['POST'])
@login_required
def learn_file():
    try:
        if 'file' not in request.files:
            return '''
            <div style="font-family:Tahoma; padding:20px;">
                <h3>❌ خطا</h3>
                <p>فایلی انتخاب نشده است</p>
                <a href="/admin">🔙 بازگشت</a>
            </div>
            '''
        
        file = request.files['file']
        if file.filename == '':
            return '''
            <div style="font-family:Tahoma; padding:20px;">
                <h3>❌ خطا</h3>
                <p>نام فایل معتبر نیست</p>
                <a href="/admin">🔙 بازگشت</a>
            </div>
            '''
        
        # ذخیره فایل
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # پردازش فایل
        result = ai.learn_from_file(filepath, filename)
        
        # پاک کردن فایل موقت
        os.remove(filepath)
        
        if result['success'] and result['total'] > 0:
            categories = ', '.join([f"{k}: {v}" for k, v in result['categories'].items()])
            return f'''
            <div style="font-family:Tahoma; padding:20px; background:#f5f5f5;">
                <div style="background:#d4edda; color:#155724; padding:20px; border-radius:10px;">
                    <h3>✅ آپلود موفق</h3>
                    <p>تعداد {result['total']} مورد با موفقیت یاد گرفته شد</p>
                    <p>دسته‌بندی: {categories}</p>
                    <p>فایل: {filename}</p>
                    <a href="/admin" style="display:inline-block; margin-top:15px; padding:10px 20px; background:#667eea; color:white; text-decoration:none; border-radius:5px;">🔙 بازگشت به پنل</a>
                </div>
            </div>
            '''
        else:
            return f'''
            <div style="font-family:Tahoma; padding:20px;">
                <div style="background:#fff3cd; color:#856404; padding:20px; border-radius:10px;">
                    <h3>⚠️ هشدار</h3>
                    <p>موردی برای یادگیری یافت نشد!</p>
                    <p>فرمت فایل را بررسی کنید.</p>
                    <a href="/admin" style="display:inline-block; margin-top:15px; padding:10px 20px; background:#667eea; color:white; text-decoration:none; border-radius:5px;">🔙 بازگشت</a>
                </div>
            </div>
            '''
        
    except Exception as e:
        return f'''
        <div style="font-family:Tahoma; padding:20px;">
            <div style="background:#f8d7da; color:#721c24; padding:20px; border-radius:10px;">
                <h3>❌ خطا</h3>
                <p>{str(e)}</p>
                <a href="/admin" style="display:inline-block; margin-top:15px; padding:10px 20px; background:#667eea; color:white; text-decoration:none; border-radius:5px;">🔙 بازگشت</a>
            </div>
        </div>
        '''

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

if __name__ == '__main__':
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║     🤖 هوش مصنوعی فوق‌العاده با قابلیت آپلود فایل       ║
    ╠══════════════════════════════════════════════════════════╣
    ║  📚 دانش: {}                                                ║
    ║  📁 فایل‌های پردازش شده: {}                                 ║
    ║  🔍 فرمت‌های پشتیبانی: txt, csv, json, md                 ║
    ║  🌐 چت: http://localhost:5000                             ║
    ║  🔐 پنل: http://localhost:5000/admin-login                ║
    ║  👤 کاربر: admin / admin123                                ║
    ╚══════════════════════════════════════════════════════════╝
    """.format(ai.get_stats()['knowledge'], ai.get_stats()['files']))
    
    app.run(debug=True, host='0.0.0.0', port=5000)
