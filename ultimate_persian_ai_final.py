# ultimate_persian_ai_final.py
from flask import Flask, render_template, request, jsonify, session, render_template_string, redirect, url_for
from flask_cors import CORS
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import json
import os
import hashlib
from datetime import datetime, timedelta
from werkzeug.utils import secure_filename
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter, defaultdict
import re
import math
import string
import random
import time
import traceback

app = Flask(__name__)
app.config['SECRET_KEY'] = 'ultra-persian-ai-super-secret-key-2024'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200MB
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=365)
app.config['SESSION_REFRESH_EACH_REQUEST'] = True
app.config['SESSION_PERMANENT'] = True
CORS(app)

# ایجاد پوشه‌ها با خطاگیری
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('data', exist_ok=True)
os.makedirs('backup', exist_ok=True)
os.makedirs('logs', exist_ok=True)

# ================ پردازشگر فارسی ساده ================
class PersianTextProcessor:
    def __init__(self):
        # الفبای کامل فارسی
        self.persian_alphabet = {
            'آ': 'الف با کلاه', 'ا': 'الف', 'ب': 'به', 'پ': 'په', 'ت': 'ته',
            'ث': 'ثه', 'ج': 'جیم', 'چ': 'چه', 'ح': 'حه', 'خ': 'خه',
            'د': 'دال', 'ذ': 'ذال', 'ر': 'ره', 'ز': 'زه', 'ژ': 'ژه',
            'س': 'سین', 'ش': 'شین', 'ص': 'صاد', 'ض': 'ضاد', 'ط': 'طا',
            'ظ': 'ظا', 'ع': 'عین', 'غ': 'غین', 'ف': 'فه', 'ق': 'قاف',
            'ک': 'کاف', 'گ': 'گاف', 'ل': 'لام', 'م': 'میم', 'ن': 'نون',
            'و': 'واو', 'ه': 'هه', 'ی': 'یه'
        }
        
        # حرکات
        self.diacritics = {
            'َ': 'فتحه', 'ِ': 'کسره', 'ُ': 'ضمه', 'ّ': 'تشدید', 'ْ': 'سکون'
        }
        
        # کلمات پرسشی
        self.question_words = ['کیست', 'کی', 'کجاست', 'چیست', 'چرا', 'چطور', 'چگونه', 'کدام', 'آیا']

    def normalize(self, text):
        """نرمال‌سازی ساده متن"""
        if not text:
            return ""
        text = text.replace('ي', 'ی').replace('ك', 'ک')
        text = re.sub(r'[^\w\s\u0600-\u06FF]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

# ================ دیتابیس دائمی ================
class PermanentDatabase:
    def __init__(self, filename='data/ai_db.json'):
        self.filename = filename
        self.data = {
            'knowledge_base': [],
            'users_questions': [],
            'stats': {
                'total_questions': 0,
                'answered': 0,
                'unanswered': 0
            }
        }
        self.load()
    
    def load(self):
        if os.path.exists(self.filename):
            try:
                with open(self.filename, 'r', encoding='utf-8') as f:
                    self.data.update(json.load(f))
                print(f"💾 {len(self.data['knowledge_base'])} دانش بارگذاری شد")
            except:
                self.save()
        else:
            self.save()
    
    def save(self):
        with open(self.filename, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)

# ================ هسته اصلی هوش مصنوعی ================
class PersianAI:
    def __init__(self):
        self.db = PermanentDatabase()
        self.processor = PersianTextProcessor()
        self.knowledge_base = self.db.data['knowledge_base']
        self.users_questions = self.db.data['users_questions']
        self.stats = self.db.data['stats']
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 3))
        self.question_vectors = None
        self.initialize_data()
        self.update_vectors()
    
    def initialize_data(self):
        """اطلاعات اولیه"""
        if len(self.knowledge_base) == 0:
            initial_data = [
                {"id": 1, "question": "سلام", "answer": "سلام! چطور میتونم کمک کنم؟", "category": "عمومی"},
                {"id": 2, "question": "چطوری", "answer": "خوبم، ممنون! شما چطورین؟", "category": "عمومی"},
                {"id": 3, "question": "خداحافظ", "answer": "خداحافظ! روز خوبی داشته باشید", "category": "عمومی"},
                {"id": 4, "question": "حرف ب چیست", "answer": "حرف 'ب' چهارمین حرف الفبای فارسی است. مثال: باران، باد، بهار", "category": "الفبا"},
                {"id": 5, "question": "کوروش کبیر که بود", "answer": "کوروش بزرگ بنیانگذار شاهنشاهی هخامنشی بود.", "category": "تاریخ"},
            ]
            self.knowledge_base = initial_data
            self.db.data['knowledge_base'] = self.knowledge_base
            self.db.save()
    
    def update_vectors(self):
        if self.knowledge_base:
            questions = [item['question'] for item in self.knowledge_base]
            try:
                self.question_vectors = self.vectorizer.fit_transform(questions)
            except:
                self.question_vectors = None
    
    def search(self, query):
        """جستجوی هوشمند با ۳ الگوریتم"""
        query = self.processor.normalize(query)
        results = []
        
        if not self.knowledge_base:
            return results, "هیچ"
        
        # الگوریتم ۱: تطابق دقیق
        for item in self.knowledge_base:
            if query == item['question']:
                results.append({
                    'id': item['id'],
                    'answer': item['answer'],
                    'score': 1.0,
                    'method': 'دقیق'
                })
        
        # الگوریتم ۲: کلمات کلیدی
        if not results:
            query_words = set(query.split())
            for item in self.knowledge_base:
                item_words = set(item['question'].split())
                common = query_words & item_words
                if common:
                    score = len(common) / max(len(query_words), len(item_words))
                    if score > 0.3:
                        results.append({
                            'id': item['id'],
                            'answer': item['answer'],
                            'score': score,
                            'method': 'کلیدی'
                        })
        
        # الگوریتم ۳: شباهت برداری
        if not results and self.question_vectors is not None:
            try:
                query_vector = self.vectorizer.transform([query])
                similarities = cosine_similarity(query_vector, self.question_vectors)[0]
                best_idx = np.argmax(similarities)
                if similarities[best_idx] > 0.2:
                    item = self.knowledge_base[best_idx]
                    results.append({
                        'id': item['id'],
                        'answer': item['answer'],
                        'score': float(similarities[best_idx]),
                        'method': 'برداری'
                    })
            except:
                pass
        
        # مرتب‌سازی نتایج
        results.sort(key=lambda x: x['score'], reverse=True)
        
        # تشخیص کیفیت
        quality = "هیچ"
        if results:
            if results[0]['score'] >= 0.8:
                quality = "عالی"
            elif results[0]['score'] >= 0.6:
                quality = "خوب"
            elif results[0]['score'] >= 0.4:
                quality = "متوسط"
            else:
                quality = "ضعیف"
        
        return results[:3], quality
    
    def process_question(self, question):
        """پردازش سوال کاربر"""
        question = question.strip()
        
        # ثبت سوال
        self.stats['total_questions'] += 1
        record = {
            'id': len(self.users_questions) + 1,
            'question': question,
            'time': datetime.now().isoformat(),
            'answered': False
        }
        self.users_questions.append(record)
        self.stats['unanswered'] += 1
        
        if len(self.users_questions) > 1000:
            self.users_questions = self.users_questions[-1000:]
        
        self.db.data['users_questions'] = self.users_questions
        self.db.data['stats'] = self.stats
        self.db.save()
        
        # جستجو
        results, quality = self.search(question)
        
        if results:
            best = results[0]
            
            # به‌روزرسانی آمار
            for item in self.knowledge_base:
                if item['id'] == best['id']:
                    item['times_used'] = item.get('times_used', 0) + 1
                    self.stats['answered'] += 1
                    self.stats['unanswered'] -= 1
                    break
            
            self.db.save()
            
            return {
                'answer': best['answer'],
                'quality': quality,
                'method': best.get('method', ''),
                'found': True
            }
        
        return {'answer': None, 'found': False}
    
    def add_knowledge(self, question, answer, category='عمومی'):
        """افزودن دانش جدید"""
        # بررسی تکراری
        for item in self.knowledge_base:
            if item['question'].lower() == question.lower():
                return False, "این سوال قبلاً ثبت شده"
        
        new_item = {
            'id': len(self.knowledge_base) + 1,
            'question': self.processor.normalize(question),
            'answer': answer,
            'category': category,
            'date': datetime.now().isoformat(),
            'times_used': 0
        }
        
        self.knowledge_base.append(new_item)
        self.update_vectors()
        self.db.data['knowledge_base'] = self.knowledge_base
        self.db.save()
        return True, "اضافه شد"
    
    def bulk_import(self, text):
        """وارد کردن گروهی از متن"""
        lines = text.strip().split('\n')
        count = 0
        errors = []
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            if '|' in line:
                parts = line.split('|', 1)
                if len(parts) == 2:
                    q, a = parts
                    success, msg = self.add_knowledge(q.strip(), a.strip(), 'imported')
                    if success:
                        count += 1
                    else:
                        errors.append(f"خط {i+1}: {msg}")
        
        return count, errors
    
    def get_unanswered(self):
        """گرفتن سوالات بی‌پاسخ"""
        return [q for q in self.users_questions if not q['answered']][-20:]
    
    def get_stats(self):
        return {
            'knowledge': len(self.knowledge_base),
            'questions': self.stats['total_questions'],
            'answered': self.stats['answered'],
            'unanswered': self.stats['unanswered']
        }

# ================ نمونه اصلی ================
ai = PersianAI()
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
            }
            
            :root {
                --primary: #6c5ce7;
                --secondary: #a363d9;
                --dark: #2d3436;
                --light: #f5f6fa;
                --glass: rgba(255, 255, 255, 0.98);
            }
            
            html, body {
                height: 100%;
                overflow: hidden;
            }
            
            body {
                font-family: 'Vazir', 'Tahoma', sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                display: flex;
                align-items: center;
                justify-content: center;
                padding: 12px;
            }
            
            .chat-container {
                width: 100%;
                max-width: 500px;
                height: 100%;
                max-height: 800px;
                background: var(--glass);
                backdrop-filter: blur(10px);
                border-radius: 40px;
                box-shadow: 0 25px 50px rgba(0,0,0,0.3);
                display: flex;
                flex-direction: column;
                overflow: hidden;
                position: relative;
            }
            
            .chat-header {
                background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
                color: white;
                padding: 20px;
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
            
            .menu-btn:hover {
                background: rgba(255,255,255,0.2);
            }
            
            .header-title {
                font-size: 1.4em;
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
                gap: 16px;
                scroll-behavior: smooth;
            }
            
            .chat-messages::-webkit-scrollbar {
                width: 6px;
            }
            
            .chat-messages::-webkit-scrollbar-track {
                background: #f1f1f1;
            }
            
            .chat-messages::-webkit-scrollbar-thumb {
                background: var(--primary);
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
            
            .message.user {
                justify-content: flex-end;
            }
            
            .message.bot {
                justify-content: flex-start;
            }
            
            .message-content {
                max-width: 85%;
                padding: 14px 18px;
                border-radius: 25px;
                position: relative;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                line-height: 1.6;
                font-size: 1rem;
                word-wrap: break-word;
                white-space: pre-wrap;
            }
            
            .user .message-content {
                background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
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
                background: var(--primary);
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
                padding: 16px 20px;
                background: white;
                border-top: 1px solid #eee;
                display: flex;
                gap: 12px;
                align-items: center;
                flex-shrink: 0;
            }
            
            .chat-input {
                flex: 1;
                padding: 14px 20px;
                border: 2px solid #e0e0e0;
                border-radius: 30px;
                font-size: 1rem;
                outline: none;
                transition: all 0.3s;
                font-family: inherit;
                background: #f8fafc;
            }
            
            .chat-input:focus {
                border-color: var(--primary);
                box-shadow: 0 0 0 3px rgba(108,92,231,0.1);
                background: white;
            }
            
            .send-btn {
                width: 52px;
                height: 52px;
                border-radius: 50%;
                background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
                color: white;
                border: none;
                cursor: pointer;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 1.4em;
                transition: all 0.3s;
                flex-shrink: 0;
            }
            
            .send-btn:hover {
                transform: scale(1.1) rotate(5deg);
                box-shadow: 0 5px 15px rgba(108,92,231,0.3);
            }
            
            /* منوی کشویی */
            .menu-overlay {
                position: fixed;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: rgba(0,0,0,0.5);
                z-index: 1000;
                display: none;
                backdrop-filter: blur(5px);
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
            
            .menu-panel.open {
                right: 0;
            }
            
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
                transition: all 0.3s;
                display: flex;
                align-items: center;
                gap: 15px;
                color: var(--dark);
                text-decoration: none;
            }
            
            .menu-item:hover {
                background: #f0f2f5;
                transform: translateX(-5px);
            }
            
            .menu-item.admin {
                background: linear-gradient(135deg, #667eea10 0%, #764ba210 100%);
                border: 1px solid var(--primary);
            }
            
            .welcome-message {
                text-align: center;
                padding: 20px;
                background: linear-gradient(135deg, #667eea08 0%, #764ba208 100%);
                border-radius: 20px;
                margin-bottom: 10px;
            }
            
            .welcome-message h3 {
                color: var(--primary);
                margin-bottom: 8px;
                font-size: 1.3em;
            }
            
            .welcome-message p {
                color: #666;
                font-size: 0.95em;
            }
            
            .quick-actions {
                display: flex;
                flex-wrap: wrap;
                gap: 8px;
                justify-content: center;
                margin-top: 12px;
            }
            
            .quick-btn {
                background: white;
                border: 1px solid var(--primary);
                color: var(--primary);
                padding: 6px 12px;
                border-radius: 30px;
                font-size: 0.85em;
                cursor: pointer;
                transition: all 0.3s;
            }
            
            .quick-btn:hover {
                background: var(--primary);
                color: white;
            }
            
            @media (max-width: 480px) {
                body { padding: 8px; }
                .chat-container { border-radius: 30px; }
                .message-content { font-size: 0.95rem; }
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
                    <h3>🌟 به هوش مصنوعی ایرانی خوش آمدید</h3>
                    <p>هر سوالی دارید بپرسید!</p>
                    <div class="quick-actions">
                        <span class="quick-btn" onclick="quickQuestion('سلام')">👋 سلام</span>
                        <span class="quick-btn" onclick="quickQuestion('چطوری')">😊 حال تو</span>
                        <span class="quick-btn" onclick="quickQuestion('حرف ب چیست')">📝 حرف ب</span>
                        <span class="quick-btn" onclick="quickQuestion('کوروش کبیر که بود')">👑 کوروش</span>
                    </div>
                </div>
            </div>
            
            <div class="chat-input-container">
                <input type="text" class="chat-input" id="message-input" 
                       placeholder="سوال خود را بپرسید..." 
                       onkeypress="if(event.key==='Enter') sendMessage()"
                       autofocus>
                <button class="send-btn" onclick="sendMessage()">
                    <span>➤</span>
                </button>
            </div>
        </div>
        
        <div class="menu-overlay" id="menuOverlay" onclick="closeMenu()"></div>
        <div class="menu-panel" id="menuPanel">
            <div class="menu-header">
                <h3>منو</h3>
                <button class="close-menu" onclick="closeMenu()">✕</button>
            </div>
            
            <a href="/m.html" class="menu-item">
                <span>📄</span> صفحه M
            </a>
            
            <a href="/admin-login" class="menu-item admin">
                <span>⚙️</span> پنل مدیریت
            </a>
            
            <div class="menu-item" onclick="clearHistory()">
                <span>🗑️</span> پاک کردن تاریخچه
            </div>
        </div>
        
        <script>
            let chatHistory = JSON.parse(localStorage.getItem('chat_history')) || [];
            
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
                
                const messageTime = time || new Date().toLocaleTimeString('fa-IR', { 
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
                    chatHistory.push({
                        text: text,
                        isUser: isUser,
                        time: messageTime
                    });
                    
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
                        body: JSON.stringify({message: message})
                    });
                    
                    const data = await response.json();
                    hideTyping();
                    
                    if (data.answer) {
                        addMessage(data.answer);
                    } else {
                        addMessage('🤔 متأسفم! نتونستم پاسخ این سوال رو پیدا کنم.');
                    }
                    
                } catch (error) {
                    hideTyping();
                    addMessage('⚠️ خطا در ارتباط با سرور');
                }
            }
            
            function quickQuestion(q) {
                document.getElementById('message-input').value = q;
                sendMessage();
            }
            
            function clearHistory() {
                if (confirm('آیا تاریخچه پاک شود؟')) {
                    localStorage.removeItem('chat_history');
                    chatHistory = [];
                    location.reload();
                }
            }
        </script>
    </body>
    </html>
    ''')

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
            }
            .container {
                background: white;
                border-radius: 30px;
                padding: 40px;
                max-width: 600px;
                text-align: center;
            }
            h1 { color: #333; margin-bottom: 20px; }
            p { color: #666; line-height: 1.8; }
            .btn {
                display: inline-block;
                padding: 12px 30px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                text-decoration: none;
                border-radius: 30px;
                margin-top: 20px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📄 صفحه M</h1>
            <p>این صفحه مخصوص منوی کشویی است.</p>
            <a href="/" class="btn">بازگشت به چت</a>
        </div>
    </body>
    </html>
    ''')

@app.route('/api/chat', methods=['POST'])
def api_chat():
    try:
        data = request.json
        question = data.get('message', '').strip()
        
        if not question:
            return jsonify({'error': 'سوال نمی‌تواند خالی باشد'})
        
        result = ai.process_question(question)
        
        if result['found']:
            return jsonify({
                'answer': result['answer'],
                'quality': result['quality'],
                'found': True
            })
        else:
            return jsonify({'answer': None, 'found': False})
            
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/admin-login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form['username']
        password = hashlib.md5(request.form['password'].encode()).hexdigest()
        
        if username == 'admin' and password == hashlib.md5('admin123'.encode()).hexdigest():
            login_user(users['1'])
            return redirect(url_for('admin_panel'))
        
        return "❌ رمز اشتباه است"
    
    return render_template_string('''
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
            }
            .login-box {
                background: white;
                padding: 40px;
                border-radius: 30px;
                width: 100%;
                max-width: 400px;
            }
            h2 { text-align: center; margin-bottom: 30px; }
            input {
                width: 100%;
                padding: 15px;
                margin: 10px 0;
                border: 2px solid #e0e0e0;
                border-radius: 15px;
            }
            button {
                width: 100%;
                padding: 15px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                border-radius: 15px;
                cursor: pointer;
                margin-top: 20px;
            }
        </style>
    </head>
    <body>
        <div class="login-box">
            <h2>🔐 پنل مدیریت</h2>
            <form method="POST">
                <input type="text" name="username" placeholder="admin" value="admin">
                <input type="password" name="password" placeholder="admin123" value="admin123">
                <button type="submit">ورود</button>
            </form>
        </div>
    </body>
    </html>
    ''')

@app.route('/admin')
@login_required
def admin_panel():
    stats = ai.get_stats()
    unanswered = ai.get_unanswered()
    
    return render_template_string('''
    <!DOCTYPE html>
    <html>
    <head>
        <title>پنل مدیریت</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body {
                font-family: Tahoma;
                background: #f5f5f5;
                padding: 20px;
            }
            .header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 20px;
                border-radius: 15px;
                margin-bottom: 20px;
                display: flex;
                justify-content: space-between;
            }
            .stats {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                gap: 15px;
                margin-bottom: 20px;
            }
            .stat-card {
                background: white;
                padding: 20px;
                border-radius: 15px;
                text-align: center;
            }
            .stat-number {
                font-size: 2em;
                color: #667eea;
                font-weight: bold;
            }
            .card {
                background: white;
                padding: 20px;
                border-radius: 15px;
                margin-bottom: 20px;
            }
            textarea, input, select {
                width: 100%;
                padding: 12px;
                margin: 10px 0;
                border: 2px solid #e0e0e0;
                border-radius: 10px;
            }
            button {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 12px 25px;
                border: none;
                border-radius: 10px;
                cursor: pointer;
            }
            .file-upload {
                border: 2px dashed #667eea;
                padding: 30px;
                text-align: center;
                border-radius: 10px;
                cursor: pointer;
                margin: 20px 0;
            }
            .unanswered-item {
                background: #fff3cd;
                padding: 15px;
                margin: 10px 0;
                border-radius: 10px;
                display: flex;
                justify-content: space-between;
            }
        </style>
    </head>
    <body>
        <div class="header">
            <h2>⚙️ پنل مدیریت</h2>
            <div>
                <a href="/" style="color: white; margin-right: 15px;">چت</a>
                <a href="/logout" style="color: white;">خروج</a>
            </div>
        </div>
        
        <div class="stats">
            <div class="stat-card">
                <div class="stat-number">{{ stats.knowledge }}</div>
                <div>کل دانش</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{{ stats.questions }}</div>
                <div>کل سوالات</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{{ stats.unanswered }}</div>
                <div>بی‌پاسخ</div>
            </div>
        </div>
        
        <div class="card">
            <h3>➕ افزودن دانش</h3>
            <form action="/admin/add" method="POST">
                <input type="text" name="question" placeholder="سوال" required>
                <textarea name="answer" rows="4" placeholder="پاسخ" required></textarea>
                <select name="category">
                    <option>عمومی</option>
                    <option>تاریخ</option>
                    <option>الفبا</option>
                </select>
                <button type="submit">افزودن</button>
            </form>
        </div>
        
        <div class="card">
            <h3>📁 آپلود فایل</h3>
            <form action="/admin/upload" method="POST" enctype="multipart/form-data">
                <div class="file-upload" onclick="document.getElementById('file').click()">
                    <p>📤 کلیک برای آپلود</p>
                    <p style="font-size:0.9em;">فرمت: سوال | جواب (هر خط)</p>
                </div>
                <input type="file" id="file" name="file" style="display:none;" accept=".txt">
                <button type="submit">آپلود</button>
            </form>
        </div>
        
        <div class="card">
            <h3>❌ سوالات بی‌پاسخ</h3>
            {% for item in unanswered %}
            <div class="unanswered-item">
                <div>{{ item.question }}</div>
                <button onclick="fillQuestion('{{ item.question }}')">پاسخ</button>
            </div>
            {% endfor %}
        </div>
        
        <script>
            function fillQuestion(q) {
                document.querySelector('[name="question"]').value = q;
                document.querySelector('[name="question"]').scrollIntoView();
            }
        </script>
    </body>
    </html>
    ''', stats=stats, unanswered=unanswered)

@app.route('/admin/add', methods=['POST'])
@login_required
def admin_add():
    question = request.form['question']
    answer = request.form['answer']
    category = request.form.get('category', 'عمومی')
    
    success, msg = ai.add_knowledge(question, answer, category)
    return redirect(url_for('admin_panel'))

@app.route('/admin/upload', methods=['POST'])
@login_required
def admin_upload():
    """آپلود فایل با خطاگیری کامل"""
    try:
        if 'file' not in request.files:
            return "❌ فایلی انتخاب نشده است"
        
        file = request.files['file']
        if file.filename == '':
            return "❌ نام فایل معتبر نیست"
        
        if not file.filename.endswith('.txt'):
            return "❌ فقط فایل .txt مجاز است"
        
        # ذخیره فایل
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # خوندن فایل با encodeهای مختلف
        content = None
        encodings = ['utf-8', 'cp1256', 'iso-8859-6']
        
        for enc in encodings:
            try:
                with open(filepath, 'r', encoding=enc) as f:
                    content = f.read()
                print(f"✅ فایل با encoding {enc} خونده شد")
                break
            except:
                continue
        
        if content is None:
            return "❌ خطا در خوندن فایل"
        
        # پردازش
        count, errors = ai.bulk_import(content)
        
        # پاک کردن فایل موقت
        os.remove(filepath)
        
        # نمایش نتیجه
        result = f"✅ {count} مورد با موفقیت اضافه شد"
        if errors:
            result += "<br>⚠️ خطاها:<br>" + "<br>".join(errors[:5])
        
        return result + '<br><a href="/admin">🔙 بازگشت</a>'
        
    except Exception as e:
        return f"❌ خطا: {str(e)}<br><a href='/admin'>🔙 بازگشت</a>"

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
    ║  📚 دانش: {}                                        ║
    ║  🌐 چت: http://localhost:5000                      ║
    ║  🔐 پنل: http://localhost:5000/admin-login         ║
    ║  👤 کاربر: admin / admin123                        ║
    ╚════════════════════════════════════════════════════╝
    """.format(len(ai.knowledge_base)))
    
    app.run(debug=True, host='0.0.0.0', port=5000)
