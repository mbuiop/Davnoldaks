# ai_scalable_million.py
from flask import Flask, render_template, request, jsonify, session, render_template_string, redirect, url_for, make_response
from flask_cors import CORS
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user
import hashlib
import os
import json
import re
import time
import uuid
import threading
import queue
from datetime import datetime, timedelta
from collections import Counter, defaultdict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import redis
import pickle
import gc
import psutil
import multiprocessing as mp

app = Flask(__name__)
app.config['SECRET_KEY'] = 'super-secret-key-for-10-million-users'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=365)
CORS(app)

# ایجاد پوشه‌ها
os.makedirs('data', exist_ok=True)
os.makedirs('uploads', exist_ok=True)
os.makedirs('cache', exist_ok=True)
os.makedirs('logs', exist_ok=True)

# ================ اتصال به Redis (برای کش و صف) ================
try:
    redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True, socket_connect_timeout=1)
    redis_client.ping()
    REDIS_AVAILABLE = True
    print("✅ Redis وصل شد")
except:
    REDIS_AVAILABLE = False
    print("⚠️ Redis وصل نشد، از کش فایل استفاده می‌شود")

# ================ مدیریت حافظه و کش ================
class MemoryCache:
    """کش هوشمند با مدیریت حافظه"""
    
    def __init__(self, max_size=100000, ttl=3600):
        self.cache = {}
        self.timestamps = {}
        self.max_size = max_size
        self.ttl = ttl
        self.lock = threading.RLock()
        self.hits = 0
        self.misses = 0
    
    def get(self, key):
        with self.lock:
            if key in self.cache:
                if time.time() - self.timestamps[key] < self.ttl:
                    self.hits += 1
                    return self.cache[key]
                else:
                    del self.cache[key]
                    del self.timestamps[key]
            self.misses += 1
            return None
    
    def set(self, key, value):
        with self.lock:
            # اگر کش پر است، قدیمی‌ترین را حذف کن
            if len(self.cache) >= self.max_size:
                oldest = min(self.timestamps.items(), key=lambda x: x[1])
                del self.cache[oldest[0]]
                del self.timestamps[oldest[0]]
            
            self.cache[key] = value
            self.timestamps[key] = time.time()
    
    def get_stats(self):
        with self.lock:
            total = self.hits + self.misses
            hit_rate = (self.hits / total * 100) if total > 0 else 0
            return {
                'size': len(self.cache),
                'hits': self.hits,
                'misses': self.misses,
                'hit_rate': f"{hit_rate:.1f}%"
            }

# ================ پایگاه داده شارد شده ================
class ShardedDatabase:
    """پایگاه داده با شاردینگ برای مقیاس‌پذیری"""
    
    def __init__(self, shard_count=10):
        self.shard_count = shard_count
        self.shards = []
        self.locks = []
        self.stats = {
            'total_docs': 0,
            'total_questions': 0,
            'queries': 0
        }
        
        for i in range(shard_count):
            os.makedirs(f'data/shard_{i}', exist_ok=True)
            self.shards.append({
                'knowledge': [],
                'users': [],
                'questions': []
            })
            self.locks.append(threading.RLock())
        
        self.load_all()
    
    def _get_shard(self, key):
        """تعیین شارد بر اساس هش کلید"""
        hash_val = hash(key) % self.shard_count
        return abs(hash_val)
    
    def save_knowledge(self, item):
        """ذخیره دانش در شارد مناسب"""
        shard_id = self._get_shard(item['q'])
        with self.locks[shard_id]:
            # بررسی تکراری
            for existing in self.shards[shard_id]['knowledge']:
                if existing['q'] == item['q']:
                    existing.update(item)
                    self._save_shard(shard_id)
                    return True
            
            self.shards[shard_id]['knowledge'].append(item)
            self.stats['total_docs'] += 1
            self._save_shard(shard_id)
            return True
    
    def search_knowledge(self, query, limit=10):
        """جستجو در همه شاردها"""
        self.stats['queries'] += 1
        results = []
        
        for shard_id in range(self.shard_count):
            with self.locks[shard_id]:
                for item in self.shards[shard_id]['knowledge']:
                    if query.lower() in item['q'].lower():
                        results.append(item)
                        if len(results) >= limit:
                            return results
        
        return results[:limit]
    
    def get_all_knowledge(self):
        """گرفتن همه دانش‌ها"""
        all_items = []
        for shard_id in range(self.shard_count):
            with self.locks[shard_id]:
                all_items.extend(self.shards[shard_id]['knowledge'])
        return all_items
    
    def _save_shard(self, shard_id):
        """ذخیره یک شارد"""
        try:
            with open(f'data/shard_{shard_id}/data.json', 'w', encoding='utf-8') as f:
                json.dump(self.shards[shard_id], f, ensure_ascii=False, indent=2)
        except:
            pass
    
    def load_all(self):
        """بارگذاری همه شاردها"""
        for i in range(self.shard_count):
            try:
                if os.path.exists(f'data/shard_{i}/data.json'):
                    with open(f'data/shard_{i}/data.json', 'r', encoding='utf-8') as f:
                        self.shards[i] = json.load(f)
                        self.stats['total_docs'] += len(self.shards[i].get('knowledge', []))
            except:
                pass
        
        print(f"📚 {self.stats['total_docs']} دانش از {self.shard_count} شارد بارگذاری شد")
    
    def record_question(self, question, user_id=None):
        """ثبت سوال کاربر برای یادگیری"""
        self.stats['total_questions'] += 1
        shard_id = self._get_shard(question)
        
        with self.locks[shard_id]:
            self.shards[shard_id]['questions'].append({
                'q': question,
                'user': user_id,
                'time': datetime.now().isoformat()
            })
            
            # محدود کردن حجم
            if len(self.shards[shard_id]['questions']) > 10000:
                self.shards[shard_id]['questions'] = self.shards[shard_id]['questions'][-10000:]
    
    def get_popular_questions(self, limit=100):
        """گرفتن سوالات پرتکرار"""
        counter = Counter()
        for shard_id in range(self.shard_count):
            with self.locks[shard_id]:
                for q in self.shards[shard_id]['questions']:
                    counter[q['q']] += 1
        return counter.most_common(limit)

# ================ موتور جستجوی برداری بهینه ================
class OptimizedVectorSearch:
    """موتور جستجوی برداری با قابلیت مقیاس‌پذیری"""
    
    def __init__(self):
        self.vectorizer = HashingVectorizer(
            n_features=2**18,  # 262144 ویژگی
            ngram_range=(1, 3),
            norm='l2',
            alternate_sign=False
        )
        self.documents = []
        self.vectors = None
        self.lock = threading.RLock()
        self.update_queue = queue.Queue()
        self.is_updating = False
    
    def add_documents(self, documents):
        """افزودن اسناد جدید"""
        with self.lock:
            start = len(self.documents)
            for doc in documents:
                self.documents.append(doc['q'])
            
            if start < len(self.documents):
                new_texts = self.documents[start:]
                new_vectors = self.vectorizer.transform(new_texts)
                
                if self.vectors is None:
                    self.vectors = new_vectors
                else:
                    from scipy.sparse import vstack
                    self.vectors = vstack([self.vectors, new_vectors])
    
    def search(self, query, top_k=5):
        """جستجوی سریع"""
        with self.lock:
            if self.vectors is None or len(self.documents) == 0:
                return []
            
            query_vec = self.vectorizer.transform([query])
            similarities = cosine_similarity(query_vec, self.vectors)[0]
            
            # گرفتن top_k با استفاده از argpartition (سریعتر)
            if len(similarities) > top_k:
                top_indices = np.argpartition(similarities, -top_k)[-top_k:]
                results = [(i, similarities[i]) for i in top_indices]
                results.sort(key=lambda x: x[1], reverse=True)
                return results[:top_k]
            else:
                indices = np.argsort(similarities)[::-1]
                return [(i, similarities[i]) for i in indices if similarities[i] > 0.1]

# ================ موتور یادگیری پیشرفته ================
class AdvancedLearningEngine:
    """موتور یادگیری با ۵ الگوریتم همزمان"""
    
    def __init__(self):
        self.db = ShardedDatabase(shard_count=10)
        self.vector_search = OptimizedVectorSearch()
        self.cache = MemoryCache(max_size=50000, ttl=300)
        self.user_profiles = defaultdict(lambda: {'questions': [], 'interests': Counter()})
        self.stats = {
            'total_queries': 0,
            'cache_hits': 0,
            'avg_response': 0
        }
        
        # بارگذاری اولیه
        all_knowledge = self.db.get_all_knowledge()
        self.vector_search.add_documents(all_knowledge)
        
        # آمار
        self.stats['total_knowledge'] = len(all_knowledge)
    
    def normalize(self, text):
        """نرمال‌سازی سریع متن"""
        if not text:
            return ""
        text = text.replace('ي', 'ی').replace('ك', 'ک')
        text = re.sub(r'[^\w\s\u0600-\u06FF]', ' ', text)
        return re.sub(r'\s+', ' ', text).strip()
    
    def extract_keywords(self, text):
        """استخراج کلمات کلیدی"""
        words = text.split()
        return [w for w in words if len(w) > 2]
    
    def learn(self, question, answer, user_id=None):
        """یادگیری دانش جدید"""
        q_norm = self.normalize(question)
        
        item = {
            'id': str(uuid.uuid4()),
            'q': q_norm,
            'a': answer,
            'keywords': self.extract_keywords(q_norm),
            'learn_count': 1,
            'use_count': 0,
            'created': time.time(),
            'user_id': user_id
        }
        
        # ذخیره در دیتابیس
        self.db.save_knowledge(item)
        
        # به‌روزرسانی موتور جستجو
        self.vector_search.add_documents([item])
        
        return True
    
    def search(self, question, user_id=None):
        """جستجوی هوشمند با ۵ الگوریتم"""
        start_time = time.time()
        self.stats['total_queries'] += 1
        
        q_norm = self.normalize(question)
        cache_key = hashlib.md5(q_norm.encode()).hexdigest()
        
        # 1. چک کردن کش
        cached = self.cache.get(cache_key)
        if cached:
            self.stats['cache_hits'] += 1
            return cached
        
        results = []
        seen_ids = set()
        
        # 2. جستجوی برداری (سریعترین)
        vector_results = self.vector_search.search(q_norm, top_k=10)
        all_knowledge = self.db.get_all_knowledge()
        
        for idx, score in vector_results:
            if idx < len(all_knowledge):
                item = all_knowledge[idx]
                if item['id'] not in seen_ids:
                    results.append({
                        'answer': item['a'],
                        'score': float(score) * 1.2,
                        'method': 'vector'
                    })
                    seen_ids.add(item['id'])
        
        # 3. جستجوی کلمات کلیدی
        keywords = self.extract_keywords(q_norm)
        if keywords:
            for item in all_knowledge:
                if item['id'] in seen_ids:
                    continue
                
                common = set(keywords) & set(item.get('keywords', []))
                if common:
                    score = len(common) / max(len(keywords), 1)
                    if score > 0.3:
                        results.append({
                            'answer': item['a'],
                            'score': score,
                            'method': 'keyword'
                        })
                        seen_ids.add(item['id'])
        
        # 4. تطابق دقیق
        for item in all_knowledge:
            if item['id'] in seen_ids:
                continue
            if item['q'] == q_norm:
                results.append({
                    'answer': item['a'],
                    'score': 1.0,
                    'method': 'exact'
                })
                seen_ids.add(item['id'])
                break
        
        # 5. سوالات مشابه کاربران
        if user_id and user_id in self.user_profiles:
            profile = self.user_profiles[user_id]
            for past_q in profile['questions'][-5:]:
                if self.similarity(q_norm, past_q) > 0.7:
                    for item in all_knowledge:
                        if item['id'] not in seen_ids and self.similarity(past_q, item['q']) > 0.5:
                            results.append({
                                'answer': item['a'],
                                'score': 0.5,
                                'method': 'collaborative'
                            })
                            seen_ids.add(item['id'])
        
        # مرتب‌سازی نهایی
        results.sort(key=lambda x: x['score'], reverse=True)
        
        # ذخیره در کش
        if results:
            self.cache.set(cache_key, results[:3])
        
        # ثبت سوال کاربر
        self.db.record_question(question, user_id)
        
        # آمار
        response_time = time.time() - start_time
        self.stats['avg_response'] = self.stats['avg_response'] * 0.95 + response_time * 0.05
        
        return results[:3]  # برگرداندن 3 نتیجه برتر
    
    def similarity(self, text1, text2):
        """محاسبه شباهت دو متن"""
        words1 = set(text1.split())
        words2 = set(text2.split())
        if not words1 or not words2:
            return 0
        return len(words1 & words2) / max(len(words1), len(words2))
    
    def ask(self, question, user_id=None):
        """پرسش و پاسخ نهایی"""
        results = self.search(question, user_id)
        
        if results:
            best = results[0]
            return {
                'answer': best['answer'],
                'method': best['method'],
                'confidence': f"{best['score']*100:.0f}%",
                'found': True
            }
        
        return {
            'answer': None,
            'found': False
        }
    
    def get_stats(self):
        """گرفتن آمار"""
        cache_stats = self.cache.get_stats()
        return {
            'knowledge': self.stats['total_knowledge'],
            'queries': self.stats['total_queries'],
            'cache_hits': cache_stats['hit_rate'],
            'cache_size': cache_stats['size'],
            'avg_response_ms': f"{self.stats['avg_response']*1000:.1f}",
            'users': len(self.user_profiles)
        }

# ================ نمونه اصلی ================
ai = AdvancedLearningEngine()

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
    user_id = request.cookies.get('user_id')
    if not user_id:
        user_id = str(uuid.uuid4())
    
    stats = ai.get_stats()
    
    resp = make_response(render_template_string('''
    <!DOCTYPE html>
    <html lang="fa" dir="rtl">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
        <title>هوش مصنوعی میلیونی</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
                -webkit-tap-highlight-color: transparent;
            }
            
            html, body {
                height: 100%;
                overflow: hidden;
                position: fixed;
                width: 100%;
            }
            
            body {
                font-family: 'Vazir', 'Tahoma', sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                display: flex;
                align-items: center;
                justify-content: center;
                padding: 10px;
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
            }
            
            .chat-input:focus {
                border-color: #667eea;
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
                flex-shrink: 0;
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
            
            .stats-bar {
                background: rgba(0,0,0,0.1);
                padding: 5px 10px;
                border-radius: 20px;
                font-size: 0.8em;
                color: white;
            }
            
            .welcome-message {
                text-align: center;
                padding: 20px;
                background: linear-gradient(135deg, #667eea08 0%, #764ba208 100%);
                border-radius: 20px;
            }
            
            .welcome-message h3 {
                color: #667eea;
                margin-bottom: 8px;
            }
        </style>
    </head>
    <body>
        <div class="chat-container">
            <div class="chat-header">
                <button class="menu-btn" onclick="toggleMenu()">☰</button>
                <div class="header-title">
                    <span>🤖</span> هوش میلیونی
                </div>
                <div class="stats-bar">
                    {{ stats.cache_hits }}
                </div>
            </div>
            
            <div class="chat-messages" id="chat-messages">
                <div class="welcome-message">
                    <h3>🌟 به هوش مصنوعی میلیونی خوش آمدید</h3>
                    <p>{{ stats.knowledge }} دانش | {{ stats.queries }} سوال</p>
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
    ''', stats=stats))
    
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
        user_id = request.cookies.get('user_id')
        
        if not question:
            return jsonify({'error': 'سوال خالی است'})
        
        result = ai.ask(question, user_id)
        
        if result['found']:
            return jsonify({
                'answer': result['answer'],
                'method': result.get('method', ''),
                'confidence': result.get('confidence', ''),
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
    <head><title>پنل مدیریت</title>
    <style>
        *{{margin:0;padding:0;box-sizing:border-box;}}
        body{{font-family:Tahoma;background:#f5f5f5;padding:20px;}}
        .header{{background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);color:white;padding:20px;border-radius:15px;margin-bottom:20px;display:flex;justify-content:space-between;}}
        .stats-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:15px;margin-bottom:20px;}}
        .stat-card{{background:white;padding:20px;border-radius:15px;text-align:center;}}
        .stat-number{{font-size:2.5em;color:#667eea;font-weight:bold;}}
        .card{{background:white;padding:20px;border-radius:15px;margin-bottom:20px;}}
        textarea,input{{width:100%;padding:12px;margin:10px 0;border:2px solid #e0e0e0;border-radius:10px;}}
        button{{background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);color:white;padding:12px 25px;border:none;border-radius:10px;cursor:pointer;}}
        .grid-2{{display:grid;grid-template-columns:1fr 1fr;gap:20px;}}
    </style>
    </head>
    <body>
        <div class="header"><h2>⚙️ پنل مدیریت</h2><div><a href="/" style="color:white;margin-right:15px;">چت</a><a href="/logout" style="color:white;">خروج</a></div></div>
        
        <div class="stats-grid">
            <div class="stat-card"><div class="stat-number">{stats['knowledge']}</div><div>دانش</div></div>
            <div class="stat-card"><div class="stat-number">{stats['queries']}</div><div>سوالات</div></div>
            <div class="stat-card"><div class="stat-number">{stats['cache_hits']}</div><div>کش</div></div>
            <div class="stat-card"><div class="stat-number">{stats['avg_response_ms']}ms</div><div>زمان پاسخ</div></div>
        </div>
        
        <div class="grid-2">
            <div class="card">
                <h3>➕ آموزش</h3>
                <form action="/admin/learn" method="POST">
                    <input name="question" placeholder="سوال" required>
                    <textarea name="answer" rows="4" placeholder="پاسخ" required></textarea>
                    <button type="submit">📚 یاد بگیر</button>
                </form>
            </div>
            <div class="card">
                <h3>📁 آپلود فایل</h3>
                <form action="/admin/learn/file" method="POST" enctype="multipart/form-data">
                    <input type="file" name="file" accept=".txt" required>
                    <button type="submit">📚 یادگیری</button>
                </form>
            </div>
        </div>
        
        <div class="card">
            <h3>⚡ آمار پیشرفته</h3>
            <p>اندازه کش: {stats['cache_size']}</p>
            <p>کاربران: {stats['users']}</p>
        </div>
    </body>
    </html>
    '''

@app.route('/admin/learn', methods=['POST'])
@login_required
def learn():
    q = request.form['question']
    a = request.form['answer']
    ai.learn(q, a)
    return redirect(url_for('admin_panel'))

@app.route('/admin/learn/file', methods=['POST'])
@login_required
def learn_file():
    try:
        file = request.files['file']
        content = file.read().decode('utf-8')
        count = 0
        for line in content.split('\n'):
            if '|' in line:
                q, a = line.split('|', 1)
                ai.learn(q.strip(), a.strip())
                count += 1
        return f"✅ {count} مورد یاد گرفته شد<br><a href='/admin'>بازگشت</a>"
    except Exception as e:
        return f"❌ خطا: {str(e)}"

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

if __name__ == '__main__':
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║     🤖 هوش مصنوعی برای ۱۰ میلیون کاربر                   ║
    ╠══════════════════════════════════════════════════════════╣
    ║  📚 دانش: {}                                               ║
    ║  ⚡ کش: {}%                                                 ║
    ║  🌐 چت: http://localhost:5000                             ║
    ║  🔐 پنل: http://localhost:5000/admin-login                ║
    ║  👤 کاربر: admin / admin123                                ║
    ║  📱 ۵ الگوریتم + پروفایل کاربر + کش هوشمند               ║
    ╚══════════════════════════════════════════════════════════╝
    """.format(ai.stats['total_knowledge'], ai.get_stats()['cache_hits']))
    
    app.run(debug=True, host='0.0.0.0', port=5000)
