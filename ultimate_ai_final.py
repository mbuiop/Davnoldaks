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

# ================ کتابخانه‌های پیشرفته تشخیص حروف ================
import easyocr
import pytesseract
from PIL import Image
import cv2
import numpy as np

# ================ کتابخانه‌های تحلیل متن پیشرفته ================
import langid
from textblob import TextBlob
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import hazm
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
import requests
from bs4 import BeautifulSoup
import wikipedia
import googlesearch
from googlesearch import search
import openai
from newspaper import Article

# ================ کتابخانه‌های تشخیص زبان و ترجمه ================
from googletrans import Translator
import langdetect

# دانلود منابع
try:
    nltk.data.find('tokenizers/punkt')
except:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')

app = Flask(__name__)
app.config['SECRET_KEY'] = 'super-secret-key-for-ai-learning'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # افزایش به 500 مگابایت
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=365)
app.config['SESSION_COOKIE_SECURE'] = True
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
CORS(app)

# ایجاد پوشه‌ها
os.makedirs('data', exist_ok=True)
os.makedirs('uploads', exist_ok=True)
os.makedirs('memory', exist_ok=True)
os.makedirs('knowledge_base', exist_ok=True)
os.makedirs('user_data', exist_ok=True)
os.makedirs('unanswered_questions', exist_ok=True)

# ================ تشخیص‌گر پیشرفته حروف ================
class AdvancedOCR:
    def __init__(self):
        self.reader = easyocr.Reader(['fa', 'en'], gpu=False)
        self.translator = Translator()
        
    def extract_text(self, image_path):
        """تشخیص حروف از تصویر با 3 روش مختلف"""
        results = []
        
        # روش 1: EasyOCR
        try:
            easy_result = self.reader.readtext(image_path)
            easy_text = ' '.join([item[1] for item in easy_result])
            if easy_text:
                results.append(('easyocr', easy_text))
        except:
            pass
        
        # روش 2: Tesseract
        try:
            img = Image.open(image_path)
            tess_text = pytesseract.image_to_string(img, lang='fas+eng')
            if tess_text.strip():
                results.append(('tesseract', tess_text))
        except:
            pass
        
        # روش 3: OpenCV preprocessing + Tesseract
        try:
            img = cv2.imread(image_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
            cv2.imwrite('temp_thresh.jpg', thresh)
            thresh_text = pytesseract.image_to_string('temp_thresh.jpg', lang='fas+eng')
            if thresh_text.strip():
                results.append(('enhanced', thresh_text))
            os.remove('temp_thresh.jpg')
        except:
            pass
        
        # انتخاب بهترین نتیجه
        best_text = ''
        max_length = 0
        for method, text in results:
            if len(text) > max_length:
                max_length = len(text)
                best_text = text
        
        return best_text if best_text else None
    
    def detect_language(self, text):
        """تشخیص زبان متن"""
        try:
            lang = langdetect.detect(text)
            return lang
        except:
            return 'fa'

# ================ موتور جستجوی پیشرفته ================
class AdvancedSearchEngine:
    def __init__(self):
        self.translator = Translator()
        self.cache = {}
        self.cache_file = 'memory/search_cache.json'
        self.load_cache()
        
    def load_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                self.cache = json.load(f)
    
    def save_cache(self):
        with open(self.cache_file, 'w', encoding='utf-8') as f:
            json.dump(self.cache, f, ensure_ascii=False)
    
    def search_google(self, query, num_results=5):
        """جستجو در گوگل"""
        try:
            results = []
            for url in search(query, num_results=num_results, lang='fa'):
                try:
                    article = Article(url)
                    article.download()
                    article.parse()
                    results.append({
                        'url': url,
                        'title': article.title,
                        'text': article.text[:1000],
                        'source': 'google'
                    })
                except:
                    continue
            return results
        except:
            return []
    
    def search_wikipedia(self, query, lang='fa'):
        """جستجو در ویکی‌پدیا"""
        try:
            wikipedia.set_lang(lang)
            results = []
            search_results = wikipedia.search(query)
            for title in search_results[:3]:
                try:
                    page = wikipedia.page(title)
                    results.append({
                        'url': page.url,
                        'title': page.title,
                        'text': page.summary,
                        'source': 'wikipedia'
                    })
                except:
                    continue
            return results
        except:
            return []
    
    def search_web(self, query):
        """جستجوی ترکیبی در وب"""
        # بررسی کش
        cache_key = hashlib.md5(query.encode()).hexdigest()
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        results = []
        
        # جستجو در گوگل
        google_results = self.search_google(query)
        results.extend(google_results)
        
        # جستجو در ویکی‌پدیا فارسی
        wiki_fa = self.search_wikipedia(query, 'fa')
        results.extend(wiki_fa)
        
        # جستجو در ویکی‌پدیا انگلیسی
        wiki_en = self.search_wikipedia(query, 'en')
        results.extend(wiki_en)
        
        # ذخیره در کش
        self.cache[cache_key] = results
        self.save_cache()
        
        return results
    
    def extract_answer(self, query, search_results):
        """استخراج بهترین پاسخ از نتایج جستجو"""
        if not search_results:
            return None
        
        best_answer = None
        max_relevance = 0
        
        for result in search_results:
            text = result.get('text', '')
            if not text:
                continue
            
            # محاسبه relevance
            words = query.lower().split()
            relevance = sum(1 for word in words if word in text.lower())
            relevance = relevance / len(words) if words else 0
            
            if relevance > max_relevance:
                max_relevance = relevance
                best_answer = text[:500]  # خلاصه کردن
        
        return best_answer

# ================ تحلیلگر متن پیشرفته ================
class AdvancedTextAnalyzer:
    def __init__(self):
        self.normalizer = hazm.Normalizer()
        self.lemmatizer = WordNetLemmatizer()
        self.translator = Translator()
        self.stop_words = set()
        try:
            self.stop_words.update(stopwords.words('persian'))
        except:
            pass
        try:
            self.stop_words.update(stopwords.words('english'))
        except:
            pass
        
        # مدل QA
        try:
            self.qa_model = pipeline("question-answering", model="m3hrdadfi/albert-fa-zwnj-qa")
        except:
            self.qa_model = None
    
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
    
    def analyze_sentiment(self, text):
        """تحلیل احساسات متن"""
        try:
            blob = TextBlob(text)
            return {
                'polarity': blob.sentiment.polarity,
                'subjectivity': blob.sentiment.subjectivity
            }
        except:
            return {'polarity': 0, 'subjectivity': 0}

# ================ حافظه ۵۰ گیگابایتی ================
class MassiveMemory:
    def __init__(self, max_size_gb=50):
        self.max_size_bytes = max_size_gb * 1024 * 1024 * 1024
        self.memory_dir = 'memory'
        self.index_file = os.path.join(self.memory_dir, 'memory_index.json')
        self.load_index()
        
    def load_index(self):
        if os.path.exists(self.index_file):
            with open(self.index_file, 'r', encoding='utf-8') as f:
                self.index = json.load(f)
        else:
            self.index = {
                'total_items': 0,
                'total_size': 0,
                'items': {}
            }
    
    def save_index(self):
        with open(self.index_file, 'w', encoding='utf-8') as f:
            json.dump(self.index, f, ensure_ascii=False)
    
    def add_to_memory(self, key, data):
        """افزودن به حافظه با مدیریت حجم"""
        item_id = hashlib.md5(key.encode()).hexdigest()
        item_file = os.path.join(self.memory_dir, f"{item_id}.json")
        
        # محاسبه حجم
        data_str = json.dumps(data, ensure_ascii=False)
        size = len(data_str.encode('utf-8'))
        
        # بررسی حجم کل
        if self.index['total_size'] + size > self.max_size_bytes:
            # حذف قدیمی‌ترین آیتم‌ها
            self.cleanup_memory(size)
        
        # ذخیره داده
        with open(item_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)
        
        # به‌روزرسانی ایندکس
        self.index['items'][item_id] = {
            'key': key,
            'size': size,
            'timestamp': datetime.now().isoformat(),
            'access_count': 0
        }
        self.index['total_items'] += 1
        self.index['total_size'] += size
        self.save_index()
        
        return True
    
    def get_from_memory(self, key):
        """بازیابی از حافظه"""
        item_id = hashlib.md5(key.encode()).hexdigest()
        item_file = os.path.join(self.memory_dir, f"{item_id}.json")
        
        if os.path.exists(item_file):
            with open(item_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # به‌روزرسانی آمار دسترسی
            if item_id in self.index['items']:
                self.index['items'][item_id]['access_count'] += 1
                self.index['items'][item_id]['last_access'] = datetime.now().isoformat()
                self.save_index()
            
            return data
        return None
    
    def cleanup_memory(self, needed_size):
        """پاکسازی حافظه برای ایجاد فضا"""
        # مرتب‌سازی بر اساس آخرین دسترسی
        items = list(self.index['items'].items())
        items.sort(key=lambda x: (
            x[1].get('access_count', 0),
            x[1].get('timestamp', '')
        ))
        
        freed_space = 0
        for item_id, info in items:
            if freed_space >= needed_size:
                break
            
            item_file = os.path.join(self.memory_dir, f"{item_id}.json")
            if os.path.exists(item_file):
                os.remove(item_file)
                freed_space += info['size']
                del self.index['items'][item_id]
                self.index['total_items'] -= 1
                self.index['total_size'] -= info['size']
        
        self.save_index()

# ================ مدیریت کاربران پیشرفته ================
class UserManager:
    def __init__(self):
        self.users_file = 'data/users.json'
        self.user_stats_file = 'data/user_stats.json'
        self.unanswered_file = 'data/unanswered.json'
        self.popular_topics_file = 'data/popular_topics.json'
        self.load_data()
    
    def load_data(self):
        if os.path.exists(self.users_file):
            with open(self.users_file, 'r', encoding='utf-8') as f:
                self.users = json.load(f)
        else:
            self.users = {}
        
        if os.path.exists(self.user_stats_file):
            with open(self.user_stats_file, 'r', encoding='utf-8') as f:
                self.user_stats = json.load(f)
        else:
            self.user_stats = {}
        
        if os.path.exists(self.unanswered_file):
            with open(self.unanswered_file, 'r', encoding='utf-8') as f:
                self.unanswered = json.load(f)
        else:
            self.unanswered = []
        
        if os.path.exists(self.popular_topics_file):
            with open(self.popular_topics_file, 'r', encoding='utf-8') as f:
                self.popular_topics = json.load(f)
        else:
            self.popular_topics = Counter()
    
    def save_data(self):
        with open(self.users_file, 'w', encoding='utf-8') as f:
            json.dump(self.users, f, ensure_ascii=False)
        
        with open(self.user_stats_file, 'w', encoding='utf-8') as f:
            json.dump(self.user_stats, f, ensure_ascii=False)
        
        with open(self.unanswered_file, 'w', encoding='utf-8') as f:
            json.dump(self.unanswered[-1000:], f, ensure_ascii=False)  # نگهداری آخرین 1000
        
        with open(self.popular_topics_file, 'w', encoding='utf-8') as f:
            json.dump(dict(self.popular_topics.most_common(100)), f, ensure_ascii=False)
    
    def get_user(self, user_id):
        if user_id not in self.users:
            self.users[user_id] = {
                'first_seen': datetime.now().isoformat(),
                'last_seen': datetime.now().isoformat(),
                'total_questions': 0,
                'total_chats': 0
            }
        else:
            self.users[user_id]['last_seen'] = datetime.now().isoformat()
        
        return self.users[user_id]
    
    def track_question(self, user_id, question, answered=False):
        """ثبت سوال کاربر"""
        # به‌روزرسانی آمار کاربر
        if user_id not in self.user_stats:
            self.user_stats[user_id] = {
                'questions': [],
                'topics': Counter(),
                'last_question': None
            }
        
        # استخراج کلمات کلیدی
        keywords = re.findall(r'\w+', question.lower())
        for word in keywords[:5]:
            if len(word) > 2:
                self.user_stats[user_id]['topics'][word] += 1
                self.popular_topics[word] += 1
        
        self.user_stats[user_id]['questions'].append({
            'question': question,
            'time': datetime.now().isoformat(),
            'answered': answered
        })
        
        # نگهداری آخرین 50 سوال
        self.user_stats[user_id]['questions'] = self.user_stats[user_id]['questions'][-50:]
        
        # اگر پاسخ داده نشده
        if not answered:
            self.unanswered.append({
                'user_id': user_id,
                'question': question,
                'time': datetime.now().isoformat(),
                'status': 'pending'
            })
        
        self.save_data()
    
    def get_unanswered(self):
        return self.unanswered[-100:]  # آخرین 100 سوال بی‌پاسخ
    
    def get_user_stats(self, user_id):
        return self.user_stats.get(user_id, {})
    
    def get_popular_topics(self, limit=20):
        return self.popular_topics.most_common(limit)
    
    def get_all_users(self):
        return self.users

# ================ پردازشگر فایل فوق پیشرفته ================
class FileProcessor:
    def __init__(self):
        self.supported_formats = ['.txt', '.csv', '.json', '.md', '.pdf', '.jpg', '.png', '.jpeg']
        self.analyzer = AdvancedTextAnalyzer()
        self.ocr = AdvancedOCR()
    
    def process_file(self, filepath, filename):
        """پردازش فایل با تشخیص خودکار فرمت"""
        ext = os.path.splitext(filename)[1].lower()
        
        if ext in ['.jpg', '.png', '.jpeg']:
            return self.process_image(filepath)
        elif ext == '.txt':
            return self.process_txt(filepath)
        elif ext == '.csv':
            return self.process_csv(filepath)
        elif ext == '.json':
            return self.process_json(filepath)
        elif ext == '.md':
            return self.process_md(filepath)
        else:
            return self.process_generic(filepath)
    
    def process_image(self, filepath):
        """پردازش تصویر با OCR"""
        items = []
        text = self.ocr.extract_text(filepath)
        
        if text:
            lang = self.ocr.detect_language(text)
            items.append({
                'type': 'info',
                'content': text,
                'category': 'image_text',
                'language': lang
            })
        
        return items
    
    def process_txt(self, filepath):
        """پردازش فایل متنی با ۵ روش مختلف"""
        items = []
        encodings = ['utf-8', 'cp1256', 'iso-8859-6', 'utf-16', 'ascii']
        
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
        
        lines = content.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            if '|' in line:
                parts = line.split('|', 1)
                if len(parts) == 2:
                    items.append({
                        'type': 'qa',
                        'question': parts[0].strip(),
                        'answer': parts[1].strip(),
                        'category': 'general'
                    })
            elif ':' in line and len(line.split(':', 1)) == 2:
                q, a = line.split(':', 1)
                items.append({
                    'type': 'qa',
                    'question': q.strip(),
                    'answer': a.strip(),
                    'category': 'general'
                })
            elif len(line) > 20:
                items.append({
                    'type': 'info',
                    'content': line,
                    'category': 'general'
                })
        
        return items
    
    def process_csv(self, filepath):
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
        items = []
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
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
        items = []
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            paragraphs = content.split('\n\n')
            for para in paragraphs:
                if len(para) > 50:
                    items.append({
                        'type': 'info',
                        'content': para.strip(),
                        'category': 'general'
                    })
        except:
            pass
        return items

# ================ موتور جستجوی پیشرفته داخلی ================
class SearchEngine:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=100000, ngram_range=(1, 4))  # افزایش به 100 هزار
        self.knowledge = []
        self.vectors = None
        self.analyzer = AdvancedTextAnalyzer()
        self.web_search = AdvancedSearchEngine()
    
    def add_knowledge(self, question, answer, category='general', source='manual'):
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
            'use_count': 0,
            'embedding': None  # برای نسخه‌های بعدی
        }
        
        self.knowledge.append(item)
        return item
    
    def add_info(self, content, category='general', source='file'):
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
        if self.knowledge:
            questions = [item['question'] for item in self.knowledge]
            self.vectors = self.vectorizer.fit_transform(questions)
    
    def search(self, query, top_k=5):
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
    
    def web_search(self, query):
        """جستجو در وب"""
        return self.web_search.search_web(query)
    
    def get_stats(self):
        return {
            'total': len(self.knowledge),
            'used': sum(item.get('use_count', 0) for item in self.knowledge)
        }

# ================ هوش مصنوعی اصلی ================
class UltimateAI:
    def __init__(self):
        self.search = SearchEngine()
        self.analyzer = AdvancedTextAnalyzer()
        self.file_processor = FileProcessor()
        self.memory = MassiveMemory(50)  # حافظه 50 گیگابایت
        self.user_manager = UserManager()
        self.stats = {
            'learned': 0,
            'asked': 0,
            'files_processed': 0,
            'web_searches': 0
        }
        self.load_knowledge()
    
    def load_knowledge(self):
        kb_file = 'knowledge_base/knowledge.json'
        if os.path.exists(kb_file):
            with open(kb_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for item in data:
                    self.search.knowledge.append(item)
        
        self.search.update_vectors()
        print(f"✅ {len(self.search.knowledge)} دانش بارگذاری شد")
    
    def save_knowledge(self):
        kb_file = 'knowledge_base/knowledge.json'
        with open(kb_file, 'w', encoding='utf-8') as f:
            json.dump(self.search.knowledge[-100000:], f, ensure_ascii=False, indent=2)  # افزایش به 100 هزار
    
    def learn(self, question, answer, category='general'):
        self.search.add_knowledge(question, answer, category, 'manual')
        self.search.update_vectors()
        self.stats['learned'] += 1
        self.save_knowledge()
        return True
    
    def learn_from_file(self, filepath, filename):
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
    
    def ask(self, question, user_id=None):
        """پرسش و پاسخ با قابلیت جستجوی وب"""
        self.stats['asked'] += 1
        
        # جستجو در دانش داخلی
        results = self.search.search(question)
        
        if results and results[0]['score'] > 0.3:  # آستانه اطمینان
            best = results[0]
            
            # ثبت سوال پاسخ داده شده
            if user_id:
                self.user_manager.track_question(user_id, question, answered=True)
            
            return {
                'answer': best['answer'],
                'confidence': best['score'],
                'found': True,
                'source': 'local'
            }
        
        # اگر در دانش داخلی نبود، جستجو در وب
        try:
            web_results = self.search.web_search(question)
            if web_results:
                answer = self.search.web_search.extract_answer(question, web_results)
                if answer:
                    self.stats['web_searches'] += 1
                    
                    # ثبت سوال پاسخ داده شده
                    if user_id:
                        self.user_manager.track_question(user_id, question, answered=True)
                    
                    return {
                        'answer': answer,
                        'confidence': 0.7,
                        'found': True,
                        'source': 'web'
                    }
        except:
            pass
        
        # ثبت سوال بی‌پاسخ
        if user_id:
            self.user_manager.track_question(user_id, question, answered=False)
        
        return {'answer': None, 'found': False}
    
    def get_stats(self):
        return {
            'knowledge': len(self.search.knowledge),
            'learned': self.stats['learned'],
            'asked': self.stats['asked'],
            'files': self.stats['files_processed'],
            'web_searches': self.stats['web_searches']
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

# ================ صفحه اصلی چت با قابلیت زوم غیرفعال ================
@app.route('/')
def index():
    user_id = request.cookies.get('user_id', str(uuid.uuid4()))
    stats = ai.get_stats()
    
    # ثبت یا به‌روزرسانی کاربر
    user = ai.user_manager.get_user(user_id)
    
    resp = make_response(render_template_string('''
    <!DOCTYPE html>
    <html lang="fa" dir="rtl">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
        <title>هوش مصنوعی فوق‌پیشرفته</title>
        <style>
            * { 
                margin: 0; 
                padding: 0; 
                box-sizing: border-box; 
                -webkit-tap-highlight-color: transparent;
                touch-action: pan-y pinch-zoom; /* غیرفعال کردن زوم دو انگشتی */
            }
            body {
                font-family: 'Vazir', 'Tahoma', sans-serif;
                background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
                height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
                padding: 0;
                margin: 0;
                overflow: hidden;
                position: fixed;
                width: 100%;
            }
            .chat-container {
                width: 100%;
                max-width: 500px;
                height: 100vh;
                background: white;
                border-radius: 0;
                box-shadow: none;
                display: flex;
                flex-direction: column;
                overflow: hidden;
                position: relative;
            }
            .chat-header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 20px;
                display: flex;
                align-items: center;
                justify-content: space-between;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
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
                background: rgba(255,255,255,0.1);
            }
            .menu-btn:active {
                background: rgba(255,255,255,0.2);
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
                font-size: 16px;
                word-break: break-word;
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
            .source-badge {
                font-size: 0.7em;
                background: rgba(255,255,255,0.2);
                padding: 2px 8px;
                border-radius: 12px;
                display: inline-block;
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
                align-items: center;
            }
            .chat-input {
                flex: 1;
                padding: 14px 18px;
                border: 2px solid #e0e0e0;
                border-radius: 30px;
                font-size: 16px;
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
                font-size: 1.4em;
                display: flex;
                align-items: center;
                justify-content: center;
                box-shadow: 0 2px 8px rgba(102,126,234,0.4);
            }
            .send-btn:active {
                transform: scale(0.95);
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
                overflow-y: auto;
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
                transition: background 0.2s;
            }
            .menu-item:active { background: #f0f2f5; }
            .stats-badge {
                background: rgba(255,255,255,0.2);
                padding: 5px 10px;
                border-radius: 20px;
                font-size: 0.8em;
            }
            .welcome-message {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 15px;
                border-radius: 15px;
                margin-bottom: 15px;
                text-align: center;
            }
        </style>
    </head>
    <body>
        <div class="chat-container">
            <div class="chat-header">
                <button class="menu-btn" onclick="toggleMenu()">☰</button>
                <div class="stats-badge">{{ stats.knowledge }} دانش | {{ stats.web_searches }} وب</div>
            </div>
            
            <div class="chat-messages" id="chat-messages">
                <div class="message bot">
                    <div class="message-content">
                        🤖 سلام! من هوش مصنوعی فوق‌پیشرفته هستم<br>
                        می‌تونم از منابع مختلف جواب بدم
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
            <div class="menu-item" onclick="clearHistory()">🗑️ پاک کردن تاریخچه</div>
            <div class="menu-item" onclick="showStats()">📊 آمار من</div>
        </div>
        
        <script>
            let chatHistory = JSON.parse(localStorage.getItem('chat_history')) || [];
            
            chatHistory.forEach(msg => {
                addMessage(msg.text, msg.isUser, msg.time, msg.source, false);
            });
            
            function toggleMenu() {
                document.getElementById('menuOverlay').style.display = 'block';
                document.getElementById('menuPanel').classList.add('open');
            }
            
            function closeMenu() {
                document.getElementById('menuOverlay').style.display = 'none';
                document.getElementById('menuPanel').classList.remove('open');
            }
            
            function addMessage(text, isUser = false, time = null, source = null, save = true) {
                const div = document.createElement('div');
                div.className = `message ${isUser ? 'user' : 'bot'}`;
                
                const msgTime = time || new Date().toLocaleTimeString('fa-IR');
                let sourceHtml = '';
                if (source && !isUser) {
                    sourceHtml = `<div class="source-badge">منبع: ${source === 'web' ? '🌐 وب' : '📚 دانش'}</div>`;
                }
                
                div.innerHTML = `
                    <div class="message-content">
                        ${text.replace(/\\n/g, '<br>')}
                        ${sourceHtml}
                        <div class="message-time">${msgTime}</div>
                    </div>
                `;
                
                document.getElementById('chat-messages').appendChild(div);
                div.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
                
                if (save) {
                    chatHistory.push({ text, isUser, time: msgTime, source });
                    if (chatHistory.length > 100) chatHistory = chatHistory.slice(-100);
                    localStorage.setItem('chat_history', JSON.stringify(chatHistory));
                }
            }
            
            function showTyping() {
                const div = document.createElement('div');
                div.className = 'message bot';
                div.id = 'typing';
                div.innerHTML = '<div class="typing-indicator"><span></span><span></span><span></span></div>';
                document.getElementById('chat-messages').appendChild(div);
                div.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
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
                        addMessage(data.answer, false, null, data.source || 'local');
                    } else {
                        addMessage('🤔 متأسفم! نتونستم جواب پیدا کنم.');
                    }
                    
                } catch (error) {
                    hideTyping();
                    addMessage('⚠️ خطا در ارتباط با سرور');
                }
            }
            
            function clearHistory() {
                if (confirm('آیا تاریخچه پاک شود؟')) {
                    localStorage.removeItem('chat_history');
                    chatHistory = [];
                    location.reload();
                }
            }
            
            function showStats() {
                const total = chatHistory.length;
                const userMsgs = chatHistory.filter(m => m.isUser).length;
                const botMsgs = total - userMsgs;
                alert(`📊 آمار شما:\nکل پیام‌ها: ${total}\nپیام‌های شما: ${userMsgs}\nپاسخ‌ها: ${botMsgs}`);
            }
            
            // جلوگیری از زوم
            document.addEventListener('gesturestart', function(e) {
                e.preventDefault();
            });
            
            document.addEventListener('touchmove', function(e) {
                if (e.scale !== 1) { e.preventDefault(); }
            }, { passive: false });
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
    <head>
        <title>صفحه M</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
        <style>
            body {
                font-family: Tahoma;
                background: linear-gradient(135deg,#667eea 0%,#764ba2 100%);
                min-height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
                padding: 20px;
                margin: 0;
                touch-action: pan-y pinch-zoom;
            }
            .container {
                background: white;
                border-radius: 30px;
                padding: 40px;
                max-width: 600px;
                text-align: center;
            }
            .btn {
                display: inline-block;
                padding: 15px 40px;
                background: linear-gradient(135deg,#667eea 0%,#764ba2 100%);
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
            <p>صفحه مخصوص منو</p>
            <a href="/" class="btn">بازگشت</a>
        </div>
    </body>
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
                'found': True,
                'source': result.get('source', 'local')
            })
        else:
            return jsonify({'answer': None, 'found': False})
            
    except Exception as e:
        return jsonify({'error': str(e)})

# ================ پنل مدیریت فوق پیشرفته ================
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
    <head>
        <title>ورود</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body {
                font-family: Tahoma;
                background: linear-gradient(135deg,#667eea 0%,#764ba2 100%);
                height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
                margin: 0;
                padding: 10px;
            }
            .login-box {
                background: white;
                padding: 40px;
                border-radius: 30px;
                width: 100%;
                max-width: 400px;
            }
            input, button {
                width: 100%;
                padding: 15px;
                margin: 10px 0;
                border-radius: 15px;
                border: 2px solid #e0e0e0;
                font-size: 16px;
            }
            button {
                background: linear-gradient(135deg,#667eea 0%,#764ba2 100%);
                color: white;
                border: none;
                cursor: pointer;
            }
        </style>
    </head>
    <body>
        <div class="login-box">
            <h2>🔐 ورود به پنل مدیریت</h2>
            <form method="POST">
                <input name="username" value="admin" placeholder="نام کاربری">
                <input name="password" type="password" value="admin123" placeholder="رمز عبور">
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
    unanswered = ai.user_manager.get_unanswered()
    popular_topics = ai.user_manager.get_popular_topics(10)
    users = ai.user_manager.get_all_users()
    
    return f'''
    <!DOCTYPE html>
    <html>
    <head>
        <title>پنل مدیریت پیشرفته</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            body {{
                font-family: Tahoma;
                background: #f5f5f5;
                padding: 20px;
            }}
            .header {{
                background: linear-gradient(135deg,#667eea 0%,#764ba2 100%);
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
                grid-template-columns: repeat(auto-fit,minmax(200px,1fr));
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
            .card-header {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 15px;
            }}
            textarea, input, select {{
                width: 100%;
                padding: 12px;
                margin: 10px 0;
                border: 2px solid #e0e0e0;
                border-radius: 10px;
                font-size: 14px;
            }}
            button {{
                background: linear-gradient(135deg,#667eea 0%,#764ba2 100%);
                color: white;
                padding: 12px 25px;
                border: none;
                border-radius: 10px;
                cursor: pointer;
                font-size: 14px;
            }}
            button.secondary {{
                background: #6c757d;
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
            .alert-success {{
                background: #d4edda;
                color: #155724;
                padding: 15px;
                border-radius: 10px;
                margin: 10px 0;
            }}
            .grid-2 {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 20px;
            }}
            .table {{
                width: 100%;
                border-collapse: collapse;
            }}
            .table th, .table td {{
                padding: 10px;
                text-align: right;
                border-bottom: 1px solid #eee;
            }}
            .table tr:hover {{
                background: #f8f9fa;
            }}
            .badge {{
                padding: 5px 10px;
                border-radius: 20px;
                font-size: 0.8em;
            }}
            .badge-primary {{ background: #667eea; color: white; }}
            .badge-warning {{ background: #ffc107; color: #333; }}
            .badge-danger {{ background: #dc3545; color: white; }}
            .tab-nav {{
                display: flex;
                gap: 10px;
                margin-bottom: 20px;
                flex-wrap: wrap;
            }}
            .tab {{
                padding: 10px 20px;
                background: #e0e0e0;
                border-radius: 10px;
                cursor: pointer;
            }}
            .tab.active {{
                background: #667eea;
                color: white;
            }}
            .tab-content {{
                display: none;
            }}
            .tab-content.active {{
                display: block;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h2>⚙️ پنل مدیریت پیشرفته</h2>
            <div>
                <a href="/" style="color:white; margin-right:15px;">🏠 چت</a>
                <a href="/logout" style="color:white;">🚪 خروج</a>
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
            <div class="stat-card">
                <div class="stat-number">{stats['files']}</div>
                <div>فایل‌ها</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{stats['web_searches']}</div>
                <div>جستجوی وب</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{len(users)}</div>
                <div>کاربران</div>
            </div>
        </div>
        
        <div class="tab-nav">
            <div class="tab active" onclick="showTab('learn')">📚 یادگیری</div>
            <div class="tab" onclick="showTab('unanswered')">❓ بی‌پاسخ ({len(unanswered)})</div>
            <div class="tab" onclick="showTab('users')">👥 کاربران</div>
            <div class="tab" onclick="showTab('topics')">📊 موضوعات پرطرفدار</div>
            <div class="tab" onclick="showTab('stats')">📈 آمار پیشرفته</div>
        </div>
        
        <div id="learn" class="tab-content active">
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
                            <option value="math">ریاضی</option>
                            <option value="literature">ادبیات</option>
                        </select>
                        <button type="submit">📚 یاد بگیر</button>
                    </form>
                </div>
                
                <div class="card">
                    <h3>📁 آپلود فایل آموزشی</h3>
                    <p style="color:#666; margin-bottom:10px;">فرمت‌های پشتیبانی: .txt, .csv, .json, .md, .jpg, .png</p>
                    <form action="/admin/learn/file" method="POST" enctype="multipart/form-data">
                        <div class="file-upload" onclick="document.getElementById('file').click()">
                            <p style="font-size:2em;">📤</p>
                            <p>کلیک برای انتخاب فایل</p>
                            <p style="color:#666; font-size:0.9em; margin-top:10px;">
                                • txt: سوال | جواب<br>
                                • csv: سوال,جواب,دسته<br>
                                • json: [{{"question":"...","answer":"..."}}]<br>
                                • jpg/png: استخراج متن با OCR
                            </p>
                        </div>
                        <input type="file" id="file" name="file" style="display:none;" accept=".txt,.csv,.json,.md,.jpg,.jpeg,.png">
                        <button type="submit" style="width:100%;">📥 آپلود و یادگیری</button>
                    </form>
                </div>
            </div>
            
            <div class="card">
                <h3>✨ نمونه فرمت فایل</h3>
                <textarea rows="6" readonly style="background:#f8fafc; font-family:monospace;">سلام | سلام! چطور میتونم کمک کنم؟
کوروش کبیر که بود | کوروش بزرگ بنیانگذار هخامنشی بود
پایتخت ایران کجاست | تهران پایتخت ایران است

# فرمت JSON:
[
    {{"question": "تاریخ تخت جمشید", "answer": "۵۱۸ پیش از میلاد", "category": "history"}}
]</textarea>
                <button onclick="copySample()" style="margin-top:10px;">📋 کپی نمونه</button>
            </div>
        </div>
        
        <div id="unanswered" class="tab-content">
            <div class="card">
                <h3>❌ سوالات بی‌پاسخ</h3>
                <table class="table">
                    <thead>
                        <tr>
                            <th>زمان</th>
                            <th>کاربر</th>
                            <th>سوال</th>
                            <th>عملیات</th>
                        </tr>
                    </thead>
                    <tbody>
                        {''.join([f'''
                        <tr>
                            <td>{item['time'][:16]}</td>
                            <td>{item['user_id'][:8]}...</td>
                            <td>{item['question'][:50]}</td>
                            <td>
                                <button class="secondary" onclick="answerQuestion('{item['question']}')">✏️ پاسخ</button>
                            </td>
                        </tr>
                        ''' for item in unanswered])}
                    </tbody>
                </table>
            </div>
        </div>
        
        <div id="users" class="tab-content">
            <div class="card">
                <h3>👥 لیست کاربران</h3>
                <table class="table">
                    <thead>
                        <tr>
                            <th>کاربر</th>
                            <th>اولین بازدید</th>
                            <th>آخرین بازدید</th>
                            <th>تعداد سوالات</th>
                        </tr>
                    </thead>
                    <tbody>
                        {''.join([f'''
                        <tr>
                            <td>{user_id[:8]}...</td>
                            <td>{info.get('first_seen', '')[:10]}</td>
                            <td>{info.get('last_seen', '')[:10]}</td>
                            <td>{info.get('total_questions', 0)}</td>
                        </tr>
                        ''' for user_id, info in list(users.items())[:20]])}
                    </tbody>
                </table>
            </div>
        </div>
        
        <div id="topics" class="tab-content">
            <div class="card">
                <h3>📊 موضوعات پرطرفدار</h3>
                <table class="table">
                    <thead>
                        <tr>
                            <th>موضوع</th>
                            <th>تعداد جستجو</th>
                            <th>درصد</th>
                        </tr>
                    </thead>
                    <tbody>
                        {''.join([f'''
                        <tr>
                            <td>{topic}</td>
                            <td>{count}</td>
                            <td>
                                <div style="background:#e0e0e0; height:20px; border-radius:10px; overflow:hidden;">
                                    <div style="background:#667eea; width:{count/max([c for t,c in popular_topics])*100}%; height:100%;"></div>
                                </div>
                            </td>
                        </tr>
                        ''' for topic, count in popular_topics])}
                    </tbody>
                </table>
            </div>
        </div>
        
        <div id="stats" class="tab-content">
            <div class="card">
                <h3>📈 آمار پیشرفته</h3>
                <div class="stats-grid" style="grid-template-columns:repeat(2,1fr);">
                    <div class="stat-card">
                        <div class="stat-number">{sum(stats.values())}</div>
                        <div>مجموع عملیات</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">{stats['knowledge']/max(1,stats['asked']):.2f}</div>
                        <div>نسبت دانش به سوال</div>
                    </div>
                </div>
            </div>
        </div>
        
        <script>
            function showTab(tabId) {
                document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
                document.querySelectorAll('.tab').forEach(el => el.classList.remove('active'));
                
                document.getElementById(tabId).classList.add('active');
                event.target.classList.add('active');
            }
            
            function copySample() {{
                const textarea = document.querySelector('textarea');
                textarea.select();
                document.execCommand('copy');
                alert('✅ نمونه کپی شد!');
            }}
            
            function answerQuestion(question) {{
                const answer = prompt('پاسخ برای: ' + question);
                if (answer) {{
                    fetch('/admin/quick-answer', {{
                        method: 'POST',
                        headers: {{'Content-Type': 'application/json'}},
                        body: JSON.stringify({{question, answer}})
                    }}).then(() => location.reload());
                }}
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
            return error_page("فایلی انتخاب نشده است")
        
        file = request.files['file']
        if file.filename == '':
            return error_page("نام فایل معتبر نیست")
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        result = ai.learn_from_file(filepath, filename)
        os.remove(filepath)
        
        if result['success'] and result['total'] > 0:
            categories = ', '.join([f"{k}: {v}" for k, v in result['categories'].items()])
            return success_page(f"✅ {result['total']} مورد یاد گرفته شد<br>دسته‌بندی: {categories}")
        else:
            return warning_page("موردی برای یادگیری یافت نشد!")
        
    except Exception as e:
        return error_page(str(e))

@app.route('/admin/quick-answer', methods=['POST'])
@login_required
def quick_answer():
    data = request.json
    question = data.get('question')
    answer = data.get('answer')
    
    if question and answer:
        ai.learn(question, answer, 'unanswered')
        return jsonify({'success': True})
    return jsonify({'success': False})

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

def success_page(message):
    return f'''
    <div style="font-family:Tahoma; padding:20px; background:#f5f5f5;">
        <div style="background:#d4edda; color:#155724; padding:20px; border-radius:10px;">
            <h3>✅ موفقیت</h3>
            <p>{message}</p>
            <a href="/admin" style="display:inline-block; margin-top:15px; padding:10px 20px; background:#667eea; color:white; text-decoration:none; border-radius:5px;">🔙 بازگشت</a>
        </div>
    </div>
    '''

def warning_page(message):
    return f'''
    <div style="font-family:Tahoma; padding:20px;">
        <div style="background:#fff3cd; color:#856404; padding:20px; border-radius:10px;">
            <h3>⚠️ هشدار</h3>
            <p>{message}</p>
            <a href="/admin" style="display:inline-block; margin-top:15px; padding:10px 20px; background:#667eea; color:white; text-decoration:none; border-radius:5px;">🔙 بازگشت</a>
        </div>
    </div>
    '''

def error_page(message):
    return f'''
    <div style="font-family:Tahoma; padding:20px;">
        <div style="background:#f8d7da; color:#721c24; padding:20px; border-radius:10px;">
            <h3>❌ خطا</h3>
            <p>{message}</p>
            <a href="/admin" style="display:inline-block; margin-top:15px; padding:10px 20px; background:#667eea; color:white; text-decoration:none; border-radius:5px;">🔙 بازگشت</a>
        </div>
    </div>
    '''

if __name__ == '__main__':
    print("""
    ╔══════════════════════════════════════════════════════════════════════════╗
    ║     🤖 هوش مصنوعی فوق‌العاده با قابلیت‌های پیشرفته                       ║
    ╠══════════════════════════════════════════════════════════════════════════╣
    ║  📚 دانش: {}                                                             ║
    ║  📁 فایل‌های پردازش شده: {}                                              ║
    ║  🌐 جستجوهای وب: {}                                                       ║
    ║  🔍 فرمت‌های پشتیبانی: txt, csv, json, md, jpg, png                     ║
    ║  🖼️ تشخیص حروف: EasyOCR + Tesseract                                      ║
    ║  💾 حافظه: 50 گیگابایت (مدیریت هوشمند)                                   ║
    ║  🌐 چت: http://localhost:5000                                             ║
    ║  🔐 پنل: http://localhost:5000/admin-login                                ║
    ║  👤 کاربر: admin / admin123                                               ║
    ║  📱 صفحه چت: بدون زوم، بدون کشیدن دو انگشتی                              ║
    ╚══════════════════════════════════════════════════════════════════════════╝
    """.format(ai.get_stats()['knowledge'], ai.get_stats()['files'], ai.get_stats()['web_searches']))
    
    app.run(debug=True, host='0.0.0.0', port=5000)
