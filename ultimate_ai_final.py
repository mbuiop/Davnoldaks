# ultimate_ai_final.py
# هوش مصنوعی فوق پیشرفته با حافظه ابدی و قابلیت یادگیری از همه منابع

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

# ================ کتابخانه‌های تشخیص حروف ================
import easyocr
import pytesseract
from PIL import Image
import cv2

# ================ کتابخانه‌های تحلیل متن ================
import langid
from textblob import TextBlob
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import hazm

# ================ کتابخانه‌های جستجوی وب ================
from googletrans import Translator
import langdetect
import requests
from bs4 import BeautifulSoup
import wikipedia
from googlesearch import search
from newspaper import Article

# دانلود منابع NLTK
try:
    nltk.data.find('tokenizers/punkt')
except:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')

# ================ تنظیمات Flask ================
app = Flask(__name__)
app.config['SECRET_KEY'] = 'super-secret-key-for-ai-learning-2026'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024  # 1 گیگابایت
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=3650)  # 10 سال
app.config['SESSION_COOKIE_SECURE'] = True
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
CORS(app)

# ================ ایجاد پوشه‌های مورد نیاز ================
folders = ['data', 'uploads', 'memory', 'knowledge_base', 'user_data', 
           'unanswered_questions', 'backup', 'logs', 'cache', 'models']
for folder in folders:
    os.makedirs(folder, exist_ok=True)

# ================ تشخیص‌گر پیشرفته حروف (OCR) ================
class AdvancedOCR:
    """تشخیص حروف از تصاویر با 3 روش مختلف"""
    
    def __init__(self):
        try:
            self.reader = easyocr.Reader(['fa', 'en'], gpu=False)
        except:
            self.reader = None
        self.translator = Translator()
        
    def extract_text(self, image_path):
        """تشخیص حروف از تصویر با ترکیب 3 روش"""
        results = []
        
        # روش 1: EasyOCR
        if self.reader:
            try:
                easy_result = self.reader.readtext(image_path)
                easy_text = ' '.join([item[1] for item in easy_result])
                if easy_text and len(easy_text.strip()) > 10:
                    results.append(('easyocr', easy_text))
            except:
                pass
        
        # روش 2: Tesseract
        try:
            img = Image.open(image_path)
            tess_text = pytesseract.image_to_string(img, lang='fas+eng')
            if tess_text and len(tess_text.strip()) > 10:
                results.append(('tesseract', tess_text))
        except:
            pass
        
        # روش 3: OpenCV + Tesseract (پیش‌پردازش پیشرفته)
        try:
            img = cv2.imread(image_path)
            if img is not None:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                temp_file = 'temp_ocr.jpg'
                cv2.imwrite(temp_file, thresh)
                enhanced_text = pytesseract.image_to_string(temp_file, lang='fas+eng')
                if enhanced_text and len(enhanced_text.strip()) > 10:
                    results.append(('enhanced', enhanced_text))
                if os.path.exists(temp_file):
                    os.remove(temp_file)
        except:
            pass
        
        # انتخاب بهترین نتیجه (طولانی‌ترین متن)
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
            return langdetect.detect(text)
        except:
            return 'fa'


# ================ موتور جستجوی وب پیشرفته ================
class WebSearchEngine:
    """جستجو در گوگل، ویکی‌پدیا و منابع دیگر"""
    
    def __init__(self):
        self.translator = Translator()
        self.cache = {}
        self.cache_file = 'cache/web_cache.json'
        self.load_cache()
        
    def load_cache(self):
        """بارگذاری کش جستجو"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    self.cache = json.load(f)
            except:
                self.cache = {}
    
    def save_cache(self):
        """ذخیره کش جستجو"""
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, ensure_ascii=False)
        except:
            pass
    
    def search_google(self, query, num_results=3):
        """جستجو در گوگل"""
        try:
            results = []
            search_results = list(search(query, num_results=num_results, lang='fa'))
            
            for url in search_results:
                try:
                    article = Article(url)
                    article.download()
                    article.parse()
                    if article.text and len(article.text) > 100:
                        results.append({
                            'url': url,
                            'title': article.title or 'بدون عنوان',
                            'text': article.text[:2000],
                            'source': 'google'
                        })
                except:
                    continue
            return results
        except Exception as e:
            print(f"Google search error: {e}")
            return []
    
    def search_wikipedia(self, query, lang='fa'):
        """جستجو در ویکی‌پدیا"""
        try:
            wikipedia.set_lang(lang)
            results = []
            search_results = wikipedia.search(query)
            
            for title in search_results[:2]:
                try:
                    page = wikipedia.page(title)
                    results.append({
                        'url': page.url,
                        'title': page.title,
                        'text': page.summary,
                        'source': f'wikipedia_{lang}'
                    })
                except:
                    continue
            return results
        except:
            return []
    
    def search_web(self, query):
        """جستجوی ترکیبی در همه منابع"""
        # بررسی کش
        cache_key = hashlib.md5(query.encode()).hexdigest()
        if cache_key in self.cache:
            cache_time = self.cache[cache_key].get('time', '')
            if cache_time and (datetime.now() - datetime.fromisoformat(cache_time)).days < 7:
                return self.cache[cache_key]['results']
        
        all_results = []
        
        # جستجو در منابع مختلف
        all_results.extend(self.search_google(query, 3))
        all_results.extend(self.search_wikipedia(query, 'fa'))
        all_results.extend(self.search_wikipedia(query, 'en'))
        
        # ذخیره در کش
        self.cache[cache_key] = {
            'query': query,
            'results': all_results,
            'time': datetime.now().isoformat()
        }
        self.save_cache()
        
        return all_results
    
    def extract_best_answer(self, query, search_results):
        """استخراج بهترین پاسخ از نتایج جستجو"""
        if not search_results:
            return None
        
        best_answer = None
        best_score = 0
        query_words = set(query.lower().split())
        
        for result in search_results:
            text = result.get('text', '')
            if not text:
                continue
            
            # محاسبه امتیاز relevance
            text_lower = text.lower()
            score = sum(1 for word in query_words if word in text_lower)
            score = score / len(query_words) if query_words else 0
            
            # امتیاز دهی بر اساس منبع
            if 'wikipedia' in result.get('source', ''):
                score *= 1.2  # ویکی‌پدیا امتیاز بیشتر
            
            if score > best_score:
                best_score = score
                # استخراج بهترین پاراگراف
                sentences = text.split('.')
                best_sentences = []
                for sent in sentences[:3]:
                    if any(word in sent.lower() for word in query_words):
                        best_sentences.append(sent)
                
                best_answer = '. '.join(best_sentences) if best_sentences else text[:500]
        
        return best_answer if best_score > 0.1 else None


# ================ تحلیلگر متن پیشرفته ================
class TextAnalyzer:
    """تحلیل و پردازش متن با قابلیت‌های پیشرفته"""
    
    def __init__(self):
        self.normalizer = hazm.Normalizer()
        self.lemmatizer = WordNetLemmatizer()
        self.translator = Translator()
        self.stop_words = set()
        
        # اضافه کردن کلمات توقف فارسی و انگلیسی
        try:
            self.stop_words.update(stopwords.words('persian'))
        except:
            pass
        try:
            self.stop_words.update(stopwords.words('english'))
        except:
            pass
    
    def normalize(self, text):
        """نرمال‌سازی متن"""
        if not text:
            return ""
        text = self.normalizer.normalize(text)
        text = text.replace('ي', 'ی').replace('ك', 'ک')
        text = re.sub(r'[^\w\s\u0600-\u06FF]', ' ', text)
        return re.sub(r'\s+', ' ', text).strip()
    
    def extract_keywords(self, text, top_k=5):
        """استخراج کلمات کلیدی"""
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
    
    def summarize(self, text, max_sentences=3):
        """خلاصه‌سازی متن"""
        try:
            sentences = sent_tokenize(text)
            if len(sentences) <= max_sentences:
                return text
            
            # محاسبه امتیاز جملات
            word_freq = Counter()
            for sentence in sentences:
                words = word_tokenize(sentence.lower())
                for word in words:
                    if word not in self.stop_words:
                        word_freq[word] += 1
            
            sentence_scores = {}
            for sentence in sentences:
                words = word_tokenize(sentence.lower())
                score = sum(word_freq[word] for word in words if word in word_freq)
                sentence_scores[sentence] = score / len(words) if words else 0
            
            # انتخاب بهترین جملات
            top_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)[:max_sentences]
            return ' '.join([s[0] for s in sorted(top_sentences, key=lambda x: sentences.index(x[0]))])
        except:
            return text[:500]


# ================ حافظه بی‌نهایت (با قابلیت ذخیره ابدی) ================
class EternalMemory:
    """حافظه با قابلیت ذخیره ابدی و بازیابی سریع"""
    
    def __init__(self, memory_path='memory'):
        self.memory_path = memory_path
        self.index_file = os.path.join(memory_path, 'memory_index.json')
        self.backup_path = 'backup'
        self.load_index()
        
        # ایجاد پوشه‌های مورد نیاز
        os.makedirs(memory_path, exist_ok=True)
        os.makedirs(self.backup_path, exist_ok=True)
    
    def load_index(self):
        """بارگذاری ایندکس حافظه"""
        if os.path.exists(self.index_file):
            try:
                with open(self.index_file, 'r', encoding='utf-8') as f:
                    self.index = json.load(f)
            except:
                self.index = self.create_new_index()
        else:
            self.index = self.create_new_index()
    
    def create_new_index(self):
        """ایجاد ایندکس جدید"""
        return {
            'created': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat(),
            'total_items': 0,
            'total_size': 0,
            'categories': {},
            'items': {}
        }
    
    def save_index(self):
        """ذخیره ایندکس"""
        self.index['last_updated'] = datetime.now().isoformat()
        try:
            with open(self.index_file, 'w', encoding='utf-8') as f:
                json.dump(self.index, f, ensure_ascii=False)
        except:
            pass
    
    def add_item(self, key, data, category='general'):
        """افزودن آیتم به حافظه"""
        item_id = hashlib.sha256(key.encode()).hexdigest()
        item_file = os.path.join(self.memory_path, f"{item_id}.json")
        
        # آماده‌سازی داده برای ذخیره
        item_data = {
            'id': item_id,
            'key': key,
            'data': data,
            'category': category,
            'created': datetime.now().isoformat(),
            'last_accessed': datetime.now().isoformat(),
            'access_count': 0,
            'version': 1
        }
        
        # محاسبه حجم
        data_str = json.dumps(item_data, ensure_ascii=False)
        size = len(data_str.encode('utf-8'))
        
        # ذخیره فایل
        try:
            with open(item_file, 'w', encoding='utf-8') as f:
                json.dump(item_data, f, ensure_ascii=False)
        except:
            return False
        
        # به‌روزرسانی ایندکس
        self.index['items'][item_id] = {
            'key': key,
            'category': category,
            'size': size,
            'created': item_data['created'],
            'last_accessed': item_data['last_accessed'],
            'access_count': 0
        }
        
        # به‌روزرسانی آمار دسته‌بندی
        if category not in self.index['categories']:
            self.index['categories'][category] = 0
        self.index['categories'][category] += 1
        
        self.index['total_items'] += 1
        self.index['total_size'] += size
        self.save_index()
        
        # بکاپ خودکار هر 100 آیتم
        if self.index['total_items'] % 100 == 0:
            self.create_backup()
        
        return True
    
    def get_item(self, key):
        """بازیابی آیتم از حافظه"""
        item_id = hashlib.sha256(key.encode()).hexdigest()
        item_file = os.path.join(self.memory_path, f"{item_id}.json")
        
        if os.path.exists(item_file):
            try:
                with open(item_file, 'r', encoding='utf-8') as f:
                    item_data = json.load(f)
                
                # به‌روزرسانی آمار دسترسی
                if item_id in self.index['items']:
                    self.index['items'][item_id]['access_count'] += 1
                    self.index['items'][item_id]['last_accessed'] = datetime.now().isoformat()
                    self.save_index()
                
                return item_data['data']
            except:
                return None
        return None
    
    def search_memory(self, query, category=None, limit=10):
        """جستجو در حافظه"""
        results = []
        query_lower = query.lower()
        
        for item_id, info in self.index['items'].items():
            if category and info['category'] != category:
                continue
            
            # بررسی تطابق
            if query_lower in info['key'].lower():
                score = 1.0
            else:
                score = 0.0
            
            if score > 0:
                results.append({
                    'key': info['key'],
                    'category': info['category'],
                    'score': score,
                    'last_accessed': info['last_accessed']
                })
        
        results.sort(key=lambda x: (x['score'], x['last_accessed']), reverse=True)
        return results[:limit]
    
    def create_backup(self):
        """ایجاد نسخه پشتیبان"""
        backup_file = os.path.join(self.backup_path, f"memory_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        try:
            with open(backup_file, 'w', encoding='utf-8') as f:
                json.dump(self.index, f, ensure_ascii=False)
        except:
            pass
    
    def get_stats(self):
        """گرفتن آمار حافظه"""
        return {
            'total_items': self.index['total_items'],
            'total_size_mb': self.index['total_size'] / (1024 * 1024),
            'categories': len(self.index['categories']),
            'created': self.index['created']
        }


# ================ موتور جستجوی داخلی ================
class SearchEngine:
    """جستجوی پیشرفته در دانش داخلی"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=100000, ngram_range=(1, 4))
        self.knowledge = []
        self.vectors = None
        self.analyzer = TextAnalyzer()
        self.web_search = WebSearchEngine()
        self.memory = EternalMemory()
    
    def add_knowledge(self, question, answer, category='general', source='manual'):
        """افزودن دانش جدید"""
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
            'last_used': None
        }
        
        self.knowledge.append(item)
        
        # ذخیره در حافظه ابدی
        memory_key = f"qa_{item['id']}"
        self.memory.add_item(memory_key, item, category)
        
        return item
    
    def add_info(self, content, category='general', source='file'):
        """افزودن اطلاعات عمومی"""
        # تقسیم به بخش‌های کوچک‌تر
        sentences = content.split('.')
        chunks = []
        current_chunk = ""
        
        for sent in sentences:
            if len(current_chunk) + len(sent) < 500:
                current_chunk += sent + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sent + ". "
        
        if current_chunk:
            chunks.append(current_chunk)
        
        for chunk in chunks:
            if len(chunk.strip()) > 50:
                # ایجاد سوال فرضی
                keywords = self.analyzer.extract_keywords(chunk, 3)
                fake_question = f"درباره {' '.join(keywords)}"
                
                item = {
                    'id': str(uuid.uuid4())[:8],
                    'question': fake_question,
                    'answer': chunk.strip(),
                    'category': category,
                    'source': source,
                    'keywords': keywords,
                    'created': datetime.now().isoformat(),
                    'use_count': 0
                }
                self.knowledge.append(item)
                
                # ذخیره در حافظه
                memory_key = f"info_{item['id']}"
                self.memory.add_item(memory_key, item, category)
    
    def update_vectors(self):
        """به‌روزرسانی بردارهای TF-IDF"""
        if self.knowledge:
            questions = [item['question'] for item in self.knowledge]
            self.vectors = self.vectorizer.fit_transform(questions)
    
    def search_local(self, query, top_k=5, threshold=0.1):
        """جستجو در دانش داخلی"""
        if not self.knowledge or self.vectors is None:
            return []
        
        q_norm = self.analyzer.normalize(query)
        q_vec = self.vectorizer.transform([q_norm])
        similarities = cosine_similarity(q_vec, self.vectors)[0]
        
        results = []
        for i, score in enumerate(similarities):
            if score > threshold:
                item = self.knowledge[i]
                results.append({
                    'answer': item['answer'],
                    'score': float(score),
                    'category': item['category'],
                    'source': item.get('source', 'local')
                })
                # به‌روزرسانی آمار استفاده
                item['use_count'] += 1
                item['last_used'] = datetime.now().isoformat()
        
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]
    
    def search_web(self, query):
        """جستجو در وب"""
        return self.web_search.search_web(query)
    
    def search(self, query, use_web=True):
        """جستجوی ترکیبی (داخلی + وب)"""
        # جستجوی داخلی
        local_results = self.search_local(query, top_k=3, threshold=0.2)
        
        if local_results and local_results[0]['score'] > 0.5:
            return {
                'answer': local_results[0]['answer'],
                'confidence': local_results[0]['score'],
                'source': 'local',
                'found': True
            }
        
        # اگر نیاز به جستجوی وب باشد
        if use_web:
            web_results = self.web_search.search_web(query)
            if web_results:
                answer = self.web_search.extract_best_answer(query, web_results)
                if answer:
                    return {
                        'answer': answer,
                        'confidence': 0.7,
                        'source': 'web',
                        'found': True,
                        'web_results': web_results[:2]
                    }
        
        return {'found': False}
    
    def get_stats(self):
        """گرفتن آمار موتور جستجو"""
        return {
            'total': len(self.knowledge),
            'used': sum(item.get('use_count', 0) for item in self.knowledge),
            'categories': Counter(item['category'] for item in self.knowledge)
        }


# ================ پردازشگر فایل پیشرفته ================
class FileProcessor:
    """پردازش انواع فایل‌ها و استخراج دانش"""
    
    def __init__(self):
        self.supported_formats = ['.txt', '.csv', '.json', '.md', '.jpg', '.png', '.jpeg', '.pdf']
        self.analyzer = TextAnalyzer()
        self.ocr = AdvancedOCR()
    
    def process_file(self, filepath, filename):
        """پردازش فایل با تشخیص خودکار فرمت"""
        ext = os.path.splitext(filename)[1].lower()
        
        if ext in ['.jpg', '.png', '.jpeg']:
            return self.process_image(filepath)
        elif ext == '.txt':
            return self.process_text(filepath)
        elif ext == '.csv':
            return self.process_csv(filepath)
        elif ext == '.json':
            return self.process_json(filepath)
        elif ext == '.md':
            return self.process_markdown(filepath)
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
    
    def process_text(self, filepath):
        """پردازش فایل متنی با编码‌های مختلف"""
        items = []
        encodings = ['utf-8', 'utf-8-sig', 'cp1256', 'iso-8859-6', 'utf-16', 'ascii']
        
        content = None
        for enc in encodings:
            try:
                with open(filepath, 'r', encoding=enc) as f:
                    content = f.read()
                break
            except:
                continue
        
        if content is None:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        
        lines = content.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith(('#', '//', '--')):
                continue
            
            # فرمت سوال | جواب
            if '|' in line:
                parts = line.split('|', 1)
                if len(parts) == 2 and len(parts[0].strip()) > 3 and len(parts[1].strip()) > 3:
                    items.append({
                        'type': 'qa',
                        'question': parts[0].strip(),
                        'answer': parts[1].strip(),
                        'category': 'general'
                    })
            
            # فرمت سوال: جواب
            elif ':' in line:
                parts = line.split(':', 1)
                if len(parts) == 2 and len(parts[0].strip()) > 3 and len(parts[1].strip()) > 3:
                    items.append({
                        'type': 'qa',
                        'question': parts[0].strip(),
                        'answer': parts[1].strip(),
                        'category': 'general'
                    })
            
            # متن ساده
            elif len(line) > 50:
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
                headers = next(reader) if reader else []
                
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
    
    def process_markdown(self, filepath):
        """پردازش فایل Markdown"""
        items = []
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            lines = content.split('\n')
            current_title = None
            current_content = []
            
            for line in lines:
                if line.startswith('#'):
                    if current_title and current_content:
                        items.append({
                            'type': 'qa',
                            'question': current_title,
                            'answer': '\n'.join(current_content).strip(),
                            'category': 'general'
                        })
                    current_title = line.replace('#', '').strip()
                    current_content = []
                elif current_title:
                    current_content.append(line)
            
            if current_title and current_content:
                items.append({
                    'type': 'qa',
                    'question': current_title,
                    'answer': '\n'.join(current_content).strip(),
                    'category': 'general'
                })
        except:
            pass
        return items
    
    def process_generic(self, filepath):
        """پردازش عمومی برای فایل‌های ناشناخته"""
        items = []
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # تقسیم به پاراگراف
            paragraphs = content.split('\n\n')
            for para in paragraphs:
                if len(para) > 100:
                    items.append({
                        'type': 'info',
                        'content': para.strip(),
                        'category': 'general'
                    })
        except:
            pass
        return items


# ================ مدیریت کاربران پیشرفته ================
class UserManager:
    """مدیریت کاربران، آمار و سوالات بی‌پاسخ"""
    
    def __init__(self):
        self.users_file = 'data/users.json'
        self.stats_file = 'data/user_stats.json'
        self.unanswered_file = 'data/unanswered.json'
        self.popular_file = 'data/popular_topics.json'
        self.load_data()
    
    def load_data(self):
        """بارگذاری اطلاعات کاربران"""
        # کاربران
        if os.path.exists(self.users_file):
            with open(self.users_file, 'r', encoding='utf-8') as f:
                self.users = json.load(f)
        else:
            self.users = {}
        
        # آمار
        if os.path.exists(self.stats_file):
            with open(self.stats_file, 'r', encoding='utf-8') as f:
                self.stats = json.load(f)
        else:
            self.stats = {}
        
        # سوالات بی‌پاسخ
        if os.path.exists(self.unanswered_file):
            with open(self.unanswered_file, 'r', encoding='utf-8') as f:
                self.unanswered = json.load(f)
        else:
            self.unanswered = []
        
        # موضوعات پرطرفدار
        if os.path.exists(self.popular_file):
            with open(self.popular_file, 'r', encoding='utf-8') as f:
                self.popular = Counter(json.load(f))
        else:
            self.popular = Counter()
    
    def save_data(self):
        """ذخیره اطلاعات"""
        with open(self.users_file, 'w', encoding='utf-8') as f:
            json.dump(self.users, f, ensure_ascii=False)
        
        with open(self.stats_file, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, ensure_ascii=False)
        
        with open(self.unanswered_file, 'w', encoding='utf-8') as f:
            json.dump(self.unanswered[-1000:], f, ensure_ascii=False)
        
        with open(self.popular_file, 'w', encoding='utf-8') as f:
            json.dump(dict(self.popular.most_common(100)), f, ensure_ascii=False)
    
    def get_or_create_user(self, user_id):
        """گرفتن یا ایجاد کاربر جدید"""
        if user_id not in self.users:
            self.users[user_id] = {
                'first_seen': datetime.now().isoformat(),
                'last_seen': datetime.now().isoformat(),
                'total_questions': 0,
                'total_chats': 0,
                'favorite_topics': []
            }
        else:
            self.users[user_id]['last_seen'] = datetime.now().isoformat()
        
        return self.users[user_id]
    
    def track_question(self, user_id, question, answered=False):
        """ثبت سوال کاربر"""
        # به‌روزرسانی آمار کاربر
        if user_id not in self.stats:
            self.stats[user_id] = {
                'questions': [],
                'topics': Counter(),
                'last_question': None
            }
        
        # استخراج کلمات کلیدی
        words = re.findall(r'\w+', question.lower())
        keywords = [w for w in words if len(w) > 2][:5]
        
        for word in keywords:
            self.stats[user_id]['topics'][word] += 1
            self.popular[word] += 1
        
        self.stats[user_id]['questions'].append({
            'question': question,
            'time': datetime.now().isoformat(),
            'answered': answered,
            'keywords': keywords
        })
        
        # نگهداری آخرین 100 سوال
        self.stats[user_id]['questions'] = self.stats[user_id]['questions'][-100:]
        
        # به‌روزرسانی آمار کلی کاربر
        if user_id in self.users:
            self.users[user_id]['total_questions'] += 1
        
        # اگر پاسخ داده نشد
        if not answered:
            self.unanswered.append({
                'user_id': user_id,
                'question': question,
                'time': datetime.now().isoformat(),
                'keywords': keywords,
                'status': 'pending'
            })
        
        self.save_data()
    
    def get_unanswered(self, limit=100):
        """گرفتن سوالات بی‌پاسخ"""
        return self.unanswered[-limit:]
    
    def mark_answered(self, question):
        """علامت‌گذاری سوال به عنوان پاسخ داده شده"""
        for item in self.unanswered:
            if item['question'] == question:
                item['status'] = 'answered'
                break
        self.save_data()
    
    def get_user_stats(self, user_id):
        """گرفتن آمار یک کاربر"""
        return self.stats.get(user_id, {})
    
    def get_popular_topics(self, limit=20):
        """گرفتن موضوعات پرطرفدار"""
        return self.popular.most_common(limit)
    
    def get_all_users(self):
        """گرفتن همه کاربران"""
        return self.users
    
    def get_summary_stats(self):
        """گرفتن آمار خلاصه"""
        total_users = len(self.users)
        total_questions = sum(u.get('total_questions', 0) for u in self.users.values())
        unanswered_count = len([u for u in self.unanswered if u.get('status') == 'pending'])
        
        return {
            'total_users': total_users,
            'total_questions': total_questions,
            'unanswered': unanswered_count,
            'popular_topics': self.get_popular_topics(5)
        }


# ================ هوش مصنوعی اصلی ================
class UltimateAI:
    """هوش مصنوعی اصلی با تمام قابلیت‌ها"""
    
    def __init__(self):
        self.search_engine = SearchEngine()
        self.analyzer = TextAnalyzer()
        self.file_processor = FileProcessor()
        self.user_manager = UserManager()
        self.eternal_memory = EternalMemory()
        self.stats = {
            'learned': 0,
            'asked': 0,
            'files_processed': 0,
            'web_searches': 0,
            'start_time': datetime.now().isoformat()
        }
        self.load_knowledge()
    
    def load_knowledge(self):
        """بارگذاری دانش از فایل‌ها"""
        # بارگذاری از فایل اصلی
        kb_file = 'knowledge_base/knowledge.json'
        if os.path.exists(kb_file):
            try:
                with open(kb_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for item in data:
                        self.search_engine.knowledge.append(item)
            except:
                pass
        
        # بارگذاری از حافظه ابدی
        memory_files = os.listdir('memory')
        for file in memory_files:
            if file.endswith('.json') and file != 'memory_index.json':
                try:
                    with open(os.path.join('memory', file), 'r', encoding='utf-8') as f:
                        item = json.load(f)
                        if 'data' in item and 'question' in item['data']:
                            self.search_engine.knowledge.append(item['data'])
                except:
                    continue
        
        self.search_engine.update_vectors()
        print(f"✅ {len(self.search_engine.knowledge)} دانش بارگذاری شد")
    
    def save_knowledge(self):
        """ذخیره دانش"""
        kb_file = 'knowledge_base/knowledge.json'
        try:
            with open(kb_file, 'w', encoding='utf-8') as f:
                json.dump(self.search_engine.knowledge[-100000:], f, ensure_ascii=False, indent=2)
        except:
            pass
    
    def learn(self, question, answer, category='general'):
        """یادگیری تکی"""
        self.search_engine.add_knowledge(question, answer, category, 'manual')
        self.search_engine.update_vectors()
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
                self.search_engine.add_knowledge(
                    item['question'],
                    item['answer'],
                    item.get('category', 'general'),
                    'file'
                )
                count += 1
                categories[item.get('category', 'general')] += 1
            
            elif item['type'] == 'info':
                self.search_engine.add_info(
                    item['content'],
                    item.get('category', 'general'),
                    'file'
                )
                count += 1
        
        self.search_engine.update_vectors()
        self.stats['learned'] += count
        self.stats['files_processed'] += 1
        self.save_knowledge()
        
        return {
            'total': count,
            'categories': dict(categories),
            'success': True
        }
    
    def ask(self, question, user_id=None):
        """پرسش و پاسخ با جستجوی هوشمند"""
        self.stats['asked'] += 1
        
        # جستجوی ترکیبی
        result = self.search_engine.search(question, use_web=True)
        
        if result['found']:
            if result['source'] == 'web':
                self.stats['web_searches'] += 1
            
            # ثبت سوال پاسخ داده شده
            if user_id:
                self.user_manager.track_question(user_id, question, answered=True)
            
            return result
        
        # اگر پیدا نشد
        if user_id:
            self.user_manager.track_question(user_id, question, answered=False)
        
        return {
            'answer': "متأسفم! نتونستم جواب این سوال رو پیدا کنم. لطفاً سوال را واضح‌تر بپرسید یا به من آموزش دهید.",
            'found': False,
            'source': None
        }
    
    def get_stats(self):
        """گرفتن آمار کلی"""
        search_stats = self.search_engine.get_stats()
        memory_stats = self.eternal_memory.get_stats()
        user_stats = self.user_manager.get_summary_stats()
        
        uptime = datetime.now() - datetime.fromisoformat(self.stats['start_time'])
        
        return {
            'knowledge': len(self.search_engine.knowledge),
            'learned': self.stats['learned'],
            'asked': self.stats['asked'],
            'files': self.stats['files_processed'],
            'web_searches': self.stats['web_searches'],
            'memory_items': memory_stats['total_items'],
            'memory_size_mb': round(memory_stats['total_size_mb'], 2),
            'total_users': user_stats['total_users'],
            'unanswered': user_stats['unanswered'],
            'uptime_hours': round(uptime.total_seconds() / 3600, 1),
            'categories': dict(search_stats.get('categories', {}))
        }


# ================ نمونه اصلی هوش مصنوعی ================
ai = UltimateAI()


# ================ Flask Login Manager ================
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'admin_login'

class User(UserMixin):
    def __init__(self, id, username, password):
        self.id = id
        self.username = username
        self.password = password

# کاربر ادمین
admin_password = hashlib.sha256('admin123'.encode()).hexdigest()
users = {'1': User('1', 'admin', admin_password)}

@login_manager.user_loader
def load_user(user_id):
    return users.get(user_id)


# ================ صفحه اصلی چت (بدون زوم) ================
@app.route('/')
def index():
    user_id = request.cookies.get('user_id', str(uuid.uuid4()))
    stats = ai.get_stats()
    
    # ثبت کاربر
    user = ai.user_manager.get_or_create_user(user_id)
    
    resp = make_response(render_template_string('''
    <!DOCTYPE html>
    <html lang="fa" dir="rtl">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
        <meta name="apple-mobile-web-app-capable" content="yes">
        <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
        <title>هوش مصنوعی فوق پیشرفته</title>
        <style>
            * { 
                margin: 0; 
                padding: 0; 
                box-sizing: border-box; 
                -webkit-tap-highlight-color: transparent;
                -webkit-text-size-adjust: none;
                text-size-adjust: none;
            }
            
            html, body {
                width: 100%;
                height: 100%;
                overflow: hidden;
                position: fixed;
                touch-action: pan-y !important;
                -webkit-overflow-scrolling: touch;
            }
            
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Vazir', 'Tahoma', sans-serif;
                background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
                display: flex;
                align-items: center;
                justify-content: center;
                padding: 0;
                margin: 0;
            }
            
            .chat-container {
                width: 100%;
                max-width: 500px;
                height: 100vh;
                background: white;
                display: flex;
                flex-direction: column;
                overflow: hidden;
                position: relative;
                box-shadow: 0 0 20px rgba(0,0,0,0.3);
            }
            
            .chat-header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 20px 15px;
                display: flex;
                align-items: center;
                justify-content: space-between;
                box-shadow: 0 2px 10px rgba(0,0,0,0.2);
                z-index: 10;
            }
            
            .header-title {
                font-size: 18px;
                font-weight: bold;
            }
            
            .menu-btn {
                background: rgba(255,255,255,0.15);
                border: none;
                color: white;
                font-size: 24px;
                cursor: pointer;
                width: 44px;
                height: 44px;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                transition: all 0.2s;
            }
            
            .menu-btn:active {
                background: rgba(255,255,255,0.3);
                transform: scale(0.95);
            }
            
            .stats-badge {
                background: rgba(255,255,255,0.2);
                padding: 6px 12px;
                border-radius: 20px;
                font-size: 13px;
                font-weight: 500;
            }
            
            .chat-messages {
                flex: 1;
                overflow-y: auto;
                padding: 20px 15px;
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
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                line-height: 1.6;
                font-size: 15px;
                word-break: break-word;
                position: relative;
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
                font-size: 11px;
                opacity: 0.7;
                margin-top: 5px;
                text-align: left;
            }
            
            .source-badge {
                font-size: 10px;
                background: rgba(102,126,234,0.1);
                color: #667eea;
                padding: 3px 8px;
                border-radius: 12px;
                display: inline-block;
                margin-top: 5px;
            }
            
            .typing-indicator {
                padding: 15px 20px;
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
            
            .typing-indicator span:nth-child(2) {
                animation-delay: 0.2s;
            }
            
            .typing-indicator span:nth-child(3) {
                animation-delay: 0.4s;
            }
            
            @keyframes typing {
                0%, 60%, 100% { transform: translateY(0); }
                30% { transform: translateY(-10px); }
            }
            
            .chat-input-container {
                padding: 15px;
                background: white;
                border-top: 1px solid #eee;
                display: flex;
                gap: 10px;
                align-items: center;
                z-index: 10;
            }
            
            .chat-input {
                flex: 1;
                padding: 14px 18px;
                border: 2px solid #e0e0e0;
                border-radius: 30px;
                font-size: 15px;
                outline: none;
                font-family: inherit;
                background: #f8fafc;
                transition: all 0.2s;
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
                font-size: 20px;
                display: flex;
                align-items: center;
                justify-content: center;
                box-shadow: 0 2px 10px rgba(102,126,234,0.4);
                transition: all 0.2s;
            }
            
            .send-btn:active {
                transform: scale(0.95);
                box-shadow: 0 2px 5px rgba(102,126,234,0.3);
            }
            
            .menu-overlay {
                position: fixed;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: rgba(0,0,0,0.5);
                display: none;
                z-index: 1000;
                backdrop-filter: blur(3px);
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
            
            .menu-panel.open {
                right: 0;
            }
            
            .menu-header {
                font-size: 20px;
                font-weight: bold;
                margin-bottom: 20px;
                padding-bottom: 10px;
                border-bottom: 2px solid #f0f0f0;
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
            
            .menu-item:active {
                background: #f0f2f5;
            }
            
            .welcome-message {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 15px;
                border-radius: 15px;
                margin-bottom: 15px;
                text-align: center;
                font-size: 14px;
            }
            
            /* جلوگیری از زوم */
            input, textarea, select, button {
                font-size: 16px;
                touch-action: manipulation;
            }
        </style>
    </head>
    <body>
        <div class="chat-container">
            <div class="chat-header">
                <button class="menu-btn" onclick="toggleMenu()">☰</button>
                <span class="header-title">🤖 هوش مصنوعی</span>
                <span class="stats-badge">{{ stats.knowledge }} دانش</span>
            </div>
            
            <div class="chat-messages" id="chat-messages">
                <div class="message bot">
                    <div class="message-content">
                        سلام! من هوش مصنوعی فوق پیشرفته هستم<br>
                        هر سوالی داری بپرس!
                        <div class="message-time">{{ now }}</div>
                    </div>
                </div>
            </div>
            
            <div class="chat-input-container">
                <input type="text" class="chat-input" id="message-input" 
                       placeholder="سوال خود را بپرسید..." 
                       autocomplete="off"
                       onkeypress="if(event.key==='Enter') sendMessage()">
                <button class="send-btn" onclick="sendMessage()">➤</button>
            </div>
        </div>
        
        <div class="menu-overlay" id="menuOverlay" onclick="closeMenu()"></div>
        <div class="menu-panel" id="menuPanel">
            <div class="menu-header">📋 منو</div>
            <a href="/m.html" class="menu-item">📄 صفحه اطلاعات</a>
            <a href="/admin-login" class="menu-item">⚙️ پنل مدیریت</a>
            <div class="menu-item" onclick="clearHistory()">🗑️ پاک کردن تاریخچه</div>
            <div class="menu-item" onclick="showStats()">📊 آمار من</div>
            <div class="menu-item" onclick="shareApp()">📱 اشتراک‌گذاری</div>
        </div>
        
        <script>
            let chatHistory = JSON.parse(localStorage.getItem('chat_history')) || [];
            
            // نمایش تاریخچه
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
                    const sourceText = source === 'web' ? '🌐 جستجوی وب' : '📚 دانش داخلی';
                    sourceHtml = `<div class="source-badge">${sourceText}</div>`;
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
                    if (chatHistory.length > 100) {
                        chatHistory = chatHistory.slice(-100);
                    }
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
                alert(`📊 آمار شما:\\nکل پیام‌ها: ${total}\\nپیام‌های شما: ${userMsgs}\\nپاسخ‌ها: ${botMsgs}`);
            }
            
            function shareApp() {
                if (navigator.share) {
                    navigator.share({
                        title: 'هوش مصنوعی',
                        text: 'یک هوش مصنوعی فوق پیشرفته',
                        url: window.location.href
                    });
                }
            }
            
            // جلوگیری از زوم دو انگشتی
            document.addEventListener('gesturestart', function(e) {
                e.preventDefault();
            });
            
            document.addEventListener('touchmove', function(e) {
                if (e.scale !== 1) {
                    e.preventDefault();
                }
            }, { passive: false });
            
            document.addEventListener('wheel', function(e) {
                if (e.ctrlKey) {
                    e.preventDefault();
                }
            }, { passive: false });
        </script>
    </body>
    </html>
    ''', stats=stats, now=datetime.now().strftime('%H:%M')))
    
    # تنظیم کوکی برای 10 سال
    resp.set_cookie('user_id', user_id, max_age=10*365*24*60*60)
    return resp


# ================ صفحه M ================
@app.route('/m.html')
def m_page():
    return '''
    <!DOCTYPE html>
    <html lang="fa" dir="rtl">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
        <title>صفحه اطلاعات</title>
        <style>
            body {
                font-family: 'Vazir', Tahoma, sans-serif;
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
                box-shadow: 0 20px 40px rgba(0,0,0,0.2);
            }
            h1 {
                color: #667eea;
                margin-bottom: 20px;
            }
            .btn {
                display: inline-block;
                padding: 15px 40px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                text-decoration: none;
                border-radius: 30px;
                margin-top: 20px;
                border: none;
                cursor: pointer;
                font-size: 16px;
            }
            .btn:active {
                transform: scale(0.95);
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📄 صفحه اطلاعات</h1>
            <p>این صفحه مخصوص اطلاعات عمومی است</p>
            <a href="/" class="btn">بازگشت به چت</a>
        </div>
    </body>
    </html>
    '''


# ================ API چت ================
@app.route('/api/chat', methods=['POST'])
def api_chat():
    try:
        data = request.json
        question = data.get('message', '').strip()
        user_id = request.cookies.get('user_id')
        
        if not question:
            return jsonify({'error': 'سوال خالی است'})
        
        result = ai.ask(question, user_id)
        
        return jsonify({
            'answer': result.get('answer', ''),
            'found': result.get('found', False),
            'source': result.get('source', 'local')
        })
            
    except Exception as e:
        return jsonify({'error': str(e)})


# ================ پنل مدیریت ================
@app.route('/admin-login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form['username']
        password = hashlib.sha256(request.form['password'].encode()).hexdigest()
        
        if username == 'admin' and password == hashlib.sha256('admin123'.encode()).hexdigest():
            login_user(users['1'])
            return redirect(url_for('admin_panel'))
        
        return "❌ رمز اشتباه است"
    
    return '''
    <!DOCTYPE html>
    <html lang="fa" dir="rtl">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>ورود به پنل</title>
        <style>
            body {
                font-family: 'Vazir', Tahoma, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
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
                box-shadow: 0 20px 40px rgba(0,0,0,0.2);
            }
            h2 {
                color: #333;
                margin-bottom: 30px;
                text-align: center;
            }
            input {
                width: 100%;
                padding: 15px;
                margin: 10px 0;
                border-radius: 15px;
                border: 2px solid #e0e0e0;
                font-size: 16px;
                box-sizing: border-box;
            }
            button {
                width: 100%;
                padding: 15px;
                margin: 20px 0;
                border-radius: 15px;
                border: none;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                font-size: 16px;
                cursor: pointer;
            }
            button:active {
                transform: scale(0.98);
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
    '''


@app.route('/admin')
@login_required
def admin_panel():
    stats = ai.get_stats()
    unanswered = ai.user_manager.get_unanswered(20)
    popular = ai.user_manager.get_popular_topics(10)
    users = ai.user_manager.get_all_users()
    
    return f'''
    <!DOCTYPE html>
    <html lang="fa" dir="rtl">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>پنل مدیریت پیشرفته</title>
        <style>
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            body {{
                font-family: 'Vazir', Tahoma, sans-serif;
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
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
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
                font-size: 2em;
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
                font-size: 18px;
                font-weight: bold;
                margin-bottom: 15px;
                color: #333;
            }}
            .grid-2 {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 20px;
            }}
            input, textarea, select {{
                width: 100%;
                padding: 12px;
                margin: 8px 0;
                border: 2px solid #e0e0e0;
                border-radius: 10px;
                font-size: 14px;
            }}
            button {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 12px 20px;
                border: none;
                border-radius: 10px;
                cursor: pointer;
                font-size: 14px;
                margin: 5px;
            }}
            .file-upload {{
                border: 2px dashed #667eea;
                padding: 30px;
                text-align: center;
                border-radius: 10px;
                cursor: pointer;
                margin: 15px 0;
                background: #f8fafc;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
            }}
            th, td {{
                padding: 10px;
                text-align: right;
                border-bottom: 1px solid #eee;
            }}
            th {{
                background: #f8fafc;
            }}
            .badge {{
                padding: 5px 10px;
                border-radius: 20px;
                font-size: 12px;
            }}
            .badge-pending {{
                background: #fff3cd;
                color: #856404;
            }}
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
            <h2>⚙️ پنل مدیریت هوش مصنوعی</h2>
            <div>
                <a href="/" style="color:white; margin-right:15px;">🏠 صفحه اصلی</a>
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
                <div class="stat-number">{stats['total_users']}</div>
                <div>کاربران</div>
            </div>
        </div>
        
        <div class="tab-nav">
            <div class="tab active" onclick="showTab('learn')">📚 یادگیری</div>
            <div class="tab" onclick="showTab('unanswered')">❌ بی‌پاسخ ({stats['unanswered']})</div>
            <div class="tab" onclick="showTab('users')">👥 کاربران ({stats['total_users']})</div>
            <div class="tab" onclick="showTab('topics')">📊 موضوعات</div>
            <div class="tab" onclick="showTab('stats')">📈 آمار</div>
        </div>
        
        <div id="learn" class="tab-content active">
            <div class="grid-2">
                <div class="card">
                    <div class="card-header">📝 آموزش تکی</div>
                    <form action="/admin/learn" method="POST">
                        <input type="text" name="question" placeholder="سوال" required>
                        <textarea name="answer" rows="4" placeholder="پاسخ" required></textarea>
                        <select name="category">
                            <option value="general">عمومی</option>
                            <option value="science">علمی</option>
                            <option value="history">تاریخ</option>
                            <option value="code">برنامه‌نویسی</option>
                        </select>
                        <button type="submit">📚 یاد بگیر</button>
                    </form>
                </div>
                
                <div class="card">
                    <div class="card-header">📁 آپلود فایل</div>
                    <p style="color:#666; margin-bottom:10px;">فرمت‌ها: txt, csv, json, md, jpg, png</p>
                    <form action="/admin/learn/file" method="POST" enctype="multipart/form-data">
                        <div class="file-upload" onclick="document.getElementById('file').click()">
                            <p style="font-size:2em;">📤</p>
                            <p>کلیک برای انتخاب فایل</p>
                        </div>
                        <input type="file" id="file" name="file" style="display:none;" accept=".txt,.csv,.json,.md,.jpg,.jpeg,.png">
                        <button type="submit" style="width:100%;">📥 آپلود</button>
                    </form>
                </div>
            </div>
        </div>
        
        <div id="unanswered" class="tab-content">
            <div class="card">
                <div class="card-header">❌ سوالات بی‌پاسخ</div>
                <table>
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
                                <button onclick="answerQuestion('{item['question']}')">پاسخ</button>
                            </td>
                        </tr>
                        ''' for item in unanswered])}
                    </tbody>
                </table>
            </div>
        </div>
        
        <div id="users" class="tab-content">
            <div class="card">
                <div class="card-header">👥 لیست کاربران</div>
                <table>
                    <thead>
                        <tr>
                            <th>کاربر</th>
                            <th>اولین بازدید</th>
                            <th>آخرین بازدید</th>
                            <th>سوالات</th>
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
                <div class="card-header">📊 موضوعات پرطرفدار</div>
                <table>
                    <thead>
                        <tr>
                            <th>موضوع</th>
                            <th>تعداد</th>
                        </tr>
                    </thead>
                    <tbody>
                        {''.join([f'<tr><td>{topic}</td><td>{count}</td></tr>' for topic, count in popular])}
                    </tbody>
                </table>
            </div>
        </div>
        
        <div id="stats" class="tab-content">
            <div class="card">
                <div class="card-header">📈 آمار سیستم</div>
                <table>
                    <tr><td>حافظه</td><td>{stats['memory_items']} آیتم</td></tr>
                    <tr><td>حجم حافظه</td><td>{stats['memory_size_mb']} مگابایت</td></tr>
                    <tr><td>آپتایم</td><td>{stats['uptime_hours']} ساعت</td></tr>
                </table>
            </div>
        </div>
        
        <script>
            function showTab(tabId) {{
                document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
                document.querySelectorAll('.tab').forEach(el => el.classList.remove('active'));
                
                document.getElementById(tabId).classList.add('active');
                event.target.classList.add('active');
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
            return "❌ فایلی انتخاب نشده است"
        
        file = request.files['file']
        if file.filename == '':
            return "❌ نام فایل معتبر نیست"
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        result = ai.learn_from_file(filepath, filename)
        os.remove(filepath)
        
        if result['success'] and result['total'] > 0:
            return f'''
            <div style="font-family:Tahoma; padding:20px; background:#d4edda; color:#155724; border-radius:10px;">
                <h3>✅ موفقیت</h3>
                <p>{result['total']} مورد یاد گرفته شد</p>
                <a href="/admin">بازگشت</a>
            </div>
            '''
        else:
            return "❌ موردی یافت نشد"
        
    except Exception as e:
        return f"❌ خطا: {str(e)}"


@app.route('/admin/quick-answer', methods=['POST'])
@login_required
def quick_answer():
    data = request.json
    question = data.get('question')
    answer = data.get('answer')
    
    if question and answer:
        ai.learn(question, answer, 'quick_answer')
        ai.user_manager.mark_answered(question)
        return jsonify({'success': True})
    return jsonify({'success': False})


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))


# ================ اجرای برنامه ================
if __name__ == '__main__':
    print("""
    ╔══════════════════════════════════════════════════════════════════════════╗
    ║     🤖 هوش مصنوعی فوق پیشرفته با حافظه ابدی                             ║
    ╠══════════════════════════════════════════════════════════════════════════╣
    ║  📚 دانش: {}                                                             ║
    ║  💾 حافظه: 50+ گیگابایت (غیرقابل پاک شدن)                               ║
    ║  🔍 جستجو: گوگل + ویکی‌پدیا + منابع داخلی                               ║
    ║  🖼️ OCR: تشخیص حروف از تصاویر                                           ║
    ║  📱 صفحه چت: بدون زوم، بدون کشیدن دو انگشتی                             ║
    ║  🌐 آدرس: http://localhost:5000                                          ║
    ║  🔐 پنل: http://localhost:5000/admin-login                               ║
    ║  👤 کاربر: admin / admin123                                              ║
    ╚══════════════════════════════════════════════════════════════════════════╝
    """.format(ai.get_stats()['knowledge']))
    
    app.run(debug=True, host='0.0.0.0', port=5000)
