# ultra_advanced_history_bot.py
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, Response, send_file, make_response
from flask_cors import CORS
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.utils import secure_filename
from werkzeug.middleware.profiler import ProfilerMiddleware
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.naive_bayes import MultinomialNB
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from nltk.sentiment import SentimentIntensityAnalyzer
import hazm
from hazm import Normalizer, WordTokenizer, SentenceTokenizer, Lemmatizer, Stemmer, POSTagger, Chunker
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from bs4 import BeautifulSoup
import json
import os
import hashlib
from datetime import datetime, timedelta
import re
import threading
import queue
import time
import uuid
import pickle
import gzip
import base64
import secrets
import string
import random
import csv
import io
import tempfile
import shutil
import zipfile
import tarfile
from collections import Counter, defaultdict, deque
from functools import wraps, lru_cache
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import asyncio
import logging
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
import signal
import sys
import gc
import tracemalloc
import resource
import platform
import cpuinfo
import psutil
import subprocess
from pathlib import Path

# ================ تنظیمات اولیه ================
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', secrets.token_urlsafe(64))
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024 * 1024  # 100GB
app.config['MAX_CHUNK_SIZE'] = 100 * 1024 * 1024  # 100MB per chunk
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=7)
app.config['SESSION_REFRESH_EACH_REQUEST'] = True
app.config['REMEMBER_COOKIE_DURATION'] = timedelta(days=30)

CORS(app, supports_credentials=True, resources={
    r"/*": {
        "origins": ["*"],
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization", "X-Requested-With"]
    }
})

# ================ تنظیمات لاگینگ ================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
handler = RotatingFileHandler('logs/history_bot.log', maxBytes=10000000, backupCount=10)
handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
))
logger.addHandler(handler)

# ================ ایجاد پوشه‌های مورد نیاز ================
for folder in ['uploads', 'data', 'logs', 'cache', 'models', 'backups', 'temp']:
    os.makedirs(folder, exist_ok=True)
    os.makedirs(os.path.join(folder, 'chunks'), exist_ok=True)

# ================ دانلود منابع NLTK ================
nltk_resources = [
    'punkt', 'stopwords', 'wordnet', 'omw-1.4', 
    'averaged_perceptron_tagger', 'maxent_ne_chunker', 'words',
    'sentiment', 'vader_lexicon'
]
for resource in nltk_resources:
    try:
        nltk.download(resource, quiet=True)
    except:
        logger.warning(f"Failed to download {resource}")

# ================ راه‌اندازی ابزارهای پردازش زبان فارسی ================
hazm_normalizer = hazm.Normalizer()
hazm_tokenizer = hazm.WordTokenizer()
hazm_sent_tokenizer = hazm.SentenceTokenizer()
hazm_lemmatizer = hazm.Lemmatizer()
hazm_stemmer = hazm.Stemmer()
hazm_tagger = hazm.POSTagger(model='resources/pos_tagger.model')
hazm_chunker = hazm.Chunker(model='resources/chunker.model')

# ================ راه‌اندازی Session با requests ================
session_requests = requests.Session()
retry_strategy = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504]
)
adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=100, pool_maxsize=100)
session_requests.mount("http://", adapter)
session_requests.mount("https://", adapter)

# ================ مدیریت کاربران پیشرفته ================
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'admin_login'
login_manager.login_message = 'لطفاً ابتدا وارد شوید'
login_manager.login_message_category = 'warning'
login_manager.refresh_view = 'admin_login'
login_manager.needs_refresh_message = 'لطفاً مجدداً وارد شوید'
login_manager.needs_refresh_message_category = 'info'

class User(UserMixin):
    def __init__(self, id, username, password, email=None, role='user', 
                 permissions=None, created_at=None, last_login=None,
                 api_key=None, api_secret=None, quota=10000, storage=10*1024**3):
        self.id = str(id)
        self.username = username
        self.password = password
        self.email = email
        self.role = role
        self.permissions = permissions or {}
        self.created_at = created_at or datetime.now()
        self.last_login = last_login
        self.api_key = api_key or self.generate_api_key()
        self.api_secret = api_secret or self.generate_api_secret()
        self.quota = quota
        self.storage = storage
        self.is_active = True
        self.is_verified = False
        
    def generate_api_key(self):
        return secrets.token_urlsafe(32)
    
    def generate_api_secret(self):
        return secrets.token_urlsafe(64)
    
    def get_id(self):
        return self.id
    
    def has_permission(self, permission):
        return self.permissions.get(permission, False) or self.role == 'admin'
    
    @property
    def is_admin(self):
        return self.role == 'admin'
    
    @property
    def is_manager(self):
        return self.role in ['admin', 'manager']

# ================ دیتابیس درون‌حافظه‌ای ================
class MemoryDatabase:
    def __init__(self):
        self.users = {}
        self.knowledge = []
        self.conversations = deque(maxlen=10000)
        self.unanswered = deque(maxlen=5000)
        self.sessions = {}
        self.cache = {}
        self.stats = {
            'total_queries': 0,
            'answered_queries': 0,
            'unanswered_queries': 0,
            'total_knowledge': 0,
            'avg_response_time': 0,
            'peak_response_time': 0,
            'total_users': 0,
            'active_users': 0,
            'total_conversations': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'start_time': datetime.now()
        }
        self.categories = defaultdict(int)
        self.init_admin()
        
    def init_admin(self):
        admin_pass = hashlib.sha256('admin123'.encode()).hexdigest()
        admin = User(
            id=1,
            username='admin',
            password=admin_pass,
            email='admin@historybot.com',
            role='admin',
            permissions={'*': True}
        )
        self.users['1'] = admin
        
    def add_user(self, user):
        self.users[str(user.id)] = user
        self.stats['total_users'] = len(self.users)
        
    def get_user(self, user_id):
        return self.users.get(str(user_id))
    
    def get_user_by_username(self, username):
        for user in self.users.values():
            if user.username == username:
                return user
        return None
    
    def add_knowledge(self, item):
        item['id'] = len(self.knowledge) + 1
        item['created_at'] = datetime.now().isoformat()
        item['times_used'] = 0
        self.knowledge.append(item)
        self.stats['total_knowledge'] = len(self.knowledge)
        self.categories[item.get('category', 'عمومی')] += 1
        
    def add_conversation(self, conv):
        self.conversations.appendleft(conv)
        self.stats['total_conversations'] += 1
        
    def add_unanswered(self, item):
        self.unanswered.appendleft(item)

db = MemoryDatabase()

# ================ موتور جستجوی فوق‌پیشرفته ================
class UltraAdvancedSearchEngine:
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=20000,
            ngram_range=(1, 4),
            analyzer='char_wb',
            token_pattern=r'(?u)\b\w+\b',
            min_df=1,
            max_df=0.95,
            sublinear_tf=True,
            use_idf=True,
            smooth_idf=True,
            norm='l2'
        )
        
        self.count_vectorizer = CountVectorizer(
            max_features=10000,
            ngram_range=(1, 3),
            analyzer='word',
            token_pattern=r'(?u)\b\w+\b',
            min_df=1,
            max_df=0.95
        )
        
        self.lda_model = LatentDirichletAllocation(
            n_components=50,
            random_state=42,
            learning_method='online',
            max_iter=50
        )
        
        self.naive_bayes = MultinomialNB()
        self.kmeans = KMeans(n_clusters=20, random_state=42)
        
        self.embeddings = None
        self.tfidf_matrix = None
        self.count_matrix = None
        self.lda_features = None
        self.cluster_labels = None
        
        self.stopwords_persian = set(hazm_stopwords())
        self.stopwords_english = set(stopwords.words('english'))
        self.stopwords = self.stopwords_persian.union(self.stopwords_english)
        
        self.lemmatizer_en = WordNetLemmatizer()
        self.stemmer_en = PorterStemmer()
        
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
        self.word_frequencies = defaultdict(int)
        self.ngram_frequencies = defaultdict(int)
        
        self.load_data()
        
    def hazm_stopwords(self):
        """دریافت کلمات توقف فارسی"""
        stopwords_list = [
            'و', 'در', 'به', 'از', 'که', 'این', 'آن', 'با', 'برای', 'را',
            'تا', 'بر', 'هم', 'نیز', 'اما', 'یا', 'اگر', 'مگر', 'پس', 'بعد',
            'قبل', 'زیر', 'روی', 'داخل', 'خارج', 'بین', 'میان', 'طی', 'بدون',
            'غیر', 'جز', 'بجز', 'الا', 'لیکن', 'ولیکن', 'بلکه', 'چون', 'زیرا',
            'چرا', 'چگونه', 'کجا', 'کی', 'چه', 'چیست', 'چی', 'چند', 'چقدر',
            'کدام', 'همه', 'تمام', 'بعضی', 'برخی', 'هیچ', 'هر', 'هردو',
            'کس', 'چیز', 'جا', 'هنگام', 'وقتی', 'زمانی', 'گاهی', 'بعضاً'
        ]
        return stopwords_list
    
    def preprocess_persian(self, text):
        """پیش‌پردازش متن فارسی"""
        text = hazm_normalizer.normalize(text)
        text = re.sub(r'[^\w\sآ-ی]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def preprocess_english(self, text):
        """پیش‌پردازش متن انگلیسی"""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def tokenize_persian(self, text):
        """توکن‌سازی متن فارسی"""
        return hazm_tokenizer.tokenize(text)
    
    def tokenize_english(self, text):
        """توکن‌سازی متن انگلیسی"""
        return word_tokenize(text)
    
    def lemmatize_persian(self, word):
        """لماتایز کردن کلمه فارسی"""
        return hazm_lemmatizer.lemmatize(word)
    
    def lemmatize_english(self, word):
        """لماتایز کردن کلمه انگلیسی"""
        return self.lemmatizer_en.lemmatize(word)
    
    def extract_keywords_tfidf(self, text, top_n=10):
        """استخراج کلمات کلیدی با TF-IDF"""
        if not self.tfidf_matrix is not None and len(self.tfidf_vectorizer.get_feature_names_out()) > 0:
            text_vector = self.tfidf_vectorizer.transform([text])
            feature_names = self.tfidf_vectorizer.get_feature_names_out()
            scores = text_vector.toarray()[0]
            
            top_indices = scores.argsort()[-top_n:][::-1]
            keywords = [(feature_names[i], scores[i]) for i in top_indices if scores[i] > 0]
            return keywords
        return []
    
    def extract_keywords_rake(self, text, top_n=10):
        """استخراج کلمات کلیدی با الگوریتم RAKE"""
        sentences = hazm_sent_tokenizer.tokenize(text) if 'آ' in text else sent_tokenize(text)
        
        word_scores = defaultdict(float)
        word_counts = defaultdict(int)
        
        for sentence in sentences:
            words = self.tokenize_persian(sentence) if 'آ' in sentence else self.tokenize_english(sentence)
            words = [w for w in words if w not in self.stopwords and len(w) > 2]
            
            for i, word in enumerate(words):
                word_counts[word] += 1
                word_scores[word] += len(words) - i
                
        keywords = []
        for word in word_counts:
            score = word_scores[word] / word_counts[word]
            keywords.append((word, score))
            
        return sorted(keywords, key=lambda x: x[1], reverse=True)[:top_n]
    
    def extract_entities(self, text):
        """استخراج موجودیت‌های اسمی"""
        entities = {
            'persons': [],
            'places': [],
            'dates': [],
            'organizations': [],
            'events': [],
            'concepts': []
        }
        
        # تشخیص با NLTK
        try:
            tokens = word_tokenize(text)
            tagged = pos_tag(tokens)
            named_entities = ne_chunk(tagged)
            
            for chunk in named_entities:
                if hasattr(chunk, 'label'):
                    if chunk.label() == 'PERSON':
                        entities['persons'].append(' '.join(c[0] for c in chunk))
                    elif chunk.label() in ['GPE', 'LOCATION']:
                        entities['places'].append(' '.join(c[0] for c in chunk))
                    elif chunk.label() == 'ORGANIZATION':
                        entities['organizations'].append(' '.join(c[0] for c in chunk))
                    elif chunk.label() == 'DATE':
                        entities['dates'].append(' '.join(c[0] for c in chunk))
        except:
            pass
        
        # تشخیص با Hazm برای فارسی
        try:
            tokens = hazm_tokenizer.tokenize(text)
            tagged = hazm_tagger.tag(tokens)
            chunks = hazm_chunker.parse(tagged)
            
            for chunk in chunks:
                if isinstance(chunk, tuple) and len(chunk) > 1:
                    entity_type = chunk[1]
                    entity_text = chunk[0]
                    if 'PERSON' in entity_type:
                        entities['persons'].append(entity_text)
                    elif 'LOC' in entity_type:
                        entities['places'].append(entity_text)
                    elif 'DATE' in entity_type:
                        entities['dates'].append(entity_text)
        except:
            pass
        
        # تشخیص با الگوهای تاریخ
        date_patterns = [
            r'(\d{4}) میلادی',
            r'(\d{1,4}) هجری',
            r'(\d{1,4}) شمسی',
            r'قرن (اول|دوم|سوم|چهارم|پنجم|ششم|هفتم|هشتم|نهم|دهم|یازدهم|دوازدهم|سیزدهم|چهاردهم|پانزدهم)'
        ]
        
        for pattern in date_patterns:
            matches = re.findall(pattern, text)
            entities['dates'].extend(matches)
        
        # حذف تکراری‌ها
        for key in entities:
            entities[key] = list(set(entities[key]))
            
        return entities
    
    def analyze_sentiment(self, text):
        """تحلیل احساسات متن"""
        try:
            return self.sentiment_analyzer.polarity_scores(text)
        except:
            return {'compound': 0, 'pos': 0, 'neg': 0, 'neu': 1}
    
    def get_similarity_score(self, text1, text2):
        """محاسبه شباهت دو متن"""
        try:
            vectors = self.tfidf_vectorizer.transform([text1, text2])
            similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
            return float(similarity)
        except:
            return 0
    
    def generate_summary(self, text, max_sentences=5):
        """تولید خلاصه متن"""
        sentences = hazm_sent_tokenizer.tokenize(text) if 'آ' in text else sent_tokenize(text)
        
        if len(sentences) <= max_sentences:
            return text
            
        # محاسبه امتیاز جملات
        sentence_scores = {}
        for i, sentence in enumerate(sentences):
            words = self.tokenize_persian(sentence) if 'آ' in sentence else self.tokenize_english(sentence)
            words = [w for w in words if w not in self.stopwords]
            
            score = 0
            for word in words:
                score += self.word_frequencies.get(word, 1)
            
            sentence_scores[i] = score / len(words) if words else 0
        
        # انتخاب بهترین جملات
        top_indices = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:max_sentences]
        top_indices.sort()
        
        summary = ' '.join([sentences[i] for i in top_indices])
        return summary
    
    def load_data(self):
        """بارگذاری داده‌ها از دیتابیس"""
        if db.knowledge:
            texts = [f"{item['question']} {item['answer']}" for item in db.knowledge]
            
            try:
                self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
                self.count_matrix = self.count_vectorizer.fit_transform(texts)
                
                try:
                    self.lda_features = self.lda_model.fit_transform(self.count_matrix)
                    self.kmeans.fit(self.tfidf_matrix)
                    self.cluster_labels = self.kmeans.labels_
                except:
                    pass
                
                # محاسبه فراوانی کلمات
                for text in texts:
                    words = self.tokenize_persian(text) + self.tokenize_english(text)
                    for word in words:
                        self.word_frequencies[word] += 1
                        
            except Exception as e:
                logger.error(f"Error loading data: {e}")
    
    def search(self, query, top_k=10):
        """جستجوی پیشرفته با ترکیب الگوریتم‌ها"""
        if not db.knowledge or self.tfidf_matrix is None:
            return []
            
        start_time = time.time()
        
        # پیش‌پردازش query
        query_persian = self.preprocess_persian(query)
        query_english = self.preprocess_english(query)
        
        try:
            # جستجوی TF-IDF
            query_vector = self.tfidf_vectorizer.transform([query_persian + ' ' + query_english])
            tfidf_scores = cosine_similarity(query_vector, self.tfidf_matrix)[0]
            
            # جستجوی LDA
            lda_scores = np.zeros(len(db.knowledge))
            if self.count_matrix is not None:
                query_count = self.count_vectorizer.transform([query_persian + ' ' + query_english])
                try:
                    query_lda = self.lda_model.transform(query_count)
                    lda_scores = cosine_similarity(query_lda, self.lda_features)[0]
                except:
                    pass
            
            # استخراج کلمات کلیدی query
            query_keywords = set(self.extract_keywords_rake(query, 20))
            query_entities = self.extract_entities(query)
            
            results = []
            for i, item in enumerate(db.knowledge):
                # محاسبه امتیازات مختلف
                tfidf_score = tfidf_scores[i]
                lda_score = lda_scores[i]
                
                # امتیاز کلمات کلیدی
                item_text = item['question'] + ' ' + item['answer']
                item_keywords = set(self.extract_keywords_rake(item_text, 20))
                
                keyword_score = 0
                if query_keywords and item_keywords:
                    common = query_keywords.intersection(item_keywords)
                    keyword_score = len(common) / max(len(query_keywords | item_keywords), 1)
                
                # امتیاز موجودیت‌ها
                item_entities = self.extract_entities(item_text)
                entity_score = 0
                total_entities = 0
                
                for entity_type in query_entities:
                    query_entity_set = set(query_entities[entity_type])
                    item_entity_set = set(item_entities.get(entity_type, []))
                    if query_entity_set:
                        entity_score += len(query_entity_set & item_entity_set)
                        total_entities += len(query_entity_set)
                
                entity_score = entity_score / max(total_entities, 1)
                
                # امتیاز نهایی
                final_score = (
                    tfidf_score * 0.4 +
                    lda_score * 0.2 +
                    keyword_score * 0.25 +
                    entity_score * 0.15
                )
                
                if final_score > 0.1:
                    results.append({
                        'id': item['id'],
                        'question': item['question'],
                        'answer': item['answer'],
                        'category': item.get('category', 'عمومی'),
                        'score': float(final_score),
                        'tfidf_score': float(tfidf_score),
                        'keyword_score': float(keyword_score),
                        'entity_score': float(entity_score)
                    })
            
            # مرتب‌سازی نتایج
            results.sort(key=lambda x: x['score'], reverse=True)
            
            # به‌روزرسانی آمار
            response_time = time.time() - start_time
            db.stats['total_queries'] += 1
            db.stats['avg_response_time'] = (
                (db.stats['avg_response_time'] * (db.stats['total_queries'] - 1) + response_time) 
                / db.stats['total_queries']
            )
            
            if response_time > db.stats['peak_response_time']:
                db.stats['peak_response_time'] = response_time
            
            if results:
                db.stats['answered_queries'] += 1
            else:
                db.stats['unanswered_queries'] += 1
            
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []

# ================ نمونه موتور جستجو ================
search_engine = UltraAdvancedSearchEngine()

# ================ صفحات HTML ================
CHAT_TEMPLATE = '''
<!DOCTYPE html>
<html lang="fa" dir="rtl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>تاریخ‌دان هوشمند فوق‌پیشرفته</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            -webkit-tap-highlight-color: transparent;
        }
        
        body {
            font-family: 'Vazir', 'Tahoma', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            height: 100vh;
            overflow: hidden;
            position: fixed;
            width: 100%;
        }
        
        .chat-container {
            width: 100%;
            height: 100%;
            display: flex;
            flex-direction: column;
            background: white;
        }
        
        .chat-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        .header-title {
            display: flex;
            align-items: center;
            gap: 10px;
            font-size: 1.2em;
        }
        
        .header-stats {
            display: flex;
            gap: 15px;
            background: rgba(255,255,255,0.2);
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.9em;
        }
        
        .admin-link {
            color: white;
            text-decoration: none;
            padding: 5px 15px;
            border-radius: 20px;
            background: rgba(255,255,255,0.2);
            transition: all 0.3s;
        }
        
        .admin-link:hover {
            background: rgba(255,255,255,0.3);
            transform: translateY(-2px);
        }
        
        .chat-messages {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
            background: #f8f9fa;
        }
        
        .message {
            display: flex;
            margin-bottom: 20px;
            animation: slideIn 0.3s ease;
        }
        
        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        .message.user {
            justify-content: flex-end;
        }
        
        .message.bot {
            justify-content: flex-start;
        }
        
        .message-content {
            max-width: 70%;
            padding: 12px 18px;
            border-radius: 20px;
            position: relative;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
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
            text-align: left;
        }
        
        .confidence-badge {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 0.7em;
            background: rgba(255,255,255,0.2);
            margin-right: 8px;
        }
        
        .chat-input-container {
            padding: 15px 20px;
            background: white;
            border-top: 1px solid #e0e0e0;
            display: flex;
            gap: 10px;
        }
        
        .chat-input {
            flex: 1;
            padding: 12px 20px;
            border: 2px solid #e0e0e0;
            border-radius: 25px;
            font-size: 1em;
            outline: none;
            transition: all 0.3s;
        }
        
        .chat-input:focus {
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102,126,234,0.1);
        }
        
        .send-btn {
            width: 50px;
            height: 50px;
            border-radius: 50%;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            cursor: pointer;
            font-size: 1.2em;
            transition: all 0.3s;
        }
        
        .send-btn:hover {
            transform: scale(1.1);
            box-shadow: 0 5px 15px rgba(102,126,234,0.4);
        }
        
        .typing-indicator {
            padding: 12px 20px;
            background: white;
            border-radius: 20px;
            display: inline-block;
        }
        
        .typing-indicator span {
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #667eea;
            margin: 0 2px;
            animation: typing 1.4s infinite;
        }
        
        .typing-indicator span:nth-child(2) { animation-delay: 0.2s; }
        .typing-indicator span:nth-child(3) { animation-delay: 0.4s; }
        
        @keyframes typing {
            0%, 60%, 100% { transform: translateY(0); }
            30% { transform: translateY(-10px); }
        }
        
        @media (max-width: 768px) {
            .header-stats {
                display: none;
            }
        }
    </style>
</head>
<body>
    <div class="chat-container">
        <div class="chat-header">
            <div class="header-title">
                <span>🤖</span>
                <span>تاریخ‌دان هوشمند</span>
            </div>
            <div class="header-stats">
                <span>📚 {{ stats.total_knowledge }} دانش</span>
                <span>⚡ {{ "%.2f"|format(stats.avg_response_time) }} ثانیه</span>
            </div>
            <a href="/admin-login" class="admin-link">⚙️ مدیریت</a>
        </div>
        
        <div class="chat-messages" id="chat-messages">
            <div class="message bot">
                <div class="message-content">
                    سلام! من تاریخ‌دان هوشمند هستم. 
                    هر سوال تاریخی داری بپرس!
                    <div class="message-time">{{ now.strftime('%H:%M') }}</div>
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
    
    <script>
        const messagesContainer = document.getElementById('chat-messages');
        const messageInput = document.getElementById('message-input');
        
        function addMessage(text, isUser = false, confidence = null) {
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${isUser ? 'user' : 'bot'}`;
            
            const time = new Date().toLocaleTimeString('fa-IR');
            
            let confidenceHtml = '';
            if (confidence && !isUser) {
                confidenceHtml = `<span class="confidence-badge">${Math.round(confidence * 100)}%</span>`;
            }
            
            messageDiv.innerHTML = `
                <div class="message-content">
                    ${text} ${confidenceHtml}
                    <div class="message-time">${time}</div>
                </div>
            `;
            
            messagesContainer.appendChild(messageDiv);
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
        }
        
        function showTyping() {
            if (document.getElementById('typing-indicator')) return;
            
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
                    addMessage(data.answer, false, data.confidence);
                } else {
                    addMessage('❌ متأسفم! پاسخی پیدا نکردم. این سوال برای مدیر ارسال شد.', false);
                }
                
            } catch (error) {
                hideTyping();
                addMessage('❌ خطا در ارتباط با سرور', false);
            }
        }
        
        // جلوگیری از زوم با دو انگشت
        document.addEventListener('gesturestart', function(e) {
            e.preventDefault();
        });
        
        document.addEventListener('touchmove', function(e) {
            if (e.touches.length > 1) {
                e.preventDefault();
            }
        }, { passive: false });
    </script>
</body>
</html>
'''

LOGIN_TEMPLATE = '''
<!DOCTYPE html>
<html lang="fa" dir="rtl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ورود به پنل مدیریت</title>
    <style>
        body {
            font-family: 'Vazir', 'Tahoma', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 0;
        }
        
        .login-container {
            background: white;
            padding: 40px;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            width: 400px;
            max-width: 90%;
        }
        
        h2 {
            text-align: center;
            color: #333;
            margin-bottom: 30px;
            font-size: 1.8em;
        }
        
        .form-group {
            margin-bottom: 20px;
        }
        
        input {
            width: 100%;
            padding: 12px 15px;
            border: 2px solid #e0e0e0;
            border-radius: 10px;
            font-size: 1em;
            font-family: inherit;
            transition: all 0.3s;
        }
        
        input:focus {
            border-color: #667eea;
            outline: none;
            box-shadow: 0 0 0 3px rgba(102,126,234,0.1);
        }
        
        button {
            width: 100%;
            padding: 12px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 10px;
            font-size: 1.1em;
            cursor: pointer;
            transition: all 0.3s;
        }
        
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 20px rgba(102,126,234,0.4);
        }
        
        .error-message {
            background: #fee;
            color: #c00;
            padding: 10px;
            border-radius: 8px;
            margin-bottom: 20px;
            text-align: center;
        }
    </style>
</head>
<body>
    <div class="login-container">
        <h2>🔐 ورود به پنل مدیریت</h2>
        
        {% if error %}
        <div class="error-message">{{ error }}</div>
        {% endif %}
        
        <form method="POST">
            <div class="form-group">
                <input type="text" name="username" placeholder="نام کاربری" value="admin" required>
            </div>
            <div class="form-group">
                <input type="password" name="password" placeholder="رمز عبور" value="admin123" required>
            </div>
            <button type="submit">ورود</button>
        </form>
    </div>
</body>
</html>
'''

ADMIN_TEMPLATE = '''
<!DOCTYPE html>
<html lang="fa" dir="rtl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>پنل مدیریت - تاریخ‌دان هوشمند</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Vazir', 'Tahoma', sans-serif;
            background: #f5f5f5;
            padding: 20px;
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px 30px;
            border-radius: 15px;
            margin-bottom: 30px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            box-shadow: 0 5px 20px rgba(0,0,0,0.1);
        }
        
        .header h1 {
            font-size: 1.5em;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .nav-links {
            display: flex;
            gap: 15px;
        }
        
        .nav-links a {
            color: white;
            text-decoration: none;
            padding: 8px 20px;
            border-radius: 25px;
            background: rgba(255,255,255,0.2);
            transition: all 0.3s;
        }
        
        .nav-links a:hover {
            background: rgba(255,255,255,0.3);
            transform: translateY(-2px);
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .stat-card {
            background: white;
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.05);
            text-align: center;
            transition: all 0.3s;
        }
        
        .stat-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }
        
        .stat-number {
            font-size: 2.5em;
            font-weight: bold;
            color: #667eea;
            margin-bottom: 5px;
        }
        
        .stat-label {
            color: #666;
            font-size: 0.9em;
        }
        
        .grid-2 {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .card {
            background: white;
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.05);
        }
        
        .card h2 {
            margin-bottom: 20px;
            color: #333;
            font-size: 1.3em;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .form-group {
            margin-bottom: 15px;
        }
        
        input, textarea, select {
            width: 100%;
            padding: 12px 15px;
            border: 2px solid #e0e0e0;
            border-radius: 10px;
            font-size: 1em;
            font-family: inherit;
            transition: all 0.3s;
        }
        
        input:focus, textarea:focus, select:focus {
            border-color: #667eea;
            outline: none;
            box-shadow: 0 0 0 3px rgba(102,126,234,0.1);
        }
        
        textarea {
            min-height: 100px;
            resize: vertical;
        }
        
        button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 12px 25px;
            border: none;
            border-radius: 10px;
            font-size: 1em;
            cursor: pointer;
            transition: all 0.3s;
        }
        
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 20px rgba(102,126,234,0.4);
        }
        
        .file-upload-area {
            border: 2px dashed #667eea;
            padding: 30px;
            text-align: center;
            border-radius: 15px;
            cursor: pointer;
            margin: 15px 0;
            transition: all 0.3s;
        }
        
        .file-upload-area:hover {
            background: #f8f9ff;
            border-color: #764ba2;
        }
        
        .conversation-stream {
            height: 500px;
            overflow-y: auto;
            border: 1px solid #e0e0e0;
            border-radius: 10px;
            padding: 15px;
            background: #fafafa;
        }
        
        .conversation-item {
            background: white;
            padding: 15px;
            margin: 10px 0;
            border-radius: 10px;
            border-right: 3px solid #667eea;
            animation: slideIn 0.3s;
        }
        
        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateY(-10px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        .conversation-question {
            font-weight: bold;
            color: #333;
            margin-bottom: 8px;
        }
        
        .conversation-answer {
            color: #666;
            padding-right: 15px;
            margin-bottom: 8px;
            line-height: 1.6;
        }
        
        .conversation-meta {
            display: flex;
            gap: 15px;
            font-size: 0.85em;
            color: #999;
            align-items: center;
        }
        
        .badge {
            display: inline-block;
            padding: 3px 10px;
            border-radius: 20px;
            font-size: 0.85em;
        }
        
        .badge.success {
            background: #d4edda;
            color: #155724;
        }
        
        .badge.danger {
            background: #f8d7da;
            color: #721c24;
        }
        
        .badge.warning {
            background: #fff3cd;
            color: #856404;
        }
        
        .progress-bar {
            width: 100%;
            height: 10px;
            background: #f0f0f0;
            border-radius: 5px;
            overflow: hidden;
            margin: 10px 0;
        }
        
        .progress-fill {
            height: 100%;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            transition: width 0.3s;
        }
        
        @media (max-width: 768px) {
            .grid-2 {
                grid-template-columns: 1fr;
            }
            
            .stats-grid {
                grid-template-columns: 1fr 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>
            <span>⚙️</span>
            <span>پنل مدیریت تاریخ‌دان هوشمند</span>
        </h1>
        <div class="nav-links">
            <a href="/" target="_blank">🌐 صفحه چت</a>
            <a href="/logout">🚪 خروج</a>
        </div>
    </div>
    
    <div class="stats-grid">
        <div class="stat-card">
            <div class="stat-number">{{ stats.total_knowledge }}</div>
            <div class="stat-label">کل دانش</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">{{ stats.total_queries }}</div>
            <div class="stat-label">کل سوالات</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">{{ stats.answered_queries }}</div>
            <div class="stat-label">پاسخ داده شده</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">{{ stats.unanswered_queries }}</div>
            <div class="stat-label">بی‌پاسخ</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">{{ "%.2f"|format(stats.avg_response_time) }}</div>
            <div class="stat-label">میانگین زمان پاسخ</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">{{ stats.total_users }}</div>
            <div class="stat-label">کاربران</div>
        </div>
    </div>
    
    <div class="grid-2">
        <div class="card">
            <h2>
                <span>📝</span>
                <span>افزودن دانش تکی</span>
            </h2>
            <form action="/admin/add" method="POST">
                <div class="form-group">
                    <input type="text" name="question" placeholder="سوال" required>
                </div>
                <div class="form-group">
                    <textarea name="answer" placeholder="پاسخ" required></textarea>
                </div>
                <div class="form-group">
                    <select name="category">
                        <option value="ایران باستان">ایران باستان</option>
                        <option value="اسلامی">اسلامی</option>
                        <option value="معاصر">معاصر</option>
                        <option value="جهان">جهان</option>
                        <option value="علمی">علمی</option>
                    </select>
                </div>
                <button type="submit">➕ افزودن دانش</button>
            </form>
        </div>
        
        <div class="card">
            <h2>
                <span>📁</span>
                <span>آپلود فایل آموزشی</span>
            </h2>
            <form id="uploadForm" enctype="multipart/form-data">
                <div class="file-upload-area" onclick="document.getElementById('file').click()">
                    <p>📤 برای آپلود کلیک کنید</p>
                    <p style="font-size:0.9em; color:#666;">فرمت‌های مجاز: txt, pdf, docx, csv</p>
                </div>
                <input type="file" id="file" name="file" style="display:none;" accept=".txt,.pdf,.docx,.csv">
                <div id="progress" style="display:none; margin:15px 0;">
                    <div class="progress-bar">
                        <div class="progress-fill" id="progress-fill" style="width:0%"></div>
                    </div>
                    <p id="progress-text" style="text-align:center; margin-top:5px;">0%</p>
                </div>
                <button type="submit">📥 آپلود</button>
            </form>
        </div>
    </div>
    
    <div class="card">
        <h2>
            <span>💬</span>
            <span>جریان زنده مکالمات</span>
        </h2>
        <div class="conversation-stream" id="conversation-stream">
            {% for conv in conversations %}
            <div class="conversation-item">
                <div class="conversation-question">❓ {{ conv.question }}</div>
                {% if conv.answer %}
                <div class="conversation-answer">✅ {{ conv.answer }}</div>
                {% else %}
                <div class="conversation-answer" style="color:#999;">⏳ بدون پاسخ</div>
                {% endif %}
                <div class="conversation-meta">
                    <span>{{ conv.timestamp }}</span>
                    <span class="badge {{ 'success' if conv.found else 'danger' }}">
                        {{ 'پاسخ داده شده' if conv.found else 'بی‌پاسخ' }}
                    </span>
                    {% if conv.confidence %}
                    <span class="badge warning">اعتماد: {{ "%.0f"|format(conv.confidence*100) }}%</span>
                    {% endif %}
                </div>
            </div>
            {% endfor %}
        </div>
    </div>
    
    <script>
        // آپلود فایل با قابلیت تکه‌تکه
        document.getElementById('uploadForm').onsubmit = async function(e) {
            e.preventDefault();
            
            const file = document.getElementById('file').files[0];
            if (!file) {
                alert('لطفاً یک فایل انتخاب کنید');
                return;
            }
            
            const chunkSize = 5 * 1024 * 1024; // 5MB per chunk
            const chunks = Math.ceil(file.size / chunkSize);
            
            document.getElementById('progress').style.display = 'block';
            
            for (let i = 0; i < chunks; i++) {
                const start = i * chunkSize;
                const end = Math.min(start + chunkSize, file.size);
                const chunk = file.slice(start, end);
                
                const formData = new FormData();
                formData.append('file', chunk, file.name);
                formData.append('chunk', i);
                formData.append('chunks', chunks);
                
                try {
                    const response = await fetch('/admin/upload', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const result = await response.json();
                    
                    const percent = Math.round(((i + 1) / chunks) * 100);
                    document.getElementById('progress-fill').style.width = percent + '%';
                    document.getElementById('progress-text').innerText = percent + '%';
                    
                } catch (error) {
                    console.error('Upload error:', error);
                    alert('خطا در آپلود فایل');
                    break;
                }
            }
            
            alert('آپلود با موفقیت کامل شد!');
            document.getElementById('progress').style.display = 'none';
            document.getElementById('file').value = '';
        };
        
        // EventSource برای دریافت زنده مکالمات
        const eventSource = new EventSource('/admin/conversations/stream');
        const stream = document.getElementById('conversation-stream');
        
        eventSource.onmessage = function(event) {
            const conv = JSON.parse(event.data);
            
            const item = document.createElement('div');
            item.className = 'conversation-item';
            item.innerHTML = `
                <div class="conversation-question">❓ ${conv.question}</div>
                ${conv.answer ? `<div class="conversation-answer">✅ ${conv.answer}</div>` : 
                               '<div class="conversation-answer" style="color:#999;">⏳ بدون پاسخ</div>'}
                <div class="conversation-meta">
                    <span>${conv.timestamp}</span>
                    <span class="badge ${conv.found ? 'success' : 'danger'}">
                        ${conv.found ? 'پاسخ داده شده' : 'بی‌پاسخ'}
                    </span>
                    ${conv.confidence ? `<span class="badge warning">اعتماد: ${Math.round(conv.confidence * 100)}%</span>` : ''}
                </div>
            `;
            
            stream.insertBefore(item, stream.firstChild);
            
            if (stream.children.length > 100) {
                stream.removeChild(stream.lastChild);
            }
        };
        
        eventSource.onerror = function() {
            console.log('EventSource connection closed');
        };
    </script>
</body>
</html>
'''

# ================ مسیرهای API ================
@app.route('/api/chat', methods=['POST'])
def api_chat():
    """API چت"""
    data = request.json
    question = data.get('message', '').strip()
    user_id = session.get('user_id', 'anonymous')
    
    if not question:
        return jsonify({'error': 'سوال نمی‌تواند خالی باشد'})
    
    # جستجو
    results = search_engine.search(question)
    
    # ثبت مکالمه
    conversation = {
        'id': str(uuid.uuid4()),
        'user_id': user_id,
        'question': question,
        'answer': results[0]['answer'] if results else None,
        'confidence': results[0]['score'] if results else 0,
        'found': bool(results),
        'timestamp': datetime.now().isoformat()
    }
    
    db.add_conversation(conversation)
    
    if not results:
        db.add_unanswered({
            'question': question,
            'user_id': user_id,
            'timestamp': datetime.now().isoformat()
        })
    
    return jsonify(conversation)

@app.route('/api/stats')
def api_stats():
    """دریافت آمار"""
    return jsonify(db.stats)

@app.route('/api/conversations/latest')
def api_conversations_latest():
    """آخرین مکالمات"""
    return jsonify(list(db.conversations)[:20])

# ================ مسیرهای مدیریت ================
@app.route('/admin-login', methods=['GET', 'POST'])
def admin_login():
    """صفحه لاگین"""
    if request.method == 'POST':
        username = request.form['username']
        password = hashlib.sha256(request.form['password'].encode()).hexdigest()
        
        user = db.get_user_by_username(username)
        if user and user.password == password:
            login_user(user, remember=True)
            session.permanent = True
            user.last_login = datetime.now()
            return redirect(url_for('admin_dashboard'))
        
        return render_template_string(LOGIN_TEMPLATE, error='نام کاربری یا رمز عبور اشتباه است')
    
    return render_template_string(LOGIN_TEMPLATE)

@app.route('/admin')
@login_required
def admin_dashboard():
    """پنل مدیریت"""
    return render_template_string(
        ADMIN_TEMPLATE,
        stats=db.stats,
        conversations=list(db.conversations)[:50],
        user=current_user
    )

@app.route('/admin/add', methods=['POST'])
@login_required
def admin_add():
    """افزودن دانش تکی"""
    question = request.form['question']
    answer = request.form['answer']
    category = request.form.get('category', 'عمومی')
    
    item = {
        'question': question,
        'answer': answer,
        'category': category
    }
    
    db.add_knowledge(item)
    search_engine.load_data()  # به‌روزرسانی موتور جستجو
    
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/upload', methods=['POST'])
@login_required
def admin_upload():
    """آپلود فایل تکه‌تکه"""
    chunk = request.form.get('chunk', 0)
    chunks = request.form.get('chunks', 1)
    file = request.files['file']
    
    filename = secure_filename(file.filename)
    chunk_path = os.path.join('uploads', 'chunks', f"{filename}.part{chunk}")
    file.save(chunk_path)
    
    if int(chunk) == int(chunks) - 1:
        # ادغام تکه‌ها
        final_path = os.path.join('uploads', filename)
        with open(final_path, 'wb') as final_file:
            for i in range(int(chunks)):
                part_path = os.path.join('uploads', 'chunks', f"{filename}.part{i}")
                with open(part_path, 'rb') as part_file:
                    final_file.write(part_file.read())
                os.remove(part_path)
        
        # پردازش فایل در پس‌زمینه
        threading.Thread(target=process_uploaded_file, args=(final_path,)).start()
        
        return jsonify({'message': 'آپلود کامل شد', 'filename': filename})
    
    return jsonify({'message': f'تکه {int(chunk)+1} از {chunks} آپلود شد'})

@app.route('/admin/conversations/stream')
@login_required
def admin_conversations_stream():
    """Stream مکالمات"""
    def generate():
        last_index = 0
        conversations = list(db.conversations)
        
        while True:
            if len(conversations) > last_index:
                new_convs = conversations[last_index:]
                for conv in new_convs:
                    yield f"data: {json.dumps(conv, ensure_ascii=False)}\n\n"
                last_index = len(conversations)
            
            time.sleep(1)
            conversations = list(db.conversations)
    
    return Response(generate(), mimetype='text/event-stream')

def process_uploaded_file(filepath):
    """پردازش فایل آپلود شده"""
    try:
        filename = os.path.basename(filepath)
        ext = os.path.splitext(filename)[1].lower()
        
        if ext == '.txt':
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # استخراج جفت سوال-جواب
            lines = content.split('\n')
            for line in lines:
                if '|' in line:
                    parts = line.split('|', 1)
                    if len(parts) == 2:
                        question, answer = parts
                        db.add_knowledge({
                            'question': question.strip(),
                            'answer': answer.strip(),
                            'category': 'آپلود شده'
                        })
        
        # پاک کردن فایل موقت
        os.remove(filepath)
        
        # به‌روزرسانی موتور جستجو
        search_engine.load_data()
        
        logger.info(f"File processed successfully: {filename}")
        
    except Exception as e:
        logger.error(f"Error processing file {filepath}: {e}")

@app.route('/logout')
@login_required
def logout():
    """خروج از حساب"""
    logout_user()
    session.clear()
    return redirect(url_for('index'))

# ================ اجرای برنامه ================
if __name__ == '__main__':
    print("""
    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║     🤖 تاریخ‌دان هوشمند فوق‌پیشرفته - Ultra Advanced Version                ║
    ╠══════════════════════════════════════════════════════════════════════════════╣
    ║  📚 دانش فعلی: {} مورد                                                        ║
    ║  🌐 صفحه چت: http://localhost:5000                                           ║
    ║  🔐 پنل مدیریت: http://localhost:5000/admin-login                            ║
    ║  👤 کاربر پیش‌فرض: admin / admin123                                          ║
    ║  📊 آپلود فایل تا ۱۰۰ گیگابایت                                               ║
    ║  🎯 الگوریتم‌های جستجو: TF-IDF, LDA, K-Means, Naive Bayes, RAKE             ║
    ║  🌍 پشتیبانی از زبان: فارسی و انگلیسی                                        ║
    ║  💬 نمایش قطاری مکالمات در پنل مدیریت                                        ║
    ║  ⚡ آپلود تکه‌تکه برای فایل‌های بزرگ                                          ║
    ╚══════════════════════════════════════════════════════════════════════════════╝
    """.format(len(db.knowledge)))
    
    # ایجاد کاربر پیش‌فرض اگر نبود
    if not db.get_user_by_username('admin'):
        admin_pass = hashlib.sha256('admin123'.encode()).hexdigest()
        admin = User(
            id=len(db.users) + 1,
            username='admin',
            password=admin_pass,
            email='admin@historybot.com',
            role='admin',
            permissions={'*': True}
        )
        db.add_user(admin)
    
    # اجرای برنامه
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
