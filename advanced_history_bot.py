# ultra_persian_ai.py
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

# ================ کتابخانه‌های فوق پیشرفته ================
# برای پردازش زبان فارسی
import hazm  # کتابخانه تخصصی پردازش زبان فارسی
from hazm import Normalizer, WordTokenizer, SentenceTokenizer, Lemmatizer, Stemmer

# برای یادگیری عمیق و تشخیص الگو
import torch
import torch.nn as nn
import torch.nn.functional as F

# برای هوش مصنوعی و یادگیری ماشین
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, chi2

# برای پردازش متن و استخراج ویژگی
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.manifold import TSNE

# برای تشخیص حروف و کلمات
import arabic_reshaper
from bidi.algorithm import get_display
import persian
import pyarabic.araby as araby

# برای تحلیل عمیق متن
import textstat
from textblob import TextBlob
import langid

# برای جستجوی پیشرفته
from whoosh.index import create_in, open_dir
from whoosh.fields import Schema, TEXT, ID, STORED
from whoosh.qparser import QueryParser, MultifieldParser, FuzzyTermPlugin
from whoosh.query import FuzzyTerm, Wildcard

# برای کش و بهینه‌سازی
from functools import lru_cache
import hashlib
import pickle

# برای API و ارتباطات
import requests
from bs4 import BeautifulSoup

app = Flask(__name__)
app.config['SECRET_KEY'] = 'ultra-persian-ai-super-secret-key-2024'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200MB
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=365)
app.config['SESSION_REFRESH_EACH_REQUEST'] = True
app.config['SESSION_PERMANENT'] = True
CORS(app)

# ایجاد پوشه‌ها
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('data', exist_ok=True)
os.makedirs('backup', exist_ok=True)
os.makedirs('logs', exist_ok=True)
os.makedirs('index', exist_ok=True)
os.makedirs('models', exist_ok=True)

# ================ پردازشگر فارسی ================
class PersianTextProcessor:
    def __init__(self):
        self.normalizer = Normalizer()
        self.tokenizer = WordTokenizer()
        self.sentence_tokenizer = SentenceTokenizer()
        self.lemmatizer = Lemmatizer()
        self.stemmer = Stemmer()
        
        # الفبای کامل فارسی
        self.persian_alphabet = {
            'آ': {'name': 'الف با کلاه', 'type': 'vowel', 'connects': True, 'forms': ['آ', 'آ', 'آ', 'آ']},
            'ا': {'name': 'الف', 'type': 'vowel', 'connects': False, 'forms': ['ا', 'ا', 'ا', 'ا']},
            'ب': {'name': 'به', 'type': 'consonant', 'connects': True, 'forms': ['ب', 'بـ', 'ـبـ', 'ـب']},
            'پ': {'name': 'په', 'type': 'consonant', 'connects': True, 'forms': ['پ', 'پـ', 'ـپـ', 'ـپ']},
            'ت': {'name': 'ته', 'type': 'consonant', 'connects': True, 'forms': ['ت', 'تـ', 'ـتـ', 'ـت']},
            'ث': {'name': 'ثه', 'type': 'consonant', 'connects': True, 'forms': ['ث', 'ثـ', 'ـثـ', 'ـث']},
            'ج': {'name': 'جیم', 'type': 'consonant', 'connects': True, 'forms': ['ج', 'جـ', 'ـجـ', 'ـج']},
            'چ': {'name': 'چه', 'type': 'consonant', 'connects': True, 'forms': ['چ', 'چـ', 'ـچـ', 'ـچ']},
            'ح': {'name': 'حه', 'type': 'consonant', 'connects': True, 'forms': ['ح', 'حـ', 'ـحـ', 'ـح']},
            'خ': {'name': 'خه', 'type': 'consonant', 'connects': True, 'forms': ['خ', 'خـ', 'ـخـ', 'ـخ']},
            'د': {'name': 'دال', 'type': 'consonant', 'connects': False, 'forms': ['د', 'د', 'د', 'د']},
            'ذ': {'name': 'ذال', 'type': 'consonant', 'connects': False, 'forms': ['ذ', 'ذ', 'ذ', 'ذ']},
            'ر': {'name': 'ره', 'type': 'consonant', 'connects': False, 'forms': ['ر', 'ر', 'ر', 'ر']},
            'ز': {'name': 'زه', 'type': 'consonant', 'connects': False, 'forms': ['ز', 'ز', 'ز', 'ز']},
            'ژ': {'name': 'ژه', 'type': 'consonant', 'connects': False, 'forms': ['ژ', 'ژ', 'ژ', 'ژ']},
            'س': {'name': 'سین', 'type': 'consonant', 'connects': True, 'forms': ['س', 'سـ', 'ـسـ', 'ـس']},
            'ش': {'name': 'شین', 'type': 'consonant', 'connects': True, 'forms': ['ش', 'شـ', 'ـشـ', 'ـش']},
            'ص': {'name': 'صاد', 'type': 'consonant', 'connects': True, 'forms': ['ص', 'صـ', 'ـصـ', 'ـص']},
            'ض': {'name': 'ضاد', 'type': 'consonant', 'connects': True, 'forms': ['ض', 'ضـ', 'ـضـ', 'ـض']},
            'ط': {'name': 'طا', 'type': 'consonant', 'connects': True, 'forms': ['ط', 'طـ', 'ـطـ', 'ـط']},
            'ظ': {'name': 'ظا', 'type': 'consonant', 'connects': True, 'forms': ['ظ', 'ظـ', 'ـظـ', 'ـظ']},
            'ع': {'name': 'عین', 'type': 'consonant', 'connects': True, 'forms': ['ع', 'عـ', 'ـعـ', 'ـع']},
            'غ': {'name': 'غین', 'type': 'consonant', 'connects': True, 'forms': ['غ', 'غـ', 'ـغـ', 'ـغ']},
            'ف': {'name': 'فه', 'type': 'consonant', 'connects': True, 'forms': ['ف', 'فـ', 'ـفـ', 'ـف']},
            'ق': {'name': 'قاف', 'type': 'consonant', 'connects': True, 'forms': ['ق', 'قـ', 'ـقـ', 'ـق']},
            'ک': {'name': 'کاف', 'type': 'consonant', 'connects': True, 'forms': ['ک', 'کـ', 'ـکـ', 'ـک']},
            'گ': {'name': 'گاف', 'type': 'consonant', 'connects': True, 'forms': ['گ', 'گـ', 'ـگـ', 'ـگ']},
            'ل': {'name': 'لام', 'type': 'consonant', 'connects': True, 'forms': ['ل', 'لـ', 'ـلـ', 'ـل']},
            'م': {'name': 'میم', 'type': 'consonant', 'connects': True, 'forms': ['م', 'مـ', 'ـمـ', 'ـم']},
            'ن': {'name': 'نون', 'type': 'consonant', 'connects': True, 'forms': ['ن', 'نـ', 'ـنـ', 'ـن']},
            'و': {'name': 'واو', 'type': 'consonant', 'connects': False, 'forms': ['و', 'و', 'و', 'و']},
            'ه': {'name': 'هه', 'type': 'consonant', 'connects': True, 'forms': ['ه', 'هـ', 'ـهـ', 'ـه']},
            'ی': {'name': 'یه', 'type': 'consonant', 'connects': True, 'forms': ['ی', 'یـ', 'ـیـ', 'ـی']}
        }
        
        # حرکات و نشانه‌ها
        self.diacritics = {
            'َ': 'فتحه',
            'ِ': 'کسره',
            'ُ': 'ضمه',
            'ّ': 'تشدید',
            'ْ': 'سکون',
            'ً': 'تنوین نصب',
            'ٍ': 'تنوین جر',
            'ٌ': 'تنوین رفع',
            'آ': 'الف ممدوده',
            'ة': 'تاء تأنیث',
            'ء': 'همزه'
        }
        
        # کلمات پرسشی فارسی
        self.question_words = {
            'کیست': 'person',
            'کی بود': 'person',
            'چه کسی': 'person',
            'کجاست': 'place',
            'کجا': 'place',
            'چیست': 'definition',
            'چه بود': 'definition',
            'چی': 'definition',
            'کی': 'time',
            'چه زمانی': 'time',
            'چرا': 'reason',
            'چطور': 'method',
            'چگونه': 'method',
            'چند': 'quantity',
            'چه قدر': 'quantity',
            'کدام': 'choice',
            'آیا': 'yesno'
        }

    def normalize_persian(self, text):
        """نرمال‌سازی کامل متن فارسی"""
        return self.normalizer.normalize(text)
    
    def tokenize_words(self, text):
        """تجزیه به کلمات"""
        return self.tokenizer.tokenize(text)
    
    def tokenize_sentences(self, text):
        """تجزیه به جملات"""
        return self.sentence_tokenizer.tokenize(text)
    
    def get_word_features(self, word):
        """استخراج ویژگی‌های کلمه"""
        features = {
            'length': len(word),
            'has_diacritic': any(d in word for d in self.diacritics),
            'persian_letters': sum(1 for c in word if c in self.persian_alphabet),
            'arabic_letters': sum(1 for c in word if c in 'ثحذصضطظعغ'),
            'question_word': word in self.question_words,
            'normalized': self.normalize_persian(word)
        }
        return features
    
    def analyze_letters(self, word):
        """تحلیل حروف تشکیل‌دهنده کلمه"""
        letters = []
        for i, char in enumerate(word):
            if char in self.persian_alphabet:
                letter_info = self.persian_alphabet[char].copy()
                # تشخیص شکل حرف در کلمه
                if i == 0 and len(word) == 1:
                    form = 'isolated'
                elif i == 0:
                    form = 'beginning'
                elif i == len(word) - 1:
                    form = 'end'
                else:
                    form = 'middle'
                
                letters.append({
                    'char': char,
                    'name': letter_info['name'],
                    'type': letter_info['type'],
                    'form': form,
                    'connects_next': letter_info['connects'] if i < len(word)-1 else False
                })
            elif char in self.diacritics:
                letters.append({
                    'char': char,
                    'name': self.diacritics[char],
                    'type': 'diacritic',
                    'form': 'above' if char in 'َُِّْ' else 'below'
                })
        return letters

# ================ سیستم جستجوی Whoosh ================
class WhooshSearchEngine:
    def __init__(self, index_dir='index'):
        self.index_dir = index_dir
        self.schema = Schema(
            id=ID(stored=True, unique=True),
            question=TEXT(stored=True, analyzer='persian'),
            answer=TEXT(stored=True),
            category=STORED,
            keywords=TEXT(analyzer='persian')
        )
        
        if not os.path.exists(index_dir):
            os.makedirs(index_dir)
            self.ix = create_in(index_dir, self.schema)
        else:
            self.ix = open_dir(index_dir)
    
    def add_document(self, doc_id, question, answer, category='عمومی'):
        """افزودن سند به ایندکس"""
        writer = self.ix.writer()
        writer.update_document(
            id=str(doc_id),
            question=question,
            answer=answer,
            category=category,
            keywords=question + ' ' + category
        )
        writer.commit()
    
    def search(self, query, limit=5):
        """جستجوی پیشرفته با فازی"""
        results = []
        with self.ix.searcher() as searcher:
            # جستجوی چندفیلدی با پشتیبانی از فازی
            parser = MultifieldParser(["question", "keywords"], schema=self.schema)
            parser.add_plugin(FuzzyTermPlugin())
            
            # ایجاد query فازی
            fuzzy_query = FuzzyTerm("question", query, maxdist=2)
            parsed_query = parser.parse(query)
            
            # ترکیب نتایج
            fuzzy_results = searcher.search(fuzzy_query, limit=limit)
            parsed_results = searcher.search(parsed_query, limit=limit)
            
            # ترکیب و حذف تکراری
            seen = set()
            for hit in fuzzy_results:
                if hit['id'] not in seen:
                    results.append({
                        'id': hit['id'],
                        'question': hit['question'],
                        'answer': hit['answer'],
                        'score': hit.score,
                        'method': 'fuzzy'
                    })
                    seen.add(hit['id'])
            
            for hit in parsed_results:
                if hit['id'] not in seen:
                    results.append({
                        'id': hit['id'],
                        'question': hit['question'],
                        'answer': hit['answer'],
                        'score': hit.score * 0.8,
                        'method': 'parsed'
                    })
        
        return sorted(results, key=lambda x: x['score'], reverse=True)

# ================ مدل‌های یادگیری عمیق ================
class PersianTextClassifier(nn.Module):
    """شبکه عصبی برای طبقه‌بندی متون فارسی"""
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(PersianTextClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(hidden_dim * 2, 128)
        self.fc2 = nn.Linear(128, output_dim)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        hidden = self.dropout(hidden)
        hidden = self.relu(self.fc1(hidden))
        hidden = self.dropout(hidden)
        output = self.fc2(hidden)
        return output

# ================ دیتابیس دائمی فوق پیشرفته ================
class UltraPermanentDatabase:
    def __init__(self, filename='data/ultra_db.json'):
        self.filename = filename
        self.backup_dir = 'backup'
        self.data = {
            'knowledge_base': [],
            'users_questions': [],
            'conversations': [],
            'alphabet_lessons': [],
            'patterns': {},
            'word_embeddings': {},
            'stats': {
                'total_questions': 0,
                'answered': 0,
                'unanswered': 0,
                'total_conversations': 0,
                'last_backup': None,
                'created_at': datetime.now().isoformat()
            },
            'models': {},
            'training_data': []
        }
        self.load()
        self.auto_backup()
    
    def load(self):
        """بارگذاری دائمی داده‌ها"""
        if os.path.exists(self.filename):
            try:
                with open(self.filename, 'r', encoding='utf-8') as f:
                    loaded = json.load(f)
                    self.data.update(loaded)
                print(f"💾 {len(self.data['knowledge_base'])} دانش بارگذاری شد")
                print(f"📊 {self.data['stats']['total_questions']} سوال ثبت شده")
                print(f"💬 {self.data['stats']['total_conversations']} مکالمه ذخیره شده")
            except Exception as e:
                print(f"⚠️ خطا در بارگذاری: {e}")
                self.save()
        else:
            self.save()
    
    def save(self):
        """ذخیره دائمی داده‌ها"""
        if len(self.data['knowledge_base']) % 50 == 0:
            self.create_backup()
        
        with open(self.filename, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
    
    def create_backup(self):
        """ایجاد بکاپ خودکار"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_file = f"{self.backup_dir}/backup_{timestamp}.json"
        with open(backup_file, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
        self.data['stats']['last_backup'] = timestamp
        print(f"📦 بکاپ ایجاد شد: {backup_file}")
    
    def auto_backup(self):
        """بکاپ خودکار هر 12 ساعت"""
        last = self.data['stats'].get('last_backup')
        if last:
            try:
                last_time = datetime.strptime(last, '%Y%m%d_%H%M%S')
                if (datetime.now() - last_time).seconds > 43200:  # 12 ساعت
                    self.create_backup()
            except:
                self.create_backup()
        else:
            self.create_backup()

# ================ هسته اصلی هوش مصنوعی ================
class UltraPersianAI:
    def __init__(self):
        self.db = UltraPermanentDatabase()
        self.processor = PersianTextProcessor()
        self.search_engine = WhooshSearchEngine()
        
        # داده‌ها
        self.knowledge_base = self.db.data['knowledge_base']
        self.users_questions = self.db.data['users_questions']
        self.conversations = self.db.data['conversations']
        self.stats = self.db.data['stats']
        
        # سیستم‌های تشخیص
        self.vectorizer_tfidf = TfidfVectorizer(
            max_features=20000,
            ngram_range=(1, 6),
            analyzer='char_wb',
            sublinear_tf=True,
            use_idf=True,
            smooth_idf=True
        )
        
        self.vectorizer_count = CountVectorizer(
            max_features=10000,
            ngram_range=(1, 4),
            analyzer='word'
        )
        
        # مدل‌های یادگیری ماشین
        self.classifiers = {
            'nb': MultinomialNB(),
            'lr': LogisticRegression(max_iter=1000),
            'rf': RandomForestClassifier(n_estimators=200, max_depth=20),
            'gb': GradientBoostingClassifier(n_estimators=100),
            'svm': SVC(kernel='linear', probability=True)
        }
        
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.question_vectors = None
        self.word_embeddings = {}
        
        # راه‌اندازی
        self.initialize_alphabet()
        self.initialize_index()
        self.update_vectors()
    
    def initialize_alphabet(self):
        """آموزش کامل حروف الفبا"""
        processor = self.processor
        
        # آموزش هر حرف
        for letter, info in processor.persian_alphabet.items():
            lesson = {
                'letter': letter,
                'name': info['name'],
                'type': info['type'],
                'examples': [],
                'description': f"حرف {info['name']} ({letter}) یکی از حروف الفبای فارسی است."
            }
            
            # مثال برای هر حرف
            examples = {
                'آ': ['آب', 'آتش', 'آسمان'],
                'ا': ['ابر', 'امید', 'ایران'],
                'ب': ['باران', 'بابا', 'بهار'],
                'پ': ['پدر', 'پنجره', 'پول'],
                'ت': ['تاریخ', 'تبریز', 'تخت'],
                'ث': ['ثروت', 'ثلث', 'مثلث'],
                'ج': ['جنگ', 'جاده', 'جوان'],
                'چ': ['چشم', 'چوب', 'چای'],
                'ح': ['حرف', 'حافظ', 'حیاط'],
                'خ': ['خورشید', 'خوب', 'خواب'],
                'د': ['دوست', 'دست', 'دل'],
                'ذ': ['ذهن', 'مذهب', 'موذن'],
                'ر': ['روز', 'رنگ', 'راه'],
                'ز': ['زمین', 'زندگی', 'زبان'],
                'ژ': ['ژرفا', 'ژاله', 'ژن'],
                'س': ['سوال', 'سعدی', 'سحر'],
                'ش': ['شب', 'شعر', 'شادی'],
                'ص': ['صبح', 'صداقت', 'صبر'],
                'ض': ['ضرورت', 'ضرب', 'حضور'],
                'ط': ['طبیعت', 'طلا', 'طراوت'],
                'ظ': ['ظرف', 'ظرافت', 'نظر'],
                'ع': ['علم', 'عشق', 'عقل'],
                'غ': ['غروب', 'غم', 'غزل'],
                'ف': ['فکر', 'فردوسی', 'فصل'],
                'ق': ['قلم', 'قدم', 'قدرت'],
                'ک': ['کتاب', 'کوه', 'کار'],
                'گ': ['گل', 'گفتگو', 'گنج'],
                'ل': ['لبخند', 'لاله', 'لطف'],
                'م': ['ماه', 'مهر', 'مردم'],
                'ن': ['نور', 'نام', 'نگاه'],
                'و': ['وطن', 'وزش', 'وجود'],
                'ه': ['هوا', 'هستی', 'هفته'],
                'ی': ['یاد', 'یاس', 'یاری']
            }
            
            if letter in examples:
                for ex in examples[letter]:
                    lesson['examples'].append({
                        'word': ex,
                        'analysis': processor.analyze_letters(ex)
                    })
            
            self.db.data['alphabet_lessons'].append(lesson)
        
        # آموزش حرکات
        for mark, name in processor.diacritics.items():
            self.db.data['alphabet_lessons'].append({
                'letter': mark,
                'name': name,
                'type': 'diacritic',
                'description': f"{name} ({mark}) یکی از حرکات زبان فارسی است.",
                'examples': []
            })
        
        self.db.save()
    
    def initialize_index(self):
        """ایندکس‌گذاری اولیه دانش"""
        for item in self.knowledge_base:
            self.search_engine.add_document(
                item['id'],
                item['question'],
                item['answer'],
                item.get('category', 'عمومی')
            )
    
    def update_vectors(self):
        """به‌روزرسانی بردارها"""
        if self.knowledge_base:
            questions = [item['question'] for item in self.knowledge_base]
            try:
                self.question_vectors = self.vectorizer_tfidf.fit_transform(questions)
            except:
                self.question_vectors = None
    
    def analyze_question_deep(self, question):
        """تحلیل عمیق سوال کاربر"""
        # نرمال‌سازی
        normalized = self.processor.normalize_persian(question)
        
        # تجزیه به کلمات
        words = self.processor.tokenize_words(normalized)
        
        # تحلیل هر کلمه
        word_analysis = []
        for word in words:
            analysis = {
                'word': word,
                'features': self.processor.get_word_features(word),
                'letters': self.processor.analyze_letters(word),
                'question_word': word in self.processor.question_words,
                'question_type': self.processor.question_words.get(word, 'unknown')
            }
            word_analysis.append(analysis)
        
        # تشخیص نوع سوال
        question_type = 'general'
        for word in words:
            if word in self.processor.question_words:
                question_type = self.processor.question_words[word]
                break
        
        # تحلیل جملات
        sentences = self.processor.tokenize_sentences(question)
        
        return {
            'original': question,
            'normalized': normalized,
            'words': words,
            'word_analysis': word_analysis,
            'sentences': sentences,
            'question_type': question_type,
            'length': len(question),
            'word_count': len(words),
            'sentence_count': len(sentences)
        }
    
    def semantic_search_tfidf(self, query, threshold=0.1):
        """جستجوی معنایی با TF-IDF"""
        if not self.knowledge_base or self.question_vectors is None:
            return []
        
        query_vector = self.vectorizer_tfidf.transform([query])
        similarities = cosine_similarity(query_vector, self.question_vectors)[0]
        
        results = []
        for i, score in enumerate(similarities):
            if score >= threshold:
                item = self.knowledge_base[i]
                results.append({
                    'id': item['id'],
                    'question': item['question'],
                    'answer': item['answer'],
                    'score': float(score),
                    'method': 'tfidf',
                    'category': item.get('category', 'عمومی')
                })
        
        return sorted(results, key=lambda x: x['score'], reverse=True)
    
    def keyword_search(self, query, threshold=0.3):
        """جستجوی کلمات کلیدی"""
        results = []
        query_words = set(query.split())
        
        for item in self.knowledge_base:
            item_words = set(item['question'].split())
            common_words = query_words & item_words
            
            if common_words:
                score = len(common_words) / max(len(query_words), len(item_words))
                if score >= threshold:
                    results.append({
                        'id': item['id'],
                        'answer': item['answer'],
                        'score': score,
                        'common_words': list(common_words),
                        'method': 'keyword'
                    })
        
        return sorted(results, key=lambda x: x['score'], reverse=True)
    
    def letter_based_search(self, query):
        """جستجوی مبتنی بر حروف (برای الفبا)"""
        results = []
        
        # بررسی اینکه سوال درباره حروف است
        if any(word in query for word in ['حرف', 'الفبا', 'نوشتن', 'املا']):
            for lesson in self.db.data['alphabet_lessons']:
                if lesson['letter'] in query or lesson['name'] in query:
                    response = f"حرف {lesson['letter']} ({lesson['name']})"
                    if lesson['type'] == 'diacritic':
                        response += f"\n\n{lesson['description']}"
                    else:
                        response += f"\n\n{lesson['description']}\n\nمثال‌ها:\n"
                        for ex in lesson.get('examples', []):
                            response += f"\n• {ex['word']}"
                    
                    results.append({
                        'id': -lesson['letter'],
                        'answer': response,
                        'score': 1.0,
                        'method': 'alphabet'
                    })
        
        return results
    
    def ensemble_search(self, query):
        """ترکیب همه روش‌های جستجو"""
        # تحلیل عمیق سوال
        analysis = self.analyze_question_deep(query)
        
        # جستجو با روش‌های مختلف
        tfidf_results = self.semantic_search_tfidf(query)
        keyword_results = self.keyword_search(query)
        whoosh_results = self.search_engine.search(query)
        alphabet_results = self.letter_based_search(query)
        
        # ترکیب وزنی
        combined = {}
        weights = {
            'tfidf': 1.2,
            'keyword': 1.0,
            'whoosh': 1.1,
            'alphabet': 1.5
        }
        
        for results, method in [
            (tfidf_results, 'tfidf'),
            (keyword_results, 'keyword'),
            (whoosh_results, 'whoosh'),
            (alphabet_results, 'alphabet')
        ]:
            weight = weights.get(method, 1.0)
            for r in results:
                rid = r['id']
                if rid not in combined or r['score'] * weight > combined[rid]['score']:
                    r['score'] = r['score'] * weight
                    r['method'] = method
                    combined[rid] = r
        
        final_results = sorted(combined.values(), key=lambda x: x['score'], reverse=True)
        
        # تحلیل کیفیت
        quality = self.analyze_quality(final_results)
        
        return final_results, quality, analysis
    
    def analyze_quality(self, results):
        """تحلیل کیفیت نتایج"""
        if not results:
            return 'none'
        
        best_score = results[0]['score']
        
        if best_score >= 0.8:
            return 'عالی'
        elif best_score >= 0.6:
            return 'خوب'
        elif best_score >= 0.4:
            return 'متوسط'
        elif best_score >= 0.2:
            return 'ضعیف'
        else:
            return 'بسیار ضعیف'
    
    def generate_response(self, query):
        """تولید پاسخ نهایی"""
        # ثبت سوال کاربر
        self.record_question(query)
        
        # جستجو
        results, quality, analysis = self.ensemble_search(query)
        
        # تولید پاسخ
        if results:
            best = results[0]
            
            # پاسخ با کیفیت
            response = best['answer']
            
            # اضافه کردن توضیح در صورت نیاز
            if quality == 'ضعیف' or quality == 'بسیار ضعیف':
                response += "\n\n⚠️ این پاسخ با اطمینان پایین پیدا شده. لطفاً سوال را واضح‌تر بپرسید."
            
            # ثبت پاسخ موفق
            if quality in ['عالی', 'خوب']:
                self.record_answer(best['id'])
            
            return {
                'answer': response,
                'quality': quality,
                'found': True,
                'analysis': analysis
            }
        else:
            # هیچ پاسخی پیدا نشد
            return {
                'answer': None,
                'found': False,
                'analysis': analysis
            }
    
    def record_question(self, question):
        """ثبت سوال کاربر"""
        analysis = self.analyze_question_deep(question)
        
        record = {
            'id': len(self.users_questions) + 1,
            'question': question,
            'analysis': analysis,
            'timestamp': datetime.now().isoformat(),
            'answered': False
        }
        
        self.users_questions.append(record)
        self.stats['total_questions'] += 1
        self.stats['unanswered'] += 1
        
        # نگهداری 2000 رکورد آخر
        if len(self.users_questions) > 2000:
            self.users_questions = self.users_questions[-2000:]
        
        self.db.data['users_questions'] = self.users_questions
        self.db.data['stats'] = self.stats
        self.db.save()
    
    def record_answer(self, knowledge_id):
        """ثبت پاسخ موفق"""
        self.stats['answered'] += 1
        self.stats['unanswered'] -= 1
        
        for item in self.knowledge_base:
            if item['id'] == knowledge_id:
                item['times_used'] = item.get('times_used', 0) + 1
                item['last_used'] = datetime.now().isoformat()
                break
        
        self.db.save()
    
    def add_knowledge(self, question, answer, category='عمومی'):
        """افزودن دانش جدید"""
        # بررسی تکراری نبودن
        for item in self.knowledge_base:
            if item['question'].lower() == question.lower():
                return False, "این سوال قبلاً ثبت شده است"
        
        new_item = {
            'id': len(self.knowledge_base) + 1,
            'question': self.processor.normalize_persian(question),
            'original_question': question,
            'answer': answer,
            'category': category,
            'date_added': datetime.now().isoformat(),
            'times_used': 0,
            'last_used': None
        }
        
        self.knowledge_base.append(new_item)
        self.search_engine.add_document(new_item['id'], question, answer, category)
        self.update_vectors()
        
        self.db.data['knowledge_base'] = self.knowledge_base
        self.db.save()
        
        return True, "دانش با موفقیت اضافه شد"
    
    def bulk_import(self, text):
        """وارد کردن گروهی"""
        lines = text.strip().split('\n')
        count = 0
        errors = []
        
        for line in lines:
            if '|' in line:
                parts = line.split('|', 1)
                if len(parts) == 2:
                    q, a = parts
                    success, msg = self.add_knowledge(q.strip(), a.strip(), 'imported')
                    if success:
                        count += 1
                    else:
                        errors.append(f"خطا در {q[:30]}...: {msg}")
        
        return count, errors
    
    def save_conversation(self, user_id, question, answer):
        """ذخیره مکالمه کامل"""
        conv = {
            'id': len(self.conversations) + 1,
            'user_id': user_id,
            'question': question,
            'answer': answer,
            'timestamp': datetime.now().isoformat()
        }
        self.conversations.append(conv)
        self.stats['total_conversations'] += 1
        
        if len(self.conversations) > 1000:
            self.conversations = self.conversations[-1000:]
        
        self.db.data['conversations'] = self.conversations
        self.db.save()
    
    def get_unanswered(self):
        """گرفتن سوالات بی‌پاسخ"""
        return [q for q in self.users_questions if not q['answered']][-50:]
    
    def get_stats(self):
        """گرفتن آمار کامل"""
        return {
            'knowledge': {
                'total': len(self.knowledge_base),
                'categories': Counter([i.get('category', 'عمومی') for i in self.knowledge_base])
            },
            'users': {
                'total_questions': self.stats['total_questions'],
                'answered': self.stats['answered'],
                'unanswered': self.stats['unanswered'],
                'total_conversations': self.stats['total_conversations']
            },
            'alphabet': {
                'letters': len(self.processor.persian_alphabet),
                'diacritics': len(self.processor.diacritics),
                'total_lessons': len(self.db.data['alphabet_lessons'])
            }
        }

# ================ نمونه اصلی ================
ai = UltraPersianAI()
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'admin_login'

class User(UserMixin):
    def __init__(self, id, username, password):
        self.id = id
        self.username = username
        self.password = password

users = {
    '1': User('1', 'admin', hashlib.md5('admin123'.encode()).hexdigest())
}

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
        <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=yes">
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
                --success: #00b894;
                --warning: #fdcb6e;
                --danger: #d63031;
                --glass: rgba(255, 255, 255, 0.95);
            }
            
            body {
                font-family: 'Vazir', 'Tahoma', sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
                padding: 10px;
            }
            
            .chat-container {
                width: 100%;
                max-width: 500px;
                height: 95vh;
                background: var(--glass);
                backdrop-filter: blur(10px);
                border-radius: 30px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                display: flex;
                flex-direction: column;
                overflow: hidden;
                position: relative;
                animation: slideUp 0.5s ease;
            }
            
            @keyframes slideUp {
                from { opacity: 0; transform: translateY(20px); }
                to { opacity: 1; transform: translateY(0); }
            }
            
            .chat-header {
                background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
                color: white;
                padding: 15px 20px;
                display: flex;
                align-items: center;
                justify-content: space-between;
                position: relative;
            }
            
            .menu-btn {
                background: none;
                border: none;
                color: white;
                font-size: 28px;
                cursor: pointer;
                width: 40px;
                height: 40px;
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
                font-size: 1.3em;
                font-weight: bold;
                display: flex;
                align-items: center;
                gap: 8px;
            }
            
            .header-title span {
                font-size: 1.5em;
            }
            
            .admin-badge {
                background: rgba(255,255,255,0.2);
                padding: 5px 12px;
                border-radius: 30px;
                font-size: 0.8em;
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
                animation: messageSlide 0.3s ease;
                width: 100%;
            }
            
            @keyframes messageSlide {
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
                padding: 15px 18px;
                border-radius: 25px;
                position: relative;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                line-height: 1.6;
                font-size: 1rem;
                word-wrap: break-word;
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
                padding: 15px 20px;
                background: white;
                border-top: 1px solid #eee;
                display: flex;
                gap: 10px;
                align-items: center;
            }
            
            .chat-input {
                flex: 1;
                padding: 12px 18px;
                border: 2px solid #e0e0e0;
                border-radius: 30px;
                font-size: 1rem;
                outline: none;
                transition: all 0.3s;
                font-family: inherit;
            }
            
            .chat-input:focus {
                border-color: var(--primary);
                box-shadow: 0 0 0 3px rgba(108,92,231,0.1);
            }
            
            .send-btn {
                width: 50px;
                height: 50px;
                border-radius: 50%;
                background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
                color: white;
                border: none;
                cursor: pointer;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 1.3em;
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
                height: 100vh;
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
            
            .menu-header h3 {
                color: var(--dark);
                font-size: 1.3em;
            }
            
            .close-menu {
                background: none;
                border: none;
                font-size: 24px;
                cursor: pointer;
                color: #666;
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
            
            .menu-item i {
                font-size: 1.5em;
                width: 30px;
                text-align: center;
            }
            
            .menu-item.admin {
                background: linear-gradient(135deg, #667eea10 0%, #764ba210 100%);
                border: 1px solid var(--primary);
            }
            
            .history-list {
                margin-top: 20px;
                max-height: 300px;
                overflow-y: auto;
            }
            
            .history-item {
                padding: 10px;
                margin: 5px 0;
                background: #f8fafc;
                border-radius: 10px;
                font-size: 0.9em;
                cursor: pointer;
                border-right: 3px solid var(--primary);
            }
            
            .history-item:hover {
                background: #eef2f6;
            }
            
            .welcome-message {
                text-align: center;
                padding: 20px;
                background: linear-gradient(135deg, #667eea10 0%, #764ba210 100%);
                border-radius: 20px;
                margin-bottom: 10px;
            }
            
            .welcome-message h3 {
                color: var(--primary);
                margin-bottom: 10px;
                font-size: 1.4em;
            }
            
            .welcome-message p {
                color: #666;
                font-size: 0.95em;
            }
            
            .quick-actions {
                display: flex;
                flex-wrap: wrap;
                gap: 10px;
                justify-content: center;
                margin-top: 15px;
            }
            
            .quick-btn {
                background: white;
                border: 2px solid var(--primary);
                color: var(--primary);
                padding: 8px 15px;
                border-radius: 30px;
                font-size: 0.9em;
                cursor: pointer;
                transition: all 0.3s;
            }
            
            .quick-btn:hover {
                background: var(--primary);
                color: white;
            }
            
            @media (max-width: 480px) {
                .chat-container {
                    height: 100vh;
                    border-radius: 0;
                }
                
                .message-content {
                    max-width: 90%;
                    font-size: 0.95rem;
                }
                
                .menu-panel {
                    width: 260px;
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
                <div class="admin-badge">نسخه ۳.۰</div>
            </div>
            
            <div class="chat-messages" id="chat-messages">
                <div class="welcome-message">
                    <h3>🌟 به هوش مصنوعی ایرانی خوش آمدید</h3>
                    <p>هر سوالی دارید بپرسید! درباره تاریخ، حروف الفبا، مفاهیم علمی و ...</p>
                    <div class="quick-actions">
                        <span class="quick-btn" onclick="quickQuestion('حرف ب چیست')">📝 حرف ب</span>
                        <span class="quick-btn" onclick="quickQuestion('کوروش کبیر که بود')">👑 کوروش</span>
                        <span class="quick-btn" onclick="quickQuestion('چگونه جمله بسازیم')">✍️ جمله‌سازی</span>
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
        
        <!-- منوی کشویی -->
        <div class="menu-overlay" id="menuOverlay" onclick="closeMenu()"></div>
        <div class="menu-panel" id="menuPanel">
            <div class="menu-header">
                <h3>منو</h3>
                <button class="close-menu" onclick="closeMenu()">✕</button>
            </div>
            
            <a href="/m.html" class="menu-item">
                <i>📄</i> صفحه M
            </a>
            
            <a href="/admin-login" class="menu-item admin">
                <i>⚙️</i> پنل مدیریت
            </a>
            
            <div class="menu-item" onclick="clearHistory()">
                <i>🗑️</i> پاک کردن تاریخچه
            </div>
            
            <div style="margin-top: 20px;">
                <h4 style="color: #666; margin-bottom: 10px;">تاریخچه چت</h4>
                <div class="history-list" id="historyList"></div>
            </div>
        </div>
        
        <script>
            // بارگذاری تاریخچه
            let chatHistory = JSON.parse(localStorage.getItem('persian_ai_chat')) || [];
            let currentSession = JSON.parse(sessionStorage.getItem('current_chat')) || [];
            
            // نمایش تاریخچه
            function loadHistory() {
                const historyList = document.getElementById('historyList');
                if (historyList) {
                    historyList.innerHTML = '';
                    const recent = chatHistory.slice(-10).reverse();
                    recent.forEach(msg => {
                        const div = document.createElement('div');
                        div.className = 'history-item';
                        div.onclick = () => loadConversation(msg.id);
                        div.innerHTML = `
                            <div style="font-weight: bold; color: var(--primary);">${msg.question.substring(0, 30)}...</div>
                            <div style="font-size: 0.8em; color: #666;">${new Date(msg.time).toLocaleString('fa-IR')}</div>
                        `;
                        historyList.appendChild(div);
                    });
                }
            }
            
            // نمایش پیام‌های فعلی
            currentSession.forEach(msg => {
                addMessage(msg.text, msg.isUser, msg.time, false);
            });
            
            function toggleMenu() {
                document.getElementById('menuOverlay').style.display = 'block';
                document.getElementById('menuPanel').classList.add('open');
                loadHistory();
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
                messagesDiv.scrollTop = messagesDiv.scrollHeight;
                
                if (save) {
                    const msgObj = {
                        text: text,
                        isUser: isUser,
                        time: messageTime
                    };
                    currentSession.push(msgObj);
                    sessionStorage.setItem('current_chat', JSON.stringify(currentSession));
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
                document.getElementById('chat-messages').scrollTop = document.getElementById('chat-messages').scrollHeight;
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
                        
                        // ذخیره در localStorage
                        chatHistory.push({
                            id: Date.now(),
                            question: message,
                            answer: data.answer,
                            time: new Date().toISOString()
                        });
                        
                        if (chatHistory.length > 100) {
                            chatHistory = chatHistory.slice(-100);
                        }
                        
                        localStorage.setItem('persian_ai_chat', JSON.stringify(chatHistory));
                    } else {
                        addMessage('🤔 متأسفم! هنوز نتونستم پاسخ این سوال رو پیدا کنم. سوال شما برای مدیر ارسال شد.');
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
            
            function loadConversation(id) {
                const conv = chatHistory.find(c => c.id === id);
                if (conv) {
                    closeMenu();
                    // پاک کردن صفحه و نمایش مکالمه
                    document.getElementById('chat-messages').innerHTML = '';
                    addMessage(conv.question, true, null, false);
                    addMessage(conv.answer, false, null, false);
                }
            }
            
            function clearHistory() {
                if (confirm('آیا تاریخچه چت پاک شود؟')) {
                    localStorage.removeItem('persian_ai_chat');
                    sessionStorage.removeItem('current_chat');
                    chatHistory = [];
                    currentSession = [];
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
    <html lang="fa">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>صفحه M - هوش ایرانی</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body {
                font-family: 'Vazir', Tahoma;
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
                max-width: 800px;
                width: 100%;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            }
            h1 {
                color: #333;
                margin-bottom: 20px;
                text-align: center;
            }
            p {
                color: #666;
                line-height: 1.8;
                margin-bottom: 15px;
            }
            .btn {
                display: inline-block;
                padding: 12px 25px;
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
            <p>این صفحه مخصوص منوی کشویی است. می‌توانید از اینجا به بخش‌های مختلف دسترسی داشته باشید.</p>
            <p>برای بازگشت به چت، روی دکمه زیر کلیک کنید.</p>
            <a href="/" class="btn">🔙 بازگشت به چت</a>
        </div>
    </body>
    </html>
    ''')

@app.route('/api/chat', methods=['POST'])
def api_chat():
    data = request.json
    question = data.get('message', '').strip()
    
    if not question:
        return jsonify({'error': 'سوال نمی‌تواند خالی باشد'})
    
    # پردازش سوال
    result = ai.generate_response(question)
    
    if result['found']:
        return jsonify({
            'answer': result['answer'],
            'quality': result['quality'],
            'found': True
        })
    else:
        return jsonify({
            'answer': None,
            'found': False
        })

@app.route('/admin-login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form['username']
        password = hashlib.md5(request.form['password'].encode()).hexdigest()
        
        if username == 'admin' and password == hashlib.md5('admin123'.encode()).hexdigest():
            user = users['1']
            login_user(user)
            session.permanent = True
            return redirect(url_for('admin_panel'))
        
        return "❌ نام کاربری یا رمز عبور اشتباه است"
    
    return render_template_string('''
    <!DOCTYPE html>
    <html>
    <head>
        <title>ورود به پنل</title>
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
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                width: 100%;
                max-width: 400px;
            }
            h2 {
                text-align: center;
                color: #333;
                margin-bottom: 30px;
            }
            input {
                width: 100%;
                padding: 15px;
                margin: 10px 0;
                border: 2px solid #e0e0e0;
                border-radius: 15px;
                font-family: Tahoma;
                font-size: 1em;
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
        </style>
    </head>
    <body>
        <div class="login-box">
            <h2>🔐 پنل مدیریت</h2>
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
                align-items: center;
            }
            .stats-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin-bottom: 20px;
            }
            .stat-card {
                background: white;
                padding: 20px;
                border-radius: 15px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
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
                font-family: Tahoma;
            }
            button {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 12px 25px;
                border: none;
                border-radius: 10px;
                cursor: pointer;
            }
            .unanswered-item {
                background: #fff3cd;
                padding: 15px;
                margin: 10px 0;
                border-radius: 10px;
            }
            .file-upload {
                border: 2px dashed #667eea;
                padding: 30px;
                text-align: center;
                border-radius: 10px;
                cursor: pointer;
                margin: 20px 0;
            }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>⚙️ پنل مدیریت</h1>
            <div>
                <a href="/" style="color: white; margin-right: 15px;">🔙 چت</a>
                <a href="/logout" style="color: white;">🚪 خروج</a>
            </div>
        </div>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-number">{{ stats.knowledge.total }}</div>
                <div>کل دانش</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{{ stats.users.total_questions }}</div>
                <div>کل سوالات</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{{ stats.users.unanswered }}</div>
                <div>بی‌پاسخ</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{{ stats.alphabet.letters }}</div>
                <div>حروف الفبا</div>
            </div>
        </div>
        
        <div class="card">
            <h2>➕ افزودن دانش جدید</h2>
            <form action="/admin/add" method="POST">
                <input type="text" name="question" placeholder="سوال" required>
                <textarea name="answer" rows="4" placeholder="پاسخ" required></textarea>
                <select name="category">
                    <option>عمومی</option>
                    <option>تاریخ</option>
                    <option>الفبا</option>
                    <option>علمی</option>
                </select>
                <button type="submit">افزودن</button>
            </form>
        </div>
        
        <div class="card">
            <h2>📁 آپلود فایل</h2>
            <form action="/admin/upload" method="POST" enctype="multipart/form-data">
                <div class="file-upload" onclick="document.getElementById('file').click()">
                    <p>📤 کلیک برای آپلود</p>
                    <p style="font-size:0.9em;">فرمت: هر خط: سوال | جواب</p>
                </div>
                <input type="file" id="file" name="file" style="display:none;" accept=".txt">
                <button type="submit">آپلود</button>
            </form>
        </div>
        
        <div class="card">
            <h2>❓ سوالات بی‌پاسخ ({{ unanswered|length }})</h2>
            {% for item in unanswered %}
            <div class="unanswered-item">
                <strong>{{ item.question }}</strong>
                <div style="margin-top: 10px;">
                    <button onclick="answerQuestion('{{ item.question }}')">پاسخ</button>
                </div>
            </div>
            {% endfor %}
        </div>
        
        <script>
            function answerQuestion(q) {
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
    if 'file' not in request.files:
        return "❌ فایلی انتخاب نشده"
    
    file = request.files['file']
    if file.filename == '':
        return "❌ نام فایل معتبر نیست"
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        count, errors = ai.bulk_import(content)
        return f"✅ {count} مورد اضافه شد <a href='/admin'>بازگشت</a>"

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

if __name__ == '__main__':
    print("""
    ╔═══════════════════════════════════════════════════════════════════╗
    ║     🤖 هوش مصنوعی فوق پیشرفته ایرانی - نسخه نهایی                ║
    ╠═══════════════════════════════════════════════════════════════════╣
    ║  📚 دانش: {} مورد                                                  ║
    ║  📊 سوالات: {}                                                      ║
    ║  🔤 حروف الفبا: {} حرف                                             ║
    ║  🌐 چت: http://localhost:5000                                     ║
    ║  🔐 پنل: http://localhost:5000/admin-login                        ║
    ║  👤 کاربر: admin / admin123                                       ║
    ║  📱 موبایل: کاملاً ریسپانسیو                                      ║
    ╚═══════════════════════════════════════════════════════════════════╝
    """.format(len(ai.knowledge_base), ai.stats['total_questions'], len(ai.processor.persian_alphabet)))
    
    app.run(debug=True, host='0.0.0.0', port=5000)
