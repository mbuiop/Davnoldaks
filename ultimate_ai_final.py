# ultimate_ai_master.py
from flask import Flask, render_template, request, jsonify, session, render_template_string, redirect, url_for, make_response
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

# ================ کتابخانه‌های تحلیل متن حرفه‌ای ================
import langid  # تشخیص زبان
from textblob import TextBlob, Word  # تحلیل متن و کلمات
import nltk  # پردازش زبان طبیعی
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk import pos_tag, ne_chunk  # برچسب‌گذاری نقش کلمات و تشخیص موجودیت‌ها
import spacy  # پردازش عمیق زبان
from collections import Counter

# ================ کتابخانه‌های فارسی ================
import hazm  # پردازش زبان فارسی
from hazm import Normalizer, WordTokenizer, SentenceTokenizer, Lemmatizer, Stemmer, POSTagger

# دانلود منابع مورد نیاز
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('averaged_perceptron_tagger')
    nltk.data.find('maxent_ne_chunker')
    nltk.data.find('words')
except:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('maxent_ne_chunker')
    nltk.download('words')
    nltk.download('wordnet')

# تلاش برای بارگذاری spacy (اختیاری)
try:
    nlp = spacy.load("en_core_web_sm")
except:
    try:
        import subprocess
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
        nlp = spacy.load("en_core_web_sm")
    except:
        nlp = None

app = Flask(__name__)
app.config['SECRET_KEY'] = 'super-secret-key-for-ultimate-ai'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=365)
CORS(app)

# ایجاد پوشه‌ها
os.makedirs('data', exist_ok=True)
os.makedirs('uploads', exist_ok=True)
os.makedirs('memory', exist_ok=True)
os.makedirs('user_profiles', exist_ok=True)
os.makedirs('backups', exist_ok=True)

# ================ تحلیلگر فوق پیشرفته متن ================
class UltraTextAnalyzer:
    """تحلیلگر عمیق متن با ۱۰+ الگوریتم"""
    
    def __init__(self):
        # ابزارهای فارسی
        self.normalizer = Normalizer()
        self.word_tokenizer = WordTokenizer()
        self.sent_tokenizer = SentenceTokenizer()
        self.lemmatizer = Lemmatizer()
        self.stemmer = Stemmer()
        
        # ابزارهای انگلیسی
        self.lemmatizer_en = WordNetLemmatizer()
        self.stemmer_en = PorterStemmer()
        
        # الگوهای سوال
        self.question_patterns = {
            'person': {
                'patterns': [r'(کیست|که بود|چه کسی|بیوگرافی|زندگینامه|افراد|شخص|name|who|biography)'],
                'weight': 1.5
            },
            'place': {
                'patterns': [r'(کجاست|کجا|مکان|شهر|کشور|استان|موقعیت|محل|where|location|place)'],
                'weight': 1.4
            },
            'time': {
                'patterns': [r'(کی|چه زمانی|تاریخ|سال|قرن|دوره|میلادی|شمسی|هجری|when|date|time|century)'],
                'weight': 1.4
            },
            'reason': {
                'patterns': [r'(چرا|دلیل|علت|چگونه|چطور|به چه دلیل|why|reason|cause|how)'],
                'weight': 1.3
            },
            'definition': {
                'patterns': [r'(چیست|چه بود|تعریف|توضیح|معنی|مفهوم|یعنی چه|what|definition|meaning)'],
                'weight': 1.3
            },
            'quantity': {
                'patterns': [r'(چند|تعداد|مقدار|چه قدر|چه اندازه|how many|how much|quantity)'],
                'weight': 1.2
            },
            'comparison': {
                'patterns': [r'(فرق|تفاوت|شباهت|مقایسه|بهتر|بدتر|compare|comparison|difference|similar)'],
                'weight': 1.3
            },
            'code': {
                'patterns': [r'(کد|برنامه|نویسی|پایتون|جاوا|php|html|css|javascript|الگوریتم|تابع|code|program|function)'],
                'weight': 1.5
            },
            'alphabet': {
                'patterns': [r'(حرف|الفبا|نوشتن|املا|خواندن|صدا|کلمه|letter|alphabet|spell|read|write)'],
                'weight': 1.4
            },
            'feeling': {
                'patterns': [r'(حس|احساس|عشق|نفرت|خوشحال|غمگین|عصبانی|happy|sad|angry|love|hate|feel)'],
                'weight': 1.6
            },
            'opinion': {
                'patterns': [r'(نظر|عقیده|فکر|باور|فکر می‌کنی|think|believe|opinion)'],
                'weight': 1.5
            }
        }
        
        # کلمات کلیدی مهم
        self.important_words = {
            'history': ['تاریخ', 'قدیم', 'باستان', 'هخامنشی', 'ساسانی', 'قاجار', 'پهلوی', 'history', 'ancient'],
            'science': ['علم', 'دانش', 'فیزیک', 'شیمی', 'زیست', 'ریاضی', 'science', 'physics', 'chemistry'],
            'art': ['هنر', 'نقاشی', 'موسیقی', 'شعر', 'ادبیات', 'art', 'music', 'painting', 'poetry'],
            'technology': ['تکنولوژی', 'کامپیوتر', 'اینترنت', 'هوش مصنوعی', 'ربات', 'technology', 'computer', 'ai'],
            'religion': ['دین', 'اسلام', 'مسیحیت', 'یهودیت', 'زرتشت', 'خدا', 'پیامبر', 'religion', 'god', 'prophet'],
            'sport': ['ورزش', 'فوتبال', 'بسکتبال', 'والیبال', 'sport', 'football', 'soccer', 'basketball'],
            'food': ['غذا', 'خوراک', 'آشپزی', 'نان', 'برنج', 'food', 'cooking', 'recipe'],
            'travel': ['سفر', 'گردشگری', 'مسافرت', 'هتل', 'travel', 'tourism', 'hotel'],
            'education': ['آموزش', 'مدرسه', 'دانشگاه', 'کلاس', 'درس', 'education', 'school', 'university', 'class'],
            'health': ['سلامت', 'بیماری', 'درمان', 'دارو', 'دکتر', 'health', 'disease', 'treatment', 'medicine']
        }
    
    def detect_language(self, text):
        """تشخیص دقیق زبان متن"""
        try:
            lang, confidence = langid.classify(text)
            return lang, confidence
        except:
            # تشخیص ساده با حروف
            persian_chars = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
            english_chars = sum(1 for c in text if 'a' <= c.lower() <= 'z')
            
            if persian_chars > english_chars:
                return 'fa', 0.8
            else:
                return 'en', 0.8
    
    def detect_question_type_deep(self, text):
        """تشخیص عمیق نوع سوال با وزن‌دهی"""
        text_lower = text.lower()
        scores = {}
        
        for q_type, data in self.question_patterns.items():
            score = 0
            for pattern in data['patterns']:
                matches = re.findall(pattern, text_lower)
                score += len(matches) * data['weight']
            if score > 0:
                scores[q_type] = score
        
        if scores:
            best_type = max(scores.items(), key=lambda x: x[1])
            return best_type[0], best_type[1]
        
        return 'general', 0
    
    def extract_keywords_advanced(self, text, top_k=10):
        """استخراج پیشرفته کلمات کلیدی با وزن"""
        # تشخیص زبان
        lang, _ = self.detect_language(text)
        
        # توکنایز
        if lang == 'fa':
            tokens = self.word_tokenizer.tokenize(text)
        else:
            tokens = word_tokenize(text)
        
        # حذف کلمات ایست
        stop_words = set()
        try:
            stop_words.update(stopwords.words('persian'))
        except:
            pass
        try:
            stop_words.update(stopwords.words('english'))
        except:
            pass
        
        # کلمات مهم
        keywords = []
        for token in tokens:
            if len(token) > 2 and token.lower() not in stop_words:
                # وزن‌دهی بر اساس موقعیت
                weight = 1.0
                
                # کلمات با حرف بزرگ (اسم خاص)
                if token[0].isupper():
                    weight *= 1.5
                
                # کلمات تکراری
                if tokens.count(token) > 1:
                    weight *= 1.3
                
                keywords.append((token, weight))
        
        # مرتب‌سازی بر اساس وزن
        keywords.sort(key=lambda x: x[1], reverse=True)
        return keywords[:top_k]
    
    def extract_entities(self, text):
        """استخراج موجودیت‌ها (اسامی خاص، مکان‌ها، ...)"""
        entities = {
            'persons': [],
            'places': [],
            'organizations': [],
            'dates': [],
            'other': []
        }
        
        # تشخیص زبان
        lang, _ = self.detect_language(text)
        
        if lang == 'en' and nlp:
            # استفاده از spacy برای انگلیسی
            doc = nlp(text)
            for ent in doc.ents:
                if ent.label_ == 'PERSON':
                    entities['persons'].append(ent.text)
                elif ent.label_ in ['GPE', 'LOC']:
                    entities['places'].append(ent.text)
                elif ent.label_ == 'ORG':
                    entities['organizations'].append(ent.text)
                elif ent.label_ == 'DATE':
                    entities['dates'].append(ent.text)
                else:
                    entities['other'].append(f"{ent.text} ({ent.label_})")
        else:
            # تشخیص ساده برای فارسی
            words = text.split()
            for i, word in enumerate(words):
                if word[0].isupper() and len(word) > 2:
                    if i < len(words) - 1 and words[i+1][0].isupper():
                        entities['persons'].append(word + " " + words[i+1])
                    else:
                        entities['persons'].append(word)
        
        return entities
    
    def analyze_sentiment_deep(self, text):
        """تحلیل عمیق احساسات"""
        try:
            blob = TextBlob(text)
            
            # احساسات اصلی
            polarity = blob.sentiment.polarity  # -1 تا 1
            subjectivity = blob.sentiment.subjectivity  # 0 تا 1
            
            # تشخیص احساس خاص
            emotion = "خنثی"
            if polarity > 0.5:
                emotion = "بسیار مثبت 😊"
            elif polarity > 0.1:
                emotion = "مثبت 🙂"
            elif polarity < -0.5:
                emotion = "بسیار منفی 😠"
            elif polarity < -0.1:
                emotion = "منفی 😞"
            
            # کلمات احساسی
            sentiment_words = []
            for sentence in blob.sentences:
                if abs(sentence.sentiment.polarity) > 0.3:
                    sentiment_words.append(str(sentence))
            
            return {
                'polarity': polarity,
                'subjectivity': subjectivity,
                'emotion': emotion,
                'sentiment_words': sentiment_words[:3]
            }
        except:
            return {
                'polarity': 0,
                'subjectivity': 0,
                'emotion': "نامشخص",
                'sentiment_words': []
            }
    
    def get_topic(self, text):
        """تشخیص موضوع اصلی متن"""
        text_lower = text.lower()
        topic_scores = {}
        
        for topic, keywords in self.important_words.items():
            score = 0
            for keyword in keywords:
                if keyword in text_lower:
                    score += 1
            if score > 0:
                topic_scores[topic] = score
        
        if topic_scores:
            return max(topic_scores.items(), key=lambda x: x[1])[0]
        
        return 'general'
    
    def analyze_context(self, text, history=None):
        """تحلیل زمینه و ارتباط با تاریخچه"""
        context = {
            'topic': self.get_topic(text),
            'type': self.detect_question_type_deep(text)[0],
            'entities': self.extract_entities(text),
            'sentiment': self.analyze_sentiment_deep(text),
            'keywords': self.extract_keywords_advanced(text, 5),
            'language': self.detect_language(text)[0]
        }
        
        # ارتباط با تاریخچه
        if history:
            # بررسی تکرار موضوع
            same_topic_count = sum(1 for h in history if h.get('topic') == context['topic'])
            context['topic_frequency'] = same_topic_count
            
            # بررسی تغییر احساسات
            if history and 'sentiment' in history[-1]:
                prev_sentiment = history[-1]['sentiment'].get('polarity', 0)
                context['sentiment_change'] = context['sentiment']['polarity'] - prev_sentiment
        
        return context
    
    def calculate_similarity(self, text1, text2):
        """محاسبه شباهت دو متن"""
        # روش 1: اشتراک کلمات
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0
        
        jaccard = len(words1 & words2) / len(words1 | words2)
        
        # روش 2: شباهت برداری (ساده)
        all_words = list(words1 | words2)
        vec1 = [1 if w in words1 else 0 for w in all_words]
        vec2 = [1 if w in words2 else 0 for w in all_words]
        
        if sum(vec1) == 0 or sum(vec2) == 0:
            return jaccard
        
        dot = sum(v1 * v2 for v1, v2 in zip(vec1, vec2))
        norm1 = sum(v1 * v1 for v1 in vec1) ** 0.5
        norm2 = sum(v2 * v2 for v2 in vec2) ** 0.5
        
        cosine = dot / (norm1 * norm2) if norm1 * norm2 > 0 else 0
        
        # ترکیب
        return (jaccard * 0.4 + cosine * 0.6)

# ================ حافظه نامحدود ================
class InfiniteMemory:
    """حافظه نامحدود با قابلیت ذخیره همه چیز"""
    
    def __init__(self, memory_dir='memory'):
        self.memory_dir = memory_dir
        self.conversations = []
        self.user_memories = defaultdict(list)
        self.global_memory = []
        self.patterns = defaultdict(int)
        self.load_all()
    
    def load_all(self):
        """بارگذاری همه خاطرات"""
        # بارگذاری مکالمات
        conv_file = f'{self.memory_dir}/conversations.json'
        if os.path.exists(conv_file):
            with open(conv_file, 'r', encoding='utf-8') as f:
                self.conversations = json.load(f)
        
        # بارگذاری الگوها
        patterns_file = f'{self.memory_dir}/patterns.json'
        if os.path.exists(patterns_file):
            with open(patterns_file, 'r', encoding='utf-8') as f:
                self.patterns = defaultdict(int, json.load(f))
        
        # بارگذاری حافظه کاربران
        users_file = f'{self.memory_dir}/users.json'
        if os.path.exists(users_file):
            with open(users_file, 'r', encoding='utf-8') as f:
                users_data = json.load(f)
                for uid, data in users_data.items():
                    self.user_memories[uid] = data
        
        print(f"💾 {len(self.conversations)} مکالمه بارگذاری شد")
        print(f"👤 {len(self.user_memories)} کاربر در حافظه")
    
    def save_all(self):
        """ذخیره همه خاطرات"""
        # ذخیره مکالمات
        with open(f'{self.memory_dir}/conversations.json', 'w', encoding='utf-8') as f:
            json.dump(self.conversations[-10000:], f, ensure_ascii=False, indent=2)
        
        # ذخیره الگوها
        with open(f'{self.memory_dir}/patterns.json', 'w', encoding='utf-8') as f:
            json.dump(dict(self.patterns), f, ensure_ascii=False, indent=2)
        
        # ذخیره حافظه کاربران
        with open(f'{self.memory_dir}/users.json', 'w', encoding='utf-8') as f:
            json.dump(dict(self.user_memories), f, ensure_ascii=False, indent=2)
    
    def add_conversation(self, user_id, question, answer, context):
        """افزودن مکالمه به حافظه"""
        conv = {
            'id': str(uuid.uuid4())[:8],
            'user': user_id,
            'question': question,
            'answer': answer,
            'context': context,
            'time': datetime.now().isoformat()
        }
        self.conversations.append(conv)
        
        # حافظه کاربر
        if user_id not in self.user_memories:
            self.user_memories[user_id] = {
                'conversations': [],
                'topics': Counter(),
                'patterns': [],
                'first_seen': datetime.now().isoformat()
            }
        
        mem = self.user_memories[user_id]
        mem['conversations'].append({
            'question': question,
            'topic': context.get('topic'),
            'type': context.get('type'),
            'time': datetime.now().isoformat()
        })
        mem['topics'][context.get('topic', 'general')] += 1
        
        # الگوها
        for word in context.get('keywords', []):
            if isinstance(word, tuple) and len(word) > 0:
                self.patterns[word[0]] += 1
        
        # ذخیره خودکار هر ۱۰۰ مکالمه
        if len(self.conversations) % 100 == 0:
            self.save_all()
    
    def get_user_context(self, user_id, limit=5):
        """گرفتن زمینه کاربر"""
        if user_id in self.user_memories:
            mem = self.user_memories[user_id]
            recent = mem['conversations'][-limit:]
            main_topic = mem['topics'].most_common(1)[0][0] if mem['topics'] else 'general'
            return {
                'recent': recent,
                'main_topic': main_topic,
                'total': len(mem['conversations'])
            }
        return {'recent': [], 'main_topic': 'general', 'total': 0}
    
    def find_similar_questions(self, question, analyzer, limit=5):
        """پیدا کردن سوالات مشابه در حافظه"""
        similarities = []
        for conv in self.conversations[-1000:]:  # آخرین ۱۰۰۰ تا
            sim = analyzer.calculate_similarity(question, conv['question'])
            if sim > 0.5:
                similarities.append((sim, conv['answer'], conv['context']))
        
        similarities.sort(reverse=True)
        return similarities[:limit]

# ================ موتور جستجوی فوق پیشرفته ================
class UltraSearchEngine:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=20000,
            ngram_range=(1, 4),
            analyzer='char_wb',
            sublinear_tf=True,
            use_idf=True,
            smooth_idf=True
        )
        self.hashing_vectorizer = HashingVectorizer(
            n_features=2**18,  # 262144
            ngram_range=(1, 3),
            norm='l2'
        )
        self.documents = []
        self.vectors = None
        self.hashing_vectors = None
        self.analyzer = UltraTextAnalyzer()
        self.memory = InfiniteMemory()
    
    def add_document(self, question, answer, category, user_id=None):
        """افزودن سند با تحلیل کامل"""
        context = self.analyzer.analyze_context(question)
        
        doc = {
            'id': str(uuid.uuid4())[:8],
            'q': question,
            'a': answer,
            'cat': category,
            'context': context,
            'user': user_id,
            'time': datetime.now().isoformat(),
            'use_count': 0
        }
        
        self.documents.append(doc)
        return doc
    
    def update_vectors(self):
        """به‌روزرسانی بردارها"""
        if self.documents:
            questions = [d['q'] for d in self.documents]
            self.vectors = self.vectorizer.fit_transform(questions)
            self.hashing_vectors = self.hashing_vectorizer.transform(questions)
    
    def search_ultra(self, query, user_id=None, history=None):
        """جستجوی فوق پیشرفته با ۷ الگوریتم"""
        # تحلیل عمیق سوال
        query_context = self.analyzer.analyze_context(query, history)
        
        # گرفتن زمینه کاربر
        user_context = self.memory.get_user_context(user_id) if user_id else None
        
        # پیدا کردن سوالات مشابه در حافظه
        similar = self.memory.find_similar_questions(query, self.analyzer)
        
        results = []
        
        if not self.documents:
            return results, query_context, user_context, similar
        
        # 1. جستجوی برداری اصلی
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.vectors)[0]
        
        # 2. جستجوی هشینگ (سریع)
        query_hash = self.hashing_vectorizer.transform([query])
        hash_similarities = cosine_similarity(query_hash, self.hashing_vectors)[0]
        
        for i, doc in enumerate(self.documents):
            score = similarities[i] * 0.5 + hash_similarities[i] * 0.3
            
            # 3. تطابق نوع سوال
            if doc['context']['type'] == query_context['type']:
                score *= 1.3
            
            # 4. تطابق موضوع
            if doc['context']['topic'] == query_context['topic']:
                score *= 1.2
            
            # 5. کلمات کلیدی مشترک
            doc_keywords = [k[0] for k in doc['context'].get('keywords', [])]
            query_keywords = [k[0] for k in query_context.get('keywords', [])]
            common = set(doc_keywords) & set(query_keywords)
            if common:
                score *= (1 + len(common) * 0.1)
            
            # 6. تطابق با تاریخچه کاربر
            if user_context and user_context['main_topic'] == doc['context']['topic']:
                score *= 1.1
            
            # 7. تطابق با سوالات مشابه
            for sim_score, sim_answer, sim_context in similar:
                if sim_answer == doc['a']:
                    score *= (1 + sim_score * 0.2)
            
            if score > 0.15:
                results.append({
                    'answer': doc['a'],
                    'score': float(score),
                    'category': doc['cat'],
                    'context': doc['context']
                })
        
        # مرتب‌سازی
        results.sort(key=lambda x: x['score'], reverse=True)
        
        # افزایش استفاده
        if results and len(self.documents) > 0:
            for doc in self.documents:
                if doc['a'] == results[0]['answer']:
                    doc['use_count'] += 1
                    break
        
        return results[:5], query_context, user_context, similar

# ================ هوش مصنوعی اصلی ================
class UltimateAI:
    def __init__(self):
        self.search = UltraSearchEngine()
        self.analyzer = UltraTextAnalyzer()
        self.memory = InfiniteMemory()
        self.db_file = 'data/ultimate_knowledge.json'
        self.load_knowledge()
        
        print(f"✅ {len(self.search.documents)} دانش بارگذاری شد")
        print(f"💾 {len(self.memory.conversations)} مکالمه در حافظه")
    
    def load_knowledge(self):
        """بارگذاری دانش"""
        if os.path.exists(self.db_file):
            with open(self.db_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for item in data:
                    doc = self.search.add_document(
                        item['q'], 
                        item['a'], 
                        item.get('cat', 'general'),
                        item.get('user')
                    )
        
        self.search.update_vectors()
    
    def save_knowledge(self):
        """ذخیره دانش"""
        data = []
        for doc in self.search.documents:
            data.append({
                'id': doc['id'],
                'q': doc['q'],
                'a': doc['a'],
                'cat': doc['cat'],
                'user': doc.get('user'),
                'time': doc.get('time')
            })
        
        with open(self.db_file, 'w', encoding='utf-8') as f:
            json.dump(data[-10000:], f, ensure_ascii=False, indent=2)
    
    def learn(self, question, answer, category='general', user_id=None):
        """یادگیری دانش جدید"""
        # نرمال‌سازی
        question = self.analyzer.normalizer.normalize(question) if hasattr(self.analyzer, 'normalizer') else question
        
        # بررسی تکراری
        for doc in self.search.documents:
            if self.analyzer.calculate_similarity(doc['q'], question) > 0.9:
                doc['a'] = answer
                doc['use_count'] = doc.get('use_count', 0) + 1
                self.save_knowledge()
                return True, "به‌روزرسانی شد"
        
        # اضافه کردن جدید
        self.search.add_document(question, answer, category, user_id)
        self.search.update_vectors()
        self.save_knowledge()
        
        return True, "یاد گرفته شد"
    
    def ask(self, question, user_id=None, history=None):
        """پرسش و پاسخ هوشمند"""
        results, query_context, user_context, similar = self.search.search_ultra(question, user_id, history)
        
        # ذخیره در حافظه
        if results:
            best = results[0]
            self.memory.add_conversation(user_id, question, best['answer'], query_context)
            
            # ساخت پاسخ هوشمند
            answer = best['answer']
            
            # اضافه کردن بر اساس زمینه
            if query_context['topic'] == 'history' and 'sentiment' in query_context:
                if query_context['sentiment']['polarity'] > 0.3:
                    answer += "\n\n📚 به نظر می‌رسد به تاریخ علاقه دارید!"
                elif query_context['sentiment']['polarity'] < -0.3:
                    answer += "\n\n😔 تاریخ پر از فراز و نشیب است..."
            
            # اگر کاربر قبلاً سوال مشابه پرسیده
            if user_context and user_context['total'] > 5:
                if user_context['main_topic'] == query_context['topic']:
                    answer += f"\n\n✨ شما {user_context['total']} بار درباره {query_context['topic']} سوال پرسیده‌اید!"
            
            return {
                'answer': answer,
                'context': query_context,
                'found': True
            }
        else:
            # ثبت سوال بی‌پاسخ
            self.memory.add_conversation(user_id, question, None, query_context)
            
            # پیشنهاد بر اساس تحلیل
            suggestion = ""
            if query_context['type'] != 'general':
                suggestion = f"\n\n💡 به نظر می‌رسد سوال شما از نوع '{query_context['type']}' است. می‌توانید واضح‌تر بپرسید."
            
            return {
                'answer': None,
                'context': query_context,
                'suggestion': suggestion,
                'found': False
            }
    
    def get_stats(self):
        """گرفتن آمار"""
        return {
            'knowledge': len(self.search.documents),
            'conversations': len(self.memory.conversations),
            'users': len(self.memory.user_memories),
            'patterns': len(self.memory.patterns)
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
    user_id = request.cookies.get('user_id')
    if not user_id:
        user_id = str(uuid.uuid4())
    
    # گرفتن تاریخچه کاربر برای زمینه
    user_context = ai.memory.get_user_context(user_id)
    
    resp = make_response(render_template_string('''
    <!DOCTYPE html>
    <html lang="fa" dir="rtl">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>هوش مصنوعی فوق پیشرفته</title>
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
            .header-title {
                font-size: 1.3em;
                font-weight: bold;
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
                <div class="header-title">🤖 هوش فوق پیشرفته</div>
                <div style="width:44px;"></div>
            </div>
            
            <div class="chat-messages" id="chat-messages">
                <div class="message bot">
                    <div class="message-content">
                        سلام! من هوش مصنوعی هستم. هر سوالی داری بپرس، من تو رو درک می‌کنم!
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
            <div style="margin-top:20px; font-size:0.9em; color:#666;">
                <p>📊 تعداد مکالمات: {{ user_total }}</p>
                <p>🎯 موضوع اصلی: {{ user_topic }}</p>
            </div>
        </div>
        
        <script>
            let chatHistory = JSON.parse(localStorage.getItem('chat_history')) || [];
            let userHistory = [];
            
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
                    userHistory.push({ text, isUser });
                    chatHistory.push({ text, isUser, time: msgTime });
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
                        body: JSON.stringify({
                            message,
                            history: userHistory.slice(-5)
                        })
                    });
                    
                    const data = await response.json();
                    hideTyping();
                    
                    if (data.answer) {
                        addMessage(data.answer);
                    } else {
                        let msg = '🤔 متأسفم! نتونستم پیدا کنم.';
                        if (data.suggestion) msg += data.suggestion;
                        addMessage(msg);
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
                    userHistory = [];
                    location.reload();
                }
            }
        </script>
    </body>
    </html>
    ''', now=datetime.now().strftime('%H:%M'), 
        user_total=user_context['total'], 
        user_topic=user_context['main_topic']))
    
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
        history = data.get('history', [])
        user_id = request.cookies.get('user_id')
        
        if not question:
            return jsonify({'error': 'سوال خالی است'})
        
        result = ai.ask(question, user_id, history)
        
        if result['found']:
            return jsonify({
                'answer': result['answer'],
                'found': True
            })
        else:
            return jsonify({
                'answer': None,
                'suggestion': result.get('suggestion', ''),
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
        textarea,input,select{{width:100%;padding:12px;margin:10px 0;border:2px solid #e0e0e0;border-radius:10px;}}
        button{{background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);color:white;padding:12px 25px;border:none;border-radius:10px;cursor:pointer;}}
    </style>
    </head>
    <body>
        <div class="header">
            <h2>⚙️ پنل مدیریت هوش فوق پیشرفته</h2>
            <div>
                <a href="/" style="color:white;margin-right:15px;">🏠 چت</a>
                <a href="/logout" style="color:white;">🚪 خروج</a>
            </div>
        </div>
        
        <div class="stats-grid">
            <div class="stat-card"><div class="stat-number">{stats['knowledge']}</div><div>دانش</div></div>
            <div class="stat-card"><div class="stat-number">{stats['conversations']}</div><div>مکالمات</div></div>
            <div class="stat-card"><div class="stat-number">{stats['users']}</div><div>کاربران</div></div>
            <div class="stat-card"><div class="stat-number">{stats['patterns']}</div><div>الگوها</div></div>
        </div>
        
        <div class="card">
            <h3>📝 آموزش</h3>
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
        
        <div class="card">
            <h3>📁 آپلود فایل</h3>
            <form action="/admin/learn/file" method="POST" enctype="multipart/form-data">
                <input type="file" name="file" accept=".txt" required>
                <button type="submit">📤 آپلود و یادگیری</button>
            </form>
        </div>
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
    ╔══════════════════════════════════════════════════════════════╗
    ║     🤖 هوش مصنوعی فوق پیشرفته با تحلیل عمیق متن             ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  📚 دانش: {}                                                    ║
    ║  💾 مکالمات: {}                                                 ║
    ║  👤 کاربران: {}                                                 ║
    ║  🔍 ۷ الگوریتم جستجو + تحلیل احساسات + تشخیص موجودیت        ║
    ║  🌐 چت: http://localhost:5000                                 ║
    ║  🔐 پنل: http://localhost:5000/admin-login                    ║
    ║  👤 کاربر: admin / admin123                                    ║
    ║  💡 حافظه نامحدود + یادگیری از مکالمات کاربران                ║
    ╚══════════════════════════════════════════════════════════════╝
    """.format(ai.get_stats()['knowledge'], ai.get_stats()['conversations'], ai.get_stats()['users']))
    
    app.run(debug=True, host='0.0.0.0', port=5000)
