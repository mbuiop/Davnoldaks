# ultimate_ai_bot.py
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

app = Flask(__name__)
app.config['SECRET_KEY'] = 'ultra-secret-ai-bot-key-2024'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=365)
app.config['SESSION_REFRESH_EACH_REQUEST'] = True
app.config['SESSION_PERMANENT'] = True
CORS(app)

# ایجاد پوشه‌ها
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('data', exist_ok=True)
os.makedirs('backup', exist_ok=True)
os.makedirs('logs', exist_ok=True)

# ================ دیتابیس دائمی با بکاپ خودکار ================
class PermanentDatabase:
    def __init__(self, filename='data/permanent_db.json'):
        self.filename = filename
        self.backup_dir = 'backup'
        self.data = {
            'knowledge_base': [],
            'users_questions': [],
            'alphabet': 'آ ا ب پ ت ث ج چ ح خ د ذ ر ز ژ س ش ص ض ط ظ ع غ ف ق ک گ ل م ن و ه ی'.split(),
            'patterns': {},
            'conversations': [],
            'stats': {
                'total_questions': 0,
                'answered': 0,
                'unanswered': 0,
                'last_backup': None
            }
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
                print(f"📊 {self.data['stats']['total_questions']} سوال تاکنون ثبت شده")
            except:
                print("⚠️ خطا در بارگذاری، ایجاد دیتابیس جدید")
                self.save()
        else:
            self.save()
    
    def save(self):
        """ذخیره دائمی داده‌ها"""
        # ایجاد بکاپ خودکار هر 100 تغییر
        if len(self.data['knowledge_base']) % 100 == 0:
            self.create_backup()
        
        with open(self.filename, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
    
    def create_backup(self):
        """ایجاد بکاپ با timestamp"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_file = f"{self.backup_dir}/backup_{timestamp}.json"
        with open(backup_file, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
        self.data['stats']['last_backup'] = timestamp
        print(f"📦 بکاپ خودکار ایجاد شد: {backup_file}")
    
    def auto_backup(self):
        """بکاپ خودکار هر 24 ساعت"""
        last = self.data['stats'].get('last_backup')
        if last:
            try:
                last_time = datetime.strptime(last, '%Y%m%d_%H%M%S')
                if (datetime.now() - last_time).days >= 1:
                    self.create_backup()
            except:
                self.create_backup()
        else:
            self.create_backup()

# ================ مدیریت کاربران ================
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'admin_login'

class User(UserMixin):
    def __init__(self, id, username, password, role='admin'):
        self.id = id
        self.username = username
        self.password = password
        self.role = role

# کاربران پیش‌فرض
users = {
    '1': User('1', 'admin', hashlib.md5('admin123'.encode()).hexdigest(), 'admin'),
}

@login_manager.user_loader
def load_user(user_id):
    return users.get(user_id)

# ================ هسته اصلی هوش مصنوعی ================
class UltimateAI:
    def __init__(self):
        self.db = PermanentDatabase()
        self.knowledge_base = self.db.data['knowledge_base']
        self.users_questions = self.db.data['users_questions']
        self.alphabet = self.db.data['alphabet']
        self.patterns = self.db.data['patterns']
        self.conversations = self.db.data['conversations']
        self.stats = self.db.data['stats']
        
        # سیستم‌های تشخیص پیشرفته
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 5),
            analyzer='char_wb',
            sublinear_tf=True,
            use_idf=True,
            smooth_idf=True
        )
        self.question_vectors = None
        self.context_vectors = {}
        self.word_patterns = {}
        self.initialize_systems()
    
    def initialize_systems(self):
        """راه‌اندازی سیستم‌های هوشمند"""
        self.build_alphabet_knowledge()
        self.build_patterns()
        self.update_vectors()
    
    def build_alphabet_knowledge(self):
        """آموزش حروف الفبا و قواعد"""
        if not any('حروف' in item['question'] for item in self.knowledge_base):
            # آموزش حروف الفبا
            for letter in self.alphabet:
                self.add_knowledge(
                    f"حرف {letter} چیست",
                    f"حرف {letter} یکی از حروف الفبای فارسی است. مثال: {self.get_example_for_letter(letter)}",
                    "الفبا",
                    auto=True
                )
            
            # آموزش اتصال حروف
            connections = [
                ("اتصال حروف در فارسی", "در خط فارسی، حروف به هم متصل می‌شوند. هر حرف در ابتدا، وسط و انتها شکل متفاوتی دارد."),
                ("حرکت‌های فارسی", "حرکت‌ها شامل فتحه (ـَ), کسره (ـِ), ضمه (ـُ), تشدید (ـّ), سکون (ـْ) هستند."),
                ("نشانه‌های فارسی", "نشانه‌های فارسی شامل: تنوین (ـً), انواع الف (ا, آ), همزه (ء), تاء تأنیث (ة) هستند.")
            ]
            for q, a in connections:
                self.add_knowledge(q, a, "الفبا", auto=True)
    
    def get_example_for_letter(self, letter):
        """مثال برای هر حرف"""
        examples = {
            'آ': 'آب', 'ا': 'ابر', 'ب': 'باران', 'پ': 'پدر', 'ت': 'تاریخ',
            'ث': 'ثروت', 'ج': 'جنگ', 'چ': 'چشم', 'ح': 'حرف', 'خ': 'خورشید',
            'د': 'دوست', 'ذ': 'ذهن', 'ر': 'روز', 'ز': 'زمین', 'ژ': 'ژرفا',
            'س': 'سوال', 'ش': 'شب', 'ص': 'صبح', 'ض': 'ضرورت', 'ط': 'طبیعت',
            'ظ': 'ظرف', 'ع': 'علم', 'غ': 'غروب', 'ف': 'فکر', 'ق': 'قلم',
            'ک': 'کتاب', 'گ': 'گل', 'ل': 'لبخند', 'م': 'ماه', 'ن': 'نور',
            'و': 'وطن', 'ه': 'هوا', 'ی': 'یاد'
        }
        return examples.get(letter, f"کلمه‌ای با حرف {letter}")
    
    def build_patterns(self):
        """ساخت الگوهای زبانی"""
        patterns = {
            'question_start': ['کیست', 'کجاست', 'چیست', 'کی', 'چرا', 'چطور', 'چگونه', 'کدام', 'آیا'],
            'question_end': ['است؟', 'هست؟', 'می‌شود؟', 'دارد؟', 'باشند؟'],
            'connectors': ['و', 'یا', 'اما', 'ولی', 'زیرا', 'چون', 'اگر', 'که'],
            'time_words': ['امروز', 'دیروز', 'فردا', 'سال', 'ماه', 'هفته', 'قرن', 'دوره'],
            'place_words': ['اینجا', 'آنجا', 'کجا', 'شهر', 'کشور', 'منطقه', 'محل'],
            'person_words': ['کسی', 'شخص', 'فرد', 'انسان', 'مرد', 'زن', 'بچه']
        }
        
        for key, value in patterns.items():
            if key not in self.patterns:
                self.patterns[key] = value
        
        self.db.data['patterns'] = self.patterns
        self.db.save()
    
    def advanced_preprocess(self, text):
        """پیش‌پردازش فوق پیشرفته"""
        if not text:
            return ""
        
        # حذف علائم نگارشی
        text = re.sub(r'[^\w\s\u0600-\u06FF]', ' ', text)
        
        # نرمال‌سازی کامل
        text = text.replace('ي', 'ی').replace('ك', 'ک')
        text = text.replace('ة', 'ه').replace('ؤ', 'و').replace('ئ', 'ی')
        text = text.replace('إ', 'ا').replace('أ', 'ا').replace('آ', 'ا')
        
        # حذف فاصله‌های اضافی
        text = re.sub(r'\s+', ' ', text).strip()
        
        # حذف کلمات خیلی کوتاه
        words = text.split()
        words = [w for w in words if len(w) > 1]
        
        return ' '.join(words)
    
    def extract_features(self, text):
        """استخراج ویژگی‌های پیشرفته"""
        features = {}
        words = text.split()
        
        # ویژگی‌های پایه
        features['word_count'] = len(words)
        features['char_count'] = len(text)
        features['unique_words'] = len(set(words))
        
        # تشخیص نوع کلمات
        features['has_question'] = any(w in text for w in self.patterns['question_start'])
        features['has_time'] = any(w in text for w in self.patterns['time_words'])
        features['has_place'] = any(w in text for w in self.patterns['place_words'])
        features['has_person'] = any(w in text for w in self.patterns['person_words'])
        
        # ویژگی‌های n-gram
        features['bigrams'] = [' '.join(words[i:i+2]) for i in range(len(words)-1)]
        features['trigrams'] = [' '.join(words[i:i+3]) for i in range(len(words)-2)]
        
        return features
    
    def calculate_similarity(self, text1, text2):
        """محاسبه شباهت با روش‌های مختلف"""
        # روش 1: شباهت کسینوسی
        try:
            vec1 = self.vectorizer.transform([text1])
            vec2 = self.vectorizer.transform([text2])
            cos_sim = cosine_similarity(vec1, vec2)[0][0]
        except:
            cos_sim = 0
        
        # روش 2: شباهت جاکارد
        set1 = set(text1.split())
        set2 = set(text2.split())
        if set1 and set2:
            jaccard = len(set1 & set2) / len(set1 | set2)
        else:
            jaccard = 0
        
        # روش 3: شباهت ترتیبی
        words1 = text1.split()
        words2 = text2.split()
        seq_score = 0
        if words1 and words2:
            common_seq = 0
            for i, w1 in enumerate(words1):
                for j, w2 in enumerate(words2):
                    if w1 == w2 and abs(i - j) <= 2:
                        common_seq += 1
            seq_score = common_seq / max(len(words1), len(words2))
        
        # ترکیب وزنی
        final_score = (cos_sim * 0.5) + (jaccard * 0.3) + (seq_score * 0.2)
        
        return final_score
    
    def semantic_search(self, query, threshold=0.2):
        """جستجوی معنایی پیشرفته"""
        query = self.advanced_preprocess(query)
        results = []
        
        for item in self.knowledge_base:
            # محاسبه شباهت با روش‌های مختلف
            sim_score = self.calculate_similarity(query, item['question'])
            
            if sim_score >= threshold:
                # بهبود امتیاز بر اساس دسته‌بندی
                if 'category' in item:
                    if item['category'] in query:
                        sim_score *= 1.2
                
                results.append({
                    'id': item['id'],
                    'question': item['question'],
                    'answer': item['answer'],
                    'score': sim_score,
                    'category': item.get('category', 'عمومی'),
                    'method': 'semantic'
                })
        
        return sorted(results, key=lambda x: x['score'], reverse=True)
    
    def context_search(self, query):
        """جستجوی زمینه‌ای با درک مفهوم"""
        results = []
        words = set(query.split())
        
        for item in self.knowledge_base:
            item_words = set(item['question'].split())
            
            # تشخیص موضوع اصلی
            common = words & item_words
            
            if common:
                # محاسبه امتیاز زمینه‌ای
                context_score = len(common) / max(len(words), len(item_words))
                
                # افزایش امتیاز برای تطابق کامل
                if query == item['question']:
                    context_score = 1.0
                
                results.append({
                    'id': item['id'],
                    'answer': item['answer'],
                    'score': context_score * 1.2,
                    'context_words': list(common),
                    'method': 'context'
                })
        
        return sorted(results, key=lambda x: x['score'], reverse=True)
    
    def pattern_search(self, query):
        """جستجوی مبتنی بر الگو"""
        results = []
        query_type = self.detect_question_type(query)
        
        for item in self.knowledge_base:
            item_type = self.detect_question_type(item['question'])
            
            if query_type == item_type:
                # شباهت در نوع سوال
                pattern_score = 0.5
                
                # بررسی الگوهای مشترک
                for pattern in self.patterns['question_start']:
                    if pattern in query and pattern in item['question']:
                        pattern_score += 0.3
                        break
                
                results.append({
                    'id': item['id'],
                    'answer': item['answer'],
                    'score': pattern_score,
                    'type': query_type,
                    'method': 'pattern'
                })
        
        return sorted(results, key=lambda x: x['score'], reverse=True)
    
    def detect_question_type(self, text):
        """تشخیص نوع سوال"""
        text = text.lower()
        
        types = {
            'person': r'(کیست|که بود|بیوگرافی|زندگینامه|چه کسی|افراد)',
            'place': r'(کجاست|مکان|موقعیت|شهر|کشور|استان|کجا)',
            'time': r'(کی|چه زمانی|تاریخ|سال|قرن|دوره|میلادی|شمسی)',
            'reason': r'(چرا|دلیل|علت|چگونه|چطور|به چه دلیل)',
            'definition': r'(چیست|چه بود|تعریف|توضیح|معنی|مفهوم)',
            'quantity': r'(چند|تعداد|مقدار|چه قدر)',
            'comparison': r'(فرق|تفاوت|شباهت|مقایسه)',
            'alphabet': r'(حرف|الفبا|نوشتن|املا|خواندن)'
        }
        
        for q_type, pattern in types.items():
            if re.search(pattern, text):
                return q_type
        
        return 'general'
    
    def ensemble_search(self, query):
        """ترکیب همه روش‌های جستجو"""
        # جستجو با روش‌های مختلف
        semantic_results = self.semantic_search(query)
        context_results = self.context_search(query)
        pattern_results = self.pattern_search(query)
        
        # ترکیب نتایج
        combined = {}
        
        for results, weight in [(semantic_results, 1.0), (context_results, 1.2), (pattern_results, 0.8)]:
            for r in results:
                rid = r['id']
                if rid not in combined or r['score'] * weight > combined[rid]['score']:
                    r['score'] = r['score'] * weight
                    combined[rid] = r
        
        final_results = sorted(combined.values(), key=lambda x: x['score'], reverse=True)
        
        # تحلیل کیفیت
        quality = self.analyze_quality(final_results)
        
        return final_results, quality
    
    def analyze_quality(self, results):
        """تحلیل کیفیت نتایج"""
        if not results:
            return 'none'
        
        best_score = results[0]['score']
        
        if best_score >= 0.8:
            return 'excellent'
        elif best_score >= 0.6:
            return 'good'
        elif best_score >= 0.4:
            return 'fair'
        elif best_score >= 0.2:
            return 'poor'
        else:
            return 'very_poor'
    
    def process_query(self, query):
        """پردازش نهایی query کاربر"""
        query = query.strip()
        
        # ثبت سوال کاربر
        self.record_user_question(query)
        
        # جستجو
        results, quality = self.ensemble_search(query)
        
        # تحلیل پیچیدگی
        complexity = self.analyze_complexity(query)
        
        return {
            'query': query,
            'results': results,
            'quality': quality,
            'complexity': complexity,
            'type': self.detect_question_type(query)
        }
    
    def analyze_complexity(self, query):
        """تحلیل پیچیدگی سوال"""
        words = query.split()
        
        score = 0
        
        # طول سوال
        if len(words) > 10:
            score += 2
        elif len(words) > 5:
            score += 1
        
        # کلمات پیچیده
        complex_words = ['چرا', 'دلیل', 'تأثیر', 'نتیجه', 'مقایسه', 'فرق', 'شباهت']
        for word in complex_words:
            if word in query:
                score += 1
        
        # سوالات چندبخشی
        if 'و' in query:
            score += 0.5
        
        if score >= 3:
            return 'high'
        elif score >= 2:
            return 'medium'
        else:
            return 'low'
    
    def record_user_question(self, question):
        """ثبت سوال کاربر برای بهبود الگوریتم"""
        self.stats['total_questions'] += 1
        
        record = {
            'id': len(self.users_questions) + 1,
            'question': question,
            'timestamp': datetime.now().isoformat(),
            'type': self.detect_question_type(question),
            'complexity': self.analyze_complexity(question),
            'answered': False,
            'ip_hash': hashlib.md5(str(random.random()).encode()).hexdigest()[:8]
        }
        
        self.users_questions.append(record)
        self.stats['unanswered'] += 1
        
        # نگهداری فقط 1000 رکورد آخر
        if len(self.users_questions) > 1000:
            self.users_questions = self.users_questions[-1000:]
        
        self.db.data['users_questions'] = self.users_questions
        self.db.data['stats'] = self.stats
        self.db.save()
    
    def record_answer(self, question_id):
        """ثبت پاسخ‌دهی موفق"""
        for q in self.users_questions:
            if q['id'] == question_id:
                q['answered'] = True
                q['answered_at'] = datetime.now().isoformat()
                break
        
        self.stats['answered'] += 1
        self.stats['unanswered'] -= 1
        self.db.save()
    
    def add_knowledge(self, question, answer, category='عمومی', auto=False):
        """اضافه کردن دانش جدید"""
        # بررسی تکراری نبودن
        for item in self.knowledge_base:
            if self.calculate_similarity(question, item['question']) > 0.9:
                return False, "این سوال مشابه قبلاً ثبت شده است"
        
        new_item = {
            'id': len(self.knowledge_base) + 1,
            'question': self.advanced_preprocess(question),
            'original_question': question,
            'answer': answer,
            'category': category,
            'date_added': datetime.now().isoformat(),
            'times_used': 0,
            'last_used': None,
            'success_rate': 0,
            'auto_generated': auto
        }
        
        self.knowledge_base.append(new_item)
        self.update_vectors()
        self.db.data['knowledge_base'] = self.knowledge_base
        self.db.save()
        
        return True, "دانش با موفقیت اضافه شد"
    
    def update_vectors(self):
        """به‌روزرسانی بردارها"""
        if self.knowledge_base:
            questions = [item['question'] for item in self.knowledge_base]
            try:
                self.question_vectors = self.vectorizer.fit_transform(questions)
            except:
                self.question_vectors = None
    
    def bulk_import(self, text):
        """وارد کردن گروهی دانش"""
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
    
    def get_unanswered_questions(self):
        """گرفتن سوالات بی‌پاسخ برای پنل مدیریت"""
        unanswered = [q for q in self.users_questions if not q['answered']]
        return sorted(unanswered, key=lambda x: x['timestamp'], reverse=True)[:50]
    
    def get_stats(self):
        """گرفتن آمار کامل"""
        total = len(self.knowledge_base)
        if total == 0:
            return {}
        
        categories = Counter([item.get('category', 'عمومی') for item in self.knowledge_base])
        most_used = sorted(self.knowledge_base, key=lambda x: x.get('times_used', 0), reverse=True)[:10]
        
        # آمار سوالات کاربران
        questions_by_type = Counter([q['type'] for q in self.users_questions])
        questions_by_complexity = Counter([q['complexity'] for q in self.users_questions])
        
        return {
            'knowledge': {
                'total': total,
                'categories': dict(categories),
                'most_used': most_used,
                'auto_generated': len([i for i in self.knowledge_base if i.get('auto_generated')])
            },
            'users': {
                'total_questions': self.stats['total_questions'],
                'answered': self.stats['answered'],
                'unanswered': self.stats['unanswered'],
                'by_type': dict(questions_by_type),
                'by_complexity': dict(questions_by_complexity)
            },
            'alphabet': {
                'letters': len(self.alphabet),
                'patterns': len(self.patterns)
            }
        }

# ================ نمونه اصلی ================
ai = UltimateAI()

# ================ صفحات اصلی ================
@app.route('/')
def index():
    """صفحه اصلی چت - تمام صفحه حرفه‌ای"""
    return render_template_string('''
    <!DOCTYPE html>
    <html lang="fa" dir="rtl">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>هوش مصنوعی همه‌کاره</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: 'Vazir', 'Tahoma', sans-serif;
                background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
                height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
                overflow: hidden;
            }
            
            .chat-container {
                width: 95%;
                max-width: 1400px;
                height: 95vh;
                background: rgba(255, 255, 255, 0.95);
                backdrop-filter: blur(10px);
                border-radius: 40px;
                box-shadow: 0 30px 60px rgba(0,0,0,0.3);
                display: flex;
                flex-direction: column;
                overflow: hidden;
                position: relative;
            }
            
            .chat-header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 25px 35px;
                display: flex;
                align-items: center;
                justify-content: space-between;
                position: relative;
                z-index: 10;
            }
            
            .chat-header h1 {
                font-size: 2em;
                display: flex;
                align-items: center;
                gap: 15px;
            }
            
            .header-stats {
                background: rgba(255,255,255,0.2);
                padding: 10px 20px;
                border-radius: 30px;
                font-size: 0.9em;
                backdrop-filter: blur(5px);
            }
            
            .admin-link {
                color: white;
                text-decoration: none;
                padding: 10px 20px;
                border-radius: 30px;
                background: rgba(255,255,255,0.2);
                transition: all 0.3s;
                font-size: 1.1em;
                display: flex;
                align-items: center;
                gap: 5px;
            }
            
            .admin-link:hover {
                background: rgba(255,255,255,0.3);
                transform: scale(1.05);
            }
            
            .chat-messages {
                flex: 1;
                overflow-y: auto;
                padding: 30px;
                background: #f8fafc;
            }
            
            .message {
                display: flex;
                margin-bottom: 25px;
                animation: slideIn 0.3s ease;
            }
            
            @keyframes slideIn {
                from { opacity: 0; transform: translateY(20px); }
                to { opacity: 1; transform: translateY(0); }
            }
            
            .message.user {
                justify-content: flex-end;
            }
            
            .message.bot {
                justify-content: flex-start;
            }
            
            .message-content {
                max-width: 70%;
                padding: 18px 25px;
                border-radius: 30px;
                position: relative;
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);
                line-height: 1.7;
                font-size: 1.05em;
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
                font-size: 0.75em;
                opacity: 0.7;
                margin-top: 8px;
                text-align: left;
            }
            
            .message-meta {
                font-size: 0.7em;
                margin-top: 5px;
                display: flex;
                gap: 10px;
                color: #666;
            }
            
            .chat-input-container {
                padding: 25px 35px;
                background: white;
                border-top: 1px solid rgba(0,0,0,0.05);
                display: flex;
                gap: 15px;
                position: relative;
            }
            
            .chat-input {
                flex: 1;
                padding: 18px 25px;
                border: 2px solid #e0e0e0;
                border-radius: 50px;
                font-size: 1.1em;
                outline: none;
                transition: all 0.3s;
                font-family: 'Vazir', 'Tahoma', sans-serif;
                background: #f8fafc;
            }
            
            .chat-input:focus {
                border-color: #667eea;
                box-shadow: 0 0 0 4px rgba(102,126,234,0.1);
                background: white;
            }
            
            .send-btn {
                width: 70px;
                height: 70px;
                border-radius: 50%;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                cursor: pointer;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 1.8em;
                transition: all 0.3s;
            }
            
            .send-btn:hover {
                transform: scale(1.1) rotate(5deg);
                box-shadow: 0 10px 25px rgba(102,126,234,0.4);
            }
            
            .typing-indicator {
                padding: 18px 25px;
                background: white;
                border-radius: 50px;
                display: inline-block;
            }
            
            .typing-indicator span {
                display: inline-block;
                width: 10px;
                height: 10px;
                border-radius: 50%;
                background: #667eea;
                margin: 0 3px;
                animation: typing 1.4s infinite;
            }
            
            .typing-indicator span:nth-child(2) { animation-delay: 0.2s; }
            .typing-indicator span:nth-child(3) { animation-delay: 0.4s; }
            
            @keyframes typing {
                0%, 60%, 100% { transform: translateY(0); }
                30% { transform: translateY(-15px); }
            }
            
            .feedback-message {
                background: #fff3cd;
                color: #856404;
                padding: 15px 25px;
                border-radius: 15px;
                margin: 10px 0;
                font-size: 0.95em;
                border-right: 5px solid #ffc107;
            }
            
            .complexity-badge {
                display: inline-block;
                padding: 3px 8px;
                border-radius: 12px;
                font-size: 0.7em;
                margin-left: 5px;
            }
            
            .complexity-high { background: #dc3545; color: white; }
            .complexity-medium { background: #ffc107; color: #333; }
            .complexity-low { background: #28a745; color: white; }
            
            .welcome-message {
                text-align: center;
                padding: 30px;
                background: linear-gradient(135deg, #667eea10 0%, #764ba210 100%);
                border-radius: 20px;
                margin-bottom: 30px;
            }
            
            .welcome-message h2 {
                color: #333;
                margin-bottom: 15px;
                font-size: 2em;
            }
            
            .welcome-message p {
                color: #666;
                font-size: 1.2em;
            }
        </style>
    </head>
    <body>
        <div class="chat-container">
            <div class="chat-header">
                <h1>
                    <span>🤖 هوش مصنوعی همه‌کاره</span>
                </h1>
                <a href="/admin-login" class="admin-link">⚙️ ورود مدیر</a>
            </div>
            
            <div class="chat-messages" id="chat-messages">
                <div class="welcome-message">
                    <h2>🌟 به هوش مصنوعی همه‌کاره خوش آمدید</h2>
                    <p>می‌توانید هر سوالی بپرسید: تاریخ، حروف الفبا، مفاهیم علمی، هر چیزی!</p>
                    <p style="font-size: 0.9em; margin-top: 15px;">✨ هر سوال به بهبود من کمک می‌کند</p>
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
        
        <script>
            const messagesContainer = document.getElementById('chat-messages');
            const messageInput = document.getElementById('message-input');
            
            // لود تاریخچه از localStorage
            let chatHistory = JSON.parse(localStorage.getItem('chat_history')) || [];
            
            // نمایش تاریخچه
            chatHistory.forEach(msg => {
                addMessage(msg.text, msg.isUser, msg.time, false);
            });
            
            function addMessage(text, isUser = false, time = null, save = true) {
                const messageDiv = document.createElement('div');
                messageDiv.className = `message ${isUser ? 'user' : 'bot'}`;
                
                const messageTime = time || new Date().toLocaleTimeString('fa-IR', { 
                    hour: '2-digit', 
                    minute: '2-digit',
                    hour12: false
                });
                
                messageDiv.innerHTML = `
                    <div class="message-content">
                        ${text}
                        <div class="message-time">${messageTime}</div>
                    </div>
                `;
                
                messagesContainer.appendChild(messageDiv);
                messagesContainer.scrollTop = messagesContainer.scrollHeight;
                
                if (save) {
                    chatHistory.push({
                        text: text,
                        isUser: isUser,
                        time: messageTime
                    });
                    
                    // نگهداری 50 پیام آخر
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
                    
                    if (data.feedback) {
                        addMessage(data.feedback, false, null, true);
                    }
                    
                    if (data.answer) {
                        addMessage(data.answer, false, null, true);
                        
                        if (data.method) {
                            console.log('🎯 روش تشخیص:', data.method);
                        }
                    } else if (!data.feedback) {
                        addMessage('🤔 متأسفم! نتونستم جوابی پیدا کنم. این سوال برای مدیر ارسال شد.', false, null, true);
                    }
                    
                } catch (error) {
                    hideTyping();
                    addMessage('⚠️ خطا در ارتباط با سرور', false, null, true);
                    console.error(error);
                }
            }
        </script>
    </body>
    </html>
    ''')
    
@app.route('/api/chat', methods=['POST'])
def api_chat():
    """API چت با الگوریتم فوق پیشرفته"""
    data = request.json
    question = data.get('message', '').strip()
    
    if not question:
        return jsonify({'error': 'سوال نمی‌تواند خالی باشد'})
    
    # پردازش سوال
    result = ai.process_query(question)
    
    if result['results']:
        best = result['results'][0]
        
        # ثبت پاسخ موفق
        if result['quality'] in ['excellent', 'good']:
            ai.record_answer(best['id'])
        
        # تولید بازخورد
        feedback = None
        if result['quality'] == 'poor':
            feedback = "🔍 پاسخ با اطمینان متوسط پیدا شد."
        elif result['quality'] == 'very_poor':
            feedback = "💡 پاسخ با اطمینان پایین پیدا شد. سوال را واضح‌تر بپرسید."
        
        return jsonify({
            'answer': best['answer'],
            'confidence': best['score'],
            'method': best.get('method', 'unknown'),
            'quality': result['quality'],
            'found': True,
            'feedback': feedback
        })
    else:
        return jsonify({
            'answer': None,
            'found': False,
            'feedback': "📝 سوال شما ثبت شد. مدیر در اسرع وقت پاسخ را اضافه خواهد کرد."
        })

# ================ پنل مدیریت ================
@app.route('/admin-login', methods=['GET', 'POST'])
def admin_login():
    """صفحه لاگین مدیریت"""
    if request.method == 'POST':
        username = request.form['username']
        password = hashlib.md5(request.form['password'].encode()).hexdigest()
        
        for user in users.values():
            if user.username == username and user.password == password:
                login_user(user)
                session.permanent = True
                return redirect(url_for('admin_panel'))
                
        return "❌ نام کاربری یا رمز عبور اشتباه است"
    
    return render_template_string('''
    <!DOCTYPE html>
    <html>
    <head>
        <title>ورود به پنل مدیریت</title>
        <style>
            body {
                font-family: Tahoma;
                background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
                height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
            }
            .login-box {
                background: white;
                padding: 50px;
                border-radius: 30px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                width: 400px;
                animation: fadeIn 0.5s;
            }
            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(-20px); }
                to { opacity: 1; transform: translateY(0); }
            }
            h2 {
                text-align: center;
                color: #333;
                margin-bottom: 30px;
                font-size: 2em;
            }
            input {
                width: 100%;
                padding: 15px;
                margin: 15px 0;
                border: 2px solid #e0e0e0;
                border-radius: 15px;
                font-family: Tahoma;
                font-size: 1.1em;
                transition: all 0.3s;
            }
            input:focus {
                border-color: #667eea;
                outline: none;
                box-shadow: 0 0 0 3px rgba(102,126,234,0.1);
            }
            button {
                width: 100%;
                padding: 15px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                border-radius: 15px;
                cursor: pointer;
                font-size: 1.2em;
                transition: all 0.3s;
            }
            button:hover {
                transform: scale(1.02);
                box-shadow: 0 10px 20px rgba(102,126,234,0.3);
            }
        </style>
    </head>
    <body>
        <div class="login-box">
            <h2>🔐 پنل مدیریت</h2>
            <form method="POST">
                <input type="text" name="username" placeholder="نام کاربری" value="admin" required>
                <input type="password" name="password" placeholder="رمز عبور" value="admin123" required>
                <button type="submit">ورود به پنل</button>
            </form>
        </div>
    </body>
    </html>
    ''')

@app.route('/admin')
@login_required
def admin_panel():
    """پنل مدیریت اصلی"""
    stats = ai.get_stats()
    unanswered = ai.get_unanswered_questions()
    
    return render_template_string('''
    <!DOCTYPE html>
    <html lang="fa" dir="rtl">
    <head>
        <meta charset="UTF-8">
        <title>پنل مدیریت - هوش مصنوعی همه‌کاره</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body {
                font-family: Tahoma;
                background: linear-gradient(135deg, #667eea10 0%, #764ba210 100%);
                min-height: 100vh;
                padding: 20px;
            }
            
            .header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 30px;
                border-radius: 20px;
                margin-bottom: 30px;
                display: flex;
                justify-content: space-between;
                align-items: center;
                box-shadow: 0 10px 30px rgba(102,126,234,0.3);
            }
            
            .header h1 {
                font-size: 2.2em;
                display: flex;
                align-items: center;
                gap: 15px;
            }
            
            .nav-links {
                display: flex;
                gap: 15px;
            }
            
            .nav-links a {
                color: white;
                text-decoration: none;
                padding: 12px 25px;
                border-radius: 15px;
                background: rgba(255,255,255,0.2);
                transition: all 0.3s;
                font-size: 1.1em;
            }
            
            .nav-links a:hover {
                background: rgba(255,255,255,0.3);
                transform: translateY(-2px);
            }
            
            .stats-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
                gap: 25px;
                margin-bottom: 30px;
            }
            
            .stat-card {
                background: white;
                padding: 25px;
                border-radius: 20px;
                box-shadow: 0 5px 20px rgba(0,0,0,0.05);
                transition: all 0.3s;
                border: 1px solid rgba(102,126,234,0.1);
            }
            
            .stat-card:hover {
                transform: translateY(-5px);
                box-shadow: 0 15px 30px rgba(102,126,234,0.15);
            }
            
            .stat-number {
                font-size: 3em;
                font-weight: bold;
                background: linear-gradient(135deg, #667eea, #764ba2);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                margin-bottom: 10px;
            }
            
            .stat-label {
                color: #666;
                font-size: 1.2em;
            }
            
            .card {
                background: white;
                padding: 30px;
                border-radius: 20px;
                box-shadow: 0 5px 20px rgba(0,0,0,0.05);
                margin-bottom: 30px;
                border: 1px solid rgba(102,126,234,0.1);
            }
            
            .card h2 {
                margin-bottom: 25px;
                color: #333;
                font-size: 1.8em;
                border-bottom: 3px solid #667eea;
                padding-bottom: 15px;
            }
            
            textarea, input[type=text], select {
                width: 100%;
                padding: 15px;
                margin: 15px 0;
                border: 2px solid #e0e0e0;
                border-radius: 15px;
                font-family: Tahoma;
                font-size: 1.1em;
                transition: all 0.3s;
            }
            
            textarea:focus, input:focus {
                border-color: #667eea;
                outline: none;
                box-shadow: 0 0 0 3px rgba(102,126,234,0.1);
            }
            
            button {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 15px 35px;
                border: none;
                border-radius: 15px;
                cursor: pointer;
                font-size: 1.1em;
                transition: all 0.3s;
                margin: 10px 0;
            }
            
            button:hover {
                transform: scale(1.02);
                box-shadow: 0 10px 20px rgba(102,126,234,0.3);
            }
            
            .file-upload {
                border: 3px dashed #667eea;
                padding: 40px;
                text-align: center;
                border-radius: 20px;
                cursor: pointer;
                margin: 20px 0;
                background: #f8fafc;
                transition: all 0.3s;
            }
            
            .file-upload:hover {
                background: #f0f4ff;
                transform: scale(1.01);
            }
            
            .unanswered-item {
                background: #fff3cd;
                padding: 15px 20px;
                margin: 10px 0;
                border-radius: 15px;
                display: flex;
                justify-content: space-between;
                align-items: center;
                border-right: 5px solid #ffc107;
            }
            
            .unanswered-item button {
                padding: 8px 20px;
                margin: 0;
            }
            
            .grid-2 {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 30px;
            }
            
            .alphabet-grid {
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(80px, 1fr));
                gap: 15px;
                padding: 20px;
                background: #f8fafc;
                border-radius: 15px;
            }
            
            .alphabet-item {
                background: white;
                padding: 20px;
                text-align: center;
                border-radius: 15px;
                font-size: 1.8em;
                font-weight: bold;
                color: #667eea;
                box-shadow: 0 5px 15px rgba(0,0,0,0.05);
                border: 2px solid transparent;
                transition: all 0.3s;
            }
            
            .alphabet-item:hover {
                border-color: #667eea;
                transform: scale(1.1);
            }
            
            .question-list {
                max-height: 500px;
                overflow-y: auto;
                padding: 20px;
                background: #f8fafc;
                border-radius: 15px;
            }
            
            .question-item {
                background: white;
                padding: 15px;
                margin: 10px 0;
                border-radius: 12px;
                border-right: 3px solid #667eea;
                font-size: 1.1em;
            }
            
            .badge {
                display: inline-block;
                padding: 5px 12px;
                border-radius: 30px;
                font-size: 0.8em;
                margin-left: 10px;
            }
            
            .badge-unanswered { background: #ffc107; color: #333; }
            .badge-high { background: #dc3545; color: white; }
            .badge-medium { background: #fd7e14; color: white; }
            .badge-low { background: #28a745; color: white; }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>
                <span>⚙️ پنل مدیریت هوشمند</span>
            </h1>
            <div class="nav-links">
                <a href="/" target="_blank">🌐 صفحه چت</a>
                <a href="/logout">🚪 خروج</a>
            </div>
        </div>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-number">{{ stats.knowledge.total }}</div>
                <div class="stat-label">کل دانش</div>
                <div style="margin-top: 15px; color: #666;">
                    خودکار: {{ stats.knowledge.auto_generated }}
                </div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{{ stats.users.total_questions }}</div>
                <div class="stat-label">کل سوالات</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{{ stats.users.unanswered }}</div>
                <div class="stat-label">بی‌پاسخ</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{{ stats.alphabet.letters }}</div>
                <div class="stat-label">حروف الفبا</div>
            </div>
        </div>
        
        <div class="grid-2">
            <div class="card">
                <h2>📝 آموزش تکی</h2>
                <form action="/admin/add" method="POST">
                    <input type="text" name="question" placeholder="سوال" required>
                    <textarea name="answer" rows="4" placeholder="پاسخ" required></textarea>
                    <select name="category">
                        <option>تاریخ</option>
                        <option>الفبا</option>
                        <option>عمومی</option>
                        <option>علمی</option>
                        <option>ادبی</option>
                    </select>
                    <button type="submit">➕ افزودن دانش</button>
                </form>
            </div>
            
            <div class="card">
                <h2>📁 آپلود فایل آموزشی</h2>
                <form action="/admin/upload" method="POST" enctype="multipart/form-data">
                    <div class="file-upload" onclick="document.getElementById('file').click()">
                        <p style="font-size: 2em; margin-bottom: 10px;">📤</p>
                        <p style="font-size: 1.2em;">برای آپلود کلیک کنید</p>
                        <p style="color: #666; margin-top: 10px;">فرمت: هر خط: سوال | جواب</p>
                    </div>
                    <input type="file" id="file" name="file" style="display:none;" accept=".txt">
                    <button type="submit">📥 آپلود و پردازش</button>
                </form>
            </div>
        </div>
        
        <div class="grid-2">
            <div class="card">
                <h2>❓ سوالات بی‌پاسخ ({{ unanswered|length }})</h2>
                <div class="question-list">
                    {% for item in unanswered %}
                    <div class="unanswered-item">
                        <div>
                            <strong>{{ item.question }}</strong>
                            <div style="margin-top: 8px;">
                                <span class="badge badge-unanswered">پیچیدگی: {{ item.complexity }}</span>
                                <span class="badge" style="background: #6c757d; color: white;">{{ item.type }}</span>
                            </div>
                        </div>
                        <button onclick="answerQuestion('{{ item.question }}')">➕ پاسخ</button>
                    </div>
                    {% endfor %}
                </div>
            </div>
            
            <div class="card">
                <h2>🔤 حروف الفبا</h2>
                <div class="alphabet-grid">
                    {% for letter in stats.alphabet.letters %}
                    <div class="alphabet-item">{{ letter }}</div>
                    {% endfor %}
                </div>
            </div>
        </div>
        
        <div class="card">
            <h2>📊 آمار پیشرفته</h2>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 30px;">
                <div>
                    <h3>دسته‌بندی دانش</h3>
                    <ul style="list-style: none; padding: 20px;">
                    {% for cat, count in stats.knowledge.categories.items() %}
                        <li style="margin: 15px 0; display: flex; justify-content: space-between;">
                            <span>{{ cat }}</span>
                            <span style="background: #667eea; color: white; padding: 5px 15px; border-radius: 30px;">{{ count }}</span>
                        </li>
                    {% endfor %}
                    </ul>
                </div>
                
                <div>
                    <h3>پراستفاده‌ترین‌ها</h3>
                    {% for item in stats.knowledge.most_used[:5] %}
                    <div style="background: #f8fafc; padding: 15px; margin: 10px 0; border-radius: 12px;">
                        <div>{{ item.question }}</div>
                        <div style="color: #667eea; margin-top: 5px;">{{ item.times_used }} بار استفاده</div>
                    </div>
                    {% endfor %}
                </div>
            </div>
        </div>
        
        <script>
            function answerQuestion(question) {
                document.querySelector('[name="question"]').value = question;
                document.querySelector('[name="question"]').scrollIntoView({behavior: 'smooth'});
            }
        </script>
    </body>
    </html>
    ''', stats=stats, unanswered=unanswered)

@app.route('/admin/add', methods=['POST'])
@login_required
def admin_add():
    """افزودن دانش جدید"""
    question = request.form['question']
    answer = request.form['answer']
    category = request.form.get('category', 'عمومی')
    
    success, msg = ai.add_knowledge(question, answer, category)
    
    if success:
        return redirect(url_for('admin_panel'))
    else:
        return f"❌ {msg} <a href='/admin'>بازگشت</a>"

@app.route('/admin/upload', methods=['POST'])
@login_required
def admin_upload():
    """آپلود فایل آموزشی"""
    if 'file' not in request.files:
        return "❌ فایلی انتخاب نشده است"
        
    file = request.files['file']
    if file.filename == '':
        return "❌ نام فایل معتبر نیست"
        
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        count, errors = ai.bulk_import(open(filepath, 'r', encoding='utf-8').read())
        
        if errors:
            return f"✅ {count} مورد اضافه شد<br>❌ خطاها: " + "<br>".join(errors) + f" <a href='/admin'>بازگشت</a>"
        else:
            return f"✅ {count} مورد با موفقیت اضافه شد <a href='/admin'>بازگشت</a>"

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

# ================ اجرای برنامه ================
if __name__ == '__main__':
    print("""
    ╔═══════════════════════════════════════════════════════════════════╗
    ║     🤖 هوش مصنوعی همه‌کاره - نسخه نهایی فوق پیشرفته              ║
    ╠═══════════════════════════════════════════════════════════════════╣
    ║  📚 دانش: {} مورد                                                  ║
    ║  📊 سوالات ثبت شده: {}                                             ║
    ║  🔤 حروف الفبا: {} حرف                                             ║
    ║  🌐 صفحه چت: http://localhost:5000                                ║
    ║  🔐 پنل مدیریت: http://localhost:5000/admin-login                 ║
    ║  👤 کاربر: admin / رمز: admin123                                  ║
    ║  💾 ذخیره سازی: دائمی در فایل + localStorage مرورگر              ║
    ╚═══════════════════════════════════════════════════════════════════╝
    """.format(len(ai.knowledge_base), ai.stats['total_questions'], len(ai.alphabet)))
    
    app.run(debug=True, host='0.0.0.0', port=5000)
