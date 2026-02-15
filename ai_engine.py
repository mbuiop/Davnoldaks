# ai_engine.py - موتور اصلی هوش مصنوعی با قابلیت مقیاس‌پذیری بالا
# --------------------------------------------------------------

import json
import os
import re
import hashlib
import threading
import queue
import time
from datetime import datetime, timedelta
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import gc

# ================ پردازشگر متن فارسی ================
class PersianTextProcessor:
    """پردازشگر فوق سریع متن فارسی"""
    
    def __init__(self):
        self.alphabet = set('آابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهی')
        self.question_words = {'کیست', 'کی', 'کجاست', 'چیست', 'چرا', 'چطور', 'چگونه', 'کدام', 'آیا'}
        self.stop_words = {'است', 'بود', 'هست', 'می', 'که', 'را', 'با', 'از', 'به', 'برای', 'و', 'یا'}
        self.cache = {}
        self.cache_size = 10000
        
    def normalize(self, text):
        """نرمال‌سازی سریع با کش"""
        if not text:
            return ""
        
        # بررسی کش
        text_hash = hashlib.md5(text.encode()).hexdigest()
        if text_hash in self.cache:
            return self.cache[text_hash]
        
        # نرمال‌سازی
        text = text.replace('ي', 'ی').replace('ك', 'ک')
        text = re.sub(r'[^\w\s\u0600-\u06FF]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        # ذخیره در کش
        if len(self.cache) < self.cache_size:
            self.cache[text_hash] = text
        
        return text
    
    def tokenize(self, text):
        """تجزیه سریع"""
        return text.split()
    
    def extract_features(self, text):
        """استخراج ویژگی با حداقل هزینه"""
        text = self.normalize(text)
        words = self.tokenize(text)
        
        return {
            'word_count': len(words),
            'words': words[:10],  # فقط ۱۰ کلمه اول
            'has_question': any(w in self.question_words for w in words)
        }

# ================ کش هوشمند ================
class SmartCache:
    """کش هوشمند با قابلیت حذف خودکار"""
    
    def __init__(self, max_size=10000, ttl=3600):
        self.max_size = max_size
        self.ttl = ttl
        self.cache = {}
        self.timestamps = {}
        self.lock = threading.Lock()
    
    def get(self, key):
        with self.lock:
            if key in self.cache:
                if time.time() - self.timestamps[key] < self.ttl:
                    return self.cache[key]
                else:
                    del self.cache[key]
                    del self.timestamps[key]
        return None
    
    def set(self, key, value):
        with self.lock:
            if len(self.cache) >= self.max_size:
                # حذف قدیمی‌ترین
                oldest = min(self.timestamps.items(), key=lambda x: x[1])
                del self.cache[oldest[0]]
                del self.timestamps[oldest[0]]
            
            self.cache[key] = value
            self.timestamps[key] = time.time()
    
    def clear(self):
        with self.lock:
            self.cache.clear()
            self.timestamps.clear()

# ================ موتور جستجوی برداری ================
class VectorSearchEngine:
    """موتور جستجوی برداری بهینه شده"""
    
    def __init__(self):
        self.vectorizer = HashingVectorizer(
            n_features=2**16,  # 65536 ویژگی
            ngram_range=(1, 3),
            norm='l2',
            alternate_sign=False
        )
        self.vectors = None
        self.documents = []
        self.doc_ids = []
        self.lock = threading.RLock()
        self.update_queue = queue.Queue()
        self.is_updating = False
    
    def add_documents(self, documents):
        """افزودن اسناد جدید"""
        with self.lock:
            start_idx = len(self.documents)
            for doc in documents:
                self.documents.append(doc['question'])
                self.doc_ids.append(doc['id'])
            
            # به‌روزرسانی بردارها
            if len(self.documents) > 0:
                texts = [d['question'] for d in self.documents[start_idx:]]
                new_vectors = self.vectorizer.transform(texts)
                
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
            
            query_vector = self.vectorizer.transform([query])
            similarities = cosine_similarity(query_vector, self.vectors)[0]
            
            # گرفتن top_k
            top_indices = np.argpartition(similarities, -top_k)[-top_k:]
            results = []
            
            for idx in top_indices:
                if similarities[idx] > 0.1:
                    results.append({
                        'id': self.doc_ids[idx],
                        'score': float(similarities[idx])
                    })
            
            return sorted(results, key=lambda x: x['score'], reverse=True)

# ================ موتور جستجوی کلمات کلیدی ================
class KeywordSearchEngine:
    """موتور جستجوی کلمات کلیدی با ایندکس معکوس"""
    
    def __init__(self):
        self.inverted_index = defaultdict(set)
        self.documents = {}
        self.lock = threading.RLock()
    
    def add_document(self, doc_id, text):
        """افزودن سند به ایندکس"""
        with self.lock:
            self.documents[doc_id] = text
            words = set(text.split())
            for word in words:
                self.inverted_index[word].add(doc_id)
    
    def search(self, query, threshold=0.3):
        """جستجوی سریع با ایندکس معکوس"""
        words = set(query.split())
        if not words:
            return []
        
        scores = defaultdict(float)
        
        for word in words:
            for doc_id in self.inverted_index.get(word, []):
                scores[doc_id] += 1.0
        
        if not scores:
            return []
        
        max_score = max(scores.values())
        results = []
        
        for doc_id, score in scores.items():
            norm_score = score / len(words)
            if norm_score >= threshold:
                results.append({
                    'id': doc_id,
                    'score': norm_score
                })
        
        return sorted(results, key=lambda x: x['score'], reverse=True)

# ================ موتور یادگیری ================
class LearningEngine:
    """موتور یادگیری با قابلیت مقیاس‌پذیری"""
    
    def __init__(self):
        self.vector_engine = VectorSearchEngine()
        self.keyword_engine = KeywordSearchEngine()
        self.processor = PersianTextProcessor()
        self.knowledge_base = []
        self.user_questions = []
        self.learning_stats = {
            'total_learned': 0,
            'total_asked': 0,
            'success_rate': 0
        }
        self.lock = threading.RLock()
        self.cache = SmartCache(max_size=5000, ttl=3600)
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    def learn(self, question, answer, source='manual'):
        """یادگیری یک دانش جدید"""
        with self.lock:
            # نرمال‌سازی
            question_norm = self.processor.normalize(question)
            
            # بررسی تکراری
            for item in self.knowledge_base:
                if item['question'] == question_norm:
                    item['answer'] = answer
                    item['learn_count'] = item.get('learn_count', 1) + 1
                    item['updated'] = datetime.now().isoformat()
                    return True, "به‌روزرسانی شد"
            
            # افزودن جدید
            doc_id = len(self.knowledge_base) + 1
            new_item = {
                'id': doc_id,
                'question': question_norm,
                'answer': answer,
                'source': source,
                'created': datetime.now().isoformat(),
                'updated': datetime.now().isoformat(),
                'used_count': 0,
                'learn_count': 1,
                'success_count': 0
            }
            
            self.knowledge_base.append(new_item)
            
            # افزودن به موتورهای جستجو
            self.vector_engine.add_documents([new_item])
            self.keyword_engine.add_document(doc_id, question_norm)
            
            self.learning_stats['total_learned'] += 1
            
            return True, "یاد گرفته شد"
    
    def bulk_learn(self, texts, source='bulk'):
        """یادگیری گروهی با پردازش موازی"""
        documents = []
        count = 0
        
        for line in texts.strip().split('\n'):
            if '|' in line:
                parts = line.split('|', 1)
                if len(parts) == 2:
                    q = self.processor.normalize(parts[0].strip())
                    a = parts[1].strip()
                    
                    doc_id = len(self.knowledge_base) + count + 1
                    documents.append({
                        'id': doc_id,
                        'question': q,
                        'answer': a,
                        'source': source
                    })
                    count += 1
        
        # افزودن یکجا
        with self.lock:
            for doc in documents:
                doc['created'] = datetime.now().isoformat()
                doc['used_count'] = 0
                doc['learn_count'] = 1
                self.knowledge_base.append(doc)
            
            # به‌روزرسانی موتورهای جستجو
            self.vector_engine.add_documents(documents)
            for doc in documents:
                self.keyword_engine.add_document(doc['id'], doc['question'])
            
            self.learning_stats['total_learned'] += count
        
        return count, []
    
    def search(self, query, threshold=0.2):
        """جستجوی ترکیبی با ۳ الگوریتم"""
        # بررسی کش
        cache_key = hashlib.md5(query.encode()).hexdigest()
        cached = self.cache.get(cache_key)
        if cached:
            return cached
        
        query_norm = self.processor.normalize(query)
        results = []
        seen_ids = set()
        
        # الگوریتم ۱: جستجوی برداری
        vector_results = self.vector_engine.search(query_norm, top_k=10)
        for r in vector_results:
            if r['id'] not in seen_ids:
                item = self.knowledge_base[r['id'] - 1]
                results.append({
                    'id': r['id'],
                    'answer': item['answer'],
                    'score': r['score'] * 1.2,
                    'method': 'vector'
                })
                seen_ids.add(r['id'])
        
        # الگوریتم ۲: جستجوی کلمات کلیدی
        keyword_results = self.keyword_engine.search(query_norm, threshold=0.3)
        for r in keyword_results:
            if r['id'] not in seen_ids:
                item = self.knowledge_base[r['id'] - 1]
                results.append({
                    'id': r['id'],
                    'answer': item['answer'],
                    'score': r['score'],
                    'method': 'keyword'
                })
                seen_ids.add(r['id'])
        
        # الگوریتم ۳: تطابق دقیق
        for item in self.knowledge_base:
            if item['question'] == query_norm and item['id'] not in seen_ids:
                results.append({
                    'id': item['id'],
                    'answer': item['answer'],
                    'score': 1.0,
                    'method': 'exact'
                })
                seen_ids.add(item['id'])
                break
        
        # مرتب‌سازی
        results = sorted(results, key=lambda x: x['score'], reverse=True)[:3]
        
        # ذخیره در کش
        if results:
            self.cache.set(cache_key, results)
        
        return results
    
    def ask(self, query):
        """پرسش و پاسخ"""
        self.learning_stats['total_asked'] += 1
        
        results = self.search(query)
        
        if results:
            best = results[0]
            
            # به‌روزرسانی آمار
            with self.lock:
                item = self.knowledge_base[best['id'] - 1]
                item['used_count'] += 1
                item['success_count'] = item.get('success_count', 0) + 1
                item['last_used'] = datetime.now().isoformat()
            
            return {
                'answer': best['answer'],
                'score': best['score'],
                'method': best['method'],
                'found': True
            }
        
        return {'answer': None, 'found': False}
    
    def record_user_question(self, question):
        """ثبت سوال کاربر برای یادگیری آینده"""
        if len(self.user_questions) > 100000:  # حداکثر ۱۰۰ هزار
            self.user_questions = self.user_questions[-50000:]
        
        self.user_questions.append({
            'question': question,
            'time': datetime.now().isoformat(),
            'asked_count': 1
        })
    
    def get_popular_questions(self, limit=100):
        """گرفتن سوالات پرتکرار کاربران"""
        counter = Counter()
        for q in self.user_questions:
            counter[q['question']] += 1
        
        return counter.most_common(limit)
    
    def get_stats(self):
        """آمار سریع"""
        return {
            'knowledge_count': len(self.knowledge_base),
            'total_learned': self.learning_stats['total_learned'],
            'total_asked': self.learning_stats['total_asked'],
            'user_questions': len(self.user_questions),
            'cache_size': len(self.cache.cache)
        }

# ================ مدیریت پایگاه داده ================
class DatabaseManager:
    """مدیریت پایگاه داده با قابلیت شاردینگ"""
    
    def __init__(self, base_dir='data'):
        self.base_dir = base_dir
        self.shards = {}
        self.current_shard = 0
        self.max_shard_size = 10000
        os.makedirs(base_dir, exist_ok=True)
        self.load_all()
    
    def get_shard(self, doc_id):
        """تعیین شارد بر اساس ID"""
        shard_num = doc_id // self.max_shard_size
        return f"shard_{shard_num}.json"
    
    def save_document(self, doc):
        """ذخیره سند در شارد مناسب"""
        shard_file = self.get_shard(doc['id'])
        shard_path = os.path.join(self.base_dir, shard_file)
        
        # بارگذاری شارد
        if shard_file not in self.shards:
            if os.path.exists(shard_path):
                with open(shard_path, 'r', encoding='utf-8') as f:
                    self.shards[shard_file] = json.load(f)
            else:
                self.shards[shard_file] = []
        
        # افزودن یا به‌روزرسانی
        found = False
        for i, item in enumerate(self.shards[shard_file]):
            if item['id'] == doc['id']:
                self.shards[shard_file][i] = doc
                found = True
                break
        
        if not found:
            self.shards[shard_file].append(doc)
        
        # ذخیره
        with open(shard_path, 'w', encoding='utf-8') as f:
            json.dump(self.shards[shard_file], f, ensure_ascii=False, indent=2)
    
    def load_all(self):
        """بارگذاری همه شاردها"""
        for f in os.listdir(self.base_dir):
            if f.startswith('shard_') and f.endswith('.json'):
                with open(os.path.join(self.base_dir, f), 'r', encoding='utf-8') as f:
                    self.shards[f] = json.load(f)

# ================ کلاس اصلی هوش مصنوعی ================
class ScalablePersianAI:
    """هوش مصنوعی مقیاس‌پذیر برای میلیون‌ها کاربر"""
    
    def __init__(self):
        self.engine = LearningEngine()
        self.db = DatabaseManager()
        self.processor = PersianTextProcessor()
        self.stats = {
            'start_time': time.time(),
            'total_requests': 0,
            'avg_response_time': 0
        }
        
        # بارگذاری دانش
        self.load_knowledge()
    
    def load_knowledge(self):
        """بارگذاری دانش از شاردها"""
        for shard_file, items in self.db.shards.items():
            for item in items:
                self.engine.knowledge_base.append(item)
                self.engine.keyword_engine.add_document(item['id'], item['question'])
        
        if self.engine.knowledge_base:
            self.engine.vector_engine.add_documents(self.engine.knowledge_base)
        
        print(f"📚 {len(self.engine.knowledge_base)} دانش بارگذاری شد")
    
    def ask(self, question):
        """پرسش و پاسخ با اندازه‌گیری زمان"""
        start = time.time()
        
        # ثبت سوال
        self.engine.record_user_question(question)
        
        # جستجو
        result = self.engine.ask(question)
        
        # آمار
        self.stats['total_requests'] += 1
        response_time = time.time() - start
        self.stats['avg_response_time'] = (
            self.stats['avg_response_time'] * 0.95 + response_time * 0.05
        )
        
        return result
    
    def learn(self, text, source='manual'):
        """یادگیری با ذخیره در پایگاه داده"""
        learned, errors = self.engine.bulk_learn(text, source)
        
        # ذخیره دانش جدید
        for i in range(len(self.engine.knowledge_base) - learned, len(self.engine.knowledge_base)):
            self.db.save_document(self.engine.knowledge_base[i])
        
        return learned, errors
    
    def get_stats(self):
        """آمار کامل"""
        engine_stats = self.engine.get_stats()
        return {
            **engine_stats,
            'uptime': time.time() - self.stats['start_time'],
            'total_requests': self.stats['total_requests'],
            'avg_response_ms': self.stats['avg_response_time'] * 1000,
            'requests_per_second': self.stats['total_requests'] / (time.time() - self.stats['start_time'])
        }
    
    def get_popular_questions(self, limit=10):
        """گرفتن سوالات پرتکرار"""
        return self.engine.get_popular_questions(limit)
