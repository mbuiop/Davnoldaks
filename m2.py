# m2.py - Models
import json
import os
import hashlib
from datetime import datetime
from collections import Counter
import threading

class KnowledgeBase:
    """مدیریت دانش"""
    
    def __init__(self, config):
        self.config = config
        self.data = []
        self.lock = threading.Lock()
        self.load()
        
    def load(self):
        """بارگذاری دانش"""
        if os.path.exists(self.config.KNOWLEDGE_FILE):
            try:
                with open(self.config.KNOWLEDGE_FILE, 'r', encoding='utf-8') as f:
                    self.data = json.load(f)
                print(f"📚 {len(self.data)} دانش بارگذاری شد")
            except:
                self.data = []
        else:
            # دانش نمونه
            self.data = [
                {
                    "id": 1,
                    "question": "کوروش کبیر که بود",
                    "answer": "کوروش بزرگ بنیانگذار شاهنشاهی هخامنشی بود",
                    "category": "ایران باستان",
                    "keywords": ["کوروش", "هخامنشی"],
                    "times_used": 0,
                    "created_at": datetime.now().isoformat()
                },
                {
                    "id": 2,
                    "question": "داریوش چه کرد",
                    "answer": "داریوش بزرگ جاده شاهی را ساخت و امپراتوری را به ساتراپی‌ها تقسیم کرد",
                    "category": "ایران باستان",
                    "keywords": ["داریوش", "جاده شاهی"],
                    "times_used": 0,
                    "created_at": datetime.now().isoformat()
                }
            ]
            self.save()
            
    def save(self):
        """ذخیره دانش"""
        with open(self.config.KNOWLEDGE_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
            
    def add(self, question, answer, category='عمومی', source='manual'):
        """افزودن دانش جدید"""
        with self.lock:
            # بررسی تکراری نبودن
            for item in self.data:
                if item['question'].lower() == question.lower():
                    return False, "این سوال قبلاً ثبت شده است"
                    
            new_id = len(self.data) + 1
            keywords = self.extract_keywords(question + ' ' + answer)
            
            item = {
                "id": new_id,
                "question": question,
                "answer": answer,
                "category": category,
                "keywords": keywords,
                "source": source,
                "times_used": 0,
                "positive_feedback": 0,
                "negative_feedback": 0,
                "created_at": datetime.now().isoformat(),
                "last_used": None
            }
            
            self.data.append(item)
            self.save()
            return True, "دانش با موفقیت اضافه شد"
            
    def search(self, query, limit=5):
        """جستجوی ساده در دانش"""
        query = query.lower()
        results = []
        
        for item in self.data:
            score = 0
            if query in item['question'].lower():
                score = 1.0 if query == item['question'].lower() else 0.8
            elif query in item['answer'].lower():
                score = 0.6
            elif any(kw in query for kw in item.get('keywords', [])):
                score = 0.5
                
            if score > 0:
                results.append({
                    'id': item['id'],
                    'question': item['question'],
                    'answer': item['answer'],
                    'score': score,
                    'category': item['category']
                })
                
        # به‌روزرسانی آمار استفاده
        for r in results[:3]:
            for item in self.data:
                if item['id'] == r['id']:
                    item['times_used'] += 1
                    item['last_used'] = datetime.now().isoformat()
                    
        self.save()
        return sorted(results, key=lambda x: x['score'], reverse=True)[:limit]
        
    def extract_keywords(self, text, top_n=10):
        """استخراج کلمات کلیدی"""
        words = text.split()
        words = [w for w in words if len(w) > 2]
        word_freq = Counter(words)
        return [w for w, _ in word_freq.most_common(top_n)]
        
    def update_feedback(self, item_id, is_positive):
        """به‌روزرسانی بازخورد"""
        for item in self.data:
            if item['id'] == item_id:
                if is_positive:
                    item['positive_feedback'] += 1
                else:
                    item['negative_feedback'] += 1
                self.save()
                return True
        return False
        
    def get_stats(self):
        """دریافت آمار"""
        return {
            'total': len(self.data),
            'categories': Counter([item['category'] for item in self.data]),
            'total_used': sum(item['times_used'] for item in self.data),
            'avg_feedback': sum(item['positive_feedback'] for item in self.data) / max(len(self.data), 1)
        }

class UnansweredManager:
    """مدیریت سوالات بی‌پاسخ"""
    
    def __init__(self, config):
        self.config = config
        self.data = []
        self.load()
        
    def load(self):
        if os.path.exists(self.config.UNANSWERED_FILE):
            try:
                with open(self.config.UNANSWERED_FILE, 'r', encoding='utf-8') as f:
                    self.data = json.load(f)
            except:
                self.data = []
                
    def save(self):
        with open(self.config.UNANSWERED_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.data[-500:], f, ensure_ascii=False, indent=2)
            
    def add(self, question, user_id='anonymous'):
        """ثبت سوال بی‌پاسخ"""
        # بررسی تکراری بودن
        for item in self.data:
            if item['question'].lower() == question.lower():
                item['count'] += 1
                item['last_seen'] = datetime.now().isoformat()
                self.save()
                return
                
        self.data.append({
            'id': len(self.data) + 1,
            'question': question,
            'user_id': user_id,
            'count': 1,
            'first_seen': datetime.now().isoformat(),
            'last_seen': datetime.now().isoformat(),
            'status': 'pending'
        })
        self.save()
        
    def get_pending(self, limit=50):
        """دریافت سوالات در انتظار"""
        return [item for item in self.data if item['status'] == 'pending'][:limit]
        
    def mark_answered(self, question_id):
        """علامت‌گذاری به عنوان پاسخ داده شده"""
        for item in self.data:
            if item['id'] == question_id:
                item['status'] = 'answered'
                self.save()
                return True
        return False

class UserManager:
    """مدیریت کاربران"""
    
    def __init__(self, config):
        self.config = config
        self.users = {}
        self.active_sessions = {}
        self.load()
        
    def load(self):
        if os.path.exists(self.config.USER_DATA_FILE):
            try:
                with open(self.config.USER_DATA_FILE, 'r', encoding='utf-8') as f:
                    self.users = json.load(f)
            except:
                self.users = {}
                
    def save(self):
        with open(self.config.USER_DATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.users, f, ensure_ascii=False, indent=2)
            
    def create_session(self, user_id):
        """ایجاد نشست جدید"""
        session_id = hashlib.md5(f"{user_id}{datetime.now()}".encode()).hexdigest()
        self.active_sessions[session_id] = {
            'user_id': user_id,
            'created_at': datetime.now().isoformat(),
            'last_activity': datetime.now().isoformat()
        }
        return session_id
        
    def get_user_stats(self):
        """آمار کاربران"""
        return {
            'total_users': len(self.users),
            'active_sessions': len(self.active_sessions)
                  }
