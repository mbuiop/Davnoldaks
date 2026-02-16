# m6.py - Learning Engine
from collections import defaultdict, Counter
import json
import os
import threading
import time
from datetime import datetime, timedelta

class LearningEngine:
    """موتور یادگیری از تعاملات کاربران"""
    
    def __init__(self, config, knowledge_base):
        self.config = config
        self.knowledge = knowledge_base
        self.user_interests = defaultdict(lambda: defaultdict(int))
        self.user_feedback = defaultdict(list)
        self.popular_queries = Counter()
        self.learning_queue = []
        self.lock = threading.Lock()
        self.load()
        
        # شروع یادگیری خودکار
        self._start_learning()
        
    def load(self):
        """بارگذاری داده‌های یادگیری"""
        learning_file = os.path.join(self.config.DATA_FOLDER, 'learning.json')
        if os.path.exists(learning_file):
            try:
                with open(learning_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.user_interests = defaultdict(lambda: defaultdict(int), data.get('interests', {}))
                    self.popular_queries = Counter(data.get('popular', {}))
            except:
                pass
                
    def save(self):
        """ذخیره داده‌های یادگیری"""
        learning_file = os.path.join(self.config.DATA_FOLDER, 'learning.json')
        with open(learning_file, 'w', encoding='utf-8') as f:
            json.dump({
                'interests': dict(self.user_interests),
                'popular': dict(self.popular_queries)
            }, f, ensure_ascii=False, indent=2)
            
    def learn_from_query(self, user_id, question, answer_found):
        """یادگیری از سوال کاربر"""
        with self.lock:
            # ثبت سوال پرتکرار
            self.popular_queries[question] += 1
            
            # استخراج کلمات کلیدی
            words = question.split()
            for word in words:
                if len(word) > 2:
                    self.user_interests[user_id][word] += 1
                    
            # اگر سوال بی‌پاسخ بود
            if not answer_found:
                self.learning_queue.append({
                    'type': 'unanswered',
                    'question': question,
                    'user_id': user_id,
                    'time': time.time()
                })
                
    def learn_from_feedback(self, user_id, question, rating):
        """یادگیری از بازخورد"""
        with self.lock:
            self.user_feedback[user_id].append({
                'question': question,
                'rating': rating,
                'time': time.time()
            })
            
            # اگر بازخورد منفی بود
            if rating < 3:
                # پیدا کردن دانش مرتبط برای بهبود
                for item in self.knowledge.data:
                    if question in item['question'] or question in item['answer']:
                        self.learning_queue.append({
                            'type': 'improve',
                            'item_id': item['id'],
                            'question': question,
                            'rating': rating
                        })
                        
    def get_recommendations(self, user_id, limit=3):
        """توصیه بر اساس علایق کاربر"""
        if user_id not in self.user_interests:
            return []
            
        interests = self.user_interests[user_id]
        if not interests:
            return []
            
        # پیدا کردن دانش مرتبط با علایق
        recommendations = []
        for item in self.knowledge.data:
            score = 0
            for word in item['question'].split():
                if word in interests:
                    score += interests[word]
            if score > 0:
                recommendations.append({
                    'question': item['question'],
                    'score': score
                })
                
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        return [r['question'] for r in recommendations[:limit]]
        
    def get_trending(self, limit=5):
        """موضوعات داغ"""
        return [q for q, _ in self.popular_queries.most_common(limit)]
        
    def _start_learning(self):
        """پردازش خودکار یادگیری"""
        def process():
            while True:
                time.sleep(300)  # هر ۵ دقیقه
                self._process_learning_queue()
                
        thread = threading.Thread(target=process, daemon=True)
        thread.start()
        
    def _process_learning_queue(self):
        """پردازش صف یادگیری"""
        with self.lock:
            if not self.learning_queue:
                return
                
            # تحلیل سوالات بی‌پاسخ پرتکرار
            unanswered = [q for q in self.learning_queue if q['type'] == 'unanswered']
            question_counter = Counter([q['question'] for q in unanswered])
            
            for question, count in question_counter.most_common(5):
                if count > 3:  # اگر بیش از ۳ بار پرسیده شده
                    print(f"📝 سوال پرتکرار بی‌پاسخ: {question} ({count} بار)")
                    
            self.save()
