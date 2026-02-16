# models/learning_engine.py
import numpy as np
from collections import defaultdict, Counter
from datetime import datetime, timedelta
import json
import os
import threading
import time
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import hashlib

class LearningEngine:
    """موتور یادگیری از رفتار و داده‌های کاربران"""
    
    def __init__(self, config, brain, gemini_service):
        self.config = config
        self.brain = brain
        self.gemini = gemini_service
        self.user_patterns = defaultdict(lambda: {
            'questions': [],
            'categories': Counter(),
            'times': [],
            'feedback': [],
            'avg_response_time': 0,
            'preferred_time': None,
            'topics': Counter()
        })
        
        self.global_patterns = {
            'popular_questions': Counter(),
            'common_mistakes': defaultdict(list),
            'trending_topics': Counter(),
            'peak_hours': Counter(),
            'user_segments': {}
        }
        
        self.learning_queue = []
        self.model_lock = threading.Lock()
        self.kmeans = KMeans(n_clusters=5, random_state=42)
        self.scaler = StandardScaler()
        self.load_models()
        
    def load_models(self):
        """بارگذاری مدل‌های آموزشی"""
        model_file = os.path.join(self.config.MODEL_FOLDER, 'learning_models.json')
        if os.path.exists(model_file):
            with open(model_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.global_patterns.update(data.get('global_patterns', {}))
                
    def save_models(self):
        """ذخیره مدل‌ها"""
        model_file = os.path.join(self.config.MODEL_FOLDER, 'learning_models.json')
        with open(model_file, 'w', encoding='utf-8') as f:
            json.dump({
                'global_patterns': dict(self.global_patterns)
            }, f, ensure_ascii=False, indent=2)
            
    def learn_from_interaction(self, user_id, question, answer, context=None):
        """یادگیری از تعامل کاربر"""
        pattern = self.user_patterns[user_id]
        
        # ثبت سوال
        pattern['questions'].append({
            'text': question,
            'timestamp': datetime.now().isoformat(),
            'context': context
        })
        
        # استخراج دسته‌بندی
        category = self._extract_category(question)
        pattern['categories'][category] += 1
        self.global_patterns['popular_questions'][question] += 1
        
        # ثبت زمان
        hour = datetime.now().hour
        pattern['times'].append(hour)
        self.global_patterns['peak_hours'][hour] += 1
        
        # محاسبه میانگین زمان پاسخ
        if len(pattern['times']) > 1:
            pattern['avg_response_time'] = sum(pattern['times']) / len(pattern['times'])
            
        # تشخیص موضوعات داغ
        topics = self._extract_topics(question)
        for topic in topics:
            pattern['topics'][topic] += 1
            self.global_patterns['trending_topics'][topic] += 1
            
        # محدود کردن تاریخچه
        if len(pattern['questions']) > 100:
            pattern['questions'] = pattern['questions'][-100:]
            
        # یادگیری از بازخورد
        if context and 'feedback' in context:
            self.learn_from_feedback(user_id, question, context['feedback'])
            
    def learn_from_feedback(self, user_id, question, feedback_score):
        """یادگیری از بازخورد کاربر"""
        pattern = self.user_patterns[user_id]
        pattern['feedback'].append({
            'question': question,
            'score': feedback_score,
            'timestamp': datetime.now().isoformat()
        })
        
        # اگر بازخورد منفی بود
        if feedback_score < 0.3:
            self.global_patterns['common_mistakes'][question].append({
                'user_id': user_id,
                'timestamp': datetime.now().isoformat()
            })
            
    def learn_from_crowd(self):
        """یادگیری از جمع کاربران"""
        with self.model_lock:
            # تحلیل سوالات پرتکرار
            popular = self.global_patterns['popular_questions'].most_common(20)
            
            # پیدا کردن سوالات بی‌پاسخ پرتکرار
            unanswered_file = os.path.join(self.config.DATA_FOLDER, 'unanswered.json')
            if os.path.exists(unanswered_file):
                with open(unanswered_file, 'r', encoding='utf-8') as f:
                    unanswered = json.load(f)
                    
                unanswered_counter = Counter([u['question'] for u in unanswered])
                for question, count in unanswered_counter.most_common(10):
                    if count > 5:  # اگر بیش از ۵ بار پرسیده شده
                        self.learning_queue.append({
                            'type': 'unanswered_pattern',
                            'question': question,
                            'count': count,
                            'status': 'pending'
                        })
                        
            # خوشه‌بندی کاربران
            self._cluster_users()
            
    def _cluster_users(self):
        """خوشه‌بندی کاربران بر اساس رفتار"""
        features = []
        user_ids = []
        
        for user_id, pattern in self.user_patterns.items():
            if len(pattern['questions']) > 5:
                feature_vector = [
                    len(pattern['questions']),  # تعداد سوالات
                    len(pattern['categories']),   # تنوع دسته‌بندی
                    pattern['avg_response_time'],  # میانگین زمان
                    np.mean(pattern['feedback']) if pattern['feedback'] else 0,  # میانگین بازخورد
                    np.std(pattern['times']) if len(pattern['times']) > 1 else 0  # تنوع زمانی
                ]
                features.append(feature_vector)
                user_ids.append(user_id)
                
        if len(features) > 5:
            try:
                features_scaled = self.scaler.fit_transform(features)
                clusters = self.kmeans.fit_predict(features_scaled)
                
                for i, user_id in enumerate(user_ids):
                    self.global_patterns['user_segments'][user_id] = int(clusters[i])
                    
            except Exception as e:
                print(f"Clustering error: {e}")
                
    def _extract_category(self, text):
        """استخراج دسته‌بندی از متن"""
        categories = ['تاریخ ایران', 'تاریخ جهان', 'اسلامی', 'معاصر', 'باستان']
        for cat in categories:
            if cat in text:
                return cat
        return 'عمومی'
        
    def _extract_topics(self, text, top_n=3):
        """استخراج موضوعات از متن"""
        # کلمات کلیدی ساده
        words = text.split()
        words = [w for w in words if len(w) > 3]
        return list(set(words))[:top_n]
        
    def get_user_profile(self, user_id):
        """دریافت پروفایل کاربر"""
        pattern = self.user_patterns.get(user_id, {})
        
        if not pattern:
            return {}
            
        # محاسبه علایق
        interests = pattern['categories'].most_common(5)
        
        # بهترین زمان
        if pattern['times']:
            best_hour = Counter(pattern['times']).most_common(1)[0][0]
            time_range = f"{best_hour}:00 - {best_hour+1}:00"
        else:
            time_range = "نامشخص"
            
        return {
            'total_questions': len(pattern['questions']),
            'interests': [{'category': cat, 'count': cnt} for cat, cnt in interests],
            'avg_response_time': pattern['avg_response_time'],
            'preferred_time': time_range,
            'feedback_avg': np.mean(pattern['feedback']) if pattern['feedback'] else 0,
            'segment': self.global_patterns['user_segments'].get(user_id, -1)
        }
        
    def get_recommendations(self, user_id, top_n=5):
        """توصیه به کاربر بر اساس یادگیری"""
        if user_id not in self.user_patterns:
            return []
            
        pattern = self.user_patterns[user_id]
        
        # پیدا کردن موضوعات مورد علاقه
        favorite_topics = [t for t, _ in pattern['topics'].most_common(3)]
        
        # پیدا کردن دانش‌های مرتبط
        recommendations = []
        for item in self.brain.knowledge_base:
            item_text = item['question'] + ' ' + item['answer']
            
            # امتیازدهی بر اساس علایق
            score = 0
            for topic in favorite_topics:
                if topic in item_text:
                    score += 1
                    
            # امتیاز بر اساس دسته‌بندی
            if item.get('category') in pattern['categories']:
                score += 0.5
                
            if score > 0:
                recommendations.append({
                    'id': item['id'],
                    'question': item['question'],
                    'score': score
                })
                
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        return recommendations[:top_n]
        
    def get_trending_topics(self, hours=24):
        """دریافت موضوعات داغ"""
        # فیلتر بر اساس زمان
        cutoff = datetime.now() - timedelta(hours=hours)
        
        recent_questions = []
        for pattern in self.user_patterns.values():
            for q in pattern['questions']:
                if datetime.fromisoformat(q['timestamp']) > cutoff:
                    recent_questions.append(q['text'])
                    
        # استخراج موضوعات
        all_topics = []
        for q in recent_questions:
            all_topics.extend(self._extract_topics(q, 2))
            
        return Counter(all_topics).most_common(10)
        
    def get_learning_stats(self):
        """دریافت آمار یادگیری"""
        return {
            'total_users': len(self.user_patterns),
            'total_interactions': sum(len(p['questions']) for p in self.user_patterns.values()),
            'avg_per_user': np.mean([len(p['questions']) for p in self.user_patterns.values()]) if self.user_patterns else 0,
            'trending_topics': self.global_patterns['trending_topics'].most_common(10),
            'peak_hours': self.global_patterns['peak_hours'].most_common(5),
            'user_segments': len(set(self.global_patterns['user_segments'].values())),
            'learning_queue': len(self.learning_queue)
                      }
