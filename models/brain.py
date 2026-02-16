# models/brain.py
import numpy as np
import json
import os
from datetime import datetime
from collections import defaultdict, Counter
import hashlib
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import asyncio
import threading

class AIBrain:
    """مغز اصلی هوش مصنوعی با قابلیت یادگیری"""
    
    def __init__(self, config):
        self.config = config
        self.knowledge_base = []
        self.user_profiles = defaultdict(dict)
        self.learning_cache = {}
        self.conversation_patterns = defaultdict(list)
        self.feedback_history = []
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=20000,
            ngram_range=(1, 4),
            analyzer='char_wb',
            sublinear_tf=True
        )
        self.tfidf_matrix = None
        self.load_knowledge()
        
    def load_knowledge(self):
        """بارگذاری دانش از فایل"""
        kb_file = os.path.join(self.config.DATA_FOLDER, 'knowledge_base.json')
        if os.path.exists(kb_file):
            with open(kb_file, 'r', encoding='utf-8') as f:
                self.knowledge_base = json.load(f)
            self.update_vectors()
            
    def save_knowledge(self):
        """ذخیره دانش"""
        kb_file = os.path.join(self.config.DATA_FOLDER, 'knowledge_base.json')
        with open(kb_file, 'w', encoding='utf-8') as f:
            json.dump(self.knowledge_base, f, ensure_ascii=False, indent=2)
            
    def update_vectors(self):
        """به‌روزرسانی بردارهای TF-IDF"""
        if self.knowledge_base:
            texts = [f"{k['question']} {k['answer']}" for k in self.knowledge_base]
            try:
                self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
            except:
                self.tfidf_matrix = None
                
    def learn_from_user(self, user_id, question, answer, feedback_score):
        """یادگیری از تعامل کاربر"""
        profile = self.user_profiles.get(user_id, {
            'interests': defaultdict(int),
            'history': [],
            'feedback_scores': []
        })
        
        # استخراج کلمات کلیدی از سوال
        keywords = self.extract_keywords(question)
        for kw in keywords:
            profile['interests'][kw] += 1
            
        # ذخیره تاریخچه
        profile['history'].append({
            'question': question,
            'answer': answer,
            'feedback': feedback_score,
            'timestamp': datetime.now().isoformat()
        })
        profile['feedback_scores'].append(feedback_score)
        
        # محدود کردن تاریخچه
        if len(profile['history']) > 100:
            profile['history'] = profile['history'][-100:]
            
        self.user_profiles[user_id] = profile
        
        # یادگیری از بازخورد منفی
        if feedback_score < 0.3:
            self.learn_from_mistake(question, answer)
            
    def learn_from_mistake(self, question, wrong_answer):
        """یادگیری از اشتباهات"""
        # پیدا کردن دانش‌های مرتبط
        results = self.search(question, top_k=5)
        
        for result in results:
            if result['answer'] == wrong_answer:
                result['confidence'] *= 0.8  # کاهش اعتماد
                
        self.save_knowledge()
        
    def learn_from_crowd(self):
        """یادگیری از الگوهای جمعی کاربران"""
        # تحلیل الگوهای پرتکرار
        all_questions = []
        for profile in self.user_profiles.values():
            all_questions.extend([h['question'] for h in profile['history']])
            
        if len(all_questions) > 100:
            question_patterns = Counter(all_questions)
            top_patterns = question_patterns.most_common(10)
            
            # ایجاد دانش جدید از سوالات پرتکرار بی‌پاسخ
            unanswered_file = os.path.join(self.config.DATA_FOLDER, 'unanswered.json')
            if os.path.exists(unanswered_file):
                with open(unanswered_file, 'r', encoding='utf-8') as f:
                    unanswered = json.load(f)
                    
                for pattern, count in top_patterns:
                    if count > 5:  # اگر بیش از ۵ بار پرسیده شده
                        # اینجا می‌تونی از Gemini برای پاسخ استفاده کنی
                        self.pending_learning.append({
                            'pattern': pattern,
                            'count': count,
                            'status': 'pending'
                        })
                        
    def extract_keywords(self, text, top_n=5):
        """استخراج کلمات کلیدی"""
        words = text.split()
        words = [w for w in words if len(w) > 2]
        word_freq = Counter(words)
        return [w for w, _ in word_freq.most_common(top_n)]
    
    def search(self, query, top_k=5, user_id=None):
        """جستجوی هوشمند با شخصی‌سازی"""
        if not self.knowledge_base or self.tfidf_matrix is None:
            return []
            
        try:
            query_vector = self.tfidf_vectorizer.transform([query])
            similarities = cosine_similarity(query_vector, self.tfidf_matrix)[0]
            
            results = []
            for i, score in enumerate(similarities):
                if score > 0.1:
                    item = self.knowledge_base[i]
                    
                    # شخصی‌سازی بر اساس کاربر
                    personal_score = 0
                    if user_id and user_id in self.user_profiles:
                        user_interests = self.user_profiles[user_id]['interests']
                        item_keywords = set(item.get('keywords', []))
                        if user_interests and item_keywords:
                            match = sum(user_interests[kw] for kw in item_keywords if kw in user_interests)
                            personal_score = min(match / 10, 0.3)  # حداکثر ۳۰٪ تاثیر
                    
                    final_score = float(score) * (0.7 + personal_score)
                    
                    results.append({
                        'id': item['id'],
                        'question': item['question'],
                        'answer': item['answer'],
                        'score': final_score,
                        'category': item.get('category', 'عمومی')
                    })
                    
            results.sort(key=lambda x: x['score'], reverse=True)
            return results[:top_k]
            
        except Exception as e:
            print(f"Search error: {e}")
            return []
            
    def get_user_recommendations(self, user_id, top_n=3):
        """توصیه به کاربر بر اساس علایقش"""
        if user_id not in self.user_profiles:
            return []
            
        interests = self.user_profiles[user_id]['interests']
        if not interests:
            return []
            
        # پیدا کردن دانش‌های مرتبط با علایق
        recommendations = []
        for item in self.knowledge_base:
            item_keywords = set(item.get('keywords', []))
            if item_keywords:
                relevance = sum(interests[kw] for kw in item_keywords if kw in interests)
                if relevance > 0:
                    recommendations.append({
                        'id': item['id'],
                        'question': item['question'],
                        'relevance': relevance
                    })
                    
        recommendations.sort(key=lambda x: x['relevance'], reverse=True)
        return recommendations[:top_n]
