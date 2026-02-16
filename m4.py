# m4.py - Search Engine
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import re
import threading

class SearchEngine:
    """موتور جستجوی پیشرفته"""
    
    def __init__(self, config):
        self.config = config
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 3),
            analyzer='char_wb',
            min_df=1,
            max_df=0.9,
            sublinear_tf=True
        )
        self.knowledge_vectors = None
        self.knowledge_texts = []
        self.lock = threading.Lock()
        
    def preprocess(self, text):
        """پیش‌پردازش متن"""
        if not text:
            return ""
        text = re.sub(r'[^\w\sآ-یa-zA-Z0-9]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.lower().strip()
        
    def update_index(self, knowledge_base):
        """به‌روزرسانی ایندکس"""
        with self.lock:
            self.knowledge_texts = [
                f"{item['question']} {item['answer']}" 
                for item in knowledge_base
            ]
            
            if self.knowledge_texts:
                try:
                    self.knowledge_vectors = self.vectorizer.fit_transform(self.knowledge_texts)
                except:
                    self.knowledge_vectors = None
                    
    def search(self, query, knowledge_base, limit=5):
        """جستجوی پیشرفته"""
        if not knowledge_base or self.knowledge_vectors is None:
            return []
            
        query = self.preprocess(query)
        if not query:
            return []
            
        try:
            query_vector = self.vectorizer.transform([query])
            similarities = cosine_similarity(query_vector, self.knowledge_vectors)[0]
            
            # جستجوی کلمات کلیدی
            query_words = set(query.split())
            keyword_scores = []
            
            for i, item in enumerate(knowledge_base):
                tfidf_score = similarities[i]
                
                # امتیاز کلمات کلیدی
                item_words = set(item['question'].lower().split())
                common = query_words.intersection(item_words)
                keyword_score = len(common) / max(len(item_words), 1) if item_words else 0
                
                # امتیاز نهایی
                final_score = (tfidf_score * 0.7) + (keyword_score * 0.3)
                
                if final_score > 0.1:
                    keyword_scores.append({
                        'id': item['id'],
                        'question': item['question'],
                        'answer': item['answer'],
                        'score': float(final_score),
                        'category': item['category']
                    })
                    
            keyword_scores.sort(key=lambda x: x['score'], reverse=True)
            return keyword_scores[:limit]
            
        except Exception as e:
            print(f"Search error: {e}")
            return []
            
    def find_similar(self, question, knowledge_base, limit=3):
        """پیدا کردن سوالات مشابه"""
        results = self.search(question, knowledge_base, limit=limit*2)
        return [r for r in results if r['score'] > 0.3][:limit]
