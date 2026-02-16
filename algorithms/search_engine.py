# algorithms/search_engine.py
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.ensemble import RandomForestClassifier
from collections import Counter, defaultdict
import re
import hashlib
import json
import pickle
import os

class AdvancedSearchEngine:
    """موتور جستجوی پیشرفته با چندین الگوریتم"""
    
    def __init__(self, config):
        self.config = config
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=50000,
            ngram_range=(1, 5),
            analyzer='char_wb',
            min_df=2,
            max_df=0.8,
            sublinear_tf=True
        )
        
        self.count_vectorizer = CountVectorizer(
            max_features=20000,
            ngram_range=(1, 3)
        )
        
        self.lda_model = LatentDirichletAllocation(
            n_components=50,
            random_state=42,
            learning_method='online'
        )
        
        self.rf_classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        self.tfidf_matrix = None
        self.lda_features = None
        self.feature_names = None
        self.word_embeddings = {}
        
    def search_hybrid(self, query, knowledge_base, top_k=10, weights=None):
        """جستجوی ترکیبی با چند الگوریتم"""
        if weights is None:
            weights = {
                'tfidf': 0.4,
                'keyword': 0.3,
                'semantic': 0.2,
                'popularity': 0.1
            }
            
        results = defaultdict(float)
        
        # 1. جستجوی TF-IDF
        tfidf_scores = self._search_tfidf(query, knowledge_base)
        for doc_id, score in tfidf_scores.items():
            results[doc_id] += score * weights['tfidf']
            
        # 2. جستجوی کلمات کلیدی
        keyword_scores = self._search_keyword(query, knowledge_base)
        for doc_id, score in keyword_scores.items():
            results[doc_id] += score * weights['keyword']
            
        # 3. جستجوی معنایی (LDA)
        semantic_scores = self._search_semantic(query, knowledge_base)
        for doc_id, score in semantic_scores.items():
            results[doc_id] += score * weights['semantic']
            
        # 4. امتیاز محبوبیت
        for i, doc in enumerate(knowledge_base):
            popularity = doc.get('times_used', 0) / (i + 1)
            results[doc['id']] += popularity * weights['popularity']
            
        # مرتب‌سازی نتایج
        sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
        
        return [knowledge_base[doc_id-1] for doc_id, score in sorted_results[:top_k] 
                if doc_id <= len(knowledge_base)]
                
    def _search_tfidf(self, query, knowledge_base):
        """جستجوی TF-IDF"""
        scores = {}
        try:
            query_vector = self.tfidf_vectorizer.transform([query])
            similarities = cosine_similarity(query_vector, self.tfidf_matrix)[0]
            
            for i, score in enumerate(similarities):
                if score > 0.1:
                    scores[i+1] = float(score)  # id از 1 شروع می‌شه
        except:
            pass
        return scores
        
    def _search_keyword(self, query, knowledge_base):
        """جستجوی کلمات کلیدی"""
        scores = {}
        query_words = set(query.lower().split())
        query_words = [w for w in query_words if len(w) > 2]
        
        for i, doc in enumerate(knowledge_base):
            doc_words = set(doc['question'].lower().split())
            common = query_words.intersection(doc_words)
            
            if common:
                score = len(common) / max(len(doc_words), 1)
                scores[i+1] = score
                
        return scores
        
    def _search_semantic(self, query, knowledge_base):
        """جستجوی معنایی با LDA"""
        scores = {}
        try:
            query_vector = self.count_vectorizer.transform([query])
            query_lda = self.lda_model.transform(query_vector)
            
            similarities = cosine_similarity(query_lda, self.lda_features)[0]
            
            for i, score in enumerate(similarities):
                if score > 0.05:
                    scores[i+1] = float(score)
        except:
            pass
        return scores
        
    def update_index(self, knowledge_base):
        """به‌روزرسانی ایندکس‌ها"""
        texts = [f"{doc['question']} {doc['answer']}" for doc in knowledge_base]
        
        if len(texts) > 1:
            # TF-IDF
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
            
            # Count Vectorizer for LDA
            count_matrix = self.count_vectorizer.fit_transform(texts)
            
            # LDA
            self.lda_features = self.lda_model.fit_transform(count_matrix)
            
            # Feature names
            self.feature_names = self.tfidf_vectorizer.get_feature_names_out()
            
    def get_keywords(self, text, top_n=10):
        """استخراج کلمات کلیدی با امتیاز TF-IDF"""
        try:
            text_vector = self.tfidf_vectorizer.transform([text])
            scores = text_vector.toarray()[0]
            
            indices = scores.argsort()[-top_n:][::-1]
            keywords = [(self.feature_names[i], scores[i]) for i in indices if scores[i] > 0]
            return keywords
        except:
            return []
            
    def find_similar(self, knowledge_base, doc_id, top_n=5):
        """پیدا کردن دانش‌های مشابه"""
        if self.tfidf_matrix is None or doc_id > self.tfidf_matrix.shape[0]:
            return []
            
        doc_vector = self.tfidf_matrix[doc_id-1:doc_id]
        similarities = cosine_similarity(doc_vector, self.tfidf_matrix)[0]
        
        similar = []
        for i, score in enumerate(similarities):
            if i != doc_id-1 and score > 0.3:
                similar.append({
                    'id': i+1,
                    'similarity': float(score),
                    'question': knowledge_base[i]['question']
                })
                
        return sorted(similar, key=lambda x: x['similarity'], reverse=True)[:top_n]
