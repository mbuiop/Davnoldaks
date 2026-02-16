# m9.py - Chat API
from flask import Blueprint, request, jsonify, session
import time
from datetime import datetime
import uuid

chat_bp = Blueprint('chat', __name__)

class ChatAPI:
    def __init__(self, knowledge, unanswered, gemini, search, cache, learning, queue):
        self.knowledge = knowledge
        self.unanswered = unanswered
        self.gemini = gemini
        self.search = search
        self.cache = cache
        self.learning = learning
        self.queue = queue
        
    def register_routes(self, bp):
        
        @bp.route('/api/chat', methods=['POST'])
        def chat():
            """پایان‌نامه اصلی چت"""
            data = request.json
            question = data.get('message', '').strip()
            user_id = data.get('user_id', session.get('user_id', 'anonymous'))
            
            if not question:
                return jsonify({'error': 'سوال نمی‌تواند خالی باشد'}), 400
                
            start = time.time()
            
            # چک کردن کش
            cache_key = self.cache.make_key('chat', user_id, question)
            cached = self.cache.get(cache_key)
            if cached:
                return jsonify(cached)
                
            # مرحله ۱: جستجو در دانش محلی
            local_results = self.search.search(question, self.knowledge.data)
            
            if local_results and local_results[0]['score'] > 0.7:
                response = {
                    'answer': local_results[0]['answer'],
                    'source': 'local',
                    'confidence': local_results[0]['score'],
                    'found': True
                }
                self.knowledge.update_feedback(local_results[0]['id'], True)
                
            elif local_results and local_results[0]['score'] > 0.3:
                # مرحله ۲: استفاده از Gemini با context
                gemini_result = self.gemini.ask_with_retry(question, local_results[:2])
                
                if gemini_result['success']:
                    response = {
                        'answer': gemini_result['answer'],
                        'source': 'gemini',
                        'confidence': 0.85,
                        'found': True
                    }
                    
                    # یادگیری خودکار
                    if len(question) > 10 and len(gemini_result['answer']) > 50:
                        self.knowledge.add(
                            question, 
                            gemini_result['answer'], 
                            category='gemini_auto',
                            source='gemini'
                        )
                else:
                    response = {
                        'answer': None,
                        'found': False,
                        'message': 'سوال شما ثبت شد'
                    }
                    self.unanswered.add(question, user_id)
                    
            else:
                # مرحله ۳: فقط Gemini
                gemini_result = self.gemini.ask_with_retry(question)
                
                if gemini_result['success']:
                    response = {
                        'answer': gemini_result['answer'],
                        'source': 'gemini',
                        'confidence': 0.8,
                        'found': True
                    }
                else:
                    response = {
                        'answer': None,
                        'found': False,
                        'message': 'سوال شما ثبت شد'
                    }
                    self.unanswered.add(question, user_id)
                    
            # اضافه کردن metadata
            response['time'] = round(time.time() - start, 2)
            response['timestamp'] = datetime.now().isoformat()
            
            # یادگیری از تعامل
            self.learning.learn_from_query(user_id, question, response['found'])
            
            # ذخیره در کش
            if response['found']:
                self.cache.set(cache_key, response, ttl=3600)
                
            # ارسال به صف
            self.queue.publish('chat', {
                'user_id': user_id,
                'question': question,
                'response': response
            })
                
            return jsonify(response)
            
        @bp.route('/api/chat/feedback', methods=['POST'])
        def feedback():
            """ثبت بازخورد کاربر"""
            data = request.json
            question = data.get('question')
            rating = data.get('rating', 3)
            user_id = data.get('user_id', 'anonymous')
            
            self.learning.learn_from_feedback(user_id, question, rating)
            self.queue.publish('feedback', {
                'user_id': user_id,
                'question': question,
                'rating': rating
            })
            
            return jsonify({'status': 'ok'})
            
        @bp.route('/api/chat/trending', methods=['GET'])
        def trending():
            """موضوعات داغ"""
            return jsonify({
                'trending': self.learning.get_trending(),
                'stats': self.learning.popular_queries.most_common(10)
            })
