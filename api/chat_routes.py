# api/chat_routes.py
from flask import Blueprint, request, jsonify, session, Response
import time
import uuid
from datetime import datetime
import json

chat_bp = Blueprint('chat', __name__)

class ChatAPI:
    def __init__(self, brain, gemini_service, cache_service, learning_engine, queue_service):
        self.brain = brain
        self.gemini = gemini_service
        self.cache = cache_service
        self.learning = learning_engine
        self.queue = queue_service
        
    def register_routes(self, bp):
        """ثبت مسیرها"""
        
        @bp.route('/api/chat', methods=['POST'])
        def chat():
            """پایان‌نامه چت اصلی"""
            data = request.json
            question = data.get('message', '').strip()
            user_id = data.get('user_id', session.get('user_id', 'anonymous'))
            use_gemini = data.get('use_gemini', False)
            
            if not question:
                return jsonify({'error': 'سوال نمی‌تواند خالی باشد'}), 400
                
            start_time = time.time()
            
            # چک کردن کش
            cache_key = self.cache.make_key('chat', user_id, question) if hasattr(self.cache, 'make_key') else f"chat:{user_id}:{question}"
            cached = self.cache.get(cache_key) if hasattr(self.cache, 'get') else None
            
            if cached:
                return jsonify(cached)
                
            # جستجو در دانش
            results = self.brain.search(question, user_id=user_id) if hasattr(self.brain, 'search') else []
            
            response = {}
            
            if results and results[0].get('score', 0) > 0.3:
                response = {
                    'answer': results[0]['answer'],
                    'confidence': results[0]['score'],
                    'source': 'knowledge_base',
                    'found': True
                }
            elif use_gemini and self.gemini:
                # استفاده از Gemini
                gemini_result = self.gemini.generate_answer(question, results)
                if gemini_result and gemini_result.get('answer'):
                    response = {
                        'answer': gemini_result['answer'],
                        'confidence': gemini_result.get('confidence', 0.7),
                        'source': 'gemini',
                        'found': True
                    }
                    
            if not response:
                response = {
                    'answer': None,
                    'found': False,
                    'message': 'سوال شما ثبت شد'
                }
                # ثبت سوال بی‌پاسخ
                if hasattr(self.brain, 'record_unanswered'):
                    self.brain.record_unanswered(question)
                
            # اضافه کردن metadata
            response['response_time'] = time.time() - start_time
            response['timestamp'] = datetime.now().isoformat()
            response['session_id'] = session.get('session_id', str(uuid.uuid4()))
            
            # یادگیری از تعامل
            if hasattr(self.learning, 'learn_from_interaction'):
                self.learning.learn_from_interaction(user_id, question, response.get('answer'))
                
            # ذخیره در کش
            if response.get('found') and hasattr(self.cache, 'set'):
                self.cache.set(cache_key, response, ttl=3600)
                
            # ارسال به صف برای تحلیل
            if hasattr(self.queue, 'publish_chat_message'):
                self.queue.publish_chat_message(user_id, {
                    'question': question,
                    'response': response,
                    'timestamp': time.time()
                })
                    
            return jsonify(response)
            
        @bp.route('/api/chat/stream', methods=['POST'])
        def chat_stream():
            """چت با پاسخ streaming"""
            data = request.json
            question = data.get('message', '').strip()
            user_id = data.get('user_id', session.get('user_id', 'anonymous'))
            
            def generate():
                yield f"data: {json.dumps({'type': 'start'})}\n\n"
                
                # جستجو در دانش
                results = self.brain.search(question, user_id=user_id) if hasattr(self.brain, 'search') else []
                
                if results:
                    yield f"data: {json.dumps({'type': 'token', 'content': results[0]['answer']})}\n\n"
                elif self.gemini:
                    # استفاده از Gemini
                    gemini_result = self.gemini.generate_answer(question)
                    if gemini_result and gemini_result.get('answer'):
                        # ارسال تکه‌تکه
                        answer = gemini_result['answer']
                        chunk_size = 50
                        for i in range(0, len(answer), chunk_size):
                            chunk = answer[i:i+chunk_size]
                            yield f"data: {json.dumps({'type': 'token', 'content': chunk})}\n\n"
                            time.sleep(0.05)
                            
                yield f"data: {json.dumps({'type': 'end'})}\n\n"
                
            return Response(generate(), mimetype='text/event-stream')
            
        @bp.route('/api/chat/history', methods=['GET'])
        def chat_history():
            """دریافت تاریخچه چت"""
            user_id = request.args.get('user_id', session.get('user_id', 'anonymous'))
            limit = int(request.args.get('limit', 50))
            
            # دریافت از کش
            cache_key = f"history:{user_id}:{limit}"
            cached = self.cache.get(cache_key) if hasattr(self.cache, 'get') else None
            
            if cached:
                return jsonify(cached)
                
            # از پروفایل کاربر
            profile = self.learning.get_user_profile(user_id) if hasattr(self.learning, 'get_user_profile') else {}
            history = profile.get('questions', [])[-limit:] if profile else []
            
            response = {
                'user_id': user_id,
                'total': len(history),
                'history': history
            }
            
            if hasattr(self.cache, 'set'):
                self.cache.set(cache_key, response, ttl=300)
                
            return jsonify(response)
            
        @bp.route('/api/chat/feedback', methods=['POST'])
        def chat_feedback():
            """ثبت بازخورد کاربر"""
            data = request.json
            question = data.get('question')
            answer = data.get('answer')
            feedback = data.get('feedback')  # 1-5
            user_id = data.get('user_id', session.get('user_id', 'anonymous'))
            
            if not all([question, feedback]):
                return jsonify({'error': 'اطلاعات ناقص'}), 400
                
            # یادگیری از بازخورد
            if hasattr(self.learning, 'learn_from_feedback'):
                self.learning.learn_from_feedback(user_id, question, feedback/5)
                
            # ارسال به صف
            if hasattr(self.queue, 'publish_learning_task'):
                self.queue.publish_learning_task({
                    'type': 'feedback',
                    'question': question,
                    'answer': answer,
                    'feedback': feedback,
                    'user_id': user_id
                })
                
            return jsonify({'status': 'ok', 'message': 'بازخورد ثبت شد'})
            
        @bp.route('/api/chat/suggestions', methods=['GET'])
        def get_suggestions():
            """دریافت پیشنهادات برای کاربر"""
            user_id = request.args.get('user_id', session.get('user_id', 'anonymous'))
            
            # توصیه‌ها
            recommendations = self.learning.get_recommendations(user_id) if hasattr(self.learning, 'get_recommendations') else []
            
            # موضوعات داغ
            trending = self.learning.get_trending_topics() if hasattr(self.learning, 'get_trending_topics') else []
            
            return jsonify({
                'recommendations': recommendations,
                'trending': trending
            })
