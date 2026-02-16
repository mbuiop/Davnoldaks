# services/gemini_service.py
import google.generativeai as genai
import asyncio
import json
import time
from typing import List, Dict, Any
import hashlib

class GeminiService:
    """سرویس هوش مصنوعی گوگل جمینی"""
    
    def __init__(self, config):
        self.config = config
        genai.configure(api_key=config.GEMINI_API_KEY)
        self.model = genai.GenerativeModel(config.GEMINI_MODEL)
        self.chat_sessions = {}
        self.cache = {}
        
    async def generate_answer(self, question: str, context: List[Dict] = None) -> Dict:
        """تولید پاسخ با استفاده از جمینی"""
        cache_key = hashlib.md5(f"{question}:{str(context)}".encode()).hexdigest()
        
        # چک کردن کش
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            prompt = self._build_prompt(question, context)
            
            # استفاده از ThreadPool برای non-blocking
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, 
                lambda: self.model.generate_content(prompt)
            )
            
            result = {
                'answer': response.text,
                'confidence': 0.9,  # Gemini confidence
                'model': 'gemini-pro',
                'timestamp': time.time()
            }
            
            # ذخیره در کش
            self.cache[cache_key] = result
            
            return result
            
        except Exception as e:
            print(f"Gemini error: {e}")
            return {
                'answer': None,
                'error': str(e),
                'confidence': 0
            }
            
    async def chat_session(self, user_id: str, message: str, history: List = None):
        """ایجاد جلسه چت با حافظه"""
        if user_id not in self.chat_sessions:
            self.chat_sessions[user_id] = self.model.start_chat(history=history or [])
            
        chat = self.chat_sessions[user_id]
        
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: chat.send_message(message)
            )
            
            return {
                'response': response.text,
                'history': chat.history
            }
            
        except Exception as e:
            print(f"Chat session error: {e}")
            return None
            
    async def analyze_sentiment(self, text: str) -> Dict:
        """تحلیل احساسات متن"""
        prompt = f"""
        احساسات این متن را تحلیل کن و به صورت JSON برگردان:
        متن: {text}
        
        فرمت خروجی:
        {{
            "sentiment": "positive/negative/neutral",
            "score": 0.0-1.0,
            "emotions": ["شادی", "غم", "خشم", ...]
        }}
        """
        
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.model.generate_content(prompt)
            )
            
            # استخراج JSON از پاسخ
            result_text = response.text
            # پیدا کردن JSON در متن
            start = result_text.find('{')
            end = result_text.rfind('}') + 1
            if start >= 0 and end > start:
                return json.loads(result_text[start:end])
                
        except Exception as e:
            print(f"Sentiment analysis error: {e}")
            
        return {
            'sentiment': 'neutral',
            'score': 0.5,
            'emotions': []
        }
        
    async def extract_entities(self, text: str) -> List[Dict]:
        """استخراج موجودیت‌ها از متن"""
        prompt = f"""
        موجودیت‌های زیر را از متن استخراج کن و به صورت JSON برگردان:
        - افراد
        - مکان‌ها
        - تاریخ‌ها
        - رویدادها
        
        متن: {text}
        
        فرمت خروجی:
        {{
            "persons": [],
            "locations": [],
            "dates": [],
            "events": []
        }}
        """
        
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.model.generate_content(prompt)
            )
            
            result_text = response.text
            start = result_text.find('{')
            end = result_text.rfind('}') + 1
            if start >= 0 and end > start:
                return json.loads(result_text[start:end])
                
        except Exception as e:
            print(f"Entity extraction error: {e}")
            
        return {
            'persons': [],
            'locations': [],
            'dates': [],
            'events': []
        }
        
    async def summarize_text(self, text: str, max_length: int = 200) -> str:
        """خلاصه‌سازی متن"""
        prompt = f"""
        این متن را در {max_length} کلمه خلاصه کن:
        
        {text}
        """
        
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.model.generate_content(prompt)
            )
            
            return response.text
            
        except Exception as e:
            print(f"Summarization error: {e}")
            return text[:max_length]
            
    def _build_prompt(self, question: str, context: List[Dict] = None) -> str:
        """ساخت پرامپت برای جمینی"""
        prompt = "تو یک دستیار هوشمند تاریخ هستی که به سوالات پاسخ می‌دهی.\n\n"
        
        if context:
            prompt += "اطلاعات موجود:\n"
            for i, item in enumerate(context, 1):
                prompt += f"{i}. سوال: {item['question']}\n"
                prompt += f"   پاسخ: {item['answer']}\n\n"
                
        prompt += f"سوال کاربر: {question}\n\n"
        prompt += "پاسخ:"
        
        return prompt
