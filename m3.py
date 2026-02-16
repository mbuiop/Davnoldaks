# m3.py - Gemini Service
import requests
import json
import time
from datetime import datetime
import threading
import hashlib

class GeminiService:
    """سرویس Google Gemini API"""
    
    def __init__(self, config):
        self.config = config
        self.api_key = config.GEMINI_API_KEY
        # لیست مدل‌های مختلف برای fallback
        self.models = [
            "gemini-pro",
            "gemini-1.5-pro",
            "gemini-1.0-pro"
        ]
        self.current_model_index = 0
        self.api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.models[0]}:generateContent"
        self.cache = {}
        self.request_count = 0
        self.error_count = 0
        self.lock = threading.Lock()
        
    def ask(self, question, context=None):
        """پرسش از Gemini با fallback مدل"""
        
        # امتحان مدل‌های مختلف
        for attempt in range(len(self.models)):
            model = self.models[attempt]
            api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
            
            try:
                # چک کردن کش
                cache_key = f"gemini:{hashlib.md5(question.encode()).hexdigest()}"
                if cache_key in self.cache:
                    cache_time, answer = self.cache[cache_key]
                    if time.time() - cache_time < 3600:  # 1 ساعت
                        return {
                            'success': True,
                            'answer': answer,
                            'source': 'cache'
                        }
                        
                headers = {'Content-Type': 'application/json'}
                
                prompt = self._build_prompt(question, context)
                
                data = {
                    "contents": [{
                        "parts": [{"text": prompt}]
                    }],
                    "generationConfig": {
                        "temperature": 0.7,
                        "maxOutputTokens": 1024,
                        "topP": 0.95
                    }
                }
                
                response = requests.post(
                    f"{api_url}?key={self.api_key}",
                    headers=headers,
                    json=data,
                    timeout=15
                )
                
                with self.lock:
                    self.request_count += 1
                    
                if response.status_code == 200:
                    result = response.json()
                    answer = result['candidates'][0]['content']['parts'][0]['text']
                    
                    # ذخیره در کش
                    self.cache[cache_key] = (time.time(), answer)
                    
                    return {
                        'success': True,
                        'answer': answer,
                        'source': 'gemini',
                        'model': model
                    }
                else:
                    print(f"Gemini API error with model {model}: {response.status_code}")
                    if attempt == len(self.models) - 1:
                        with self.lock:
                            self.error_count += 1
                        return {
                            'success': False,
                            'error': f"API error: {response.status_code}"
                        }
                    continue  # امتحان مدل بعدی
                    
            except Exception as e:
                print(f"Gemini service error with model {model}: {e}")
                if attempt == len(self.models) - 1:
                    with self.lock:
                        self.error_count += 1
                    return {
                        'success': False,
                        'error': str(e)
                    }
                continue  # امتحان مدل بعدی
                
        return {
            'success': False,
            'error': "All models failed"
        }
            
    def ask_with_retry(self, question, context=None, retries=3):
        """پرسش با تلاش مجدد"""
        for i in range(retries):
            result = self.ask(question, context)
            if result['success']:
                return result
            time.sleep(1 * (i + 1))
        return result
        
    def _build_prompt(self, question, context):
        """ساخت پرامپت"""
        prompt = "تو یک تاریخ‌دان حرفه‌ای هستی که به زبان فارسی مسلط هستی.\n\n"
        
        if context:
            prompt += "اطلاعات موجود:\n"
            for i, item in enumerate(context[:3], 1):
                prompt += f"{i}. {item['question']} -> {item['answer']}\n"
            prompt += "\n"
            
        prompt += f"سوال کاربر: {question}\n\n"
        prompt += "پاسخ دقیق و مستند بده:"
        
        return prompt
        
    def get_stats(self):
        """آمار سرویس"""
        return {
            'total_requests': self.request_count,
            'error_count': self.error_count,
            'cache_size': len(self.cache)
      }
