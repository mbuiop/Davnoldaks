# services/cache_service.py
import redis
import json
import pickle
import hashlib
import time
from typing import Any, Optional
import threading

class CacheService:
    """سرویس کش توزیع شده با Redis"""
    
    def __init__(self, config):
        self.config = config
        self.redis_client = redis.Redis.from_url(config.REDIS_URL, decode_responses=True)
        self.local_cache = {}
        self.local_cache_ttl = {}
        self.hit_count = 0
        self.miss_count = 0
        self.lock = threading.Lock()
        
    def get(self, key: str, use_local: bool = True) -> Optional[Any]:
        """دریافت از کش"""
        # چک کردن کش محلی
        if use_local and key in self.local_cache:
            if self.local_cache_ttl[key] > time.time():
                self.hit_count += 1
                return self.local_cache[key]
            else:
                del self.local_cache[key]
                del self.local_cache_ttl[key]
                
        # چک کردن Redis
        try:
            value = self.redis_client.get(key)
            if value:
                self.hit_count += 1
                data = json.loads(value)
                
                # ذخیره در کش محلی
                if use_local:
                    with self.lock:
                        self.local_cache[key] = data
                        self.local_cache_ttl[key] = time.time() + 60
                        
                return data
            else:
                self.miss_count += 1
                return None
        except Exception as e:
            print(f"Redis get error: {e}")
            self.miss_count += 1
            return None
            
    def set(self, key: str, value: Any, ttl: int = 3600):
        """ذخیره در کش"""
        try:
            serialized = json.dumps(value, ensure_ascii=False)
            self.redis_client.setex(key, ttl, serialized)
            
            # ذخیره در کش محلی
            with self.lock:
                self.local_cache[key] = value
                self.local_cache_ttl[key] = time.time() + min(ttl, 60)
                
        except Exception as e:
            print(f"Redis set error: {e}")
            
    def delete(self, key: str):
        """حذف از کش"""
        try:
            self.redis_client.delete(key)
            if key in self.local_cache:
                del self.local_cache[key]
                del self.local_cache_ttl[key]
        except Exception as e:
            print(f"Redis delete error: {e}")
            
    def clear_pattern(self, pattern: str):
        """پاک کردن کلیدهای با الگوی مشخص"""
        try:
            keys = self.redis_client.keys(pattern)
            if keys:
                self.redis_client.delete(*keys)
        except Exception as e:
            print(f"Redis clear pattern error: {e}")
            
    def get_or_set(self, key: str, callback, ttl: int = 3600):
        """دریافت از کش یا محاسبه و ذخیره"""
        value = self.get(key)
        if value is not None:
            return value
            
        value = callback()
        if value:
            self.set(key, value, ttl)
        return value
        
    def remember_user(self, user_id: str, data: Any, ttl: int = 86400):
        """ذخیره اطلاعات کاربر"""
        key = f"user:{user_id}:data"
        self.set(key, data, ttl)
        
    def get_user(self, user_id: str) -> Optional[Any]:
        """دریافت اطلاعات کاربر"""
        key = f"user:{user_id}:data"
        return self.get(key)
        
    def cache_conversation(self, conversation_id: str, data: Any, ttl: int = 3600):
        """کش کردن مکالمه"""
        key = f"conv:{conversation_id}"
        self.set(key, data, ttl)
        
    def get_conversation(self, conversation_id: str) -> Optional[Any]:
        """دریافت مکالمه از کش"""
        key = f"conv:{conversation_id}"
        return self.get(key)
        
    def get_stats(self) -> dict:
        """دریافت آمار کش"""
        total = self.hit_count + self.miss_count
        hit_rate = (self.hit_count / total * 100) if total > 0 else 0
        
        return {
            'hit_count': self.hit_count,
            'miss_count': self.miss_count,
            'hit_rate': hit_rate,
            'local_cache_size': len(self.local_cache),
            'redis_info': self.redis_client.info()
        }
        
    def make_key(self, *args) -> str:
        """ساخت کلید یکتا"""
        key_string = ':'.join(str(arg) for arg in args)
        return hashlib.md5(key_string.encode()).hexdigest()
