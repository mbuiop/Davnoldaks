# m5.py - Cache Service
import time
import threading
from collections import OrderedDict
import hashlib
import json

class CacheService:
    """سرویس کش درون‌حافظه‌ای"""
    
    def __init__(self, config):
        self.config = config
        self.cache = OrderedDict()
        self.ttl = {}
        self.hits = 0
        self.misses = 0
        self.lock = threading.Lock()
        self.max_size = config.LOCAL_CACHE_SIZE
        
        # شروع تمیزکننده خودکار
        self._start_cleaner()
        
    def get(self, key):
        """دریافت از کش"""
        with self.lock:
            if key in self.cache:
                if time.time() < self.ttl[key]:
                    self.hits += 1
                    # انتقال به انتها (اخیراً استفاده شده)
                    self.cache.move_to_end(key)
                    return self.cache[key]
                else:
                    # حذف منقضی شده
                    del self.cache[key]
                    del self.ttl[key]
                    
            self.misses += 1
            return None
            
    def set(self, key, value, ttl=3600):
        """ذخیره در کش"""
        with self.lock:
            # مدیریت حجم کش
            if len(self.cache) >= self.max_size:
                # حذف قدیمی‌ترین
                oldest = next(iter(self.cache))
                del self.cache[oldest]
                del self.ttl[oldest]
                
            self.cache[key] = value
            self.ttl[key] = time.time() + ttl
            self.cache.move_to_end(key)
            
    def delete(self, key):
        """حذف از کش"""
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                del self.ttl[key]
                return True
        return False
        
    def clear(self):
        """پاک کردن کل کش"""
        with self.lock:
            self.cache.clear()
            self.ttl.clear()
            
    def make_key(self, *args):
        """ساخت کلید یکتا"""
        key_string = ':'.join(str(arg) for arg in args)
        return hashlib.md5(key_string.encode()).hexdigest()
        
    def get_or_set(self, key, func, ttl=3600):
        """دریافت یا محاسبه و ذخیره"""
        value = self.get(key)
        if value is not None:
            return value
            
        value = func()
        if value:
            self.set(key, value, ttl)
        return value
        
    def _start_cleaner(self):
        """تمیزکننده خودکار"""
        def cleaner():
            while True:
                time.sleep(60)  # هر دقیقه
                self._clean_expired()
                
        thread = threading.Thread(target=cleaner, daemon=True)
        thread.start()
        
    def _clean_expired(self):
        """حذف موارد منقضی"""
        with self.lock:
            now = time.time()
            expired = [k for k, t in self.ttl.items() if now >= t]
            for k in expired:
                del self.cache[k]
                del self.ttl[k]
                
    def get_stats(self):
        """آمار کش"""
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        
        return {
            'size': len(self.cache),
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': round(hit_rate, 2)
        }
