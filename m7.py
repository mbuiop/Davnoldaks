# m7.py - Queue Service
import queue
import threading
import time
from datetime import datetime
import uuid

class QueueService:
    """سرویس صف پیام درون‌حافظه‌ای"""
    
    def __init__(self, config):
        self.config = config
        self.queues = {
            'chat': queue.Queue(maxsize=10000),
            'learning': queue.Queue(maxsize=5000),
            'feedback': queue.Queue(maxsize=5000),
            'file': queue.Queue(maxsize=1000)
        }
        self.results = {}
        self.workers = {}
        self.stats = defaultdict(int)
        self.lock = threading.Lock()
        
        # شروع workerها
        self._start_workers()
        
    def publish(self, queue_name, message):
        """انتشار پیام به صف"""
        if queue_name in self.queues:
            try:
                message_id = str(uuid.uuid4())
                message['id'] = message_id
                message['timestamp'] = time.time()
                
                self.queues[queue_name].put(message, block=False)
                
                with self.lock:
                    self.stats[f'published_{queue_name}'] += 1
                    
                return message_id
            except queue.Full:
                return None
        return None
        
    def subscribe(self, queue_name, callback):
        """ثبت callback برای مصرف پیام"""
        def worker():
            while True:
                try:
                    message = self.queues[queue_name].get(timeout=1)
                    try:
                        result = callback(message)
                        with self.lock:
                            self.stats[f'processed_{queue_name}'] += 1
                    except Exception as e:
                        print(f"Error processing message: {e}")
                except queue.Empty:
                    pass
                    
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        self.workers[queue_name] = thread
        
    def _start_workers(self):
        """شروع workerهای پیش‌فرض"""
        # worker برای یادگیری
        def learning_worker(message):
            print(f"Learning from: {message.get('question', '')}")
            time.sleep(0.1)  # شبیه‌سازی پردازش
            
        self.subscribe('learning', learning_worker)
        
        # worker برای بازخورد
        def feedback_worker(message):
            print(f"Feedback received: {message.get('rating', 0)}")
            
        self.subscribe('feedback', feedback_worker)
        
    def get_stats(self):
        """آمار صف"""
        with self.lock:
            return {
                'queue_sizes': {name: q.qsize() for name, q in self.queues.items()},
                'stats': dict(self.stats)
            }
