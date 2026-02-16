# services/queue_service.py
import pika
import json
import threading
import time
from collections import deque
import asyncio

class QueueService:
    """سرویس صف پیام برای پردازش غیرهمزمان"""
    
    def __init__(self, config):
        self.config = config
        self.connection = None
        self.channel = None
        self.queues = {}
        self.consumers = {}
        self.pending_tasks = deque(maxlen=10000)
        self.completed_tasks = deque(maxlen=1000)
        self.connect()
        
    def connect(self):
        """اتصال به RabbitMQ"""
        try:
            params = pika.URLParameters(self.config.RABBITMQ_URL)
            self.connection = pika.BlockingConnection(params)
            self.channel = self.connection.channel()
            
            # اعلام صف‌ها
            self._declare_queues()
            
        except Exception as e:
            print(f"RabbitMQ connection error: {e}")
            # Fallback به صف درون‌حافظه‌ای
            self.use_memory_queue = True
            
    def _declare_queues(self):
        """اعلام صف‌های مورد نیاز"""
        queues = [
            'chat_messages',
            'file_processing',
            'email_notifications',
            'analytics',
            'learning_tasks',
            'feedback'
        ]
        
        for queue in queues:
            self.channel.queue_declare(queue=queue, durable=True)
            self.queues[queue] = queue
            
    def publish(self, queue: str, message: dict, priority: int = 5):
        """انتشار پیام به صف"""
        try:
            if hasattr(self, 'use_memory_queue'):
                self.pending_tasks.append({
                    'queue': queue,
                    'message': message,
                    'timestamp': time.time()
                })
                return
                
            self.channel.basic_publish(
                exchange='',
                routing_key=queue,
                body=json.dumps(message, ensure_ascii=False),
                properties=pika.BasicProperties(
                    delivery_mode=2,  # persistent
                    priority=priority,
                    timestamp=int(time.time())
                )
            )
        except Exception as e:
            print(f"Publish error: {e}")
            
    def consume(self, queue: str, callback, auto_ack: bool = True):
        """مصرف پیام از صف"""
        if hasattr(self, 'use_memory_queue'):
            self._memory_consumer(queue, callback)
            return
            
        def wrapped_callback(ch, method, properties, body):
            try:
                message = json.loads(body)
                result = callback(message)
                if result:
                    ch.basic_ack(delivery_tag=method.delivery_tag)
                    self.completed_tasks.append({
                        'queue': queue,
                        'message': message,
                        'result': result,
                        'timestamp': time.time()
                    })
            except Exception as e:
                print(f"Consume error: {e}")
                ch.basic_nack(delivery_tag=method.delivery_tag)
                
        self.channel.basic_consume(
            queue=queue,
            on_message_callback=wrapped_callback,
            auto_ack=auto_ack
        )
        
        self.consumers[queue] = True
        
    def _memory_consumer(self, queue: str, callback):
        """مصرف کننده درون‌حافظه‌ای (Fallback)"""
        def process():
            while True:
                if self.pending_tasks:
                    task = self.pending_tasks.popleft()
                    if task['queue'] == queue:
                        try:
                            result = callback(task['message'])
                            self.completed_tasks.append({
                                'queue': queue,
                                'message': task['message'],
                                'result': result,
                                'timestamp': time.time()
                            })
                        except Exception as e:
                            print(f"Memory consumer error: {e}")
                time.sleep(0.1)
                
        thread = threading.Thread(target=process, daemon=True)
        thread.start()
        
    def start_consuming(self):
        """شروع مصرف پیام‌ها"""
        if hasattr(self, 'use_memory_queue'):
            print("Using in-memory queue")
            return
            
        def run():
            try:
                self.channel.start_consuming()
            except Exception as e:
                print(f"Consuming error: {e}")
                
        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        
    def get_queue_length(self, queue: str) -> int:
        """دریافت طول صف"""
        if hasattr(self, 'use_memory_queue'):
            return len([t for t in self.pending_tasks if t['queue'] == queue])
            
        try:
            result = self.channel.queue_declare(queue=queue, passive=True)
            return result.method.message_count
        except:
            return 0
            
    def publish_chat_message(self, user_id: str, message: str):
        """انتشار پیام چت"""
        self.publish('chat_messages', {
            'user_id': user_id,
            'message': message,
            'timestamp': time.time()
        })
        
    def publish_file_task(self, file_id: str, filename: str, user_id: str):
        """انتشار تسک پردازش فایل"""
        self.publish('file_processing', {
            'file_id': file_id,
            'filename': filename,
            'user_id': user_id,
            'timestamp': time.time()
        }, priority=8)
        
    def publish_learning_task(self, data: dict):
        """انتشار تسک یادگیری"""
        self.publish('learning_tasks', {
            'data': data,
            'timestamp': time.time()
        }, priority=6)
        
    def get_stats(self) -> dict:
        """دریافت آمار صف"""
        return {
            'pending_tasks': len(self.pending_tasks),
            'completed_tasks': len(self.completed_tasks),
            'queues': {q: self.get_queue_length(q) for q in self.queues.keys()}
                          }
