# m8.py - File Processor
import os
import chardet
import threading
import time
from datetime import datetime

class FileProcessor:
    """پردازش فایل‌های آپلودی"""
    
    def __init__(self, config, knowledge_base):
        self.config = config
        self.knowledge = knowledge_base
        self.processing_queue = []
        self.results = {}
        self.lock = threading.Lock()
        
        # شروع پردازشگر
        self._start_processor()
        
    def process(self, filepath, filename, user_id='admin'):
        """افزودن فایل به صف پردازش"""
        with self.lock:
            task_id = f"task_{len(self.processing_queue)}_{time.time()}"
            self.processing_queue.append({
                'id': task_id,
                'filepath': filepath,
                'filename': filename,
                'user_id': user_id,
                'status': 'pending',
                'progress': 0,
                'created_at': datetime.now().isoformat()
            })
            return task_id
            
    def _process_file(self, task):
        """پردازش فایل"""
        try:
            task['status'] = 'processing'
            task['progress'] = 10
            
            # تشخیص encoding
            with open(task['filepath'], 'rb') as f:
                raw = f.read()
                encoding = chardet.detect(raw)['encoding'] or 'utf-8'
                
            task['progress'] = 30
            
            # خواندن فایل
            with open(task['filepath'], 'r', encoding=encoding, errors='ignore') as f:
                content = f.read()
                
            task['progress'] = 50
            
            # پردازش خط به خط
            lines = content.split('\n')
            added = 0
            
            for i, line in enumerate(lines):
                if '|' in line:
                    parts = line.split('|', 1)
                    if len(parts) == 2:
                        q, a = parts
                        success, _ = self.knowledge.add(
                            q.strip(), 
                            a.strip(), 
                            category='file_upload',
                            source=task['filename']
                        )
                        if success:
                            added += 1
                            
                # به‌روزرسانی پیشرفت
                if i % 10 == 0:
                    task['progress'] = 50 + int((i / len(lines)) * 40)
                    
            task['progress'] = 100
            task['status'] = 'completed'
            task['result'] = f"{added} مورد اضافه شد"
            
            # پاک کردن فایل موقت
            os.remove(task['filepath'])
            
        except Exception as e:
            task['status'] = 'error'
            task['error'] = str(e)
            
    def _start_processor(self):
        """پردازشگر پس‌زمینه"""
        def processor():
            while True:
                if self.processing_queue:
                    task = self.processing_queue.pop(0)
                    self._process_file(task)
                time.sleep(1)
                
        thread = threading.Thread(target=processor, daemon=True)
        thread.start()
        
    def get_status(self, task_id):
        """دریافت وضعیت پردازش"""
        for task in self.processing_queue:
            if task['id'] == task_id:
                return task
        return None
        
    def get_all_status(self):
        """دریافت همه وضعیت‌ها"""
        return self.processing_queue
