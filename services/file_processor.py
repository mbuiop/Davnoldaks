# services/file_processor.py
import os
import uuid
import json
import pandas as pd
import PyPDF2
import docx
from PIL import Image
import pytesseract
import csv
from datetime import datetime
import threading
import hashlib
from werkzeug.utils import secure_filename

class FileProcessor:
    """پردازش فایل‌های آپلودی با قابلیت مقیاس‌پذیری"""
    
    def __init__(self, config, brain, queue_service):
        self.config = config
        self.brain = brain
        self.queue = queue_service
        self.processing_status = {}
        self.supported_formats = {
            '.txt': self._process_txt,
            '.csv': self._process_csv,
            '.json': self._process_json,
            '.pdf': self._process_pdf,
            '.docx': self._process_docx,
            '.xlsx': self._process_excel,
            '.jpg': self._process_image,
            '.png': self._process_image
        }
        
    def process_files(self, files, user_id: str):
        """پردازش چند فایل همزمان"""
        results = []
        
        for file in files:
            if file and file.filename:
                file_id = str(uuid.uuid4())
                filename = secure_filename(file.filename)
                ext = os.path.splitext(filename)[1].lower()
                
                # ذخیره فایل
                filepath = os.path.join(self.config.UPLOAD_FOLDER, f"{file_id}_{filename}")
                file.save(filepath)
                
                # ثبت وضعیت
                self.processing_status[file_id] = {
                    'filename': filename,
                    'status': 'pending',
                    'progress': 0,
                    'user_id': user_id,
                    'started_at': datetime.now().isoformat()
                }
                
                # ارسال به صف پردازش
                self.queue.publish_file_task(file_id, filepath, user_id)
                
                results.append({
                    'file_id': file_id,
                    'filename': filename,
                    'status': 'queued'
                })
                
        return results
        
    def process_file_sync(self, file_id: str, filepath: str, user_id: str):
        """پردازش همزمان یک فایل"""
        try:
            self.processing_status[file_id]['status'] = 'processing'
            self.processing_status[file_id]['progress'] = 10
            
            ext = os.path.splitext(filepath)[1].lower()
            
            if ext in self.supported_formats:
                processor = self.supported_formats[ext]
                result = processor(filepath)
                
                self.processing_status[file_id]['progress'] = 80
                
                # اضافه کردن به دانش
                count = 0
                errors = []
                
                for item in result:
                    if 'question' in item and 'answer' in item:
                        success, msg = self.brain.add_knowledge(
                            item['question'],
                            item['answer'],
                            item.get('category', 'file_upload')
                        )
                        if success:
                            count += 1
                        else:
                            errors.append(msg)
                            
                self.processing_status[file_id]['status'] = 'completed'
                self.processing_status[file_id]['progress'] = 100
                self.processing_status[file_id]['result'] = {
                    'added': count,
                    'errors': errors[:5]
                }
                
            else:
                self.processing_status[file_id]['status'] = 'error'
                self.processing_status[file_id]['error'] = f"فرمت {ext} پشتیبانی نمی‌شود"
                
        except Exception as e:
            self.processing_status[file_id]['status'] = 'error'
            self.processing_status[file_id]['error'] = str(e)
            
        finally:
            # پاک کردن فایل موقت
            try:
                os.remove(filepath)
            except:
                pass
                
    def _process_txt(self, filepath: str) -> list:
        """پردازش فایل متنی"""
        results = []
        
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        for line in lines:
            line = line.strip()
            if '|' in line:
                parts = line.split('|', 1)
                if len(parts) == 2:
                    results.append({
                        'question': parts[0].strip(),
                        'answer': parts[1].strip(),
                        'category': 'file_upload'
                    })
                    
        return results
        
    def _process_csv(self, filepath: str) -> list:
        """پردازش فایل CSV"""
        results = []
        
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2:
                    results.append({
                        'question': row[0].strip(),
                        'answer': row[1].strip(),
                        'category': row[2].strip() if len(row) > 2 else 'file_upload'
                    })
                    
        return results
        
    def _process_json(self, filepath: str) -> list:
        """پردازش فایل JSON"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            return [data]
        else:
            return []
            
    def _process_pdf(self, filepath: str) -> list:
        """پردازش فایل PDF"""
        results = []
        
        with open(filepath, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            text = ""
            
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
                
            # استخراج جفت سوال-جواب از متن
            lines = text.split('\n')
            for line in lines:
                if '?' in line or '؟' in line:
                    parts = line.split('?', 1) if '?' in line else line.split('؟', 1)
                    if len(parts) == 2:
                        results.append({
                            'question': parts[0].strip() + '؟',
                            'answer': parts[1].strip(),
                            'category': 'pdf_upload'
                        })
                        
        return results
        
    def _process_docx(self, filepath: str) -> list:
        """پردازش فایل Word"""
        results = []
        
        doc = docx.Document(filepath)
        text = ""
        
        for para in doc.paragraphs:
            text += para.text + "\n"
            
        lines = text.split('\n')
        for line in lines:
            if '|' in line:
                parts = line.split('|', 1)
                if len(parts) == 2:
                    results.append({
                        'question': parts[0].strip(),
                        'answer': parts[1].strip(),
                        'category': 'docx_upload'
                    })
                    
        return results
        
    def _process_excel(self, filepath: str) -> list:
        """پردازش فایل Excel"""
        results = []
        
        df = pd.read_excel(filepath)
        
        if 'question' in df.columns and 'answer' in df.columns:
            for _, row in df.iterrows():
                results.append({
                    'question': str(row['question']),
                    'answer': str(row['answer']),
                    'category': row.get('category', 'excel_upload')
                })
                
        return results
        
    def _process_image(self, filepath: str) -> list:
        """پردازش تصویر با OCR"""
        results = []
        
        image = Image.open(filepath)
        text = pytesseract.image_to_string(image, lang='fas+eng')
        
        lines = text.split('\n')
        for line in lines:
            if '|' in line:
                parts = line.split('|', 1)
                if len(parts) == 2:
                    results.append({
                        'question': parts[0].strip(),
                        'answer': parts[1].strip(),
                        'category': 'image_ocr'
                    })
                    
        return results
        
    def get_status(self, file_id: str) -> dict:
        """دریافت وضعیت پردازش"""
        return self.processing_status.get(file_id, {
            'status': 'not_found'
        })
        
    def get_all_status(self, user_id: str = None) -> list:
        """دریافت همه وضعیت‌ها"""
        if user_id:
            return [v for v in self.processing_status.values() if v.get('user_id') == user_id]
        return list(self.processing_status.values())
