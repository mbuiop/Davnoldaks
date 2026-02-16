# m10.py - Admin API
from flask import Blueprint, request, jsonify, redirect, url_for, render_template
from flask_login import login_required, current_user
import os
from werkzeug.utils import secure_filename

admin_bp = Blueprint('admin', __name__)

class AdminAPI:
    def __init__(self, knowledge, unanswered, gemini, file_processor, learning):
        self.knowledge = knowledge
        self.unanswered = unanswered
        self.gemini = gemini
        self.file_processor = file_processor
        self.learning = learning
        
    def register_routes(self, bp):
        
        @bp.route('/admin')
        @login_required
        def admin_panel():
            """پنل مدیریت"""
            stats = {
                'total': len(self.knowledge.data),
                'unanswered': len(self.unanswered.get_pending()),
                'gemini_stats': self.gemini.get_stats()
            }
            return render_template('index.html', 
                                 stats=stats, 
                                 user=current_user,
                                 page='admin')
        
        @bp.route('/admin/add', methods=['POST'])
        @login_required
        def add_knowledge():
            """افزودن دانش تکی"""
            question = request.form['question']
            answer = request.form['answer']
            category = request.form.get('category', 'عمومی')
            
            success, message = self.knowledge.add(question, answer, category)
            
            return jsonify({'success': success, 'message': message})
        
        @bp.route('/admin/upload', methods=['POST'])
        @login_required
        def upload_file():
            """آپلود فایل"""
            if 'file' not in request.files:
                return jsonify({'error': 'فایلی انتخاب نشده'}), 400
                
            files = request.files.getlist('file')
            tasks = []
            
            for file in files:
                if file and file.filename:
                    filename = secure_filename(file.filename)
                    filepath = os.path.join('uploads', filename)
                    file.save(filepath)
                    
                    task_id = self.file_processor.process(
                        filepath, filename, current_user.id
                    )
                    tasks.append({'filename': filename, 'task_id': task_id})
                    
            return jsonify({'tasks': tasks})
        
        @bp.route('/admin/unanswered')
        @login_required
        def get_unanswered():
            """دریافت سوالات بی‌پاسخ"""
            return jsonify(self.unanswered.get_pending())
        
        @bp.route('/admin/stats')
        @login_required
        def get_stats():
            """آمار کامل"""
            return jsonify({
                'knowledge': self.knowledge.get_stats(),
                'unanswered': len(self.unanswered.data),
                'gemini': self.gemini.get_stats(),
                'learning': self.learning.popular_queries.most_common(20)
            })
        
        @bp.route('/admin/learn/<int:item_id>', methods=['POST'])
        @login_required
        def mark_learned(item_id):
            """علامت‌گذاری به عنوان یادگرفته شده"""
            success = self.unanswered.mark_answered(item_id)
            return jsonify({'success': success})
