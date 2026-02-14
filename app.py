from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
import torch
import os
import uuid
from datetime import datetime
import threading
import queue

from brain import GiantBrain, Config, create_brain, load_brain

app = Flask(__name__)
app.secret_key = 'super-secret-key-giant-brain'
CORS(app)

# ========== بارگذاری مغز ==========
print("🚀 Loading Giant Brain...")
config = Config()
brain = None

if os.path.exists(os.path.join(config.model_path, "best_model.pt")):
    brain = load_brain("best_model.pt", config)
    print("✅ Brain loaded from checkpoint")
else:
    brain = create_brain(config)
    print("✅ New brain created")

# ========== صف آموزش ==========
training_queue = queue.Queue()
training_status = {
    'is_training': False,
    'current_file': '',
    'progress': 0,
    'total_files': 0,
    'loss': 0
}

# ========== صفحات ==========

@app.route('/')
def index():
    """صفحه اصلی چت"""
    return render_template('index.html')

@app.route('/admin')
def admin():
    """پنل مدیریت"""
    return render_template('index.html')

# ========== API چت ==========

@app.route('/api/chat', methods=['POST'])
def chat():
    """چت با مغز"""
    data = request.json
    message = data.get('message', '').strip()
    
    if not message:
        return jsonify({'error': 'Message is empty'}), 400
    
    try:
        response = brain.generate(message, max_length=200, temperature=0.7)
        return jsonify({'response': response})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ========== API یادگیری ==========

@app.route('/api/learn/text', methods=['POST'])
def learn_text():
    """یادگیری از متن"""
    data = request.json
    text = data.get('text', '').strip()
    
    if not text:
        return jsonify({'error': 'Text is empty'}), 400
    
    try:
        loss = brain.learn_from_text(text)
        return jsonify({
            'success': True,
            'loss': loss,
            'stats': brain.get_stats()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/learn/file', methods=['POST'])
def learn_file():
    """یادگیری از فایل"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Save file temporarily
    temp_path = f"temp_{file.filename}"
    file.save(temp_path)
    
    try:
        loss = brain.learn_from_file(temp_path)
        os.remove(temp_path)
        
        return jsonify({
            'success': True,
            'loss': loss,
            'stats': brain.get_stats()
        })
    except Exception as e:
        os.remove(temp_path)
        return jsonify({'error': str(e)}), 500

@app.route('/api/learn/directory', methods=['POST'])
def learn_directory():
    """یادگیری از پوشه"""
    data = request.json
    directory = data.get('directory', '').strip()
    
    if not directory or not os.path.exists(directory):
        return jsonify({'error': 'Invalid directory'}), 400
    
    def train_in_background():
        global training_status
        training_status['is_training'] = True
        
        try:
            loss = brain.learn_from_directory(directory)
            training_status['loss'] = loss
        except Exception as e:
            training_status['error'] = str(e)
        finally:
            training_status['is_training'] = False
    
    if not training_status['is_training']:
        thread = threading.Thread(target=train_in_background)
        thread.start()
        return jsonify({'success': True, 'message': 'Training started'})
    else:
        return jsonify({'error': 'Already training'}), 400

@app.route('/api/learn/status', methods=['GET'])
def training_status_api():
    """وضعیت آموزش"""
    return jsonify(training_status)

@app.route('/api/learn/stop', methods=['POST'])
def stop_training():
    """توقف آموزش"""
    global training_status
    training_status['is_training'] = False
    return jsonify({'success': True})

# ========== API مدیریت ==========

@app.route('/api/brain/save', methods=['POST'])
def save_brain():
    """ذخیره مغز"""
    data = request.json
    filename = data.get('filename', 'manual_save.pt')
    brain.save_checkpoint(filename)
    return jsonify({'success': True})

@app.route('/api/brain/load', methods=['POST'])
def load_brain_api():
    """بارگذاری مغز"""
    data = request.json
    filename = data.get('filename', 'best_model.pt')
    success = brain.load_checkpoint(filename)
    return jsonify({'success': success})

@app.route('/api/brain/stats', methods=['GET'])
def brain_stats():
    """آمار مغز"""
    return jsonify(brain.get_stats())

@app.route('/api/brain/config', methods=['GET'])
def brain_config():
    """تنظیمات مغز"""
    config_dict = {
        'vocab_size': config.vocab_size,
        'hidden_size': config.hidden_size,
        'num_layers': config.num_layers,
        'num_heads': config.num_heads,
        'memory_size': config.memory_size
    }
    return jsonify(config_dict)

@app.route('/api/brain/generate', methods=['POST'])
def generate_text():
    """تولید متن (برای تست)"""
    data = request.json
    prompt = data.get('prompt', '')
    max_length = data.get('max_length', 100)
    temperature = data.get('temperature', 0.7)
    
    try:
        response = brain.generate(prompt, max_length, temperature)
        return jsonify({'response': response})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ========== اجرا ==========

if __name__ == '__main__':
    print("="*60)
    print("🚀 GIANT BRAIN SERVER READY")
    print(f"📊 Stats: {brain.get_stats()}")
    print(f"🌐 Chat: http://127.0.0.1:5000")
    print(f"⚙️  Admin: http://127.0.0.1:5000/admin")
    print("="*60)
    app.run(debug=True, host='0.0.0.0', port=5000)
