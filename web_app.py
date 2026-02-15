# web_app.py - رابط کاربری وب با قابلیت مقیاس‌پذیری
# ------------------------------------------------

from flask import Flask, render_template, request, jsonify, session, render_template_string, redirect, url_for
from flask_cors import CORS
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import hashlib
import os
import time
from werkzeug.utils import secure_filename
from ai_engine import ScalablePersianAI
import redis
import json

app = Flask(__name__)
app.config['SECRET_KEY'] = 'super-secret-key-for-production'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024
CORS(app)

# محدود کننده نرخ برای جلوگیری از حمله
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

# ایجاد پوشه‌ها
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ================ هوش مصنوعی ================
ai = ScalablePersianAI()

# ================ مدیریت کاربران ================
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'admin_login'

class User(UserMixin):
    def __init__(self, id, username, password):
        self.id = id
        self.username = username
        self.password = password

users = {
    '1': User('1', 'admin', hashlib.md5('admin123'.encode()).hexdigest())
}

@login_manager.user_loader
def load_user(user_id):
    return users.get(user_id)

# ================ صفحه اصلی چت ================
@app.route('/')
@limiter.limit("30 per minute")
def index():
    return render_template_string('''
    <!DOCTYPE html>
    <html lang="fa" dir="rtl">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>هوش مصنوعی ایرانی</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            
            :root {
                --primary: #6c5ce7;
                --secondary: #a363d9;
                --bg: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            }
            
            body {
                font-family: 'Vazir', 'Tahoma', sans-serif;
                background: var(--bg);
                min-height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
                padding: 12px;
            }
            
            .chat-container {
                width: 100%;
                max-width: 500px;
                height: 90vh;
                background: rgba(255, 255, 255, 0.98);
                border-radius: 30px;
                box-shadow: 0 25px 50px rgba(0,0,0,0.3);
                display: flex;
                flex-direction: column;
                overflow: hidden;
            }
            
            .chat-header {
                background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
                color: white;
                padding: 20px;
                display: flex;
                align-items: center;
                justify-content: space-between;
            }
            
            .menu-btn {
                background: none;
                border: none;
                color: white;
                font-size: 28px;
                cursor: pointer;
                width: 44px;
                height: 44px;
                border-radius: 50%;
            }
            
            .chat-messages {
                flex: 1;
                overflow-y: auto;
                padding: 20px;
                background: #f8fafc;
                display: flex;
                flex-direction: column;
                gap: 15px;
            }
            
            .message {
                display: flex;
                animation: slideIn 0.3s ease;
            }
            
            @keyframes slideIn {
                from { opacity: 0; transform: translateY(10px); }
                to { opacity: 1; transform: translateY(0); }
            }
            
            .message.user { justify-content: flex-end; }
            .message.bot { justify-content: flex-start; }
            
            .message-content {
                max-width: 85%;
                padding: 14px 18px;
                border-radius: 25px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                line-height: 1.6;
                word-wrap: break-word;
            }
            
            .user .message-content {
                background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
                color: white;
                border-bottom-right-radius: 5px;
            }
            
            .bot .message-content {
                background: white;
                border-bottom-left-radius: 5px;
            }
            
            .message-time {
                font-size: 0.7em;
                opacity: 0.7;
                margin-top: 5px;
            }
            
            .typing-indicator {
                padding: 14px 20px;
                background: white;
                border-radius: 25px;
                display: inline-block;
            }
            
            .typing-indicator span {
                display: inline-block;
                width: 8px;
                height: 8px;
                border-radius: 50%;
                background: var(--primary);
                margin: 0 3px;
                animation: typing 1.4s infinite;
            }
            
            .chat-input-container {
                padding: 16px 20px;
                background: white;
                border-top: 1px solid #eee;
                display: flex;
                gap: 12px;
            }
            
            .chat-input {
                flex: 1;
                padding: 14px 20px;
                border: 2px solid #e0e0e0;
                border-radius: 30px;
                font-size: 1rem;
                outline: none;
                font-family: inherit;
            }
            
            .chat-input:focus {
                border-color: var(--primary);
                box-shadow: 0 0 0 3px rgba(108,92,231,0.1);
            }
            
            .send-btn {
                width: 52px;
                height: 52px;
                border-radius: 50%;
                background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
                color: white;
                border: none;
                cursor: pointer;
                font-size: 1.4em;
            }
            
            .menu-panel {
                position: fixed;
                top: 0;
                right: -300px;
                width: 280px;
                height: 100%;
                background: white;
                transition: right 0.3s ease;
                box-shadow: -5px 0 30px rgba(0,0,0,0.2);
                padding: 20px;
                z-index: 1001;
            }
            
            .menu-panel.open { right: 0; }
            
            .menu-overlay {
                position: fixed;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: rgba(0,0,0,0.5);
                display: none;
                z-index: 1000;
            }
            
            .menu-item {
                padding: 15px;
                margin: 5px 0;
                border-radius: 15px;
                cursor: pointer;
                display: flex;
                align-items: center;
                gap: 15px;
                text-decoration: none;
                color: #333;
            }
            
            .menu-item:hover { background: #f0f2f5; }
            
            .stats-badge {
                position: fixed;
                bottom: 10px;
                left: 10px;
                background: rgba(0,0,0,0.5);
                color: white;
                padding: 5px 10px;
                border-radius: 20px;
                font-size: 0.8em;
            }
        </style>
    </head>
    <body>
        <div class="chat-container">
            <div class="chat-header">
                <button class="menu-btn" onclick="toggleMenu()">☰</button>
                <div style="font-size: 1.4em;">🤖 هوش ایرانی</div>
                <div style="width: 44px;"></div>
            </div>
            
            <div class="chat-messages" id="chat-messages">
                <div class="message bot">
                    <div class="message-content">
                        سلام! هر سوالی داری بپرس. من یاد می‌گیرم و بهتر جواب می‌دم.
                        <div class="message-time">{{ now }}</div>
                    </div>
                </div>
            </div>
            
            <div class="chat-input-container">
                <input type="text" class="chat-input" id="message-input" 
                       placeholder="سوال خود را بپرسید..." 
                       onkeypress="if(event.key==='Enter') sendMessage()">
                <button class="send-btn" onclick="sendMessage()">➤</button>
            </div>
        </div>
        
        <div class="menu-overlay" id="menuOverlay" onclick="closeMenu()"></div>
        <div class="menu-panel" id="menuPanel">
            <h3 style="margin-bottom: 20px;">منو</h3>
            <a href="/m.html" class="menu-item">📄 صفحه M</a>
            <a href="/admin-login" class="menu-item">⚙️ پنل مدیریت</a>
            <div class="menu-item" onclick="clearHistory()">🗑️ پاک کردن تاریخچه</div>
        </div>
        
        <div class="stats-badge" id="stats">
            {{ stats.total_requests }} درخواست
        </div>
        
        <script>
            let chatHistory = JSON.parse(localStorage.getItem('chat_history')) || [];
            
            chatHistory.forEach(msg => {
                addMessage(msg.text, msg.isUser, msg.time, false);
            });
            
            function toggleMenu() {
                document.getElementById('menuOverlay').style.display = 'block';
                document.getElementById('menuPanel').classList.add('open');
            }
            
            function closeMenu() {
                document.getElementById('menuOverlay').style.display = 'none';
                document.getElementById('menuPanel').classList.remove('open');
            }
            
            function addMessage(text, isUser = false, time = null, save = true) {
                const div = document.createElement('div');
                div.className = `message ${isUser ? 'user' : 'bot'}`;
                
                const msgTime = time || new Date().toLocaleTimeString('fa-IR');
                
                div.innerHTML = `
                    <div class="message-content">
                        ${text}
                        <div class="message-time">${msgTime}</div>
                    </div>
                `;
                
                document.getElementById('chat-messages').appendChild(div);
                div.scrollIntoView({ behavior: 'smooth' });
                
                if (save) {
                    chatHistory.push({ text, isUser, time: msgTime });
                    if (chatHistory.length > 50) chatHistory = chatHistory.slice(-50);
                    localStorage.setItem('chat_history', JSON.stringify(chatHistory));
                }
            }
            
            function showTyping() {
                const div = document.createElement('div');
                div.className = 'message bot';
                div.id = 'typing';
                div.innerHTML = '<div class="typing-indicator"><span></span><span></span><span></span></div>';
                document.getElementById('chat-messages').appendChild(div);
                div.scrollIntoView({ behavior: 'smooth' });
            }
            
            function hideTyping() {
                const typing = document.getElementById('typing');
                if (typing) typing.remove();
            }
            
            async function sendMessage() {
                const input = document.getElementById('message-input');
                const message = input.value.trim();
                if (!message) return;
                
                addMessage(message, true);
                input.value = '';
                showTyping();
                
                try {
                    const response = await fetch('/api/chat', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({message})
                    });
                    
                    const data = await response.json();
                    hideTyping();
                    
                    if (data.answer) {
                        addMessage(data.answer);
                    } else {
                        addMessage('🤔 متأسفم! نتونستم پیدا کنم.');
                    }
                    
                    // به‌روزرسانی آمار
                    document.getElementById('stats').innerText = data.stats + ' درخواست';
                    
                } catch (error) {
                    hideTyping();
                    addMessage('⚠️ خطا');
                }
            }
            
            function clearHistory() {
                if (confirm('پاک شود؟')) {
                    localStorage.removeItem('chat_history');
                    chatHistory = [];
                    location.reload();
                }
            }
        </script>
    </body>
    </html>
    ''', now=time.strftime('%H:%M'), stats=ai.get_stats())

# ================ صفحه M ================
@app.route('/m.html')
def m_page():
    return render_template_string('''
    <!DOCTYPE html>
    <html>
    <head>
        <title>صفحه M</title>
        <style>
            body {
                font-family: Tahoma;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
                padding: 20px;
            }
            .container {
                background: white;
                border-radius: 30px;
                padding: 40px;
                max-width: 600px;
                text-align: center;
            }
            .btn {
                display: inline-block;
                padding: 15px 40px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                text-decoration: none;
                border-radius: 30px;
                margin-top: 20px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📄 صفحه M</h1>
            <p>این صفحه مخصوص منوی کشویی است</p>
            <a href="/" class="btn">بازگشت</a>
        </div>
    </body>
    </html>
    ''')

# ================ API چت ================
@app.route('/api/chat', methods=['POST'])
@limiter.limit("10 per minute")
def api_chat():
    try:
        data = request.json
        question = data.get('message', '').strip()
        
        if not question:
            return jsonify({'error': 'سوال نمی‌تواند خالی باشد'})
        
        result = ai.ask(question)
        stats = ai.get_stats()
        
        if result['found']:
            return jsonify({
                'answer': result['answer'],
                'method': result.get('method', ''),
                'found': True,
                'stats': stats['total_requests']
            })
        else:
            return jsonify({
                'answer': None,
                'found': False,
                'stats': stats['total_requests']
            })
            
    except Exception as e:
        return jsonify({'error': str(e)})

# ================ پنل مدیریت ================
@app.route('/admin-login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form['username']
        password = hashlib.md5(request.form['password'].encode()).hexdigest()
        
        if username == 'admin' and password == hashlib.md5('admin123'.encode()).hexdigest():
            login_user(users['1'])
            return redirect(url_for('admin_panel'))
        
        return "❌ رمز اشتباه است"
    
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>ورود</title>
        <style>
            body {
                font-family: Tahoma;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
            }
            .login-box {
                background: white;
                padding: 40px;
                border-radius: 30px;
                width: 400px;
            }
            input, button {
                width: 100%;
                padding: 15px;
                margin: 10px 0;
                border-radius: 15px;
                border: 2px solid #e0e0e0;
            }
            button {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                cursor: pointer;
            }
        </style>
    </head>
    <body>
        <div class="login-box">
            <h2>🔐 پنل مدیریت</h2>
            <form method="POST">
                <input type="text" name="username" value="admin">
                <input type="password" name="password" value="admin123">
                <button type="submit">ورود</button>
            </form>
        </div>
    </body>
    </html>
    '''

@app.route('/admin')
@login_required
def admin_panel():
    stats = ai.get_stats()
    popular = ai.get_popular_questions(10)
    
    return f'''
    <!DOCTYPE html>
    <html>
    <head>
        <title>پنل مدیریت</title>
        <style>
            body {{
                font-family: Tahoma;
                background: #f5f5f5;
                padding: 20px;
            }}
            .header {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 20px;
                border-radius: 15px;
                margin-bottom: 20px;
                display: flex;
                justify-content: space-between;
            }}
            .stats-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin-bottom: 20px;
            }}
            .stat-card {{
                background: white;
                padding: 20px;
                border-radius: 15px;
                text-align: center;
            }}
            .stat-number {{
                font-size: 2em;
                color: #667eea;
                font-weight: bold;
            }}
            .card {{
                background: white;
                padding: 20px;
                border-radius: 15px;
                margin-bottom: 20px;
            }}
            textarea, input, select {{
                width: 100%;
                padding: 12px;
                margin: 10px 0;
                border: 2px solid #e0e0e0;
                border-radius: 10px;
            }}
            button {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 12px 25px;
                border: none;
                border-radius: 10px;
                cursor: pointer;
            }}
            .file-upload {{
                border: 2px dashed #667eea;
                padding: 30px;
                text-align: center;
                border-radius: 10px;
                cursor: pointer;
                margin: 20px 0;
            }}
            .grid-2 {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 20px;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h2>⚙️ پنل مدیریت</h2>
            <div>
                <a href="/" style="color: white; margin-right: 15px;">چت</a>
                <a href="/logout" style="color: white;">خروج</a>
            </div>
        </div>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-number">{stats['knowledge_count']}</div>
                <div>کل دانش</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{stats['total_learned']}</div>
                <div>یادگیری‌ها</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{stats['total_asked']}</div>
                <div>سوالات</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{stats['avg_response_ms']:.1f}ms</div>
                <div>زمان پاسخ</div>
            </div>
        </div>
        
        <div class="grid-2">
            <div class="card">
                <h3>➕ آموزش تکی</h3>
                <form action="/admin/learn" method="POST">
                    <input type="text" name="question" placeholder="سوال" required>
                    <textarea name="answer" rows="4" placeholder="پاسخ" required></textarea>
                    <button type="submit">📚 یاد بگیر</button>
                </form>
            </div>
            
            <div class="card">
                <h3>📁 آپلود فایل</h3>
                <form action="/admin/learn/file" method="POST" enctype="multipart/form-data">
                    <div class="file-upload" onclick="document.getElementById('file').click()">
                        <p>📤 کلیک برای آپلود</p>
                        <p style="font-size:0.9em;">فرمت: سوال | جواب</p>
                    </div>
                    <input type="file" id="file" name="file" style="display:none;" accept=".txt">
                    <button type="submit">📚 یادگیری از فایل</button>
                </form>
            </div>
        </div>
        
        <div class="card">
            <h3>🔥 سوالات پرتکرار</h3>
            <ul>
                {''.join([f'<li style="margin:10px 0;">{q} - {c} بار</li>' for q, c in popular])}
            </ul>
        </div>
        
        <div class="card">
            <h3>⚡ آمار پیشرفته</h3>
            <p>درخواست‌ها: {stats['total_requests']}</p>
            <p>درخواست/ثانیه: {stats['requests_per_second']:.2f}</p>
            <p>آپتایم: {stats['uptime']/3600:.1f} ساعت</p>
            <p>کش: {stats['cache_size']} آیتم</p>
        </div>
        
        <script>
            function copyCode() {{
                const code = document.querySelector('textarea').select();
                document.execCommand('copy');
                alert('✅ کپی شد');
            }}
        </script>
    </body>
    </html>
    '''

# ================ یادگیری ================
@app.route('/admin/learn', methods=['POST'])
@login_required
def learn():
    question = request.form['question']
    answer = request.form['answer']
    
    learned, _ = ai.learn(f"{question} | {answer}", 'manual')
    return redirect(url_for('admin_panel'))

@app.route('/admin/learn/file', methods=['POST'])
@login_required
def learn_file():
    try:
        if 'file' not in request.files:
            return "❌ فایلی انتخاب نشده"
        
        file = request.files['file']
        if file.filename == '':
            return "❌ نام فایل معتبر نیست"
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        learned, _ = ai.learn(content, 'file')
        os.remove(filepath)
        
        return f"✅ {learned} مورد یاد گرفته شد<br><a href='/admin'>بازگشت</a>"
        
    except Exception as e:
        return f"❌ خطا: {str(e)}<br><a href='/admin'>بازگشت</a>"

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

if __name__ == '__main__':
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║     🤖 هوش مصنوعی مقیاس‌پذیر برای ۱ میلیون کاربر        ║
    ╠══════════════════════════════════════════════════════════╣
    ║  📚 دانش: {} مورد                                         ║
    ║  ⚡ زمان پاسخ: {:.1f}ms                                   ║
    ║  🌐 چت: http://localhost:5000                           ║
    ║  🔐 پنل: http://localhost:5000/admin-login              ║
    ║  👤 کاربر: admin / admin123                              ║
    ║  📁 فایل‌ها: ۴ فایل - مقیاس‌پذیر کامل                   ║
    ╚══════════════════════════════════════════════════════════╝
    """.format(len(ai.engine.knowledge_base), ai.get_stats()['avg_response_ms']))
    
    app.run(debug=True, host='0.0.0.0', port=5000)
