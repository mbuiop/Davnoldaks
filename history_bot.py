# history_bot.py
from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
import json
import os
from datetime import datetime
import numpy as np

app = Flask(__name__)
app.secret_key = 'history-bot-secret'
CORS(app)

# ================ دیتابیس ساده تاریخ ================
class HistoryDatabase:
    def __init__(self, filename='history_knowledge.json'):
        self.filename = filename
        self.knowledge = []
        self.load()
    
    def load(self):
        """بارگذاری دانش از فایل"""
        if os.path.exists(self.filename):
            with open(self.filename, 'r', encoding='utf-8') as f:
                self.knowledge = json.load(f)
            print(f"📚 {len(self.knowledge)} دانش تاریخی بارگذاری شد")
    
    def save(self):
        """ذخیره دانش در فایل"""
        with open(self.filename, 'w', encoding='utf-8') as f:
            json.dump(self.knowledge, f, ensure_ascii=False, indent=2)
    
    def add_dialogue(self, question, answer, category='general', source='admin'):
        """اضافه کردن دیالوگ جدید به دانش"""
        entry = {
            'id': len(self.knowledge) + 1,
            'question': question.lower().strip(),
            'answer': answer,
            'category': category,
            'source': source,
            'date_added': datetime.now().isoformat(),
            'times_used': 0
        }
        self.knowledge.append(entry)
        self.save()
        return entry
    
    def add_bulk_dialogues(self, dialogues):
        """اضافه کردن چندین دیالوگ یکجا"""
        count = 0
        for q, a in dialogues:
            if q and a:
                self.add_dialogue(q, a, source='admin_bulk')
                count += 1
        return count
    
    def search(self, query, threshold=0.7):
        """جستجوی ساده بر اساس کلمات کلیدی"""
        query = query.lower().strip()
        query_words = set(query.split())
        
        results = []
        for entry in self.knowledge:
            # به‌روزرسانی تعداد استفاده
            entry['times_used'] = entry.get('times_used', 0)
            
            # بررسی تطابق
            q_words = set(entry['question'].split())
            common_words = query_words.intersection(q_words)
            
            if common_words:
                score = len(common_words) / max(len(q_words), 1)
                if score >= threshold:
                    results.append({
                        'answer': entry['answer'],
                        'score': score,
                        'category': entry['category'],
                        'id': entry['id']
                    })
        
        # مرتب‌سازی بر اساس امتیاز
        results.sort(key=lambda x: x['score'], reverse=True)
        return results

# ================ پنل مدیریت ================
db = HistoryDatabase()

@app.route('/')
def index():
    """صفحه اصلی برای کاربران"""
    return render_template_string('''
    <!DOCTYPE html>
    <html>
    <head>
        <title>🤖 تاریخ‌دان - ربات تاریخ</title>
        <style>
            body { font-family: Vazir, Tahoma; max-width: 800px; margin: 0 auto; padding: 20px; background: #f5f5f5; }
            .chat-container { background: white; border-radius: 10px; padding: 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            .message { margin: 10px 0; padding: 10px; border-radius: 5px; }
            .user { background: #e3f2fd; text-align: right; }
            .bot { background: #f1f1f1; text-align: left; }
            input[type=text] { width: 80%; padding: 10px; border: 1px solid #ddd; border-radius: 5px; }
            button { padding: 10px 20px; background: #4CAF50; color: white; border: none; border-radius: 5px; cursor: pointer; }
            .admin-link { position: fixed; top: 10px; right: 10px; background: #2196F3; color: white; padding: 10px; border-radius: 5px; text-decoration: none; }
            .stats { background: #fff3cd; padding: 10px; border-radius: 5px; margin-bottom: 20px; }
        </style>
    </head>
    <body>
        <a href="/admin" class="admin-link">⚙️ پنل مدیریت</a>
        <div class="chat-container">
            <h1>🤖 تاریخ‌دان</h1>
            <div class="stats">
                📚 تعداد دانش: {{ db_size }} | 🗓️ آخرین به‌روزرسانی: {{ last_update }}
            </div>
            <div id="chat-history">
                {% for msg in history %}
                <div class="message {% if msg.role == 'user' %}user{% else %}bot{% endif %}">
                    <strong>{% if msg.role == 'user' %}شما{% else %}ربات{% endif %}:</strong> {{ msg.content }}
                </div>
                {% endfor %}
            </div>
            <form method="POST" action="/chat">
                <input type="text" name="message" placeholder="سوال تاریخی خود را بپرسید..." required>
                <button type="submit">ارسال</button>
            </form>
        </div>
    </body>
    </html>
    ''', db_size=len(db.knowledge), last_update=datetime.now().strftime('%Y-%m-%d %H:%M'))

@app.route('/chat', methods=['POST'])
def chat():
    """پاسخ به سوالات کاربران"""
    question = request.form['message']
    
    # جستجو در دانش
    results = db.search(question)
    
    if results:
        answer = results[0]['answer']
        # به‌روزرسانی آمار استفاده
        for entry in db.knowledge:
            if entry['id'] == results[0]['id']:
                entry['times_used'] += 1
                break
        db.save()
    else:
        answer = "متأسفم! هنوز این موضوع تاریخی رو یاد نگرفتم. لطفاً از پنل مدیریت به من یاد بده 🙏"
    
    # ذخیره در تاریخچه (با session ساده)
    if 'history' not in session:
        session['history'] = []
    session['history'].append({'role': 'user', 'content': question})
    session['history'].append({'role': 'bot', 'content': answer})
    
    return index()

@app.route('/admin')
def admin_panel():
    """پنل مدیریت برای آموزش"""
    return render_template_string('''
    <!DOCTYPE html>
    <html>
    <head>
        <title>⚙️ پنل مدیریت - آموزش تاریخ</title>
        <style>
            body { font-family: Vazir, Tahoma; max-width: 1200px; margin: 0 auto; padding: 20px; }
            .container { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
            .card { background: white; border-radius: 10px; padding: 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            input, textarea { width: 100%; padding: 8px; margin: 5px 0; border: 1px solid #ddd; border-radius: 4px; }
            button { background: #4CAF50; color: white; padding: 10px; border: none; border-radius: 4px; cursor: pointer; }
            .knowledge-item { border-bottom: 1px solid #eee; padding: 10px; }
            .stats { background: #e3f2fd; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
            .bulk-input { width: 100%; height: 200px; font-family: monospace; }
            .success { background: #d4edda; color: #155724; padding: 10px; border-radius: 5px; margin: 10px 0; }
        </style>
    </head>
    <body>
        <h1>⚙️ پنل مدیریت - آموزش تاریخ</h1>
        <div class="stats">
            <p>📚 تعداد کل دانش: {{ db_size }}</p>
            <p>📊 پراستفاده‌ترین: {% if most_used %}{{ most_used.question }} ({{ most_used.times_used }} بار){% endif %}</p>
            <p>🗓️ آخرین آموزش: {{ last_added }}</p>
        </div>
        
        <div class="container">
            <div class="card">
                <h2>📝 آموزش تکی</h2>
                <form action="/admin/add" method="POST">
                    <label>دسته‌بندی:</label>
                    <select name="category">
                        <option>ایران باستان</option>
                        <option>اسلامی</option>
                        <option>معاصر</option>
                        <option>جهان</option>
                        <option>عمومی</option>
                    </select>
                    
                    <label>سوال:</label>
                    <input type="text" name="question" required placeholder="مثال: کوروش کبیر که بود؟">
                    
                    <label>جواب:</label>
                    <textarea name="answer" required rows="4" placeholder="جواب دقیق تاریخی..."></textarea>
                    
                    <button type="submit">➕ اضافه کن</button>
                </form>
            </div>
            
            <div class="card">
                <h2>📚 آموزش گروهی (۱۰۰۰ دیالوگ)</h2>
                <form action="/admin/bulk" method="POST">
                    <label>فرمت: هر خط: سوال | جواب</label>
                    <textarea class="bulk-input" name="dialogues" placeholder="کوروش کبیر که بود؟ | کوروش بزرگ بنیانگذار شاهنشاهی هخامنشی بود...
داریوش چه کرد؟ | داریوش بزرگ امپراتوری را به ساتراپی‌ها تقسیم کرد...
..." required></textarea>
                    <button type="submit">📥 آموزش گروهی</button>
                </form>
            </div>
        </div>
        
        <div class="card">
            <h2>📖 دانش فعلی ({{ db_size }} مورد)</h2>
            <input type="text" id="search" placeholder="جستجو در دانش..." onkeyup="filterKnowledge()">
            <div id="knowledge-list">
                {% for item in knowledge %}
                <div class="knowledge-item" data-text="{{ item.question }} {{ item.answer }}">
                    <strong>{{ item.question }}</strong> ({{ item.category }}) - {{ item.times_used }} بار استفاده
                    <p>{{ item.answer[:100] }}...</p>
                </div>
                {% endfor %}
            </div>
        </div>
        
        <script>
        function filterKnowledge() {
            let search = document.getElementById('search').value.toLowerCase();
            let items = document.getElementsByClassName('knowledge-item');
            for(let item of items) {
                let text = item.getAttribute('data-text').toLowerCase();
                item.style.display = text.includes(search) ? 'block' : 'none';
            }
        }
        </script>
    </body>
    </html>
    ''', 
    db_size=len(db.knowledge),
    most_used=max(db.knowledge, key=lambda x: x.get('times_used', 0)) if db.knowledge else None,
    last_added=db.knowledge[-1]['date_added'][:10] if db.knowledge else 'هیچ',
    knowledge=db.knowledge[-20:]  # ۲۰ مورد آخر
    )

@app.route('/admin/add', methods=['POST'])
def admin_add():
    """اضافه کردن دیالوگ تکی"""
    question = request.form['question']
    answer = request.form['answer']
    category = request.form['category']
    
    db.add_dialogue(question, answer, category)
    return admin_panel()

@app.route('/admin/bulk', methods=['POST'])
def admin_bulk():
    """اضافه کردن گروهی دیالوگ"""
    dialogues_text = request.form['dialogues']
    dialogues = []
    
    for line in dialogues_text.strip().split('\n'):
        if '|' in line:
            q, a = line.split('|', 1)
            dialogues.append((q.strip(), a.strip()))
    
    count = db.add_bulk_dialogues(dialogues)
    
    return f'''
    <html>
    <body style="font-family: Vazir; text-align: center; padding: 50px;">
        <div class="success">
            <h2>✅ {count} دیالوگ با موفقیت اضافه شد!</h2>
            <a href="/admin">🔙 بازگشت به پنل</a>
        </div>
        <script>setTimeout(() => window.location='/admin', 2000);</script>
    </body>
    </html>
    '''

@app.route('/admin/stats')
def admin_stats():
    """آمار استفاده"""
    stats = {
        'total': len(db.knowledge),
        'by_category': {},
        'most_used': [],
        'never_used': []
    }
    
    for entry in db.knowledge:
        cat = entry['category']
        stats['by_category'][cat] = stats['by_category'].get(cat, 0) + 1
        
        if entry.get('times_used', 0) > 0:
            stats['most_used'].append(entry)
        else:
            stats['never_used'].append(entry)
    
    return jsonify(stats)

# ================ راه‌اندازی ================
if __name__ == '__main__':
    # ایجاد چند نمونه اولیه
    if len(db.knowledge) == 0:
        sample_data = [
            ("کوروش کبیر که بود؟", "کوروش بزرگ بنیانگذار شاهنشاهی هخامنشی بود که در سال ۵۵۹ پیش از میلاد تاسیس شد."),
            ("داریوش بزرگ چه کرد؟", "داریوش بزرگ امپراتوری را به ساتراپی‌ها تقسیم کرد و جاده شاهی را ساخت."),
            ("خشایارشا که بود؟", "خشایارشا پسر داریوش بزرگ بود که به یونان لشکر کشید."),
        ]
        db.add_bulk_dialogues(sample_data)
    
    print("""
    ╔══════════════════════════════════════╗
    ║   🤖 ربات تاریخ‌دان آماده کار است    ║
    ╠══════════════════════════════════════╣
    ║ 📚 دانش: {} مورد                      ║
    ║ 🌐 آدرس: http://localhost:5000        ║
    ║ ⚙️ پنل: http://localhost:5000/admin   ║
    ╚══════════════════════════════════════╝
    """.format(len(db.knowledge)))
    
    app.run(debug=True, host='0.0.0.0', port=5000)
