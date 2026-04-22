from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session
from flask_login import LoginManager, login_user, logout_user, login_required, current_user, UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from textblob import TextBlob
import sqlite3
import os
import re
import math
from datetime import datetime
from collections import deque

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your-secret-key-here')
app.config['SESSION_TYPE'] = 'filesystem'

# Database path - works on both local and cloud
if 'RENDER' in os.environ or 'SNAPDEPLOY' in os.environ:
    DATABASE = '/tmp/helpdesk.db'
else:
    DATABASE = 'instance/helpdesk.db'

os.makedirs(os.path.dirname(DATABASE) if os.path.dirname(DATABASE) else '.', exist_ok=True)

def get_db():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    c = conn.cursor()
    
    c.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        role TEXT DEFAULT 'user'
    )''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS chat_sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        title TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS chat_messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id INTEGER,
        user_id INTEGER,
        user_message TEXT NOT NULL,
        bot_response TEXT NOT NULL,
        category TEXT,
        confidence REAL,
        sentiment REAL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS escalations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        user_message TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS feedback (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        message_id INTEGER,
        rating INTEGER,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )''')
    
    admin = c.execute("SELECT * FROM users WHERE username = 'admin'").fetchone()
    if not admin:
        c.execute("INSERT INTO users (username, email, password_hash, role) VALUES (?, ?, ?, ?)",
                  ('admin', 'admin@helpdesk.com', generate_password_hash('admin123'), 'admin'))
        print("Admin created: admin / admin123")
    
    conn.commit()
    conn.close()
    print("Database ready")

# ========== PURE ML MODEL ==========
class PureMLModel:
    def __init__(self):
        self.documents = []
        self.categories = []
        self.idf = {}
        self.trained = False
        
    def clean(self, text):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        return text.split()
    
    def train(self):
        data = [
            ("my printer is not working", "printer"),
            ("printer won't print", "printer"),
            ("printer says offline", "printer"),
            ("paper jam", "printer"),
            ("forgot my password", "password"),
            ("can't login", "password"),
            ("reset password", "password"),
            ("wifi not connecting", "network"),
            ("internet is down", "network"),
            ("no internet", "network"),
            ("computer is slow", "performance"),
            ("slow computer", "performance"),
            ("outlook not opening", "software"),
            ("excel crashing", "software"),
            ("email not sending", "email"),
            ("vpn not connecting", "vpn"),
            ("talk to human", "escalation"),
            ("speak to agent", "escalation"),
        ]
        
        for text, cat in data:
            tokens = self.clean(text)
            self.documents.append(tokens)
            self.categories.append(cat)
        
        all_tokens = [t for doc in self.documents for t in doc]
        for token in set(all_tokens):
            count = sum(1 for doc in self.documents if token in doc)
            self.idf[token] = math.log(len(self.documents) / (1 + count))
        
        self.trained = True
        print(f"ML Model trained with {len(data)} examples")
        return True
    
    def predict(self, text):
        if not self.trained:
            self.train()
        
        tokens = self.clean(text)
        if not tokens:
            return 'unknown', 0.2
        
        scores = {}
        for i, doc in enumerate(self.documents):
            score = 0
            for token in tokens:
                if token in doc:
                    score += self.idf.get(token, 0)
            cat = self.categories[i]
            scores[cat] = scores.get(cat, 0) + score
        
        if not scores:
            return 'unknown', 0.2
        
        best = max(scores, key=scores.get)
        max_score = scores[best]
        confidence = min(0.95, max_score / 10 + 0.3)
        return best, confidence

ml_model = PureMLModel()

def analyze_sentiment(text):
    try:
        blob = TextBlob(text)
        return blob.sentiment.polarity
    except:
        return 0.0

def get_response(category, confidence, sentiment):
    responses = {
        'printer': f"🔧 Printer Fix ({confidence:.0f}%)\n\n1. Check power and paper\n2. Restart printer and computer\n3. Clear print queue\n\nType 'human' for IT support",
        'password': f"🔐 Password Reset ({confidence:.0f}%)\n\n1. Click Forgot Password\n2. Check email (Spam folder too)\n3. Create strong password\n\nType 'human' if locked out",
        'network': f"🌐 WiFi Fix ({confidence:.0f}%)\n\n1. Restart router (unplug 30 sec)\n2. Restart computer\n3. Forget and reconnect to WiFi\n\nType 'human' for help",
        'performance': f"⚡ Slow Computer ({confidence:.0f}%)\n\n1. Restart computer\n2. Close unused programs\n3. Run Disk Cleanup\n\nType 'human' for help",
        'software': f"💻 App Fix ({confidence:.0f}%)\n\n1. Restart the app\n2. Restart computer\n3. Reinstall if needed\n\nType 'human' for help",
        'email': f"📧 Email Fix ({confidence:.0f}%)\n\n1. Close and reopen Outlook\n2. Check internet connection\n3. Try web version\n\nType 'human' for support",
        'vpn': f"🔒 VPN Fix ({confidence:.0f}%)\n\n1. Check internet\n2. Restart VPN app\n3. Try different server\n\nType 'human' for support",
        'escalation': f"👤 Human Support Requested\n\nAgent notified. Will contact within 15 minutes.\nTicket: {datetime.now().strftime('%H%M%S')}",
        'unknown': f"🤔 Need more details ({confidence:.0f}%)\n\nWhat happened? Any error message?\n\nTry: printer, password, or wifi"
    }
    return responses.get(category, responses['unknown'])

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

class User(UserMixin):
    def __init__(self, id, username, email, role):
        self.id = id
        self.username = username
        self.email = email
        self.role = role

@login_manager.user_loader
def load_user(user_id):
    conn = get_db()
    user = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    conn.close()
    if user:
        return User(user['id'], user['username'], user['email'], user['role'])
    return None

# ========== ROUTES ==========
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat')
@login_required
def chat():
    return render_template('chat.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        conn = get_db()
        user = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        conn.close()
        if user and check_password_hash(user['password_hash'], password):
            login_user(User(user['id'], user['username'], user['email'], user['role']))
            flash('Login successful!')
            return redirect(url_for('chat'))
        else:
            flash('Invalid username or password')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        conn = get_db()
        existing = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        if existing:
            conn.close()
            flash('Username exists')
            return redirect(url_for('register'))
        conn.execute("INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)", 
                     (username, email, generate_password_hash(password)))
        conn.commit()
        conn.close()
        flash('Registration successful!')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/api/chat', methods=['POST'])
@login_required
def chat_api():
    try:
        data = request.get_json()
        msg = data.get('message', '').strip()
        session_id = data.get('session_id')
        
        if not msg:
            return jsonify({'error': 'Empty'}), 400
        
        if not session_id:
            conn = get_db()
            cursor = conn.execute(
                "INSERT INTO chat_sessions (user_id, title) VALUES (?, ?)",
                (current_user.id, 'New Chat')
            )
            session_id = cursor.lastrowid
            conn.commit()
            conn.close()
        
        if msg.lower() in ['human', 'help', 'support', 'agent']:
            conn = get_db()
            conn.execute("INSERT INTO escalations (user_id, user_message) VALUES (?, ?)", 
                        (current_user.id, msg))
            conn.commit()
            conn.close()
            response = get_response('escalation', 1.0, 0)
            return jsonify({'response': response, 'category': 'escalation', 'confidence': 1.0, 'session_id': session_id})
        
        sentiment = analyze_sentiment(msg)
        category, confidence = ml_model.predict(msg)
        response = get_response(category, confidence, sentiment)
        
        conn = get_db()
        session = conn.execute("SELECT title FROM chat_sessions WHERE id = ?", (session_id,)).fetchone()
        if session and session['title'] == 'New Chat':
            new_title = msg[:40] + ('...' if len(msg) > 40 else '')
            conn.execute("UPDATE chat_sessions SET title = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?", 
                        (new_title, session_id))
        
        cursor = conn.execute(
            "INSERT INTO chat_messages (session_id, user_id, user_message, bot_response, category, confidence, sentiment) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (session_id, current_user.id, msg, response, category, confidence, sentiment)
        )
        message_id = cursor.lastrowid
        conn.execute("UPDATE chat_sessions SET updated_at = CURRENT_TIMESTAMP WHERE id = ?", (session_id,))
        conn.commit()
        conn.close()
        
        return jsonify({
            'response': response,
            'category': category,
            'confidence': confidence,
            'session_id': session_id,
            'message_id': message_id
        })
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/sessions')
@login_required
def get_sessions():
    conn = get_db()
    sessions = conn.execute(
        "SELECT id, title, created_at, updated_at FROM chat_sessions WHERE user_id = ? ORDER BY updated_at DESC",
        (current_user.id,)
    ).fetchall()
    conn.close()
    return jsonify([{
        'id': s['id'],
        'title': s['title'],
        'created_at': s['created_at'],
        'updated_at': s['updated_at']
    } for s in sessions])

@app.route('/api/sessions', methods=['POST'])
@login_required
def create_session():
    data = request.get_json()
    title = data.get('title', 'New Chat')
    conn = get_db()
    cursor = conn.execute(
        "INSERT INTO chat_sessions (user_id, title) VALUES (?, ?)",
        (current_user.id, title)
    )
    session_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return jsonify({'id': session_id, 'title': title})

@app.route('/api/sessions/<int:session_id>', methods=['DELETE'])
@login_required
def delete_session(session_id):
    conn = get_db()
    conn.execute("DELETE FROM chat_messages WHERE session_id = ?", (session_id,))
    conn.execute("DELETE FROM chat_sessions WHERE id = ? AND user_id = ?", (session_id, current_user.id))
    conn.commit()
    conn.close()
    return jsonify({'success': True})

@app.route('/api/sessions/<int:session_id>/messages')
@login_required
def get_session_messages(session_id):
    conn = get_db()
    messages = conn.execute(
        "SELECT id, user_message, bot_response, category, confidence FROM chat_messages WHERE session_id = ? ORDER BY timestamp ASC",
        (session_id,)
    ).fetchall()
    conn.close()
    return jsonify([{
        'id': m['id'],
        'message': m['user_message'],
        'response': m['bot_response'],
        'category': m['category'],
        'confidence': m['confidence']
    } for m in messages])

@app.route('/api/history')
@login_required
def get_history():
    conn = get_db()
    logs = conn.execute(
        "SELECT user_message, bot_response FROM chat_messages WHERE user_id = ? ORDER BY timestamp DESC LIMIT 50",
        (current_user.id,)
    ).fetchall()
    conn.close()
    history = []
    for log in reversed(logs):
        history.append({'message': log['user_message'], 'response': log['bot_response']})
    return jsonify(history)

@app.route('/api/feedback', methods=['POST'])
@login_required
def submit_feedback():
    data = request.get_json()
    message_id = data.get('message_id')
    rating = data.get('rating')
    if message_id and rating is not None:
        conn = get_db()
        conn.execute("INSERT INTO feedback (message_id, rating) VALUES (?, ?)", (message_id, rating))
        conn.commit()
        conn.close()
        return jsonify({'success': True})
    return jsonify({'success': False}), 400

@app.route('/admin')
@login_required
def admin():
    if current_user.role != 'admin':
        flash('Access denied')
        return redirect(url_for('chat'))
    conn = get_db()
    total_chats = conn.execute("SELECT COUNT(*) FROM chat_messages").fetchone()[0]
    total_users = conn.execute("SELECT COUNT(*) FROM users WHERE role = 'user'").fetchone()[0]
    total_escalations = conn.execute("SELECT COUNT(*) FROM escalations").fetchone()[0]
    conn.close()
    return render_template('admin.html', total_chats=total_chats, total_users=total_users, total_escalations=total_escalations)

# ========== RUN THE APP ==========
if __name__ == '__main__':
    init_db()
    ml_model.train()
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)