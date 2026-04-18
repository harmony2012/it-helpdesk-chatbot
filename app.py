import os
import re
import math
from datetime import datetime
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session
from flask_login import LoginManager, login_user, logout_user, login_required, current_user, UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from textblob import TextBlob
import firebase_admin
from firebase_admin import credentials, firestore, auth
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key-here')

# ========== FIREBASE INITIALIZATION ==========
# Check if running on cloud or local
if os.path.exists('firebase-key.json'):
    cred = credentials.Certificate('firebase-key.json')
else:
    # For cloud deployment - use environment variable
    cred = credentials.Certificate({
        "type": os.getenv("FIREBASE_TYPE"),
        "project_id": os.getenv("FIREBASE_PROJECT_ID"),
        "private_key_id": os.getenv("FIREBASE_PRIVATE_KEY_ID"),
        "private_key": os.getenv("FIREBASE_PRIVATE_KEY").replace('\\n', '\n'),
        "client_email": os.getenv("FIREBASE_CLIENT_EMAIL"),
        "client_id": os.getenv("FIREBASE_CLIENT_ID"),
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
    })

firebase_admin.initialize_app(cred)
db = firestore.client()

# ========== HELPER FUNCTIONS ==========
def get_user_by_username(username):
    users_ref = db.collection('users')
    query = users_ref.where('username', '==', username).limit(1).get()
    for doc in query:
        return {**doc.to_dict(), 'id': doc.id}
    return None

def get_user_by_email(email):
    users_ref = db.collection('users')
    query = users_ref.where('email', '==', email).limit(1).get()
    for doc in query:
        return {**doc.to_dict(), 'id': doc.id}
    return None

def create_user(username, email, password_hash):
    doc_ref = db.collection('users').document()
    doc_ref.set({
        'username': username,
        'email': email,
        'password_hash': password_hash,
        'role': 'user',
        'created_at': datetime.now().isoformat()
    })
    return doc_ref.id

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
            ("printer out of ink", "printer"),
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

# ========== SENTIMENT ANALYSIS ==========
def analyze_sentiment(text):
    try:
        blob = TextBlob(text)
        return blob.sentiment.polarity
    except:
        return 0.0

# ========== RESPONSES ==========
def get_response(category, confidence, sentiment):
    responses = {
        'printer': f"""🔧 **Printer Troubleshooting** ({confidence:.0f}% sure)

1. Check power and paper
2. Restart printer and computer
3. Clear print queue

Type 'human' for IT support""",

        'password': f"""🔐 **Password Reset** ({confidence:.0f}% sure)

1. Click "Forgot Password"
2. Check email (Spam folder too)
3. Create strong password

Type 'human' if locked out""",

        'network': f"""🌐 **WiFi Fix** ({confidence:.0f}% sure)

1. Restart router (unplug 30 sec)
2. Run: ipconfig /release && ipconfig /renew
3. Restart computer

Type 'human' for help""",

        'performance': f"""⚡ **Slow Computer** ({confidence:.0f}% sure)

1. Restart computer
2. Close unused programs
3. Run Disk Cleanup

Type 'human' for help""",

        'software': f"""💻 **App Fix** ({confidence:.0f}% sure)

1. Restart the app
2. Restart computer
3. Run: sfc /scannow

Type 'human' for help""",

        'email': f"""📧 **Email Fix** ({confidence:.0f}% sure)

1. Close and reopen Outlook
2. Run: outlook.exe /safe
3. Create new profile

Type 'human' for support""",

        'vpn': f"""🔒 **VPN Fix** ({confidence:.0f}% sure)

1. Check internet
2. Restart VPN app
3. Try different server

Type 'human' for support""",

        'escalation': f"""👤 **Human Support Requested**

Agent notified. Will contact within 15 minutes.
Ticket: {datetime.now().strftime('%H%M%S')}""",

        'unknown': f"""🤔 Need more details ({confidence:.0f}%)

What happened? Any error message?

Try: printer, password, or wifi"""
    }
    return responses.get(category, responses['unknown'])

# ========== FLASK-LOGIN ==========
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
    doc_ref = db.collection('users').document(user_id)
    doc = doc_ref.get()
    if doc.exists:
        data = doc.to_dict()
        return User(doc.id, data['username'], data['email'], data['role'])
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
        user = get_user_by_username(username)
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
        existing = get_user_by_username(username)
        if existing:
            flash('Username exists')
            return redirect(url_for('register'))
        create_user(username, email, generate_password_hash(password))
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
            session_ref = db.collection('chat_sessions').document()
            session_ref.set({
                'user_id': current_user.id,
                'title': 'New Chat',
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat()
            })
            session_id = session_ref.id
        
        if msg.lower() in ['human', 'help', 'support', 'agent']:
            db.collection('escalations').add({
                'user_id': current_user.id,
                'user_message': msg,
                'timestamp': datetime.now().isoformat()
            })
            response = get_response('escalation', 1.0, 0)
            return jsonify({'response': response, 'category': 'escalation', 'confidence': 1.0, 'session_id': session_id})
        
        sentiment = analyze_sentiment(msg)
        category, confidence = ml_model.predict(msg)
        response = get_response(category, confidence, sentiment)
        
        # Update session title if new
        session_doc = db.collection('chat_sessions').document(session_id).get()
        if session_doc.exists and session_doc.to_dict().get('title') == 'New Chat':
            new_title = msg[:40] + ('...' if len(msg) > 40 else '')
            db.collection('chat_sessions').document(session_id).update({
                'title': new_title,
                'updated_at': datetime.now().isoformat()
            })
        
        # Save message
        msg_ref = db.collection('chat_messages').document()
        msg_ref.set({
            'session_id': session_id,
            'user_id': current_user.id,
            'user_message': msg,
            'bot_response': response,
            'category': category,
            'confidence': confidence,
            'sentiment': sentiment,
            'timestamp': datetime.now().isoformat()
        })
        
        return jsonify({
            'response': response,
            'category': category,
            'confidence': confidence,
            'session_id': session_id,
            'message_id': msg_ref.id
        })
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/sessions')
@login_required
def get_sessions():
    sessions_ref = db.collection('chat_sessions').where('user_id', '==', current_user.id).order_by('updated_at', direction=firestore.Query.DESCENDING)
    sessions = []
    for doc in sessions_ref.stream():
        data = doc.to_dict()
        sessions.append({
            'id': doc.id,
            'title': data.get('title', 'Chat'),
            'created_at': data.get('created_at'),
            'updated_at': data.get('updated_at')
        })
    return jsonify(sessions)

@app.route('/api/sessions', methods=['POST'])
@login_required
def create_session():
    data = request.get_json()
    title = data.get('title', 'New Chat')
    session_ref = db.collection('chat_sessions').document()
    session_ref.set({
        'user_id': current_user.id,
        'title': title,
        'created_at': datetime.now().isoformat(),
        'updated_at': datetime.now().isoformat()
    })
    return jsonify({'id': session_ref.id, 'title': title})

@app.route('/api/sessions/<session_id>', methods=['DELETE'])
@login_required
def delete_session(session_id):
    # Delete messages first
    messages = db.collection('chat_messages').where('session_id', '==', session_id).stream()
    for msg in messages:
        db.collection('chat_messages').document(msg.id).delete()
    # Delete session
    db.collection('chat_sessions').document(session_id).delete()
    return jsonify({'success': True})

@app.route('/api/sessions/<session_id>/messages')
@login_required
def get_session_messages(session_id):
    messages_ref = db.collection('chat_messages').where('session_id', '==', session_id).order_by('timestamp')
    messages = []
    for doc in messages_ref.stream():
        data = doc.to_dict()
        messages.append({
            'id': doc.id,
            'message': data.get('user_message'),
            'response': data.get('bot_response'),
            'category': data.get('category'),
            'confidence': data.get('confidence')
        })
    return jsonify(messages)

@app.route('/api/feedback', methods=['POST'])
@login_required
def submit_feedback():
    data = request.get_json()
    message_id = data.get('message_id')
    rating = data.get('rating')
    if message_id and rating is not None:
        db.collection('feedback').add({
            'message_id': message_id,
            'rating': rating,
            'timestamp': datetime.now().isoformat()
        })
        return jsonify({'success': True})
    return jsonify({'success': False}), 400

@app.route('/admin')
@login_required
def admin():
    if current_user.role != 'admin':
        flash('Access denied')
        return redirect(url_for('chat'))
    
    # Get stats from Firebase
    chats = db.collection('chat_messages').stream()
    users = db.collection('users').stream()
    escalations = db.collection('escalations').stream()
    
    total_chats = sum(1 for _ in chats)
    total_users = sum(1 for _ in users)
    total_escalations = sum(1 for _ in escalations)
    
    return render_template('admin.html', 
                         total_chats=total_chats, 
                         total_users=total_users, 
                         total_escalations=total_escalations)

if __name__ == '__main__':
    ml_model.train()
    print("🚀 Server starting at http://localhost:5000")
    print("🔐 Login: admin / admin123")
    app.run(debug=True, port=5000)