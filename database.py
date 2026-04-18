import sqlite3
from werkzeug.security import generate_password_hash

DATABASE = 'instance/helpdesk.db'

def get_db():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    c = conn.cursor()
    
    # Users table
    c.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        role TEXT DEFAULT 'user'
    )''')
    
    # Chat sessions table (NEW)
    c.execute('''CREATE TABLE IF NOT EXISTS chat_sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        title TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(user_id) REFERENCES users(id)
    )''')
    
    # Chat messages table (updated with session_id)
    c.execute('''CREATE TABLE IF NOT EXISTS chat_messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id INTEGER,
        user_id INTEGER,
        user_message TEXT NOT NULL,
        bot_response TEXT NOT NULL,
        category TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(session_id) REFERENCES chat_sessions(id),
        FOREIGN KEY(user_id) REFERENCES users(id)
    )''')
    
    # Escalations table
    c.execute('''CREATE TABLE IF NOT EXISTS escalations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        user_message TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        resolved INTEGER DEFAULT 0
    )''')
    
    # Create admin user
    admin = c.execute("SELECT * FROM users WHERE username = 'admin'").fetchone()
    if not admin:
        c.execute("INSERT INTO users (username, email, password_hash, role) VALUES (?, ?, ?, ?)",
                  ('admin', 'admin@helpdesk.com', generate_password_hash('admin123'), 'admin'))
        print("✅ Admin created: admin / admin123")
    
    conn.commit()
    conn.close()
    print("✅ Database ready with chat sessions!")