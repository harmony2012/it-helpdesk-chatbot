import pandas as pd
import numpy as np
import joblib
import re
import nltk
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except:
    nltk.download('punkt', quiet=True)

class AIModel:
    def __init__(self, model_path='models/ai_model.pkl'):
        self.model_path = model_path
        self.pipeline = None
        self.categories = None
        
    def clean_text(self, text):
        """Clean and preprocess text"""
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def load_training_data(self, csv_path='data/training_data.csv'):
        """Load and preprocess training data"""
        if not os.path.exists(csv_path):
            self.create_default_training_data(csv_path)
        
        df = pd.read_csv(csv_path)
        df['clean'] = df['question'].apply(self.clean_text)
        return df
    
    def create_default_training_data(self, csv_path):
        """Create default training data if none exists"""
        os.makedirs('data', exist_ok=True)
        
        data = [
            # Printer issues
            ("my printer is not working", "printer"),
            ("printer won't print", "printer"),
            ("printer offline", "printer"),
            ("paper jam", "printer"),
            ("printer out of ink", "printer"),
            ("can't print", "printer"),
            ("print spooler error", "printer"),
            ("printer not responding", "printer"),
            
            # Password issues
            ("forgot my password", "password"),
            ("can't login", "password"),
            ("reset password", "password"),
            ("account locked", "password"),
            ("login failed", "password"),
            ("password not working", "password"),
            ("forgot username", "password"),
            
            # Network issues
            ("wifi not connecting", "network"),
            ("internet is down", "network"),
            ("no internet connection", "network"),
            ("wifi keeps disconnecting", "network"),
            ("can't connect to wifi", "network"),
            ("ethernet not working", "network"),
            ("network adapter issue", "network"),
            
            # Performance issues
            ("computer is slow", "performance"),
            ("slow performance", "performance"),
            ("computer freezing", "performance"),
            ("high cpu usage", "performance"),
            ("ram full", "performance"),
            ("system lag", "performance"),
            
            # Software issues
            ("outlook not opening", "software"),
            ("excel crashing", "software"),
            ("word not saving", "software"),
            ("software installation failed", "software"),
            ("application not responding", "software"),
            
            # Email issues
            ("email not sending", "email"),
            ("outlook not syncing", "email"),
            ("can't receive emails", "email"),
            ("email stuck in outbox", "email"),
            
            # Hardware issues
            ("computer won't start", "hardware"),
            ("blue screen error", "hardware"),
            ("laptop overheating", "hardware"),
            ("monitor not working", "hardware"),
            ("keyboard not responding", "hardware"),
            
            # VPN issues
            ("vpn not connecting", "vpn"),
            ("vpn connection failed", "vpn"),
            ("can't access vpn", "vpn"),
            
            # Human escalation
            ("talk to human", "escalation"),
            ("speak to agent", "escalation"),
            ("need help from person", "escalation")
        ]
        
        df = pd.DataFrame(data, columns=['question', 'category'])
        df.to_csv(csv_path, index=False)
        print(f"✅ Created default training data: {csv_path}")
    
    def train(self, csv_path='data/training_data.csv'):
        """Train the AI model"""
        print("🧠 Training AI model...")
        
        df = self.load_training_data(csv_path)
        X = df['clean']
        y = df['category']
        
        self.categories = sorted(y.unique())
        
        # Use RandomForest for better accuracy
        self.pipeline = Pipeline([
            ('vectorizer', TfidfVectorizer(
                max_features=3000,
                ngram_range=(1, 3),
                stop_words='english',
                sublinear_tf=True
            )),
            ('classifier', RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                random_state=42,
                n_jobs=-1
            ))
        ])
        
        self.pipeline.fit(X, y)
        
        # Evaluate
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        temp_pipeline = Pipeline([
            ('vectorizer', TfidfVectorizer(max_features=3000, ngram_range=(1, 3), stop_words='english')),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
        ])
        
        temp_pipeline.fit(X_train, y_train)
        y_pred = temp_pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"✅ AI Model Accuracy: {accuracy:.2%}")
        print(f"📊 Categories: {self.categories}")
        
        os.makedirs('models', exist_ok=True)
        joblib.dump({'pipeline': self.pipeline, 'categories': self.categories}, self.model_path)
        
        return {'accuracy': accuracy, 'samples': len(df)}
    
    def load_model(self):
        """Load trained model"""
        if os.path.exists(self.model_path):
            data = joblib.load(self.model_path)
            self.pipeline = data['pipeline']
            self.categories = data['categories']
            return True
        return False
    
    def predict(self, text):
        """Predict category with confidence"""
        if not self.pipeline:
            if not self.load_model():
                return 'unknown', 0.0
        
        cleaned = self.clean_text(text)
        proba = self.pipeline.predict_proba([cleaned])[0]
        pred = self.pipeline.predict([cleaned])[0]
        confidence = max(proba)
        
        return pred, confidence
    
    def predict_with_context(self, text, context=None):
        """Predict with conversation context"""
        pred, confidence = self.predict(text)
        
        # Add context awareness
        if context and confidence < 0.6:
            # If low confidence and we have context, check context
            for ctx in context[-2:]:  # Last 2 messages
                ctx_pred, ctx_conf = self.predict(ctx)
                if ctx_pred == pred:
                    confidence = min(confidence + 0.2, 0.9)
        
        return pred, confidence

# Create global model instance
ai_model = AIModel()