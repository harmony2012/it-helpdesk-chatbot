import pandas as pd
import numpy as np
import joblib
import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class AdvancedAIModel:
    def __init__(self, model_path='models/advanced_ai.pkl'):
        self.model_path = model_path
        self.pipeline = None
        self.categories = None
        self.label_encoder = LabelEncoder()
        
    def clean_text(self, text):
        """Advanced text cleaning"""
        if not text:
            return ""
        text = text.lower()
        # Remove special characters but keep important ones
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def augment_data(self, df):
        """Create variations of existing questions for better learning"""
        augmented = []
        
        synonyms = {
            'printer': ['printer', 'print', 'printing', 'printed', 'printout', 'document print'],
            'password': ['password', 'login', 'log in', 'sign in', 'access', 'credentials', 'passcode'],
            'wifi': ['wifi', 'internet', 'network', 'connection', 'wireless', 'ethernet', 'signal'],
            'slow': ['slow', 'lag', 'freeze', 'hang', 'unresponsive', 'stuck', 'not responding'],
            'outlook': ['outlook', 'email', 'mail', 'inbox', 'message', 'gmail']
        }
        
        for _, row in df.iterrows():
            augmented.append(row)
            text = row['question']
            cat = row['category']
            
            # Add variations
            if cat == 'printer':
                augmented.append({'question': f"my {text}", 'category': cat})
                augmented.append({'question': f"why won't my {text}", 'category': cat})
            elif cat == 'password':
                augmented.append({'question': f"can't {text}", 'category': cat})
                augmented.append({'question': f"need help with {text}", 'category': cat})
            elif cat == 'network':
                augmented.append({'question': f"my {text}", 'category': cat})
                augmented.append({'question': f"{text} problem", 'category': cat})
        
        return pd.DataFrame(augmented)
    
    def train(self, csv_path='data/dataset.csv'):
        """Advanced ML training"""
        print("=" * 60)
        print("🤖 TRAINING ADVANCED AI MODEL")
        print("=" * 60)
        
        # Load data
        if not os.path.exists(csv_path):
            self.create_training_data(csv_path)
        
        df = pd.read_csv(csv_path)
        print(f"📊 Original samples: {len(df)}")
        
        # Data augmentation
        df = self.augment_data(df)
        print(f"📊 Augmented samples: {len(df)}")
        
        # Clean text
        df['clean'] = df['question'].apply(self.clean_text)
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['clean'])
        print(f"📊 Unique samples: {len(df)}")
        
        # Get categories
        self.categories = sorted(df['category'].unique())
        print(f"📊 Categories: {self.categories}")
        
        # Advanced pipeline
        self.pipeline = Pipeline([
            ('vectorizer', TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 3),
                stop_words='english',
                sublinear_tf=True,
                use_idf=True,
                smooth_idf=True
            )),
            ('classifier', RandomForestClassifier(
                n_estimators=200,
                max_depth=30,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'
            ))
        ])
        
        # Train
        X = df['clean']
        y = df['category']
        
        print("\n🔄 Training model...")
        self.pipeline.fit(X, y)
        
        # Cross-validation
        cv_scores = cross_val_score(self.pipeline, X, y, cv=5)
        print(f"📊 Cross-validation scores: {cv_scores}")
        print(f"📊 Mean CV accuracy: {cv_scores.mean():.2%}")
        
        # Train/Test split evaluation
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        self.pipeline.fit(X_train, y_train)
        y_pred = self.pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n✅ Test Accuracy: {accuracy:.2%}")
        
        # Detailed report
        print("\n📋 CLASSIFICATION REPORT:")
        print(classification_report(y_test, y_pred))
        
        # Save model
        os.makedirs('models', exist_ok=True)
        joblib.dump({
            'pipeline': self.pipeline,
            'categories': self.categories
        }, self.model_path)
        
        print(f"\n✅ Model saved to {self.model_path}")
        
        return {
            'accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'categories': self.categories,
            'samples': len(df)
        }
    
    def create_training_data(self, csv_path):
        """Create comprehensive training data"""
        os.makedirs('data', exist_ok=True)
        
        data = [
            # Printer issues (15 examples)
            ("my printer is not working", "printer"),
            ("printer won't print", "printer"),
            ("printer says offline", "printer"),
            ("paper jam in printer", "printer"),
            ("printer out of ink", "printer"),
            ("can't print to my printer", "printer"),
            ("print job stuck in queue", "printer"),
            ("printer not responding", "printer"),
            ("printer driver error", "printer"),
            ("how to fix printer", "printer"),
            ("printer not connecting", "printer"),
            ("print spooler error", "printer"),
            ("printer keeps jamming", "printer"),
            ("printer shows error", "printer"),
            ("document won't print", "printer"),
            
            # Password issues (15 examples)
            ("forgot my password", "password"),
            ("can't login to my account", "password"),
            ("reset my password", "password"),
            ("password not working", "password"),
            ("account locked out", "password"),
            ("login failed", "password"),
            ("forgot windows password", "password"),
            ("how to reset password", "password"),
            ("mfa code not working", "password"),
            ("two factor authentication failed", "password"),
            ("can't access my account", "password"),
            ("password reset link not working", "password"),
            ("old password not accepted", "password"),
            ("account temporarily locked", "password"),
            ("forgot email password", "password"),
            
            # Network issues (15 examples)
            ("wifi not connecting", "network"),
            ("internet is down", "network"),
            ("no internet connection", "network"),
            ("wifi keeps disconnecting", "network"),
            ("can't connect to wifi", "network"),
            ("ethernet not working", "network"),
            ("network adapter not found", "network"),
            ("dns server not responding", "network"),
            ("ip address conflict", "network"),
            ("slow internet speed", "network"),
            ("wifi signal weak", "network"),
            ("can't find wifi network", "network"),
            ("network connection drops", "network"),
            ("limited connectivity", "network"),
            ("no network access", "network"),
            
            # Performance issues (10 examples)
            ("computer is slow", "performance"),
            ("slow performance", "performance"),
            ("computer freezing", "performance"),
            ("high cpu usage", "performance"),
            ("memory full", "performance"),
            ("disk 100 percent usage", "performance"),
            ("computer takes forever to start", "performance"),
            ("system lag", "performance"),
            ("applications running slow", "performance"),
            ("pc very slow", "performance"),
            
            # Software issues (10 examples)
            ("outlook not opening", "software"),
            ("excel keeps crashing", "software"),
            ("word document not saving", "software"),
            ("software installation failed", "software"),
            ("application not responding", "software"),
            ("can't open attachment", "software"),
            ("program keeps closing", "software"),
            ("software update failed", "software"),
            ("app crashing", "software"),
            ("browser not working", "software"),
            
            # Email issues (8 examples)
            ("email not sending", "email"),
            ("outlook not syncing", "email"),
            ("cannot receive emails", "email"),
            ("email stuck in outbox", "email"),
            ("gmail not loading", "email"),
            ("email account hacked", "email"),
            ("can't send attachments", "email"),
            ("email signature missing", "email"),
            
            # Hardware issues (10 examples)
            ("monitor not turning on", "hardware"),
            ("computer won't start", "hardware"),
            ("blue screen error", "hardware"),
            ("laptop overheating", "hardware"),
            ("usb device not recognized", "hardware"),
            ("mouse not working", "hardware"),
            ("keyboard not responding", "hardware"),
            ("battery not charging", "hardware"),
            ("computer making noise", "hardware"),
            ("screen flickering", "hardware"),
            
            # VPN issues (7 examples)
            ("vpn not connecting", "vpn"),
            ("vpn connection failed", "vpn"),
            ("can't access vpn", "vpn"),
            ("remote desktop not working", "vpn"),
            ("vpn keeps disconnecting", "vpn"),
            ("vpn authentication failed", "vpn"),
            ("cannot connect to office vpn", "vpn"),
            
            # Human escalation (5 examples)
            ("talk to a human", "escalation"),
            ("speak to an agent", "escalation"),
            ("need real person", "escalation"),
            ("connect me to support", "escalation"),
            ("i need help from a person", "escalation")
        ]
        
        df = pd.DataFrame(data, columns=['question', 'category'])
        df.to_csv(csv_path, index=False)
        print(f"✅ Created training data: {len(df)} examples")
    
    def load_model(self):
        """Load trained model"""
        if os.path.exists(self.model_path):
            data = joblib.load(self.model_path)
            self.pipeline = data['pipeline']
            self.categories = data['categories']
            return True
        return False
    
    def predict(self, text, threshold=0.2):
        """Predict category with confidence score"""
        if not self.pipeline:
            if not self.load_model():
                return 'unknown', 0.0
        
        cleaned = self.clean_text(text)
        proba = self.pipeline.predict_proba([cleaned])[0]
        pred = self.pipeline.predict([cleaned])[0]
        confidence = max(proba)
        
        # Get top 2 predictions for context
        top_indices = np.argsort(proba)[-2:][::-1]
        top_categories = [self.categories[i] for i in top_indices]
        top_confidences = [proba[i] for i in top_indices]
        
        if confidence < threshold:
            return 'unknown', confidence, top_categories
        
        return pred, confidence, top_categories
    
    def predict_with_confidence(self, text):
        """Get full prediction details"""
        pred, conf, alternatives = self.predict(text)
        return {
            'category': pred,
            'confidence': conf,
            'alternatives': alternatives,
            'is_confident': conf > 0.6,
            'needs_clarification': conf < 0.4
        }

# Create global instance
ai_model = AdvancedAIModel()