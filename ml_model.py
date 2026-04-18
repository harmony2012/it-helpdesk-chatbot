"""
MACHINE LEARNING MODEL FOR IT HELPDESK
Uses Random Forest Classifier with NLP
"""

import pandas as pd
import numpy as np
import joblib
import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

class MLModel:
    def __init__(self, model_path='models/ml_model.pkl'):
        self.model_path = model_path
        self.pipeline = None
        self.categories = None
        
    def clean_text(self, text):
        """Clean and preprocess text for ML"""
        if not text:
            return ""
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def create_dataset(self):
        """Create training dataset with 30+ examples"""
        os.makedirs('data', exist_ok=True)
        
        data = [
            # PRINTER (5 examples)
            ("my printer is not working", "printer"),
            ("printer won't print", "printer"),
            ("printer says offline", "printer"),
            ("paper jam in printer", "printer"),
            ("printer out of ink", "printer"),
            
            # PASSWORD (5 examples)
            ("forgot my password", "password"),
            ("can't login to account", "password"),
            ("reset my password", "password"),
            ("password not working", "password"),
            ("account locked out", "password"),
            
            # NETWORK (5 examples)
            ("wifi not connecting", "network"),
            ("internet is down", "network"),
            ("no internet connection", "network"),
            ("wifi keeps disconnecting", "network"),
            ("ethernet not working", "network"),
            
            # PERFORMANCE (3 examples)
            ("computer is slow", "performance"),
            ("slow performance", "performance"),
            ("computer freezing", "performance"),
            
            # SOFTWARE (3 examples)
            ("outlook not opening", "software"),
            ("excel keeps crashing", "software"),
            ("word not saving", "software"),
            
            # EMAIL (3 examples)
            ("email not sending", "email"),
            ("outlook not syncing", "email"),
            ("cannot receive emails", "email"),
            
            # HARDWARE (3 examples)
            ("monitor not turning on", "hardware"),
            ("computer won't start", "hardware"),
            ("blue screen error", "hardware"),
            
            # VPN (2 examples)
            ("vpn not connecting", "vpn"),
            ("vpn connection failed", "vpn"),
            
            # ESCALATION (2 examples)
            ("talk to a human", "escalation"),
            ("speak to an agent", "escalation"),
        ]
        
        df = pd.DataFrame(data, columns=['question', 'category'])
        df.to_csv('data/dataset.csv', index=False)
        print(f"✅ Created dataset with {len(df)} examples")
        print(f"✅ Categories: {df['category'].unique().tolist()}")
        return df
    
    def train(self):
        """Train the Machine Learning model"""
        print("\n" + "="*60)
        print("🤖 TRAINING MACHINE LEARNING MODEL")
        print("="*60)
        
        # Load or create dataset
        if not os.path.exists('data/dataset.csv'):
            df = self.create_dataset()
        else:
            df = pd.read_csv('data/dataset.csv')
        
        # Clean text
        df['clean'] = df['question'].apply(self.clean_text)
        
        # Get categories
        self.categories = sorted(df['category'].unique())
        print(f"📊 Categories: {self.categories}")
        print(f"📊 Training samples: {len(df)}")
        
        # Create ML Pipeline with Random Forest
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
        
        # Train on ALL data (no test split to avoid stratification issues)
        self.pipeline.fit(df['clean'], df['category'])
        
        # Simple evaluation using cross validation
        from sklearn.model_selection import cross_val_score
        cv_scores = cross_val_score(self.pipeline, df['clean'], df['category'], cv=3)
        accuracy = cv_scores.mean()
        
        print(f"\n✅ Model Accuracy (3-fold CV): {accuracy:.2%}")
        
        # Save model
        os.makedirs('models', exist_ok=True)
        joblib.dump(self.pipeline, self.model_path)
        print(f"✅ Model saved to {self.model_path}")
        
        return {'accuracy': accuracy, 'samples': len(df)}
    
    def load_model(self):
        """Load trained model"""
        if os.path.exists(self.model_path):
            self.pipeline = joblib.load(self.model_path)
            return True
        return False
    
    def predict(self, text):
        """Predict category with confidence score"""
        if not self.pipeline:
            if not self.load_model():
                return 'unknown', 0.0, []
        
        cleaned = self.clean_text(text)
        proba = self.pipeline.predict_proba([cleaned])[0]
        pred = self.pipeline.predict([cleaned])[0]
        confidence = max(proba)
        
        # Get top 3 alternatives
        top_indices = np.argsort(proba)[-3:][::-1]
        alternatives = [self.categories[i] for i in top_indices if self.categories[i] != pred][:2]
        
        return pred, confidence, alternatives
    
    def predict_with_details(self, text):
        """Full prediction details"""
        category, confidence, alternatives = self.predict(text)
        return {
            'category': category,
            'confidence': confidence,
            'alternatives': alternatives,
            'is_confident': confidence > 0.6,
            'needs_clarification': confidence < 0.4
        }

# Create global instance
ml_model = MLModel()