"""
PURE PYTHON MACHINE LEARNING - No C Compiler Needed
Works on Python 3.13
"""

import json
import re
import math
import os
from collections import Counter

class PureMLModel:
    def __init__(self, model_path='models/pure_ml.json'):
        self.model_path = model_path
        self.documents = []
        self.categories = []
        self.idf = {}
        self.vocabulary = {}
        
    def clean_text(self, text):
        """Clean and tokenize text"""
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        tokens = text.split()
        return tokens
    
    def create_dataset(self):
        """Create training data"""
        os.makedirs('data', exist_ok=True)
        
        data = [
            # Printer
            ("my printer is not working", "printer"),
            ("printer won't print", "printer"),
            ("printer says offline", "printer"),
            ("paper jam in printer", "printer"),
            ("printer out of ink", "printer"),
            ("can't print document", "printer"),
            ("print job stuck", "printer"),
            
            # Password
            ("forgot my password", "password"),
            ("can't login to account", "password"),
            ("reset my password", "password"),
            ("password not working", "password"),
            ("account locked out", "password"),
            ("login failed", "password"),
            
            # Network
            ("wifi not connecting", "network"),
            ("internet is down", "network"),
            ("no internet connection", "network"),
            ("wifi keeps disconnecting", "network"),
            ("can't connect to wifi", "network"),
            ("ethernet not working", "network"),
            
            # Performance
            ("computer is slow", "performance"),
            ("slow performance", "performance"),
            ("computer freezing", "performance"),
            ("high cpu usage", "performance"),
            
            # Software
            ("outlook not opening", "software"),
            ("excel keeps crashing", "software"),
            ("word not saving", "software"),
            ("app not responding", "software"),
            
            # Email
            ("email not sending", "email"),
            ("outlook not syncing", "email"),
            ("cannot receive emails", "email"),
            
            # Hardware
            ("monitor not turning on", "hardware"),
            ("computer won't start", "hardware"),
            ("blue screen error", "hardware"),
            
            # VPN
            ("vpn not connecting", "vpn"),
            ("vpn connection failed", "vpn"),
            
            # Escalation
            ("talk to a human", "escalation"),
            ("speak to an agent", "escalation"),
        ]
        
        with open('data/dataset.json', 'w') as f:
            json.dump(data, f)
        print(f"✅ Created dataset with {len(data)} examples")
        return data
    
    def compute_tf(self, tokens):
        """Term Frequency"""
        tf = {}
        for token in tokens:
            tf[token] = tf.get(token, 0) + 1
        # Normalize
        for token in tf:
            tf[token] = tf[token] / len(tokens)
        return tf
    
    def compute_idf(self, documents):
        """Inverse Document Frequency"""
        doc_count = len(documents)
        token_doc_count = {}
        
        for doc in documents:
            unique_tokens = set(doc)
            for token in unique_tokens:
                token_doc_count[token] = token_doc_count.get(token, 0) + 1
        
        idf = {}
        for token, count in token_doc_count.items():
            idf[token] = math.log(doc_count / (1 + count))
        
        return idf
    
    def compute_tfidf(self, tf, idf):
        """TF-IDF Vector"""
        tfidf = {}
        for token, tf_val in tf.items():
            tfidf[token] = tf_val * idf.get(token, 0)
        return tfidf
    
    def cosine_similarity(self, vec1, vec2):
        """Calculate cosine similarity between two vectors"""
        if not vec1 or not vec2:
            return 0
        
        common_keys = set(vec1.keys()) & set(vec2.keys())
        dot_product = sum(vec1[k] * vec2[k] for k in common_keys)
        
        norm1 = math.sqrt(sum(v ** 2 for v in vec1.values()))
        norm2 = math.sqrt(sum(v ** 2 for v in vec2.values()))
        
        if norm1 == 0 or norm2 == 0:
            return 0
        
        return dot_product / (norm1 * norm2)
    
    def train(self):
        """Train the pure ML model"""
        print("\n" + "="*60)
        print("🤖 TRAINING PURE ML MODEL (No scikit-learn)")
        print("="*60)
        
        # Load or create dataset
        if os.path.exists('data/dataset.json'):
            with open('data/dataset.json', 'r') as f:
                data = json.load(f)
        else:
            data = self.create_dataset()
        
        print(f"📊 Training samples: {len(data)}")
        
        # Process each document
        self.documents = []
        self.categories = []
        
        for text, category in data:
            tokens = self.clean_text(text)
            self.documents.append(tokens)
            self.categories.append(category)
        
        # Build vocabulary and compute IDF
        all_tokens = [token for doc in self.documents for token in doc]
        self.vocabulary = {token: idx for idx, token in enumerate(set(all_tokens))}
        self.idf = self.compute_idf(self.documents)
        
        # Compute TF-IDF for each document
        self.doc_vectors = []
        for doc in self.documents:
            tf = self.compute_tf(doc)
            tfidf = self.compute_tfidf(tf, self.idf)
            self.doc_vectors.append(tfidf)
        
        print(f"✅ Vocabulary size: {len(self.vocabulary)}")
        print(f"✅ Categories: {list(set(self.categories))}")
        
        # Save model
        os.makedirs('models', exist_ok=True)
        model_data = {
            'documents': [list(doc) for doc in self.documents],
            'categories': self.categories,
            'idf': self.idf,
            'vocabulary': self.vocabulary
        }
        with open(self.model_path, 'w') as f:
            json.dump(model_data, f)
        
        print(f"✅ Model saved to {self.model_path}")
        
        return {'samples': len(data)}
    
    def load_model(self):
        """Load trained model"""
        if os.path.exists(self.model_path):
            with open(self.model_path, 'r') as f:
                model_data = json.load(f)
            
            self.documents = [list(doc) for doc in model_data['documents']]
            self.categories = model_data['categories']
            self.idf = model_data['idf']
            self.vocabulary = model_data['vocabulary']
            
            # Recompute vectors
            self.doc_vectors = []
            for doc in self.documents:
                tf = self.compute_tf(doc)
                tfidf = self.compute_tfidf(tf, self.idf)
                self.doc_vectors.append(tfidf)
            
            return True
        return False
    
    def predict(self, text):
        """Predict category with confidence"""
        tokens = self.clean_text(text)
        if not tokens:
            return 'unknown', 0.0, []
        
        tf = self.compute_tf(tokens)
        query_vector = self.compute_tfidf(tf, self.idf)
        
        # Find most similar document
        similarities = []
        for idx, doc_vec in enumerate(self.doc_vectors):
            sim = self.cosine_similarity(query_vector, doc_vec)
            similarities.append((sim, self.categories[idx]))
        
        # Sort by similarity
        similarities.sort(reverse=True, key=lambda x: x[0])
        
        if not similarities or similarities[0][0] < 0.05:
            return 'unknown', 0.0, []
        
        best_match = similarities[0]
        category = best_match[1]
        confidence = best_match[0]
        
        # Get alternatives
        alternatives = []
        seen = set([category])
        for sim, cat in similarities[1:4]:
            if cat not in seen and len(alternatives) < 2:
                alternatives.append(cat)
                seen.add(cat)
        
        return category, min(confidence, 0.99), alternatives
    
    def predict_with_details(self, text):
        """Full prediction details"""
        category, confidence, alternatives = self.predict(text)
        return {
            'category': category,
            'confidence': confidence,
            'alternatives': alternatives,
            'is_confident': confidence > 0.4,
            'needs_clarification': confidence < 0.2
        }

# Create global instance
ml_model = PureMLModel()