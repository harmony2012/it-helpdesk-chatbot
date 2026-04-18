# test_all.py
print("Testing all imports...")

try:
    from flask import Flask
    print("✅ Flask - OK")
except ImportError as e:
    print(f"❌ Flask - {e}")

try:
    from flask_sqlalchemy import SQLAlchemy
    print("✅ SQLAlchemy - OK")
except ImportError as e:
    print(f"❌ SQLAlchemy - {e}")

try:
    from flask_login import LoginManager
    print("✅ Flask-Login - OK")
except ImportError as e:
    print(f"❌ Flask-Login - {e}")

try:
    from model import ChatbotModel
    print("✅ Model module - OK")
    
    # Test model initialization
    model = ChatbotModel()
    print("✅ Model initialization - OK")
except Exception as e:
    print(f"❌ Model - {e}")

try:
    from database import db, User, ChatLog, Response, TrainingData, UnansweredQuestion
    print("✅ Database module - OK")
except ImportError as e:
    print(f"❌ Database - {e}")

print("\nAll tests completed!")