# test_install.py
print("Testing imports...")

try:
    from flask import Flask
    print("✅ Flask imported successfully")
except ImportError as e:
    print(f"❌ Flask import failed: {e}")

try:
    from flask_sqlalchemy import SQLAlchemy
    print("✅ SQLAlchemy imported successfully")
except ImportError as e:
    print(f"❌ SQLAlchemy import failed: {e}")

try:
    from flask_login import LoginManager
    print("✅ Flask-Login imported successfully")
except ImportError as e:
    print(f"❌ Flask-Login import failed: {e}")

try:
    import sklearn
    print(f"✅ scikit-learn {sklearn.__version__} imported successfully")
except ImportError as e:
    print(f"❌ scikit-learn import failed: {e}")

print("\nTest complete!")