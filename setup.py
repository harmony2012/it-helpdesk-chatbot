import os
import subprocess
import sys

def setup_project():
    """Setup script to initialize the project"""
    
    print("🚀 Setting up IT Helpdesk Chatbot...")
    
    # Create necessary directories
    directories = ['data', 'models', 'templates', 'static', 'instance']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    # Install requirements
    print("\n📦 Installing requirements...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    
    # Set Flask app environment variable
    os.environ['FLASK_APP'] = 'app.py'
    
    # Initialize database
    print("\n🗄️  Initializing database...")
    try:
        # Import and run init_db directly
        from app import app, init_db
        with app.app_context():
            init_db()
        print("✅ Database initialized")
    except Exception as e:
        print(f"❌ Error initializing database: {e}")
        return False
    
    # Train model
    print("\n🤖 Training machine learning model...")
    try:
        subprocess.check_call([sys.executable, "train_model.py"])
        print("✅ Model trained successfully")
    except Exception as e:
        print(f"❌ Error training model: {e}")
        return False
    
    print("\n" + "="*50)
    print("✅ SETUP COMPLETE!")
    print("="*50)
    print("\nTo start the application:")
    print("1. Run: python app.py")
    print("2. Open browser: http://localhost:5000")
    print("\nDefault admin credentials:")
    print("Username: admin")
    print("Password: admin123")
    
    return True

if __name__ == "__main__":
    setup_project()