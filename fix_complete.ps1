# fix_complete.ps1
Write-Host "🔧 Fixing Python 3.13.5 compatibility issues..." -ForegroundColor Yellow

# Deactivate if active
if (Get-Command deactivate -ErrorAction SilentlyContinue) {
    deactivate
}

# Remove old virtual environment
if (Test-Path ".venv") {
    Remove-Item -Recurse -Force .venv
    Write-Host "✅ Removed old virtual environment" -ForegroundColor Green
}

# Create fresh venv
Write-Host "🚀 Creating fresh virtual environment..." -ForegroundColor Yellow
python -m venv .venv

# Activate
Write-Host "🔌 Activating..." -ForegroundColor Yellow
& .\.venv\Scripts\Activate.ps1

# Upgrade pip
Write-Host "📦 Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

# Install packages with Python 3.13 compatibility
Write-Host "📚 Installing packages for Python 3.13..." -ForegroundColor Yellow
pip install Flask==3.0.3
pip install Werkzeug==3.0.3
pip install Flask-Login==0.6.3
pip install Flask-SQLAlchemy==3.1.1
pip install pandas==2.2.0
pip install scikit-learn==1.4.0
pip install numpy==1.26.3
pip install joblib==1.3.2
pip install nltk==3.8.1

# Verify installations
Write-Host "`n✅ Verifying..." -ForegroundColor Yellow
python -c "
import sys
print(f'Python: {sys.version}')
import flask
print(f'Flask: {flask.__version__}')
import werkzeug
print(f'Werkzeug: {werkzeug.__version__}')
import flask_login
print(f'Flask-Login: {flask_login.__version__}')
from werkzeug.urls import url_decode
print('✅ url_decode imported successfully!')
import pandas
print(f'Pandas: {pandas.__version__}')
import sklearn
print(f'Scikit-learn: {sklearn.__version__}')
print('\n✅ All packages installed correctly!')
"

Write-Host "`n🗄️  Initializing database..." -ForegroundColor Yellow
$env:FLASK_APP = "app.py"
python -c "
from app import app, init_db
with app.app_context():
    init_db()
print('✅ Database initialized')
"

Write-Host "`n🤖 Training model..." -ForegroundColor Yellow
python train_model.py

Write-Host "`n" + "="*50 -ForegroundColor Green
Write-Host "✅ FIX COMPLETE!" -ForegroundColor Green
Write-Host "="*50 -ForegroundColor Green
Write-Host "`nRun: python run.py" -ForegroundColor Cyan
Write-Host "Open: http://localhost:5000" -ForegroundColor Cyan