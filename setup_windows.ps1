# setup_complete.ps1
Write-Host "🚀 Complete Setup for IT Helpdesk Chatbot" -ForegroundColor Green

# Check if running in PowerShell
if ($PSVersionTable.PSVersion.Major -lt 5) {
    Write-Host "Please use PowerShell 5.0 or higher" -ForegroundColor Red
    exit
}

# Step 1: Create necessary directories
Write-Host "`n📁 Creating directories..." -ForegroundColor Yellow
$directories = @("data", "models", "templates", "static", "instance")
foreach ($dir in $directories) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir | Out-Null
        Write-Host "  Created: $dir" -ForegroundColor Green
    } else {
        Write-Host "  Exists: $dir" -ForegroundColor Gray
    }
}

# Step 2: Check if virtual environment exists
if (-not (Test-Path ".venv")) {
    Write-Host "`n🐍 Creating virtual environment..." -ForegroundColor Yellow
    python -m venv .venv
}

# Step 3: Activate virtual environment
Write-Host "`n🔌 Activating virtual environment..." -ForegroundColor Yellow
& .\.venv\Scripts\Activate.ps1

# Step 4: Upgrade pip
Write-Host "`n📦 Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

# Step 5: Install requirements
Write-Host "`n📚 Installing requirements..." -ForegroundColor Yellow
pip install Flask==2.2.5
pip install Werkzeug==2.2.3
pip install Flask-SQLAlchemy==3.0.5
pip install Flask-Login==0.6.2
pip install scikit-learn==1.3.0
pip install pandas==2.0.3
pip install numpy==1.24.3
pip install nltk==3.8.1
pip install joblib==1.3.1

# Step 6: Verify installations
Write-Host "`n✅ Verifying installations..." -ForegroundColor Yellow
python -c "import flask; print(f'  Flask {flask.__version__}')"
python -c "import sklearn; print(f'  scikit-learn {sklearn.__version__}')"

# Step 7: Initialize database
Write-Host "`n🗄️  Initializing database..." -ForegroundColor Yellow
$env:FLASK_APP = "app.py"
python -c "
from app import app, init_db
with app.app_context():
    init_db()
print('  Database initialized successfully')
"

# Step 8: Train model
Write-Host "`n🤖 Training model..." -ForegroundColor Yellow
python train_model.py

# Step 9: Run tests
Write-Host "`n🧪 Running tests..." -ForegroundColor Yellow
python test_all.py

Write-Host "`n" + "="*50 -ForegroundColor Green
Write-Host "✅ SETUP COMPLETE!" -ForegroundColor Green
Write-Host "="*50 -ForegroundColor Green
Write-Host "`nTo start the application:" -ForegroundColor Cyan
Write-Host "1. Run: python run.py" -ForegroundColor White
Write-Host "2. Open browser: http://localhost:5000" -ForegroundColor White
Write-Host "`nDefault admin credentials:" -ForegroundColor Cyan
Write-Host "Username: admin" -ForegroundColor White
Write-Host "Password: admin123" -ForegroundColor White