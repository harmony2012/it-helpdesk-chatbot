@echo off
echo ========================================
echo   IT Helpdesk Chatbot - Quick Start
echo ========================================
echo.

REM Check if virtual environment exists
if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
)

REM Activate virtual environment
echo Activating virtual environment...
call .venv\Scripts\activate.bat

REM Check if Flask is installed
python -c "import flask" 2>nul
if errorlevel 1 (
    echo Installing requirements...
    pip install -r requirements.txt
)

REM Initialize database if not exists
if not exist "instance\helpdesk.db" (
    echo Initializing database...
    set FLASK_APP=app.py
    flask init-db
)

REM Train model if not exists
if not exist "models\chatbot_model.pkl" (
    echo Training model...
    python train_model.py
)

echo.
echo Starting application...
echo Open http://localhost:5000 in your browser
echo.
python run.py

pause