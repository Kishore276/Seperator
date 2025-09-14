@echo off
REM ========== Weed Classifier Project Launcher ==========

REM 1. Set up Python virtual environment (optional but recommended)
IF NOT EXIST venv (
    echo Creating virtual environment...
    python -m venv venv
)

REM 2. Activate virtual environment
call venv\Scripts\activate

REM 3. Install dependencies
cd Web-application
pip install --upgrade pip
pip install -r requirements.txt

REM 4. Set up environment variables (edit .env for API keys)
IF NOT EXIST .env (
    echo PLANT_ID_API_KEY=your_api_key_here > .env
    echo Created default .env file. Please update with your API key if needed.
)

REM 5. Run the main application
python app.py

REM 6. Deactivate virtual environment on exit
call ..\venv\Scripts\deactivate

pause
