@echo off
echo Installing required AI dependencies...
pip install flask-login Flask-Mail flask_sqlalchemy
echo.
echo Starting Athlete Vision 5.0...
cd /d "%~dp0"
start http://127.0.0.1:5000
python app.py
pause
