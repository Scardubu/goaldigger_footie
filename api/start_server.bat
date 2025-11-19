@echo off
echo Starting GoalDiggers API Server...

REM Change directory to the script directory
cd /d "%~dp0"

echo Setting up environment...
REM Activate virtual environment if it exists
if exist ..\venv\Scripts\activate.bat (
    call ..\venv\Scripts\activate.bat
    echo Virtual environment activated
) else (
    echo No virtual environment found, using system Python
)

REM Set default port
set API_PORT=5000
set API_HOST=0.0.0.0

REM Start the API server
echo Starting API server on %API_HOST%:%API_PORT%...
python server.py

REM If the server exits, pause to see any error messages
pause
