@echo off
REM ===========================================
REM  AirLLM Chat - Run Script (no build needed)
REM  Runs directly from Python source
REM ===========================================

echo.
echo  AirLLM Chat - Starting...
echo.

REM Check for virtual environment
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
) else (
    echo [Warning] No virtual environment found. Using system Python.
)

REM Check if airllm is installed
python -c "import airllm" >nul 2>&1
if errorlevel 1 (
    echo [Installing] airllm not found. Installing dependencies...
    pip install airllm
)

REM Run the app
python app.py %*

if errorlevel 1 (
    echo.
    echo [Error] Application crashed. Check the error above.
    pause
)
