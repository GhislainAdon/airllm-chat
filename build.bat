@echo off
REM ===========================================
REM  AirLLM Chat - Build Script for Windows
REM  Creates a standalone .exe binary
REM ===========================================

echo.
echo  ============================================
echo   AirLLM Chat - Windows Build Script
echo  ============================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH.
    echo Please install Python 3.9+ from https://python.org
    pause
    exit /b 1
)

REM Check pip
pip --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] pip is not available.
    pause
    exit /b 1
)

echo [Step 1/4] Creating virtual environment...
if not exist "venv" (
    python -m venv venv
    echo Virtual environment created.
) else (
    echo Virtual environment already exists.
)

echo.
echo [Step 2/4] Installing dependencies...
call venv\Scripts\activate.bat
pip install --upgrade pip
pip install airllm
pip install pyinstaller

echo.
echo [Step 3/4] Building executable with PyInstaller...
pyinstaller airllm-chat.spec --clean

echo.
echo [Step 4/4] Done!
echo.
echo  The executable is located at:
echo    dist\airllm-chat\airllm-chat.exe
echo.
echo  To run it:
echo    dist\airllm-chat\airllm-chat.exe
echo.

pause
