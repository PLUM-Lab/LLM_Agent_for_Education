@echo off
echo Starting local server...
echo.
echo Please choose one of the following options:
echo 1. Python (if installed)
echo 2. Node.js (if installed)
echo 3. PHP (if installed)
echo.
echo Trying Python...
python -m http.server 8000 2>nul
if %errorlevel% neq 0 (
    echo Python not found, trying Node.js...
    npx http-server -p 8000 2>nul
    if %errorlevel% neq 0 (
        echo Node.js not found, trying PHP...
        php -S localhost:8000 2>nul
        if %errorlevel% neq 0 (
            echo.
            echo ERROR: No server found!
            echo Please install one of the following:
            echo - Python 3: https://www.python.org/downloads/
            echo - Node.js: https://nodejs.org/
            echo - PHP: https://www.php.net/downloads.php
            echo.
            pause
        )
    )
)

