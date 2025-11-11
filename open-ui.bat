@echo off
cd /d "%~dp0"
echo Starting server and opening UI...
echo.

REM Try Node.js http-server first
where npx >nul 2>&1
if %errorlevel% equ 0 (
    echo Using Node.js http-server...
    start /B npx --yes http-server -p 8000 --cors
    timeout /t 3 /nobreak >nul
    start http://localhost:8000/medical-quiz.html
    echo.
    echo Server started! The page should open in your browser.
    echo If it doesn't open, manually go to: http://localhost:8000/medical-quiz.html
    echo.
    echo Press any key to stop the server...
    pause >nul
    taskkill /F /IM node.exe >nul 2>&1
    exit
)

REM If Node.js not found, try Python
where python >nul 2>&1
if %errorlevel% equ 0 (
    echo Using Python http.server...
    start /B python -m http.server 8000
    timeout /t 3 /nobreak >nul
    start http://localhost:8000/medical-quiz.html
    echo.
    echo Server started! The page should open in your browser.
    echo If it doesn't open, manually go to: http://localhost:8000/medical-quiz.html
    echo.
    echo Press any key to stop the server...
    pause >nul
    taskkill /F /IM python.exe >nul 2>&1
    exit
)

REM If nothing works, just open the file directly
echo No server found. Opening file directly...
start "" "medical-quiz.html"
echo.
echo File opened directly. Some features may not work without a server.
pause

