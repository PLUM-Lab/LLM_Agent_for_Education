@echo off
echo Starting server...
start /B python -m http.server 8000
timeout /t 2 /nobreak >nul
echo Opening clear page...
start http://localhost:8000/auto-clear-api-key.html
echo Done! The page should open in your browser.
pause

