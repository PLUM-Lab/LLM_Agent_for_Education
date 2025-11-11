# PowerShell script to start server and open UI
Write-Host "Starting server and opening UI..." -ForegroundColor Green
Write-Host ""

# Change to script directory
Set-Location $PSScriptRoot

# Try Node.js http-server first
if (Get-Command npx -ErrorAction SilentlyContinue) {
    Write-Host "Using Node.js http-server..." -ForegroundColor Yellow
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PSScriptRoot'; npx --yes http-server -p 8000 --cors"
    Start-Sleep -Seconds 3
    Start-Process "http://localhost:8000/medical-quiz.html"
    Write-Host ""
    Write-Host "Server started! The page should open in your browser." -ForegroundColor Green
    Write-Host "If it doesn't open, manually go to: http://localhost:8000/medical-quiz.html" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Press any key to exit (server will keep running)..." -ForegroundColor Cyan
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    exit
}

# Try Python if Node.js not found
if (Get-Command python -ErrorAction SilentlyContinue) {
    Write-Host "Using Python http.server..." -ForegroundColor Yellow
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PSScriptRoot'; python -m http.server 8000"
    Start-Sleep -Seconds 3
    Start-Process "http://localhost:8000/medical-quiz.html"
    Write-Host ""
    Write-Host "Server started! The page should open in your browser." -ForegroundColor Green
    Write-Host "If it doesn't open, manually go to: http://localhost:8000/medical-quiz.html" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Press any key to exit (server will keep running)..." -ForegroundColor Cyan
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    exit
}

# If nothing works, open file directly
Write-Host "No server found. Opening file directly..." -ForegroundColor Yellow
Start-Process "medical-quiz.html"
Write-Host ""
Write-Host "File opened directly. Some features may not work without a server." -ForegroundColor Yellow
Write-Host ""
Write-Host "Press any key to exit..." -ForegroundColor Cyan
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")

