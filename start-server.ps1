# PowerShell script to start local server
Write-Host "Starting local server..." -ForegroundColor Green
Write-Host ""

# Try Python first
Write-Host "Trying Python..." -ForegroundColor Yellow
try {
    python -m http.server 8000
} catch {
    # Try Node.js
    Write-Host "Python not found, trying Node.js..." -ForegroundColor Yellow
    try {
        npx http-server -p 8000
    } catch {
        # Try PHP
        Write-Host "Node.js not found, trying PHP..." -ForegroundColor Yellow
        try {
            php -S localhost:8000
        } catch {
            Write-Host ""
            Write-Host "ERROR: No server found!" -ForegroundColor Red
            Write-Host "Please install one of the following:" -ForegroundColor Yellow
            Write-Host "- Python 3: https://www.python.org/downloads/"
            Write-Host "- Node.js: https://nodejs.org/"
            Write-Host "- PHP: https://www.php.net/downloads.php"
            Write-Host ""
            Read-Host "Press Enter to exit"
        }
    }
}

