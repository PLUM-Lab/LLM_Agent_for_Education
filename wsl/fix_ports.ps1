# 端口修复：释放 8000/5000 并刷新端口转发（WSL IP 可能变化）
# 以管理员身份运行可同时设置端口转发

Write-Host "=== Port check ===" -ForegroundColor Cyan

# 1. Show current port usage
$on8000 = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue
$on5000 = Get-NetTCPConnection -LocalPort 5000 -State Listen -ErrorAction SilentlyContinue
if ($on8000) { Write-Host "Port 8000: in use (PID $($on8000.OwningProcess))" } else { Write-Host "Port 8000: free" }
if ($on5000) { Write-Host "Port 5000: in use (proxy/RAG, PID $($on5000.OwningProcess))" } else { Write-Host "Port 5000: free" }

# 2. Port forwarding (need admin to modify)
$wslIp = (wsl hostname -I 2>$null).Trim().Split()[0]
if ($wslIp) {
    Write-Host "`nWSL IP: $wslIp" -ForegroundColor Cyan
    netsh interface portproxy show all
    # Try to update (fails without admin)
    netsh interface portproxy delete v4tov4 listenport=5000 listenaddress=127.0.0.1 2>$null
    $r = netsh interface portproxy add v4tov4 listenport=5000 listenaddress=127.0.0.1 connectport=5000 connectaddress=$wslIp 2>&1
    if ($LASTEXITCODE -eq 0) { Write-Host "[OK] Port forwarding updated." -ForegroundColor Green } 
    else { Write-Host "[!] To update port forwarding, run as Administrator: wsl\setup_port_forward.bat" -ForegroundColor Yellow }
} else {
    Write-Host "[!] WSL IP not found. Is WSL running?" -ForegroundColor Yellow
}

# 3. Test connectivity
Write-Host "`n=== Connectivity ===" -ForegroundColor Cyan
try {
    $u = Invoke-WebRequest -Uri "http://127.0.0.1:8000" -UseBasicParsing -TimeoutSec 2; Write-Host "UI (8000): OK" -ForegroundColor Green
} catch { Write-Host "UI (8000): not reachable" -ForegroundColor Red }
try {
    $r = Invoke-WebRequest -Uri "http://127.0.0.1:5000/health" -UseBasicParsing -TimeoutSec 2
    $j = $r.Content | ConvertFrom-Json; Write-Host "RAG (5000): OK (reranker: $($j.reranker))" -ForegroundColor Green
} catch { Write-Host "RAG (5000): not reachable. Run wsl\setup_port_forward.bat as Admin, then start.py" -ForegroundColor Red }

Write-Host "`nDone. Start UI: python start.py" -ForegroundColor Cyan
