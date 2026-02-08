# 设置 localhost -> WSL 端口转发（需管理员权限）
# 转发 8000(UI) 和 5000(RAG/Tutor)，这样浏览器用 127.0.0.1 即可访问 WSL 里的服务

$wslIp = (wsl hostname -I).Trim().Split()[0]
if (-not $wslIp) {
    Write-Host "[Error] Cannot get WSL IP. Is WSL running?" -ForegroundColor Red
    exit 1
}

Write-Host "WSL IP: $wslIp"

# 删除已有规则
netsh interface portproxy delete v4tov4 listenport=5000 listenaddress=127.0.0.1 2>$null

# 添加转发：5000->5001(RAG/Tutor，WSL 内 rag_server 使用 RAG_PORT=5001)
# 注：UI 由 start.py 在 Windows 运行于 8000，不需转发
netsh interface portproxy add v4tov4 listenport=5000 listenaddress=127.0.0.1 connectport=5001 connectaddress=$wslIp

if ($LASTEXITCODE -eq 0) {
    Write-Host "[OK] Port forwarding:" -ForegroundColor Green
    Write-Host "  127.0.0.1:5000 -> ${wslIp}:5001 (RAG/Tutor)"
    Write-Host ""
    Write-Host "Then run: python start.py" -ForegroundColor Cyan
    Write-Host "Then open in browser: http://127.0.0.1:8000/medical-quiz.html" -ForegroundColor Cyan
    netsh interface portproxy show all
} else {
    Write-Host "[Error] Run as Administrator (right-click -> Run as administrator)" -ForegroundColor Red
    exit 1
}
