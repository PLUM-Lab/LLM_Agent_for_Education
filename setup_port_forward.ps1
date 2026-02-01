# 设置 localhost:5000 -> WSL RAG 端口转发（需要管理员权限）
# 右键此文件 -> 使用 PowerShell 运行，或管理员 PowerShell 执行

$wslIp = (wsl hostname -I).Trim().Split()[0]
if (-not $wslIp) {
    Write-Host "[Error] Cannot get WSL IP. Is WSL running?" -ForegroundColor Red
    exit 1
}

Write-Host "WSL IP: $wslIp"

# 删除已有规则
netsh interface portproxy delete v4tov4 listenport=5000 listenaddress=127.0.0.1 2>$null

# 添加转发
netsh interface portproxy add v4tov4 listenport=5000 listenaddress=127.0.0.1 connectport=5000 connectaddress=$wslIp

if ($LASTEXITCODE -eq 0) {
    Write-Host "[OK] Port forwarding: 127.0.0.1:5000 -> ${wslIp}:5000" -ForegroundColor Green
    netsh interface portproxy show all
} else {
    Write-Host "[Error] Run this script as Administrator (right-click -> Run as administrator)" -ForegroundColor Red
    exit 1
}
