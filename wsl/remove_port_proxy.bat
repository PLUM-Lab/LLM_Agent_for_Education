@echo off
:: 删除 5000/8000 端口代理，让本机 Flask/UI 直接响应（用于 --no-wsl-rag 模式）
:: 请右键 -> 以管理员身份运行

netsh interface portproxy delete v4tov4 listenport=5000 listenaddress=127.0.0.1
netsh interface portproxy delete v4tov4 listenport=8000 listenaddress=127.0.0.1
echo.
echo [OK] 已删除端口代理。请重启: python start.py --no-wsl-rag
pause
