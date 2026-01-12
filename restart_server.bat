@echo off
echo ========================================
echo 重启 RAG 服务器
echo ========================================
echo.

echo [1/3] 正在停止旧服务器进程...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :5000 ^| findstr LISTENING') do (
    echo 找到进程ID: %%a
    taskkill /F /PID %%a >nul 2>&1
    if errorlevel 1 (
        echo 警告: 无法停止进程 %%a（可能已经停止）
    ) else (
        echo 成功停止进程 %%a
    )
)

echo.
echo [2/3] 等待端口释放...
timeout /t 2 /nobreak >nul

echo.
echo [3/3] 启动新服务器...
echo 注意: 服务器将在当前窗口运行
echo 按 Ctrl+C 可以停止服务器
echo.
python rag_server.py

