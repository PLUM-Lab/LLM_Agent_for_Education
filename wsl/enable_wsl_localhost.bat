@echo off
chcp 65001 >nul
echo ============================================================
echo 启用 WSL 镜像网络（localhost 互通）
echo ============================================================
echo.

set WSLCONFIG=%UserProfile%\.wslconfig
if exist "%WSLCONFIG%" (
    echo 当前 .wslconfig 内容：
    type "%WSLCONFIG"
    echo.
)

(
echo [wsl2]
echo networkingMode=mirrored
echo.
echo [experimental]
echo hostAddressLoopback=true
) > "%WSLCONFIG%"

echo 已写入 %WSLCONFIG%
echo.
echo 请执行以下步骤：
echo   1. 关闭所有 WSL 窗口
echo   2. 在 PowerShell 或 CMD 中执行: wsl --shutdown
echo   3. 重新打开 WSL，进入项目目录执行: ./wsl/start_wsl.sh 或 python3 start.py
echo   4. 在 Windows 浏览器打开: http://127.0.0.1:8000/medical-quiz.html
echo.
pause
