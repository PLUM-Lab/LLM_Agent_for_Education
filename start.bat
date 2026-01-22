@echo off
REM 统一启动脚本 - Windows批处理版本
REM 使用Python启动脚本

echo ========================================
echo 医学教育测验系统 - 启动器
echo ========================================
echo.

REM 检查Python是否安装
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到Python，请先安装Python
    pause
    exit /b 1
)

REM 启动统一启动脚本
python start.py %*

pause
