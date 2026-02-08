@echo off
REM 设置 Visual Studio 编译器环境并启动服务
REM 适用于 Visual Studio 2022 Build Tools

setlocal enabledelayedexpansion

REM 设置编译器路径
set "MSVC_PATH=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64"

REM 检查编译器是否存在
if exist "%MSVC_PATH%\cl.exe" (
    echo [OK] 找到 C++ 编译器
    set "PATH=%MSVC_PATH%;%PATH%"
) else (
    echo [!] 未找到 C++ 编译器，仅使用 FAISS 模式
)

REM 启动 Python 服务
cd /d "d:\LLM_Agent_for_Education"
python start.py

pause
