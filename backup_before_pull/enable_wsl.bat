@echo off
REM 以管理员权限启用 WSL

echo 启用 WSL 功能...
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart

echo.
echo 下载并设置 WSL 2 Linux 内核...
wsl --update
wsl --set-default-version 2

echo.
echo 安装 Ubuntu-22.04...
wsl --install Ubuntu-22.04

echo.
echo 完成！计算机将重启。
timeout /t 10
shutdown /r /t 30

pause
