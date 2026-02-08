@echo off
:: 以管理员身份运行 PowerShell 设置端口转发
:: 双击此文件会请求管理员权限

powershell -Command "Start-Process powershell -ArgumentList '-ExecutionPolicy Bypass -File \"%~dp0setup_port_forward.ps1\"' -Verb RunAs"
