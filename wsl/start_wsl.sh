#!/bin/bash
# 在 WSL 内一键启动（UI 8000 + RAG 5000，含 ColBERTv2）
# 用法：在 WSL 里执行 ./start_wsl.sh 或 bash start_wsl.sh

cd "$(cd "$(dirname "$0")/.." && pwd)"
WSL_IP=$(hostname -I | awk '{print $1}')

echo "============================================================"
echo "Medical Quiz - 全部在 WSL 内运行"
echo "============================================================"
echo ""
echo "在 Windows 浏览器中打开："
echo "  http://127.0.0.1:8000/medical-quiz.html"
echo ""
echo "（若 127.0.0.1 打不开，可试 http://${WSL_IP}:8000/medical-quiz.html）"
echo "（首次使用 127.0.0.1 需先运行 enable_wsl_localhost.bat 并执行 wsl --shutdown）"
echo "按 Ctrl+C 停止"
echo "============================================================"
echo ""

exec python3 start.py
