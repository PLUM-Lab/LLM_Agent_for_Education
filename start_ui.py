#!/usr/bin/env python3
"""
启动医学教育测验系统的UI和RAG服务器

同时启动：
1. HTTP 服务器（端口 8000）- 提供前端UI
2. RAG 服务器（端口 5000）- 提供语义搜索功能（包含 ColBERTv2 重排序器）

**重要：此脚本必须在 WSL 或 Linux 环境中运行**
重排序器需要 Linux 环境才能正常工作（pthread.h 支持）

使用方法（在 WSL 中）：
    python3 start_ui.py

或从 Windows PowerShell：
    wsl -d Ubuntu-22.04 -e bash -c "cd /mnt/d/LLM_Agent_for_Education && python3 start_ui.py"
"""

import threading
import time
import sys
import os
from pathlib import Path

def start_http_server(port=8000):
    """启动HTTP服务器"""
    print(f"\n{'='*60}")
    print(f"[HTTP 服务器] 正在启动在端口 {port}...")
    print(f"{'='*60}\n")
    
    try:
        # 使用 Python 内置的 http.server
        import http.server
        import socketserver
        
        handler = http.server.SimpleHTTPRequestHandler
        with socketserver.TCPServer(("", port), handler) as httpd:
            print(f"✓ HTTP 服务器已启动")
            print(f"  访问地址: http://localhost:{port}/medical-quiz.html")
            print(f"\n按 Ctrl+C 停止所有服务器\n")
            httpd.serve_forever()
    except OSError as e:
        if "Address already in use" in str(e) or "address already in use" in str(e).lower():
            print(f"⚠ 端口 {port} 已被占用，尝试使用其他端口...")
            # 尝试下一个端口
            try:
                with socketserver.TCPServer(("", port + 1), handler) as httpd:
                    print(f"✓ HTTP 服务器已启动在端口 {port + 1}")
                    print(f"  访问地址: http://localhost:{port + 1}/medical-quiz.html")
                    print(f"\n按 Ctrl+C 停止所有服务器\n")
                    httpd.serve_forever()
            except Exception as e2:
                print(f"✗ 无法启动 HTTP 服务器: {e2}")
        else:
            print(f"✗ HTTP 服务器启动失败: {e}")
    except KeyboardInterrupt:
        print("\n[HTTP 服务器] 正在关闭...")
    except Exception as e:
        print(f"✗ HTTP 服务器错误: {e}")

def start_rag_server():
    """启动RAG服务器"""
    print(f"\n{'='*60}")
    print(f"[RAG 服务器] 正在启动...")
    print(f"{'='*60}\n")
    
    try:
        # 导入 RAG 服务器模块
        import rag_server
        
        # 初始化 RAG 系统
        if rag_server.initialize_rag():
            print("\n[RAG 服务器] 服务器启动在 http://localhost:5000")
            print("\nAPI 端点：")
            print("  POST /search     - 搜索相关块")
            print("  GET  /health     - 健康检查")
            print("  POST /rebuild    - 从 PDF 重建索引")
            print()
            
            # 启动 Flask 应用（不使用reloader，避免多进程问题）
            rag_server.app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
        else:
            print("\n[警告] RAG 系统初始化失败")
            print("UI 仍可使用，但 RAG 功能将不可用")
            
    except ImportError as e:
        print(f"\n[警告] 无法导入 rag_server: {e}")
        print("UI 仍可使用，但 RAG 功能将不可用")
    except KeyboardInterrupt:
        print("\n[RAG 服务器] 正在关闭...")
    except Exception as e:
        print(f"\n[警告] RAG 服务器启动失败: {e}")
        print("UI 仍可使用，但 RAG 功能将不可用")
        print("你可以稍后单独运行: python rag_server.py")

def check_wsl_environment():
    """检查是否在 WSL 或 Linux 环境中运行"""
    # 检查平台
    if sys.platform == "win32":
        return False, "Windows"
    
    # 检查是否在 WSL 中（通过检查 /proc/version 或环境变量）
    try:
        with open('/proc/version', 'r') as f:
            version_info = f.read().lower()
            if 'microsoft' in version_info or 'wsl' in version_info:
                return True, "WSL"
            elif 'linux' in version_info:
                return True, "Linux"
    except:
        pass
    
    # 检查环境变量
    if os.environ.get('WSL_DISTRO_NAME'):
        return True, "WSL"
    
    # 默认：如果是非 Windows 平台，假设是 Linux
    if sys.platform.startswith('linux'):
        return True, "Linux"
    
    return False, "Unknown"

def main():
    """主函数：同时启动两个服务器（必须在 WSL/Linux 中运行）"""
    print("\n" + "="*60)
    print("医学教育测验系统 - 启动器")
    print("="*60)
    
    # 检查运行环境
    is_linux, env_type = check_wsl_environment()
    
    if not is_linux:
        print("\n" + "="*60)
        print("❌ 错误：此脚本必须在 WSL 或 Linux 环境中运行")
        print("="*60)
        print("\n原因：")
        print("  - ColBERTv2 重排序器需要 Linux 环境（pthread.h 支持）")
        print("  - Windows 环境不支持重排序器的 C++ 扩展")
        print("\n解决方法：")
        print("  1. 在 WSL 终端中运行：")
        print("     cd /mnt/d/LLM_Agent_for_Education")
        print("     python3 start_ui.py")
        print("\n  2. 或从 Windows PowerShell 运行：")
        print("     wsl -d Ubuntu-22.04 -e bash -c \"cd /mnt/d/LLM_Agent_for_Education && python3 start_ui.py\"")
        print("\n" + "="*60)
        sys.exit(1)
    
    print(f"\n✓ 运行环境：{env_type}")
    print("✓ 使用系统 C++ 编译器（gcc/g++）")
    print("✓ ColBERTv2 重排序器将可用")
    
    print("\n正在启动服务器...")
    print("  - HTTP 服务器 (端口 8000): 前端UI")
    print("  - RAG 服务器 (端口 5000): 语义搜索功能 + ColBERTv2 重排序")
    print("\n提示: 按 Ctrl+C 停止所有服务器")
    
    # 创建线程来运行 RAG 服务器
    rag_thread = threading.Thread(target=start_rag_server, daemon=True)
    rag_thread.start()
    
    # 等待一下让 RAG 服务器开始初始化
    time.sleep(2)
    
    # 在主线程运行 HTTP 服务器（这样 Ctrl+C 可以正确捕获）
    try:
        start_http_server()
    except KeyboardInterrupt:
        print("\n\n" + "="*60)
        print("正在关闭所有服务器...")
        print("="*60)
        sys.exit(0)

if __name__ == '__main__':
    main()

