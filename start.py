#!/usr/bin/env python3
"""
统一启动脚本 - 医学教育测验系统
=====================================

支持启动以下服务：
1. 主UI服务器 (端口 8000)
2. RAG服务器 (端口 5000) - 语义搜索功能
3. 评估界面服务器 (端口 8001)

使用方法：
    # 启动所有服务（默认）
    python start.py
    
    # 只启动主UI和RAG服务器
    python start.py --ui --rag
    
    # 只启动评估界面
    python start.py --evaluator
    
    # 指定端口
    python start.py --ui --port 8000
    
    # Windows环境（自动检测，无需ColBERTv2）
    python start.py
    
    # WSL/Linux环境（支持ColBERTv2重排序）
    python start.py
"""

import argparse
import threading
import time
import sys
import os
import subprocess
import signal
from pathlib import Path

def check_wsl_environment():
    """检查是否在 WSL 或 Linux 环境中运行"""
    if sys.platform == "win32":
        return False, "Windows"
    
    try:
        with open('/proc/version', 'r') as f:
            version_info = f.read().lower()
            if 'microsoft' in version_info or 'wsl' in version_info:
                return True, "WSL"
            elif 'linux' in version_info:
                return True, "Linux"
    except:
        pass
    
    if os.environ.get('WSL_DISTRO_NAME'):
        return True, "WSL"
    
    if sys.platform.startswith('linux'):
        return True, "Linux"
    
    return False, "Unknown"

def start_http_server(port=8000):
    """启动HTTP服务器"""
    print(f"\n{'='*60}")
    print(f"[HTTP 服务器] 正在启动在端口 {port}...")
    print(f"{'='*60}\n")
    
    try:
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
            print(f"⚠ 端口 {port} 已被占用")
            print(f"请尝试使用其他端口或停止占用该端口的进程")
            return False
        else:
            print(f"✗ HTTP 服务器启动失败: {e}")
            return False
    except KeyboardInterrupt:
        print("\n[HTTP 服务器] 正在关闭...")
        return True
    except Exception as e:
        print(f"✗ HTTP 服务器错误: {e}")
        return False

def start_rag_server():
    """启动RAG服务器"""
    print(f"\n{'='*60}")
    print(f"[RAG 服务器] 正在启动...")
    print(f"{'='*60}\n")
    
    try:
        import rag_server
        
        if rag_server.initialize_rag():
            print("\n[RAG 服务器] 服务器启动在 http://localhost:5000")
            print("\nAPI 端点：")
            print("  POST /search     - 搜索相关块")
            print("  GET  /health     - 健康检查")
            print("  POST /rebuild    - 从 PDF 重建索引")
            print()
            
            rag_server.app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
        else:
            print("\n[警告] RAG 系统初始化失败")
            print("UI 仍可使用，但 RAG 功能将不可用")
            return False
            
    except ImportError as e:
        print(f"\n[警告] 无法导入 rag_server: {e}")
        print("UI 仍可使用，但 RAG 功能将不可用")
        return False
    except KeyboardInterrupt:
        print("\n[RAG 服务器] 正在关闭...")
        return True
    except Exception as e:
        print(f"\n[警告] RAG 服务器启动失败: {e}")
        print("UI 仍可使用，但 RAG 功能将不可用")
        return False

def start_evaluator_server(port=8001):
    """启动评估界面服务器"""
    print(f"\n{'='*60}")
    print(f"[评估界面服务器] 正在启动在端口 {port}...")
    print(f"{'='*60}\n")
    
    try:
        import http.server
        import socketserver
        
        class CORSRequestHandler(http.server.SimpleHTTPRequestHandler):
            def end_headers(self):
                self.send_header('Access-Control-Allow-Origin', '*')
                self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
                self.send_header('Access-Control-Allow-Headers', 'Content-Type')
                super().end_headers()
        
        with socketserver.TCPServer(("", port), CORSRequestHandler) as httpd:
            print(f"✓ 评估界面服务器已启动")
            print(f"  访问地址: http://localhost:{port}/question_evaluator.html")
            print(f"\n按 Ctrl+C 停止服务器\n")
            httpd.serve_forever()
            
    except OSError as e:
        if "Address already in use" in str(e) or "address already in use" in str(e).lower():
            print(f"⚠ 端口 {port} 已被占用")
            print(f"请尝试使用其他端口: python start.py --evaluator --port {port + 1}")
            return False
        else:
            print(f"✗ 服务器启动失败: {e}")
            return False
    except KeyboardInterrupt:
        print("\n[评估界面服务器] 正在关闭...")
        return True
    except Exception as e:
        print(f"✗ 服务器错误: {e}")
        return False

def restart_rag_server():
    """重启RAG服务器（停止现有进程后重新启动）"""
    print("\n" + "="*60)
    print("重启 RAG 服务器")
    print("="*60)
    
    # 检查并停止占用端口5000的进程
    if sys.platform == "win32":
        try:
            import subprocess
            result = subprocess.run(
                ['netstat', '-ano'],
                capture_output=True,
                text=True
            )
            for line in result.stdout.split('\n'):
                if ':5000' in line and 'LISTENING' in line:
                    parts = line.split()
                    if len(parts) >= 5:
                        pid = parts[-1]
                        print(f"\n[1/3] 找到占用端口5000的进程: PID {pid}")
                        try:
                            subprocess.run(['taskkill', '/F', '/PID', pid], 
                                         check=True, capture_output=True)
                            print(f"✓ 已停止进程 {pid}")
                        except:
                            print(f"⚠ 无法停止进程 {pid}（可能已经停止）")
            
            print("\n[2/3] 等待端口释放...")
            time.sleep(2)
        except Exception as e:
            print(f"⚠ 停止进程时出错: {e}")
    else:
        # Linux/WSL
        try:
            import subprocess
            result = subprocess.run(
                ['lsof', '-ti:5000'],
                capture_output=True,
                text=True
            )
            if result.stdout.strip():
                pids = result.stdout.strip().split('\n')
                print(f"\n[1/3] 找到占用端口5000的进程: {', '.join(pids)}")
                for pid in pids:
                    try:
                        os.kill(int(pid), signal.SIGTERM)
                        print(f"✓ 已停止进程 {pid}")
                    except:
                        print(f"⚠ 无法停止进程 {pid}（可能已经停止）")
                
                print("\n[2/3] 等待端口释放...")
                time.sleep(2)
        except Exception as e:
            print(f"⚠ 停止进程时出错: {e}")
    
    print("\n[3/3] 启动新RAG服务器...")
    print("注意: 服务器将在当前窗口运行")
    print("按 Ctrl+C 可以停止服务器\n")
    
    # 启动RAG服务器
    start_rag_server()

def main():
    parser = argparse.ArgumentParser(
        description='统一启动脚本 - 医学教育测验系统',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 启动所有服务（默认）
  python start.py
  
  # 只启动主UI和RAG服务器
  python start.py --ui --rag
  
  # 只启动评估界面
  python start.py --evaluator
  
  # 指定端口
  python start.py --ui --port 8000 --evaluator --evaluator-port 8002
        """
    )
    
    parser.add_argument('--ui', action='store_true', 
                       help='启动主UI服务器 (端口 8000)')
    parser.add_argument('--rag', action='store_true',
                       help='启动RAG服务器 (端口 5000)')
    parser.add_argument('--evaluator', action='store_true',
                       help='启动评估界面服务器 (端口 8001)')
    parser.add_argument('--port', type=int, default=8000,
                       help='主UI服务器端口 (默认: 8000)')
    parser.add_argument('--evaluator-port', type=int, default=8001,
                       help='评估界面服务器端口 (默认: 8001)')
    parser.add_argument('--restart-rag', action='store_true',
                       help='重启RAG服务器（停止现有进程后重新启动）')
    
    args = parser.parse_args()
    
    # 如果只是重启RAG服务器，直接执行并退出
    if args.restart_rag:
        restart_rag_server()
        return
    
    # 如果没有指定任何服务，默认启动所有服务
    if not (args.ui or args.rag or args.evaluator):
        args.ui = True
        args.rag = True
    
    print("\n" + "="*60)
    print("医学教育测验系统 - 统一启动脚本")
    print("="*60)
    
    # 检查运行环境
    is_linux, env_type = check_wsl_environment()
    print(f"\n运行环境: {env_type}")
    
    if args.rag and not is_linux:
        print("\n[警告] Windows 环境检测到")
        print("  - RAG 服务器可以启动，但 ColBERTv2 重排序器可能不可用")
        print("  - 如需完整功能，请在 WSL/Linux 环境中运行")
        print("  - 继续启动...\n")
    
    if is_linux and args.rag:
        print("✓ ColBERTv2 重排序器将可用\n")
    
    # 启动服务
    threads = []
    
    # 启动RAG服务器（如果需要）
    if args.rag:
        rag_thread = threading.Thread(target=start_rag_server, daemon=True)
        rag_thread.start()
        threads.append(rag_thread)
        time.sleep(2)  # 等待RAG服务器开始初始化
    
    # 启动评估界面服务器（如果需要，在独立线程）
    if args.evaluator:
        eval_thread = threading.Thread(
            target=start_evaluator_server, 
            args=(args.evaluator_port,), 
            daemon=True
        )
        eval_thread.start()
        threads.append(eval_thread)
        time.sleep(1)
    
    # 在主线程运行HTTP服务器（这样 Ctrl+C 可以正确捕获）
    try:
        if args.ui:
            start_http_server(args.port)
        else:
            # 如果没有启动UI，保持脚本运行
            print("\n所有服务已在后台启动")
            print("按 Ctrl+C 停止所有服务\n")
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n" + "="*60)
        print("正在关闭所有服务器...")
        print("="*60)
        sys.exit(0)

if __name__ == '__main__':
    main()
