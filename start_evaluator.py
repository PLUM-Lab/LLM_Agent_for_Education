#!/usr/bin/env python3
"""
启动问题评估界面服务器

使用端口 8001，避免与主 UI（端口 8000）冲突

使用方法：
    python start_evaluator.py

或指定端口：
    python start_evaluator.py --port 8001
"""

import argparse
import http.server
import socketserver
import sys
from pathlib import Path

def start_evaluator_server(port=8001):
    """启动评估界面 HTTP 服务器"""
    print(f"\n{'='*60}")
    print(f"问题评估界面服务器")
    print(f"{'='*60}\n")
    print(f"正在启动在端口 {port}...")
    
    try:
        handler = http.server.SimpleHTTPRequestHandler
        
        # 自定义处理程序，添加 CORS 支持
        class CORSRequestHandler(http.server.SimpleHTTPRequestHandler):
            def end_headers(self):
                self.send_header('Access-Control-Allow-Origin', '*')
                self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
                self.send_header('Access-Control-Allow-Headers', 'Content-Type')
                super().end_headers()
        
        with socketserver.TCPServer(("", port), CORSRequestHandler) as httpd:
            print(f"\n✓ 服务器已启动")
            print(f"\n访问地址:")
            print(f"  http://localhost:{port}/question_evaluator.html")
            print(f"\n按 Ctrl+C 停止服务器\n")
            httpd.serve_forever()
            
    except OSError as e:
        if "Address already in use" in str(e) or "address already in use" in str(e).lower():
            print(f"\n⚠ 端口 {port} 已被占用")
            print(f"请尝试使用其他端口:")
            print(f"  python start_evaluator.py --port {port + 1}")
            sys.exit(1)
        else:
            print(f"\n✗ 服务器启动失败: {e}")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n正在关闭服务器...")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ 服务器错误: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description='启动问题评估界面服务器',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 使用默认端口 8001
  python start_evaluator.py
  
  # 指定端口
  python start_evaluator.py --port 8002
        """
    )
    
    parser.add_argument(
        '--port', '-p',
        type=int,
        default=8001,
        help='服务器端口（默认：8001）'
    )
    
    args = parser.parse_args()
    start_evaluator_server(args.port)

if __name__ == '__main__':
    main()

