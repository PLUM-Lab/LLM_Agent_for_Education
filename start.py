#!/usr/bin/env python3
"""
Unified Startup Script - Medical Education Quiz System
=======================================================

Supports starting the following services:
1. Main UI server (port 8000)
2. RAG server (port 5000) - Semantic search functionality
3. Evaluator interface server (port 8001)

Usage:
    # Start all services (default)
    python start.py
    
    # Start only main UI and RAG server
    python start.py --ui --rag
    
    # Start only evaluator interface
    python start.py --evaluator
    
    # Specify ports
    python start.py --ui --port 8000
    
    # Windows environment (auto-detected, ColBERTv2 not required)
    python start.py
    
    # WSL/Linux environment (supports ColBERTv2 reranking)
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

def windows_path_to_wsl(win_path):
    """Convert Windows path to WSL path (e.g. C:\\Users\\foo -> /mnt/c/Users/foo)."""
    path = os.path.normpath(win_path).replace("\\", "/")
    if len(path) >= 2 and path[1] == ":":
        drive = path[0].lower()
        return f"/mnt/{drive}" + (path[2:] if len(path) > 2 else "")
    return path

def start_rag_server_in_wsl():
    """
    Start RAG server inside WSL so ColBERTv2 reranker loads (Windows-only).
    Returns the subprocess.Popen instance, or None on failure.
    """
    cwd = os.getcwd()
    wsl_path = windows_path_to_wsl(cwd)
    env = os.environ.copy()
    env["REQUIRE_RERANKER"] = "1"
    cmd = [
        "wsl", "-e", "bash", "-c",
        f"cd '{wsl_path}' && export REQUIRE_RERANKER=1 && python3 rag_server.py"
    ]
    try:
        p = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=cwd,
        )
        return p
    except FileNotFoundError:
        print("[Warning] WSL not found. Install WSL or use --no-wsl-rag to run RAG without reranker.")
        return None
    except Exception as e:
        print(f"[Warning] Failed to start RAG in WSL: {e}")
        return None

def check_wsl_environment():
    """Check if running in WSL or Linux environment"""
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

def wait_for_http(host='127.0.0.1', port=8000, timeout=5):
    """Check if HTTP server is reachable within timeout seconds."""
    import socket
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1):
                return True
        except Exception:
            time.sleep(0.3)
    return False

def start_http_server(port=8000):
    """Start HTTP server"""
    print(f"\n{'='*60}")
    print(f"[HTTP Server] Starting on port {port}...")
    print(f"{'='*60}\n")
    
    try:
        import http.server
        import socketserver

        class QuietHandler(http.server.SimpleHTTPRequestHandler):
            def log_message(self, format, *args):
                pass

        with socketserver.TCPServer(("", port), QuietHandler) as httpd:
            print("[OK] HTTP server started")
            print(f"  Access: http://localhost:{port}/medical-quiz.html")
            print(f"\nPress Ctrl+C to stop all servers\n")
            httpd.serve_forever()
    except OSError as e:
        if "Address already in use" in str(e) or "address already in use" in str(e).lower():
            print(f"[WARN] Port {port} is already in use")
            print(f"Please try a different port or stop the process using this port")
            return False
        else:
            print(f"✗ HTTP server startup failed: {e}")
            return False
    except KeyboardInterrupt:
        print("\n[HTTP Server] Shutting down...")
        return True
    except Exception as e:
        print(f"[FAIL] HTTP server error: {e}")
        return False

def start_rag_server():
    """Start RAG server"""
    print(f"\n{'='*60}")
    print(f"[RAG Server] Starting...")
    print(f"{'='*60}\n")
    
    try:
        import rag_server
        
        if rag_server.initialize_rag():
            print("\n[RAG Server] Server started at http://localhost:5000")
            print("\nAPI Endpoints:")
            print("  POST /search     - Search relevant chunks")
            print("  GET  /health     - Health check")
            print("  POST /rebuild    - Rebuild index from PDFs")
            print()
            try:
                from waitress import serve
                print("[RAG Server] Using waitress on port 5000")
                serve(rag_server.app, host='0.0.0.0', port=5000)
            except Exception:
                print("[RAG Server] Using Flask dev server on port 5000")
                rag_server.app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
        else:
            print("\n[Warning] RAG system initialization failed")
            print("UI is still usable, but RAG functionality will be unavailable")
            return False
            
    except ImportError as e:
        print(f"\n[Warning] Cannot import rag_server: {e}")
        print("UI is still usable, but RAG functionality will be unavailable")
        return False
    except KeyboardInterrupt:
        print("\n[RAG Server] Shutting down...")
        return True
    except Exception as e:
        print(f"\n[Warning] RAG server startup failed: {e}")
        print("UI is still usable, but RAG functionality will be unavailable")
        return False

def start_evaluator_server(port=8001):
    """Start evaluator interface server"""
    print(f"\n{'='*60}")
    print(f"[Evaluator Interface Server] Starting on port {port}...")
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
            print("[OK] Evaluator interface server started")
            print(f"  Access: http://localhost:{port}/question_evaluator.html")
            print(f"\nPress Ctrl+C to stop server\n")
            httpd.serve_forever()
            
    except OSError as e:
        if "Address already in use" in str(e) or "address already in use" in str(e).lower():
            print(f"⚠ Port {port} is already in use")
            print(f"Please try a different port: python start.py --evaluator --port {port + 1}")
            return False
        else:
            print(f"✗ Server startup failed: {e}")
            return False
    except KeyboardInterrupt:
        print("\n[Evaluator Interface Server] Shutting down...")
        return True
    except Exception as e:
        print(f"[FAIL] Server error: {e}")
        return False

def restart_rag_server():
    """Restart RAG server (stop existing process and restart)"""
    print("\n" + "="*60)
    print("Restart RAG Server")
    print("="*60)
    
    # Check and stop processes using port 5000
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
                        print(f"\n[1/3] Found process using port 5000: PID {pid}")
                        try:
                            subprocess.run(['taskkill', '/F', '/PID', pid], 
                                         check=True, capture_output=True)
                            print(f"[OK] Stopped process {pid}")
                        except:
                            print(f"[WARNING] Cannot stop process {pid} (may already be stopped)")
            
            print("\n[2/3] Waiting for port to be released...")
            time.sleep(2)
        except Exception as e:
            print(f"[WARNING] Error stopping process: {e}")
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
                print(f"\n[1/3] Found processes using port 5000: {', '.join(pids)}")
                for pid in pids:
                    try:
                        os.kill(int(pid), signal.SIGTERM)
                        print(f"[OK] Stopped process {pid}")
                    except:
                        print(f"[WARNING] Cannot stop process {pid} (may already be stopped)")
                
                print("\n[2/3] Waiting for port to be released...")
                time.sleep(2)
        except Exception as e:
            print(f"[WARNING] Error stopping process: {e}")
    
    print("\n[3/3] Starting new RAG server...")
    print("Note: Server will run in current window")
    print("Press Ctrl+C to stop server\n")
    
    # Start RAG server
    start_rag_server()

def main():
    parser = argparse.ArgumentParser(
        description='Unified Startup Script - Medical Education Quiz System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start all services (default)
  python start.py
  
  # Start only main UI and RAG server
  python start.py --ui --rag
  
  # Start only evaluator interface
  python start.py --evaluator
  
  # Specify ports
  python start.py --ui --port 8000 --evaluator --evaluator-port 8002
        """
    )
    
    parser.add_argument('--ui', action='store_true', 
                       help='Start main UI server (port 8000)')
    parser.add_argument('--rag', action='store_true',
                       help='Start RAG server (port 5000)')
    parser.add_argument('--evaluator', action='store_true',
                       help='Start evaluator interface server (port 8001)')
    parser.add_argument('--port', type=int, default=8000,
                       help='Main UI server port (default: 8000)')
    parser.add_argument('--evaluator-port', type=int, default=8001,
                       help='Evaluator interface server port (default: 8001)')
    parser.add_argument('--restart-rag', action='store_true',
                       help='Restart RAG server (stop existing process and restart)')
    parser.add_argument('--no-wsl-rag', action='store_true',
                       help='On Windows: do not start RAG in WSL (RAG runs in current process, reranker may be unavailable)')
    
    args = parser.parse_args()
    
    # If only restarting RAG server, execute and exit
    if args.restart_rag:
        restart_rag_server()
        return
    
    # If no service specified, start all services by default
    if not (args.ui or args.rag or args.evaluator):
        args.ui = True
        args.rag = True
    
    print("\n" + "="*60)
    print("Medical Education Quiz System - Unified Startup Script")
    print("="*60)
    
    # Check runtime environment
    is_linux, env_type = check_wsl_environment()
    print(f"\nRuntime environment: {env_type}")
    
    rag_wsl_process = None  # When on Windows, RAG may run in WSL subprocess
    
    if args.rag and not is_linux and not args.no_wsl_rag:
        print("\n[Info] Windows detected: starting RAG in WSL so ColBERTv2 reranker is used")
        rag_wsl_process = start_rag_server_in_wsl()
        if rag_wsl_process is not None:
            print("  RAG server starting in WSL (port 5000)...")
            time.sleep(3)  # WSL init takes longer
        else:
            print("  Falling back to RAG in current process (reranker may be unavailable)\n")
            rag_wsl_process = None
    elif args.rag and not is_linux:
        print("\n[Info] --no-wsl-rag: RAG runs in current process (ColBERTv2 may be unavailable)\n")
    elif is_linux and args.rag:
        print("[OK] ColBERTv2 reranker will be available\n")
    
    # Start services
    threads = []
    
    # Start RAG server (if needed, and not already started in WSL)
    if args.rag and rag_wsl_process is None:
        rag_thread = threading.Thread(target=start_rag_server, daemon=True)
        rag_thread.start()
        threads.append(rag_thread)
        time.sleep(2)  # Wait for RAG server to start initializing
    
    # Start evaluator interface server (if needed, in separate thread)
    if args.evaluator:
        eval_thread = threading.Thread(
            target=start_evaluator_server, 
            args=(args.evaluator_port,), 
            daemon=True
        )
        eval_thread.start()
        threads.append(eval_thread)
        time.sleep(1)

    # Health checks
    time.sleep(1.0)
    print("\n[Health] Checking services...")
    if args.ui:
        ok = wait_for_http('127.0.0.1', args.port)
        print(f"  UI (http://127.0.0.1:{args.port}) -> {'OK' if ok else 'NO RESPONSE'}")
    if args.rag:
        ok = wait_for_http('127.0.0.1', 5000)
        print(f"  RAG (http://127.0.0.1:5000) -> {'OK' if ok else 'NO RESPONSE'}")
    if args.evaluator:
        ok = wait_for_http('127.0.0.1', args.evaluator_port)
        print(f"  Evaluator (http://127.0.0.1:{args.evaluator_port}) -> {'OK' if ok else 'NO RESPONSE'}")
    print()

    # Run HTTP server in main thread (so Ctrl+C can be properly caught)
    try:
        if args.ui:
            start_http_server(args.port)
        else:
            # If UI not started, keep script running
            print("\nAll services started in background")
            print("Press Ctrl+C to stop all services\n")
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n" + "="*60)
        print("Shutting down all servers...")
        if rag_wsl_process is not None:
            try:
                rag_wsl_process.terminate()
                rag_wsl_process.wait(timeout=5)
            except Exception:
                rag_wsl_process.kill()
        print("="*60)
        sys.exit(0)

if __name__ == '__main__':
    main()
