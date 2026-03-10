#!/usr/bin/env python3
"""
Unified Startup Script - Medical Education Quiz System
=======================================================

Single-port mode: one server (port 8000) serves UI + RAG API.

Usage:
    python start.py                    # Start at http://localhost:8000
    python start.py --port 8080        # Use another port
    python start.py --evaluator        # Also start evaluator (port 8001)
    python start.py --restart-rag      # Restart main server only
    python start.py --wsl             # Windows: run server in WSL (ColBERTv2 重排序可用)
    python start.py --no-wsl-rag      # Windows: run in current process (no WSL)
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
        f"cd '{wsl_path}' && export REQUIRE_RERANKER=1 && export RAG_PORT=5001 && python3 rag_server.py"
    ]
    try:
        # 不使用 PIPE，让 RAG 的 traceback 直接输出到终端，便于排查 Tutor 500 错误
        p = subprocess.Popen(
            cmd,
            stdout=None,
            stderr=None,
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

def get_wsl_ip():
    """Get WSL IP address. Returns None on failure."""
    try:
        r = subprocess.run(
            ["wsl", "hostname", "-I"],
            capture_output=True, text=True, timeout=10, cwd=os.getcwd()
        )
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip().split()[0]
    except Exception:
        pass
    return None

def setup_port_forward_to_wsl():
    """
    Set up port forwarding: localhost:5000 -> WSL:5000 (Windows only).
    Required for Tutor to access WSL RAG. May trigger UAC if not running as admin.
    Returns True if successful or already accessible.
    """
    wsl_ip = get_wsl_ip()
    if not wsl_ip:
        print("[Warning] Cannot get WSL IP. Port forwarding skipped.")
        return False
    # Try netsh (works if Python has admin)
    try:
        subprocess.run(
            ["netsh", "interface", "portproxy", "delete", "v4tov4",
             "listenport=5000", "listenaddress=127.0.0.1"],
            capture_output=True, timeout=5,
        )
        r = subprocess.run(
            ["netsh", "interface", "portproxy", "add", "v4tov4",
             "listenport=5000", "listenaddress=127.0.0.1",
             f"connectport=5001", f"connectaddress={wsl_ip}"],
            capture_output=True, timeout=5,
        )
        if r.returncode == 0:
            print(f"[OK] Port forwarding: 127.0.0.1:5000 -> {wsl_ip}:5001")
            return True
    except Exception:
        pass
    # Need admin: launch setup_port_forward.ps1 with elevation (UAC popup)
    script_dir = Path(__file__).resolve().parent
    ps1 = script_dir / "wsl" / "setup_port_forward.ps1"
    if ps1.exists():
        print("[Info] Setting up port forwarding (may show UAC prompt)...")
        try:
            subprocess.run(
                ["powershell", "-ExecutionPolicy", "Bypass", "-Command",
                 f"Start-Process powershell -ArgumentList '-ExecutionPolicy Bypass -NoProfile -File \"{ps1}\"' -Verb RunAs -Wait"],
                cwd=str(script_dir), timeout=30,
            )
            time.sleep(1)
            return wait_for_http("127.0.0.1", 5000, timeout=3)
        except Exception as e:
            print(f"[Warning] Port forwarding failed: {e}")
    return False

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


def write_rag_config(port, single_port=False):
    """Write rag-config.js so HTML pages know which port the RAG server is on (and single-port mode)."""
    config_path = Path(__file__).resolve().parent / "rag-config.js"
    single_line = "\nwindow.SINGLE_PORT = true;\n" if single_port else "\n"
    config_path.write_text(
        "// RAG server port configuration\n"
        "// Auto-generated by start.py -- do not edit manually.\n"
        f"window.RAG_PORT = {port};{single_line}",
        encoding="utf-8"
    )
    print(f"[Config] rag-config.js written with RAG_PORT={port}" + (" (single-port mode)" if single_port else ""))


def start_rag_server(port=5001):
    """Start RAG server"""
    print(f"\n{'='*60}")
    print(f"[RAG Server] Starting on port {port}...")
    print(f"{'='*60}\n")

    os.environ['RAG_PORT'] = str(port)

    try:
        import rag_server

        if rag_server.initialize_rag():
            print(f"\n[RAG Server] Server started at http://localhost:{port}")
            print("\nAPI Endpoints:")
            print("  POST /search     - Search relevant chunks")
            print("  GET  /health        - Health check")
            print("  GET /user_log, POST /user_log/event - Per-user log (user_logs/)")
            print("  POST /rebuild       - Rebuild index from PDFs")
            print()
            try:
                from waitress import serve
                print(f"[RAG Server] Using waitress on port {port}")
                serve(rag_server.app, host='0.0.0.0', port=port)
            except Exception:
                print(f"[RAG Server] Using Flask dev server on port {port}")
                rag_server.app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)
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
            print(f"[FAIL] Server startup failed: {e}")
            return False
    except KeyboardInterrupt:
        print("\n[Evaluator Interface Server] Shutting down...")
        return True
    except Exception as e:
        print(f"[FAIL] Server error: {e}")
        return False

def restart_rag_server(port=8000):
    """Restart main server (UI + API) on the given port."""
    print("\n" + "="*60)
    print(f"Restart RAG Server (port {port})")
    print("="*60)

    # Check and stop processes using the RAG port
    if sys.platform == "win32":
        try:
            import subprocess
            result = subprocess.run(
                ['netstat', '-ano'],
                capture_output=True,
                text=True
            )
            for line in result.stdout.split('\n'):
                if f':{port}' in line and 'LISTENING' in line:
                    parts = line.split()
                    if len(parts) >= 5:
                        pid = parts[-1]
                        print(f"\n[1/3] Found process using port {port}: PID {pid}")
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
                [f'lsof', f'-ti:{port}'],
                capture_output=True,
                text=True
            )
            if result.stdout.strip():
                pids = result.stdout.strip().split('\n')
                print(f"\n[1/3] Found processes using port {port}: {', '.join(pids)}")
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
    start_rag_server(port)

def main():
    parser = argparse.ArgumentParser(
        description='Unified Startup Script - Medical Education Quiz System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python start.py
  python start.py --wsl              # Windows: 在 WSL 中运行，启用 ColBERTv2 重排序
  python start.py --port 8080
  python start.py --evaluator
  python start.py --restart-rag
        """
    )
    
    parser.add_argument('--evaluator', action='store_true',
                       help='Also start evaluator interface (port 8001)')
    parser.add_argument('--port', type=int, default=8000,
                       help='Server port for UI + API (default: 8000)')
    parser.add_argument('--evaluator-port', type=int, default=8001,
                       help='Evaluator interface port (default: 8001)')
    parser.add_argument('--restart-rag', action='store_true',
                       help='Restart main server (same as --port) and exit')
    parser.add_argument('--wsl', action='store_true',
                       help='Windows: run server in WSL so ColBERTv2 reranking works')
    parser.add_argument('--no-wsl-rag', action='store_true',
                       help='Windows: run server in current process (no WSL/ColBERTv2)')

    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    os.chdir(project_root)

    # Single-port only: RAG serves UI + API on --port
    write_rag_config(args.port, single_port=True)

    # If only restarting main server, execute and exit
    if args.restart_rag:
        restart_rag_server(args.port)
        return

    # Windows: 在 WSL 中运行整个服务，使 ColBERTv2 重排序可用
    if sys.platform == 'win32' and getattr(args, 'wsl', False):
        wsl_path = windows_path_to_wsl(os.getcwd())
        print("\n" + "="*60)
        print("在 WSL 中启动服务（ColBERTv2 重排序可用）")
        print("="*60)
        print(f"  项目路径 (WSL): {wsl_path}")
        print(f"  浏览器访问: http://localhost:{args.port}/")
        print("  按 Ctrl+C 停止\n")
        cmd = f"cd '{wsl_path}' && export RAG_PORT={args.port} && python3 start.py"
        rc = subprocess.call(["wsl", "-e", "bash", "-c", cmd])
        sys.exit(rc if rc is not None else 0)

    # By default start main server (UI + API). Optionally add evaluator.
    args.rag = True
    args.rag_port = args.port

    print("\n" + "="*60)
    print("Medical Education Quiz System - Unified Startup Script")
    print("="*60)
    
    # Check runtime environment
    is_linux, env_type = check_wsl_environment()
    print(f"\nRuntime environment: {env_type}")
    
    # Single-port: one process serves UI + API
    args.no_wsl_rag = True

    # Start services
    threads = []

    # Start main server (UI + API on one port)
    if args.rag:
        rag_thread = threading.Thread(target=start_rag_server, args=(args.rag_port,), daemon=True)
        rag_thread.start()
        threads.append(rag_thread)
        time.sleep(2)
        print(f"\n[Server] UI + API at http://127.0.0.1:{args.rag_port}/ (e.g. /medical-quiz.html)")

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
    if args.rag:
        ok = wait_for_http('127.0.0.1', args.rag_port)
        print(f"  Main (http://127.0.0.1:{args.rag_port}) -> {'OK' if ok else 'NO RESPONSE'}")
    if args.evaluator:
        ok = wait_for_http('127.0.0.1', args.evaluator_port)
        print(f"  Evaluator (http://127.0.0.1:{args.evaluator_port}) -> {'OK' if ok else 'NO RESPONSE'}")
    print()

    # Main thread: keep running until Ctrl+C
    try:
        print("\nPress Ctrl+C to stop.\n")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n" + "="*60)
        print("Shutting down...")
        print("="*60)
        sys.exit(0)

if __name__ == '__main__':
    main()
