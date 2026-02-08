#!/usr/bin/env python3
"""
Integrated startup script to launch all servers (UI, RAG, Evaluator)

Usage:
    python start_all.py           # start UI + RAG (with reranker when available)
    python start_all.py --no-rag  # don't start RAG
    python start_all.py --ui --evaluator --port 8000 --evaluator-port 8001

The script will:
 - start a simple HTTP server serving the repository (UI) on `--port`
 - import and initialize `rag_server` (calls `initialize_rag()` to load FAISS and ColBERTv2)
 - start the Flask app from `rag_server` on port 5000
 - start an evaluator static HTTP server on `--evaluator-port`
 - perform quick health checks and print status

This is a convenience wrapper; for production use consider using a process manager.
"""

import argparse
import threading
import time
import sys
import os

def start_ui_server(port: int):
    import http.server
    import socketserver

    class QuietHandler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

    print(f"[UI] Starting HTTP server on port {port} (serving repository)")
    try:
        with socketserver.TCPServer(("", port), QuietHandler) as httpd:
            print(f"[UI] Access: http://localhost:{port}/medical-quiz.html")
            httpd.serve_forever()
    except Exception as e:
        print(f"[UI] Failed to start: {e}")


def start_evaluator_server(port: int):
    import http.server
    import socketserver

    class CORSRequestHandler(http.server.SimpleHTTPRequestHandler):
        def end_headers(self):
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type')
            super().end_headers()

    print(f"[Evaluator] Starting evaluator interface on port {port}")
    try:
        with socketserver.TCPServer(("", port), CORSRequestHandler) as httpd:
            print(f"[Evaluator] Access: http://localhost:{port}/question_evaluator.html")
            httpd.serve_forever()
    except Exception as e:
        print(f"[Evaluator] Failed to start: {e}")


def start_rag_server_blocking():
    # Import and initialize rag_server; this should expose `app` and `initialize_rag()`
    try:
        import rag_server
    except Exception as e:
        print(f"[RAG] Cannot import rag_server: {e}")
        return

    print("[RAG] Initializing RAG subsystem (this may take some time on first run)")
    ok = False
    try:
        ok = rag_server.initialize_rag()
    except Exception as e:
        print(f"[RAG] initialize_rag() raised an exception: {e}")

    if not ok:
        print("[RAG] Initialization failed; RAG API will not start")
        return

    # Start Flask app; prefer waitress if available for threaded serving
    try:
        from waitress import serve
        print("[RAG] Starting Flask app with waitress on port 5000")
        serve(rag_server.app, host='0.0.0.0', port=5000)
    except Exception:
        print("[RAG] Starting Flask development server on port 5000")
        rag_server.app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)


def wait_for_http(host: str, port: int, path: str = '/', timeout: int = 5):
    import socket
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1) as s:
                return True
        except Exception:
            time.sleep(0.3)
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ui', action='store_true', default=True, help='Start UI server')
    parser.add_argument('--no-ui', action='store_true', help="Don't start UI server")
    parser.add_argument('--rag', action='store_true', default=True, help='Start RAG server')
    parser.add_argument('--no-rag', action='store_true', help="Don't start RAG server")
    parser.add_argument('--evaluator', action='store_true', help='Start evaluator interface')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--evaluator-port', type=int, default=8001)
    parser.add_argument('--reranker', action='store_true', help='Force attempt to load ColBERTv2 reranker')
    args = parser.parse_args()

    start_ui = args.ui and not args.no_ui
    start_rag = args.rag and not args.no_rag
    start_eval = args.evaluator

    # If reranker explicitly requested, set env var to hint rag_server
    if args.reranker:
        os.environ['FORCE_RERANKER'] = '1'

    threads = []

    # UI thread
    if start_ui:
        t = threading.Thread(target=start_ui_server, args=(args.port,), daemon=True)
        t.start()
        threads.append(('ui', args.port, t))

    # Evaluator thread
    if start_eval:
        t = threading.Thread(target=start_evaluator_server, args=(args.evaluator_port,), daemon=True)
        t.start()
        threads.append(('evaluator', args.evaluator_port, t))

    # RAG thread (blocking inside)
    if start_rag:
        t = threading.Thread(target=start_rag_server_blocking, daemon=True)
        t.start()
        threads.append(('rag', 5000, t))

    # Health checks
    time.sleep(1.0)
    print('\n[Health] Checking services...')
    if start_ui:
        ok = wait_for_http('127.0.0.1', args.port)
        print(f"  UI (http://127.0.0.1:{args.port}) -> {'OK' if ok else 'NO RESPONSE'}")
    if start_rag:
        ok = wait_for_http('127.0.0.1', 5000)
        print(f"  RAG (http://127.0.0.1:5000/health) -> {'OK' if ok else 'NO RESPONSE'}")
    if start_eval:
        ok = wait_for_http('127.0.0.1', args.evaluator_port)
        print(f"  Evaluator (http://127.0.0.1:{args.evaluator_port}) -> {'OK' if ok else 'NO RESPONSE'}")

    print('\n[Info] Servers started (threads):')
    for name, port, th in threads:
        print(f"  - {name}: port {port} -> alive={th.is_alive()}")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print('\n[Shutdown] Stopping all servers (press Ctrl+C again to force)')


if __name__ == '__main__':
    main()
