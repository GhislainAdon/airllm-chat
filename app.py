#!/usr/bin/env python3
"""
ChatGPT-style API server using airllm as the backend engine.
Provides REST endpoints for chat, model management, and settings.
OpenAI-compatible API at /v1/ for Cline, Continue, Cursor, etc.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from http.server import HTTPServer, SimpleHTTPRequestHandler
from socketserver import ThreadingMixIn
from urllib.parse import urlparse, parse_qs
import os
import sys
import webbrowser
import argparse
from typing import Optional

from airllm_engine import AirLLMEngine


def _read_version() -> str:
    """Read version from VERSION file next to this script."""
    version_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "VERSION")
    try:
        with open(version_file, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        return "0.0.0-dev"


__version__ = _read_version()

# Global engine instance
engine: Optional[AirLLMEngine] = None
PORT = 7860
HOST = "127.0.0.1"
MAX_BODY_SIZE = 10 * 1024 * 1024  # 10 MB


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle each request in a new thread — critical for streaming."""
    daemon_threads = True
    allow_reuse_address = True


def get_static_dir():
    if getattr(sys, 'frozen', False):
        return os.path.join(sys._MEIPASS, 'static')
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')


def get_template_content(filename: str) -> str:
    if getattr(sys, 'frozen', False):
        filepath = os.path.join(sys._MEIPASS, 'templates', filename)
    else:
        filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates', filename)
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()


class ChatHandler(SimpleHTTPRequestHandler):
    """HTTP request handler with API endpoints for chat."""

    def __init__(self, *args, **kwargs):
        static_dir = get_static_dir()
        super().__init__(*args, directory=static_dir, **kwargs)

    def log_message(self, format, *args):
        sys.stderr.write(f"[{self.log_date_time_string()}] {format % args}\n")

    # ---- Response helpers ----

    def _send_json(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html_content, status=200):
        body = html_content.encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self):
        """Read and parse JSON body. Returns dict, or None on error/oversized."""
        content_length = int(self.headers.get('Content-Length', 0))
        if content_length > MAX_BODY_SIZE:
            return None
        if content_length > 0:
            raw = self.rfile.read(content_length)
            try:
                return json.loads(raw.decode('utf-8'))
            except json.JSONDecodeError:
                return None
        return {}

    def _bad_request(self, message="Invalid request body"):
        self._send_json({
            "error": {"message": message, "type": "invalid_request_error"}
        }, 400)

    # ---- GET routes ----

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path in ('/', ''):
            html = get_template_content('index.html')
            self._send_html(html)
            return

        if path == '/api/info':
            self._send_json(engine.info)
            return

        if path == '/api/models':
            models = [
                {"id": "garage-bAInd/Platypus2-70B-instruct", "name": "Platypus2 70B Instruct", "size": "70B"},
                {"id": "garage-bAInd/Platypus2-13B-instruct", "name": "Platypus2 13B Instruct", "size": "13B"},
                {"id": "TheBloke/Llama-2-7B-Chat-GPTQ", "name": "Llama 2 7B Chat (GPTQ)", "size": "7B"},
            ]
            self._send_json({"models": models})
            return

        # ---- Ollama GET endpoints ----

        if path == '/api/ollama/status':
            self._send_json(engine.get_ollama_info())
            return

        if path == '/api/ollama/models':
            self._send_json(engine.get_ollama_models())
            return

        # ---- System info GET endpoint ----

        if path == '/api/system/info':
            self._send_json(engine.get_system_info())
            return

        # OpenAI-compatible: /v1/models
        if path in ('/v1/models', '/v1/models/'):
            now = int(time.time())
            model_id = engine.model_path if engine.is_loaded else "no-model-loaded"
            data = {
                "object": "list",
                "data": [{"id": model_id, "object": "model", "created": now,
                          "owned_by": "airllm-local", "permission": [], "root": model_id, "parent": None}],
            }
            for m in ["garage-bAInd/Platypus2-70B-instruct",
                       "garage-bAInd/Platypus2-13B-instruct",
                       "TheBloke/Llama-2-7B-Chat-GPTQ"]:
                if m != model_id:
                    data["data"].append({"id": m, "object": "model", "created": now,
                                         "owned_by": "airllm-local", "permission": [],
                                         "root": m, "parent": None})
            self._send_json(data)
            return

        if path in ('/v1', '/v1/'):
            self._send_json({"status": "ok", "engine": "airllm",
                             "message": "OpenAI-compatible API running. Use /v1/chat/completions and /v1/models."})
            return

        super().do_GET()

    # ---- POST routes ----

    def do_POST(self):
        path = urlparse(self.path).path

        # OpenAI-compatible: /v1/chat/completions
        if path == '/v1/chat/completions':
            body = self._read_body()
            if body is None:
                self._bad_request()
                return

            messages = body.get('messages', [])
            stream = body.get('stream', False)
            req_model = body.get('model', '')

            if req_model and '..' in req_model:
                self._bad_request("Invalid model path")
                return

            if req_model and req_model != engine.model_path and req_model != 'no-model-loaded':
                print(f"[OpenAI API] Switching model to: {req_model}")
                engine.set_params(model_path=req_model)

            if body.get('temperature') is not None:
                engine.temperature = max(0.0, min(2.0, float(body['temperature'])))
            if body.get('max_tokens') is not None:
                engine.max_new_tokens = max(1, min(32768, int(body['max_tokens'])))

            if stream:
                self._handle_openai_stream(messages)
            else:
                response = engine.generate(messages, stream=False)
                self._send_json(self._build_openai_response(response))
            return

        if path == '/api/load':
            body = self._read_body()
            if body is None:
                self._bad_request()
                return
            model_path = body.get('model', '')
            if model_path and '..' in model_path:
                self._bad_request("Invalid model path")
                return
            if model_path:
                result = engine.set_params(model_path=model_path)
            else:
                result = engine.load_model()
            self._send_json(result)
            return

        if path == '/api/chat':
            body = self._read_body()
            if body is None:
                self._bad_request()
                return
            messages = body.get('messages', [])
            stream = body.get('stream', False)
            if stream:
                self._handle_stream_chat(messages)
            else:
                response = engine.generate(messages, stream=False)
                self._send_json({"choices": [{"message": {"role": "assistant", "content": response}}],
                                 "model": engine.model_path})
            return

        if path == '/api/settings':
            body = self._read_body()
            if body is None:
                self._bad_request()
                return
            result = engine.set_params(
                max_new_tokens=body.get('max_tokens'),
                temperature=body.get('temperature'),
            )
            self._send_json(result)
            return

        if path == '/api/stop':
            engine.cancel_generation()
            self._send_json({"status": "ok", "message": "Generation stopped."})
            return

        if path == '/api/unload':
            engine.unload_model()
            self._send_json({"status": "ok", "message": "Model unloaded."})
            return

        # ---- Ollama POST endpoints ----

        if path == '/api/ollama/pull':
            body = self._read_body()
            if body is None:
                self._bad_request()
                return
            model_name = body.get('model', '')
            if not model_name:
                self._bad_request("Missing 'model' field in request body.")
                return
            self._handle_ollama_pull(model_name)
            return

        if path == '/api/ollama/delete':
            body = self._read_body()
            if body is None:
                self._bad_request()
                return
            model_name = body.get('model', '')
            if not model_name:
                self._bad_request("Missing 'model' field in request body.")
                return
            result = engine.delete_ollama_model(model_name)
            self._send_json(result)
            return

        if path == '/api/ollama/load':
            body = self._read_body()
            if body is None:
                self._bad_request()
                return
            model_name = body.get('model', '')
            if not model_name:
                self._bad_request("Missing 'model' field in request body.")
                return
            # Resolve the GGUF path from Ollama, then load into airllm
            path_result = engine.get_ollama_model_path(model_name)
            if path_result.get("status") != "success":
                self._send_json(path_result, 400)
                return
            gguf_path = path_result["path"]
            result = engine.set_params(model_path=gguf_path)
            result["ollama_model"] = model_name
            result["gguf_path"] = gguf_path
            self._send_json(result)
            return

        if path == '/api/ollama/path':
            body = self._read_body()
            if body is None:
                self._bad_request()
                return
            custom_path = body.get('path', '')
            if not custom_path:
                self._bad_request("Missing 'path' field in request body.")
                return
            if '..' in custom_path:
                self._bad_request("Invalid path: path traversal detected.")
                return
            expanded = os.path.expanduser(custom_path)
            if os.path.isdir(expanded):
                engine._ollama_models_dir = expanded
                self._send_json({
                    "status": "success",
                    "message": f"Ollama models directory set to: {expanded}",
                    "path": expanded,
                })
            else:
                self._send_json({
                    "status": "error",
                    "message": f"Directory does not exist: {expanded}",
                }, 400)
            return

        self._send_json({"error": {"message": "Not found", "type": "invalid_request_error", "code": "not_found"}}, 404)

    # ---- Ollama pull SSE handler ----

    def _handle_ollama_pull(self, model_name: str):
        """Stream ``ollama pull`` progress as Server-Sent Events (SSE)."""
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream; charset=utf-8')
        self.send_header('Cache-Control', 'no-cache')
        self.send_header('Connection', 'keep-alive')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()

        try:
            for event in engine.pull_ollama_model(model_name):
                self.wfile.write(f"data: {json.dumps(event, ensure_ascii=False)}\n\n".encode('utf-8'))
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            return

        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    # ---- OpenAI-compatible response builders ----

    def _build_openai_response(self, content: str, finish_reason: str = "stop") -> dict:
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": engine.model_path,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                         "finish_reason": finish_reason}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

    def _handle_openai_stream(self, messages):
        """OpenAI-compatible SSE streaming (for Cline, Continue, etc.)."""
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream; charset=utf-8')
        self.send_header('Cache-Control', 'no-cache')
        self.send_header('Connection', 'keep-alive')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()

        chat_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())
        model_name = engine.model_path

        # Initial role chunk (OpenAI convention)
        initial = {"id": chat_id, "object": "chat.completion.chunk", "created": created,
                    "model": model_name, "choices": [{"index": 0, "delta": {"role": "assistant"},
                                                      "finish_reason": None}]}
        self.wfile.write(f"data: {json.dumps(initial)}\n\n".encode('utf-8'))
        self.wfile.flush()

        # Stream tokens
        try:
            for token in engine.generate(messages, stream=True):
                if engine._cancel_requested:
                    break
                chunk = {"id": chat_id, "object": "chat.completion.chunk", "created": created,
                         "model": model_name, "choices": [{"index": 0, "delta": {"content": token},
                                                           "finish_reason": None}]}
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode('utf-8'))
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            return

        finish_reason = "stop" if not engine._cancel_requested else "cancelled"
        final = {"id": chat_id, "object": "chat.completion.chunk", "created": created,
                 "model": model_name, "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}]}
        self.wfile.write(f"data: {json.dumps(final)}\n\n".encode('utf-8'))
        self.wfile.flush()

        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()
        engine._cancel_requested = False

    def _handle_stream_chat(self, messages):
        """Internal SSE streaming for the web UI."""
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream; charset=utf-8')
        self.send_header('Cache-Control', 'no-cache')
        self.send_header('Connection', 'keep-alive')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()

        try:
            for token in engine.generate(messages, stream=True):
                if engine._cancel_requested:
                    break
                data = json.dumps({"choices": [{"delta": {"content": token}}],
                                   "model": engine.model_path}, ensure_ascii=False)
                self.wfile.write(f"data: {data}\n\n".encode('utf-8'))
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            return

        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()
        engine._cancel_requested = False

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')
        self.end_headers()


def open_browser():
    time.sleep(1.5)
    webbrowser.open(f"http://{HOST}:{PORT}")


def main():
    global engine, PORT, HOST

    parser = argparse.ArgumentParser(description="AirLLM Chat - ChatGPT-style interface powered by airllm")
    parser.add_argument('--port', type=int, default=7860)
    parser.add_argument('--host', type=str, default='127.0.0.1')
    parser.add_argument('--model', type=str, default='')
    parser.add_argument('--max-tokens', type=int, default=512)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--no-browser', action='store_true')
    args = parser.parse_args()

    PORT = args.port
    HOST = args.host

    print("=" * 60)
    print("   AirLLM Chat  -  Local AI Chat Interface")
    print("=" * 60)

    engine = AirLLMEngine(
        model_path=args.model,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    print(f"[Config] max_tokens={args.max_tokens}  temperature={args.temperature}")

    if args.model:
        print(f"[Loading] {args.model} ...")
        result = engine.load_model()
        print(f"[Result] {result['message']}")
    else:
        print("[Info] No model specified. Load one from the UI or let clients auto-load.")

    server = ThreadedHTTPServer((HOST, PORT), ChatHandler)
    url = f"http://{HOST}:{PORT}"
    print(f"\n[Server] Running at {url}")
    print(f"\n  Web UI:        {url}")
    print(f"  OpenAI API:    {url}/v1/chat/completions")
    print(f"  Models list:   {url}/v1/models")
    print(f"  Ollama status: {url}/api/ollama/status")
    print(f"  System info:   {url}/api/system/info")
    print(f"\n  Compatible with: Cline, Continue, Cursor, OpenInterpreter, etc.")
    print(f"  Just set Base URL to: {url}/v1")
    print(f"\n[Server] Press Ctrl+C to stop.\n")

    if not args.no_browser:
        threading.Thread(target=open_browser, daemon=True).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[Server] Shutting down...")
        engine.unload_model()
        server.server_close()


if __name__ == '__main__':
    main()
