"""
ChatGPT-style API server using airllm as the backend engine.
Provides REST endpoints for chat, model management, and settings.
"""

import json
import queue
import threading
import time
import uuid
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import os
import sys
import webbrowser
import argparse

from airllm_engine import AirLLMEngine

# Global engine instance
engine: AirLLMEngine = None
PORT = 7860
HOST = "127.0.0.1"


def get_static_dir():
    """Get the directory where static files are located."""
    if getattr(sys, 'frozen', False):
        # Running as a PyInstaller bundle
        return os.path.join(sys._MEIPASS, 'static')
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')


def get_template_content(filename: str) -> str:
    """Read an HTML template file."""
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
        # Suppress default logging for cleaner output
        sys.stderr.write(f"[{self.log_date_time_string()}] {format % args}\n")

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
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self):
        content_length = int(self.headers.get('Content-Length', 0))
        if content_length > 0:
            raw = self.rfile.read(content_length)
            try:
                return json.loads(raw.decode('utf-8'))
            except json.JSONDecodeError:
                return {}
        return {}

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        params = parse_qs(parsed.query)

        # Serve the main chat page
        if path == '/' or path == '':
            html = get_template_content('index.html')
            self._send_html(html)
            return

        # API: Get engine info
        if path == '/api/info':
            self._send_json(engine.info)
            return

        # API: List models (placeholder - airllm uses HuggingFace paths)
        if path == '/api/models':
            models = [
                {
                    "id": "garage-bAInd/Platypus2-70B-instruct",
                    "name": "Platypus2 70B Instruct",
                    "size": "70B",
                },
                {
                    "id": "garage-bAInd/Platypus2-13B-instruct",
                    "name": "Platypus2 13B Instruct",
                    "size": "13B",
                },
                {
                    "id": "TheBloke/Llama-2-7B-Chat-GPTQ",
                    "name": "Llama 2 7B Chat (GPTQ)",
                    "size": "7B",
                },
            ]
            self._send_json({"models": models})
            return

        # ==============================================
        # OpenAI-Compatible API: /v1/models
        # Used by Cline, Continue, Cursor, OpenInterpreter, etc.
        # ==============================================
        if path == '/v1/models' or path == '/v1/models/':
            now = int(time.time())
            model_id = engine.model_path if engine.is_loaded else "no-model-loaded"
            data = {
                "object": "list",
                "data": [
                    {
                        "id": model_id,
                        "object": "model",
                        "created": now,
                        "owned_by": "airllm-local",
                        "permission": [],
                        "root": model_id,
                        "parent": None,
                    }
                ]
            }
            # Also include known models that can be loaded
            known = [
                "garage-bAInd/Platypus2-70B-instruct",
                "garage-bAInd/Platypus2-13B-instruct",
                "TheBloke/Llama-2-7B-Chat-GPTQ",
            ]
            for m in known:
                if m != model_id:
                    data["data"].append({
                        "id": m,
                        "object": "model",
                        "created": now,
                        "owned_by": "airllm-local",
                        "permission": [],
                        "root": m,
                        "parent": None,
                    })
            self._send_json(data)
            return

        # OpenAI health/compatibility check endpoint
        if path == '/v1' or path == '/v1/':
            self._send_json({"status": "ok", "engine": "airllm", "message": "OpenAI-compatible API is running. Use /v1/chat/completions for chat and /v1/models to list models."})
            return

        # Default: serve static files
        super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        # ==============================================
        # OpenAI-Compatible API: /v1/chat/completions
        # This is the KEY endpoint that makes Cline,
        # Continue, Cursor, OpenInterpreter, etc. work.
        # ==============================================
        if path == '/v1/chat/completions':
            body = self._read_body()
            messages = body.get('messages', [])
            stream = body.get('stream', False)
            req_model = body.get('model', '')
            req_temperature = body.get('temperature')
            req_max_tokens = body.get('max_tokens')

            # If a model is requested and it's different, try loading it
            if req_model and req_model != engine.model_path and req_model != 'no-model-loaded':
                print(f"[OpenAI API] Switching model to: {req_model}")
                engine.set_params(model_path=req_model)

            # Override temperature/max_tokens if provided
            if req_temperature is not None:
                engine.temperature = float(req_temperature)
            if req_max_tokens is not None:
                engine.max_new_tokens = int(req_max_tokens)

            if stream:
                self._handle_openai_stream(messages)
            else:
                response = engine.generate(messages, stream=False)
                self._send_json(self._build_openai_response(response))
            return

        # API: Load model
        if path == '/api/load':
            body = self._read_body()
            model_path = body.get('model', '')
            if model_path:
                engine.set_params(model_path=model_path)
            else:
                result = engine.load_model()
            self._send_json(engine.load_model() if not model_path else result)
            return

        # API: Chat completion (internal)
        if path == '/api/chat':
            body = self._read_body()
            messages = body.get('messages', [])
            stream = body.get('stream', False)

            if stream:
                self._handle_stream_chat(messages)
            else:
                response = engine.generate(messages, stream=False)
                self._send_json({
                    "choices": [{"message": {"role": "assistant", "content": response}}],
                    "model": engine.model_path,
                })
            return

        # API: Update settings
        if path == '/api/settings':
            body = self._read_body()
            max_tokens = body.get('max_tokens')
            temperature = body.get('temperature')
            result = engine.set_params(
                max_new_tokens=max_tokens,
                temperature=temperature,
            )
            self._send_json(result)
            return

        # API: Stop generation (for streaming)
        if path == '/api/stop':
            self._send_json({"status": "ok", "message": "Generation stop requested."})
            return

        self._send_json({"error": "Not found"}, 404)

    def _build_openai_response(self, content, finish_reason="stop"):
        """Build an OpenAI-compatible chat completion response."""
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": engine.model_path,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content,
                    },
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        }

    def _handle_openai_stream(self, messages):
        """
        Handle OpenAI-compatible streaming chat completion.
        This is what Cline, Continue, Cursor, etc. expect.
        Format: SSE with 'data: {json}\n\n' and final 'data: [DONE]\n\n'
        """
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream; charset=utf-8')
        self.send_header('Cache-Control', 'no-cache')
        self.send_header('Connection', 'keep-alive')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()

        chat_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())
        model_name = engine.model_path

        # Send the initial role chunk (OpenAI convention)
        initial_chunk = {
            "id": chat_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant"},
                    "finish_reason": None,
                }
            ],
        }
        self.wfile.write(f"data: {json.dumps(initial_chunk)}\n\n".encode('utf-8'))
        self.wfile.flush()

        # Stream tokens
        full_response = ""
        for token in engine.generate(messages, stream=True):
            full_response += token
            chunk = {
                "id": chat_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": token},
                        "finish_reason": None,
                    }
                ],
            }
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode('utf-8'))
            self.wfile.flush()

        # Send final chunk with finish_reason
        final_chunk = {
            "id": chat_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }
            ],
        }
        self.wfile.write(f"data: {json.dumps(final_chunk)}\n\n".encode('utf-8'))
        self.wfile.flush()

        # Send [DONE] marker
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def _handle_stream_chat(self, messages):
        """Handle streaming chat with Server-Sent Events (SSE) - internal API format."""
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream; charset=utf-8')
        self.send_header('Cache-Control', 'no-cache')
        self.send_header('Connection', 'keep-alive')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()

        full_response = ""
        for token in engine.generate(messages, stream=True):
            full_response += token
            data = json.dumps({
                "choices": [{"delta": {"content": token}}],
                "model": engine.model_path,
            }, ensure_ascii=False)
            sse_line = f"data: {data}\n\n"
            self.wfile.write(sse_line.encode('utf-8'))
            self.wfile.flush()

        # Send final [DONE] marker
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def do_OPTIONS(self):
        """Handle CORS preflight requests."""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()


def open_browser():
    """Open the default browser after a short delay."""
    import time
    time.sleep(1.5)
    webbrowser.open(f"http://{HOST}:{PORT}")


def main():
    global engine, PORT, HOST

    parser = argparse.ArgumentParser(description="AirLLM Chat - ChatGPT-style interface powered by airllm")
    parser.add_argument('--port', type=int, default=7860, help='Port to run the server on (default: 7860)')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Host to bind to (default: 127.0.0.1)')
    parser.add_argument('--model', type=str, default='', help='Model path or HuggingFace repo ID to load on startup')
    parser.add_argument('--max-tokens', type=int, default=512, help='Max new tokens for generation (default: 512)')
    parser.add_argument('--temperature', type=float, default=0.7, help='Temperature for generation (default: 0.7)')
    parser.add_argument('--no-browser', action='store_true', help='Do not open browser automatically')
    args = parser.parse_args()

    PORT = args.port
    HOST = args.host

    print("=" * 60)
    print("   AirLLM Chat  -  Local AI Chat Interface")
    print("=" * 60)

    # Initialize engine
    engine = AirLLMEngine(
        model_path=args.model,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    print(f"[Config] max_tokens={args.max_tokens}  temperature={args.temperature}")

    # Auto-load model if specified
    if args.model:
        print(f"[Loading] {args.model} ...")
        result = engine.load_model()
        print(f"[Result] {result['message']}")
    else:
        print("[Info] No model specified. Load one from the UI.")

    # Start server
    server = HTTPServer((HOST, PORT), ChatHandler)
    url = f"http://{HOST}:{PORT}"
    print(f"\n[Server] Running at {url}")
    print(f"\n  Web UI:        {url}")
    print(f"  OpenAI API:    {url}/v1/chat/completions")
    print(f"  Models list:   {url}/v1/models")
    print(f"\n  Compatible with: Cline, Continue, Cursor, OpenInterpreter, etc.")
    print(f"  Just set Base URL to: {url}/v1")
    print(f"\n[Server] Press Ctrl+C to stop.\n")

    if not args.no_browser:
        threading.Thread(target=open_browser, daemon=True).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[Server] Shutting down...")
        server.server_close()


if __name__ == '__main__':
    main()
