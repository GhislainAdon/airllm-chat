"""
Comprehensive unit tests for app.py.

Covers:
- _read_version() function
- get_static_dir() / get_template_content()
- ChatHandler._send_json() / _send_html() / _read_body() / _bad_request()
- ChatHandler.do_GET() - all routes
- ChatHandler.do_POST() - all routes
- ChatHandler.do_OPTIONS()
- Path traversal protection
- Model switching via /v1/chat/completions
- Temperature/max_tokens clamping
- SSE streaming format
- ThreadedHTTPServer
- open_browser()
- main() argument parsing
"""

from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import threading
import unittest
from unittest.mock import MagicMock, patch, PropertyMock, call
from http.server import SimpleHTTPRequestHandler

# We need to import app, but it imports AirLLMEngine from airllm_engine.
import app
from app import (
    ChatHandler,
    ThreadedHTTPServer,
    _read_version,
    get_static_dir,
    get_template_content,
    MAX_BODY_SIZE,
    HOST,
    PORT,
)


# ---------------------------------------------------------------------------
#  Helper: create a mock HTTP handler for testing
# ---------------------------------------------------------------------------

def make_mock_handler(path="/", method="GET", body=None, headers=None):
    """Create a ChatHandler instance with mocked request/response for testing.

    We bypass __init__ entirely and set up only the attributes our tests need.
    We override send_response/send_header/end_headers to capture output
    without touching real sockets.
    """
    # Build body bytes
    body_bytes = b""
    if body is not None:
        body_bytes = json.dumps(body).encode("utf-8")

    # Create handler bypassing __init__ to avoid socket I/O
    handler = ChatHandler.__new__(ChatHandler)

    # --- Core attributes ---
    handler.command = method
    handler.path = path
    handler.request_version = "HTTP/1.1"
    handler.requestline = f"{method} {path} HTTP/1.1"

    # rfile for reading request body
    handler.rfile = io.BytesIO(body_bytes)

    # wfile for writing response body
    handler.wfile = io.BytesIO()

    # Mock headers object
    handler.headers = MagicMock()
    if body_bytes:
        handler.headers.get.return_value = str(len(body_bytes))
    else:
        handler.headers.get.return_value = "0"

    # --- Response capture ---
    handler._response_status = []
    handler._response_headers = []
    handler._headers_sent = False

    # Override methods to capture instead of writing to socket
    def mock_send_response(code, message=None):
        handler._response_status.append(code)

    def mock_send_header(keyword, value):
        handler._response_headers.append((keyword, value))

    def mock_end_headers():
        handler._headers_sent = True

    def mock_log_request(code='-', size='-'):
        pass  # Suppress logging

    handler.send_response = mock_send_response
    handler.send_header = mock_send_header
    handler.end_headers = mock_end_headers
    handler.log_request = mock_log_request
    handler.log_message = lambda format, *args: None

    # Other attributes that may be needed
    handler.client_address = ("127.0.0.1", 12345)
    handler.server = MagicMock()

    return handler


def get_response_body(handler):
    """Extract the response body written to the mock wfile."""
    handler.wfile.seek(0)
    return handler.wfile.read()


def get_response_json(handler):
    """Extract and parse JSON from the response body."""
    body = get_response_body(handler)
    return json.loads(body.decode("utf-8"))


# ---------------------------------------------------------------------------
#  _read_version() tests (app module copy)
# ---------------------------------------------------------------------------

class TestAppReadVersion(unittest.TestCase):
    """Tests for app module's _read_version() function."""

    def test_reads_version_file(self):
        """Normal case: VERSION file exists and is read."""
        with patch("builtins.open", unittest.mock.mock_open(read_data="1.2.2\n")):
            with patch("os.path.join", return_value="/fake/VERSION"):
                result = _read_version()
        self.assertEqual(result, "1.2.2")

    def test_strips_whitespace(self):
        """Version with whitespace should be stripped."""
        with patch("builtins.open", unittest.mock.mock_open(read_data="  3.0.0  \n")):
            with patch("os.path.join", return_value="/fake/VERSION"):
                result = _read_version()
        self.assertEqual(result, "3.0.0")

    def test_missing_file_returns_dev(self):
        """Missing VERSION file should return dev version."""
        with patch("builtins.open", side_effect=FileNotFoundError):
            with patch("os.path.join", return_value="/fake/VERSION"):
                result = _read_version()
        self.assertEqual(result, "0.0.0-dev")

    def test_frozen_mode(self):
        """In frozen mode, should use sys._MEIPASS."""
        with patch.object(sys, "frozen", True, create=True):
            with patch.object(sys, "_MEIPASS", "/frozen/dir", create=True):
                with patch("builtins.open", unittest.mock.mock_open(read_data="1.0.0")):
                    result = _read_version()
        self.assertIsInstance(result, str)


# ---------------------------------------------------------------------------
#  get_static_dir / get_template_content tests
# ---------------------------------------------------------------------------

class TestStaticAndTemplate(unittest.TestCase):
    """Tests for get_static_dir() and get_template_content()."""

    def test_get_static_dir_development(self):
        """get_static_dir in development mode should return static/ dir."""
        with patch.object(sys, "frozen", False, create=True):
            result = get_static_dir()
        self.assertTrue(result.endswith("static"))

    def test_get_static_dir_frozen(self):
        """get_static_dir in frozen mode should use _MEIPASS."""
        with patch.object(sys, "frozen", True, create=True):
            with patch.object(sys, "_MEIPASS", "/frozen/app", create=True):
                result = get_static_dir()
        self.assertEqual(result, "/frozen/app/static")

    def test_get_template_content(self):
        """get_template_content should read a template file."""
        with patch.object(sys, "frozen", False, create=True):
            with patch("builtins.open", unittest.mock.mock_open(read_data="<html>test</html>")):
                content = get_template_content("index.html")
        self.assertEqual(content, "<html>test</html>")

    def test_get_template_content_frozen(self):
        """get_template_content in frozen mode should use _MEIPASS."""
        with patch.object(sys, "frozen", True, create=True):
            with patch.object(sys, "_MEIPASS", "/frozen/app", create=True):
                with patch("builtins.open", unittest.mock.mock_open(read_data="<html>frozen</html>")):
                    content = get_template_content("index.html")
        self.assertEqual(content, "<html>frozen</html>")


# ---------------------------------------------------------------------------
#  ChatHandler._send_json tests
# ---------------------------------------------------------------------------

class TestSendJson(unittest.TestCase):
    """Tests for ChatHandler._send_json()."""

    def test_sends_json_response(self):
        """_send_json should write JSON body with correct headers."""
        handler = make_mock_handler()
        handler._send_json({"status": "ok"})
        data = get_response_json(handler)
        self.assertEqual(data["status"], "ok")

    def test_sends_json_with_status(self):
        """_send_json should use the specified status code."""
        handler = make_mock_handler()
        handler._send_json({"error": "bad"}, status=400)
        self.assertIn(400, handler._response_status)

    def test_json_content_type(self):
        """_send_json should set Content-Type to application/json."""
        handler = make_mock_handler()
        handler._send_json({"test": True})
        header_dict = dict(handler._response_headers)
        self.assertIn("application/json", header_dict.get("Content-Type", ""))

    def test_cors_header(self):
        """_send_json should set CORS header."""
        handler = make_mock_handler()
        handler._send_json({})
        header_dict = dict(handler._response_headers)
        self.assertEqual(header_dict.get("Access-Control-Allow-Origin"), "*")

    def test_content_length_header(self):
        """_send_json should set Content-Length header."""
        handler = make_mock_handler()
        handler._send_json({"key": "value"})
        header_dict = dict(handler._response_headers)
        self.assertIn("Content-Length", header_dict)


# ---------------------------------------------------------------------------
#  ChatHandler._send_html tests
# ---------------------------------------------------------------------------

class TestSendHtml(unittest.TestCase):
    """Tests for ChatHandler._send_html()."""

    def test_sends_html_response(self):
        """_send_html should write HTML body."""
        handler = make_mock_handler()
        handler._send_html("<html>Hello</html>")
        body = get_response_body(handler)
        self.assertIn(b"Hello", body)

    def test_html_content_type(self):
        """_send_html should set Content-Type to text/html."""
        handler = make_mock_handler()
        handler._send_html("<html></html>")
        header_dict = dict(handler._response_headers)
        self.assertIn("text/html", header_dict.get("Content-Type", ""))

    def test_html_with_status(self):
        """_send_html should use the specified status code."""
        handler = make_mock_handler()
        handler._send_html("<html></html>", status=500)
        self.assertIn(500, handler._response_status)


# ---------------------------------------------------------------------------
#  ChatHandler._read_body tests
# ---------------------------------------------------------------------------

class TestReadBody(unittest.TestCase):
    """Tests for ChatHandler._read_body()."""

    def test_read_valid_json(self):
        """_read_body should parse valid JSON body."""
        handler = make_mock_handler(body={"key": "value"})
        result = handler._read_body()
        self.assertEqual(result, {"key": "value"})

    def test_read_empty_body(self):
        """_read_body with no body should return empty dict."""
        handler = make_mock_handler()
        handler.headers.get.return_value = "0"
        result = handler._read_body()
        self.assertEqual(result, {})

    def test_read_oversized_body(self):
        """_read_body should return None for oversized body."""
        handler = make_mock_handler()
        handler.headers.get.return_value = str(MAX_BODY_SIZE + 1)
        result = handler._read_body()
        self.assertIsNone(result)

    def test_read_invalid_json(self):
        """_read_body should return None for invalid JSON."""
        handler = make_mock_handler()
        handler.headers.get.return_value = "5"
        handler.rfile = io.BytesIO(b"xxxxx")
        result = handler._read_body()
        self.assertIsNone(result)

    def test_read_zero_content_length(self):
        """_read_body with Content-Length 0 should return empty dict."""
        handler = make_mock_handler()
        handler.headers.get.return_value = "0"
        result = handler._read_body()
        self.assertEqual(result, {})


# ---------------------------------------------------------------------------
#  ChatHandler._bad_request tests
# ---------------------------------------------------------------------------

class TestBadRequest(unittest.TestCase):
    """Tests for ChatHandler._bad_request()."""

    def test_default_message(self):
        """_bad_request with default message should return 400."""
        handler = make_mock_handler()
        handler._bad_request()
        self.assertIn(400, handler._response_status)

    def test_custom_message(self):
        """_bad_request with custom message should include it in response."""
        handler = make_mock_handler()
        handler._bad_request("Custom error")
        data = get_response_json(handler)
        self.assertEqual(data["error"]["message"], "Custom error")

    def test_error_type(self):
        """_bad_request should set error type to invalid_request_error."""
        handler = make_mock_handler()
        handler._bad_request()
        data = get_response_json(handler)
        self.assertEqual(data["error"]["type"], "invalid_request_error")


# ---------------------------------------------------------------------------
#  ChatHandler.do_GET tests
# ---------------------------------------------------------------------------

class TestDoGet(unittest.TestCase):
    """Tests for ChatHandler.do_GET() routes."""

    def setUp(self):
        """Set up a fresh engine mock for each test."""
        self.engine_patcher = patch("app.engine")
        self.mock_engine = self.engine_patcher.start()
        self.mock_engine.info = {
            "model_path": "test/model",
            "loaded": False,
            "max_new_tokens": 512,
            "temperature": 0.7,
            "error": None,
            "version": "1.2.2",
        }
        self.mock_engine.model_path = "test/model"
        self.mock_engine.is_loaded = False
        self.mock_engine.get_ollama_info.return_value = {
            "ollama_installed": False,
            "ollama_running": False,
        }
        self.mock_engine.get_ollama_models.return_value = {"models": []}
        self.mock_engine.get_system_info.return_value = {"gpu_name": "", "cpu_count": 8}

    def tearDown(self):
        self.engine_patcher.stop()

    def test_root_returns_html(self):
        """GET / should return HTML content."""
        handler = make_mock_handler(path="/")
        with patch("app.get_template_content", return_value="<html>Index</html>"):
            handler.do_GET()
        body = get_response_body(handler)
        self.assertIn(b"Index", body)

    def test_api_info(self):
        """GET /api/info should return engine info as JSON."""
        handler = make_mock_handler(path="/api/info")
        handler.do_GET()
        data = get_response_json(handler)
        self.assertIn("model_path", data)

    def test_api_models(self):
        """GET /api/models should return model list."""
        handler = make_mock_handler(path="/api/models")
        handler.do_GET()
        data = get_response_json(handler)
        self.assertIn("models", data)
        self.assertTrue(len(data["models"]) > 0)

    def test_v1_models(self):
        """GET /v1/models should return OpenAI-compatible model list."""
        handler = make_mock_handler(path="/v1/models")
        handler.do_GET()
        data = get_response_json(handler)
        self.assertEqual(data["object"], "list")
        self.assertIn("data", data)
        self.assertTrue(len(data["data"]) > 0)
        self.assertEqual(data["data"][0]["object"], "model")

    def test_v1_models_with_trailing_slash(self):
        """GET /v1/models/ should also work."""
        handler = make_mock_handler(path="/v1/models/")
        handler.do_GET()
        data = get_response_json(handler)
        self.assertEqual(data["object"], "list")

    def test_v1_root(self):
        """GET /v1/ should return API info."""
        handler = make_mock_handler(path="/v1/")
        handler.do_GET()
        data = get_response_json(handler)
        self.assertEqual(data["status"], "ok")
        self.assertIn("airllm", data["engine"])

    def test_v1_root_no_slash(self):
        """GET /v1 should also return API info."""
        handler = make_mock_handler(path="/v1")
        handler.do_GET()
        data = get_response_json(handler)
        self.assertEqual(data["status"], "ok")

    def test_api_ollama_status(self):
        """GET /api/ollama/status should return ollama info."""
        handler = make_mock_handler(path="/api/ollama/status")
        handler.do_GET()
        self.mock_engine.get_ollama_info.assert_called_once()

    def test_api_ollama_models(self):
        """GET /api/ollama/models should return ollama model list."""
        handler = make_mock_handler(path="/api/ollama/models")
        handler.do_GET()
        self.mock_engine.get_ollama_models.assert_called_once()

    def test_api_system_info(self):
        """GET /api/system/info should return system info."""
        handler = make_mock_handler(path="/api/system/info")
        handler.do_GET()
        self.mock_engine.get_system_info.assert_called_once()

    def test_v1_models_loaded_model(self):
        """GET /v1/models should use loaded model as first entry."""
        self.mock_engine.is_loaded = True
        self.mock_engine.model_path = "my-loaded-model"
        handler = make_mock_handler(path="/v1/models")
        handler.do_GET()
        data = get_response_json(handler)
        self.assertEqual(data["data"][0]["id"], "my-loaded-model")

    def test_v1_models_no_model_loaded(self):
        """GET /v1/models with no model should use 'no-model-loaded'."""
        self.mock_engine.is_loaded = False
        handler = make_mock_handler(path="/v1/models")
        handler.do_GET()
        data = get_response_json(handler)
        self.assertEqual(data["data"][0]["id"], "no-model-loaded")

    def test_unknown_get_falls_through(self):
        """GET /unknown/path should fall through to super().do_GET()."""
        handler = make_mock_handler(path="/unknown/path")
        # super().do_GET() will be called, which may raise an error.
        # We just ensure it doesn't crash our handler setup.
        with patch.object(SimpleHTTPRequestHandler, "do_GET", return_value=None):
            handler.do_GET()


# ---------------------------------------------------------------------------
#  ChatHandler.do_POST tests
# ---------------------------------------------------------------------------

class TestDoPost(unittest.TestCase):
    """Tests for ChatHandler.do_POST() routes."""

    def setUp(self):
        self.engine_patcher = patch("app.engine")
        self.mock_engine = self.engine_patcher.start()
        self.mock_engine.model_path = "test/model"
        self.mock_engine.is_loaded = True
        self.mock_engine.max_new_tokens = 512
        self.mock_engine.temperature = 0.7
        self.mock_engine._cancel_requested = False
        self.mock_engine.generate.return_value = "Hello!"
        self.mock_engine.load_model.return_value = {"status": "ok", "message": "Model loaded"}
        self.mock_engine.set_params.return_value = {"status": "ok", "message": "Parameters updated"}
        self.mock_engine.unload_model.return_value = {"status": "ok", "message": "Model unloaded"}
        self.mock_engine.cancel_generation.return_value = None
        self.mock_engine.delete_ollama_model.return_value = {"status": "success", "message": "Deleted"}
        self.mock_engine.get_ollama_model_path.return_value = {
            "status": "success", "model": "llama2", "digest": "abc", "path": "/path/to/blob", "size_bytes": 4096
        }

    def tearDown(self):
        self.engine_patcher.stop()

    def test_chat_completions_non_stream(self):
        """POST /v1/chat/completions non-streaming should return OpenAI response."""
        handler = make_mock_handler(path="/v1/chat/completions", method="POST",
                                     body={"messages": [{"role": "user", "content": "Hi"}]})
        handler.do_POST()
        data = get_response_json(handler)
        self.assertIn("id", data)
        self.assertEqual(data["object"], "chat.completion")
        self.assertIn("choices", data)

    def test_chat_completions_invalid_body(self):
        """POST /v1/chat/completions with invalid body should return 400."""
        handler = make_mock_handler(path="/v1/chat/completions", method="POST")
        handler.headers.get.return_value = str(MAX_BODY_SIZE + 1)
        handler.do_POST()
        self.assertIn(400, handler._response_status)

    def test_chat_completions_path_traversal(self):
        """POST /v1/chat/completions with '..' in model should be blocked."""
        handler = make_mock_handler(path="/v1/chat/completions", method="POST",
                                     body={"messages": [{"role": "user", "content": "Hi"}],
                                           "model": "../../../etc/passwd"})
        handler.do_POST()
        self.assertIn(400, handler._response_status)

    def test_chat_completions_model_switch(self):
        """POST /v1/chat/completions with different model should trigger switch."""
        handler = make_mock_handler(path="/v1/chat/completions", method="POST",
                                     body={"messages": [{"role": "user", "content": "Hi"}],
                                           "model": "new/model"})
        handler.do_POST()
        self.mock_engine.set_params.assert_called()

    def test_chat_completions_no_model_switch_same(self):
        """POST /v1/chat/completions with same model should not switch."""
        handler = make_mock_handler(path="/v1/chat/completions", method="POST",
                                     body={"messages": [{"role": "user", "content": "Hi"}],
                                           "model": "test/model"})
        handler.do_POST()
        # set_params should NOT be called for model_path since it's the same
        self.mock_engine.set_params.assert_not_called()

    def test_chat_completions_temperature_clamping(self):
        """POST /v1/chat/completions should clamp temperature to [0, 2]."""
        handler = make_mock_handler(path="/v1/chat/completions", method="POST",
                                     body={"messages": [{"role": "user", "content": "Hi"}],
                                           "temperature": 5.0})
        handler.do_POST()
        # Temperature should be clamped to 2.0
        self.assertLessEqual(self.mock_engine.temperature, 2.0)

    def test_chat_completions_temperature_negative(self):
        """POST /v1/chat/completions should clamp negative temperature to 0."""
        handler = make_mock_handler(path="/v1/chat/completions", method="POST",
                                     body={"messages": [{"role": "user", "content": "Hi"}],
                                           "temperature": -1.0})
        handler.do_POST()
        self.assertGreaterEqual(self.mock_engine.temperature, 0.0)

    def test_chat_completions_max_tokens_clamping(self):
        """POST /v1/chat/completions should clamp max_tokens to [1, 32768]."""
        handler = make_mock_handler(path="/v1/chat/completions", method="POST",
                                     body={"messages": [{"role": "user", "content": "Hi"}],
                                           "max_tokens": 100000})
        handler.do_POST()
        self.assertLessEqual(self.mock_engine.max_new_tokens, 32768)

    def test_chat_completions_max_tokens_minimum(self):
        """POST /v1/chat/completions should clamp max_tokens to at least 1."""
        handler = make_mock_handler(path="/v1/chat/completions", method="POST",
                                     body={"messages": [{"role": "user", "content": "Hi"}],
                                           "max_tokens": 0})
        handler.do_POST()
        self.assertGreaterEqual(self.mock_engine.max_new_tokens, 1)

    def test_chat_completions_streaming(self):
        """POST /v1/chat/completions with stream=True should return SSE."""
        self.mock_engine.generate.return_value = iter(["Hello", " world"])
        handler = make_mock_handler(path="/v1/chat/completions", method="POST",
                                     body={"messages": [{"role": "user", "content": "Hi"}],
                                           "stream": True})
        handler.do_POST()
        body = get_response_body(handler)
        text = body.decode("utf-8")
        # Should have SSE content type in headers
        header_dict = dict(handler._response_headers)
        self.assertIn("text/event-stream", header_dict.get("Content-Type", ""))
        self.assertIn("data:", text)

    def test_chat_completions_no_model_switch_no_model_loaded(self):
        """Model value 'no-model-loaded' should not trigger model switch."""
        handler = make_mock_handler(path="/v1/chat/completions", method="POST",
                                     body={"messages": [{"role": "user", "content": "Hi"}],
                                           "model": "no-model-loaded"})
        handler.do_POST()
        self.mock_engine.set_params.assert_not_called()

    def test_api_load_with_model(self):
        """POST /api/load with model should call set_params."""
        handler = make_mock_handler(path="/api/load", method="POST",
                                     body={"model": "new/model"})
        handler.do_POST()
        self.mock_engine.set_params.assert_called_once_with(model_path="new/model")

    def test_api_load_without_model(self):
        """POST /api/load without model should call load_model."""
        handler = make_mock_handler(path="/api/load", method="POST",
                                     body={})
        handler.do_POST()
        self.mock_engine.load_model.assert_called_once()

    def test_api_load_path_traversal(self):
        """POST /api/load with '..' in model should be blocked."""
        handler = make_mock_handler(path="/api/load", method="POST",
                                     body={"model": "../etc/passwd"})
        handler.do_POST()
        self.assertIn(400, handler._response_status)

    def test_api_load_invalid_body(self):
        """POST /api/load with invalid body should return 400."""
        handler = make_mock_handler(path="/api/load", method="POST")
        handler.headers.get.return_value = str(MAX_BODY_SIZE + 1)
        handler.do_POST()
        self.assertIn(400, handler._response_status)

    def test_api_chat_non_stream(self):
        """POST /api/chat non-streaming should return response."""
        handler = make_mock_handler(path="/api/chat", method="POST",
                                     body={"messages": [{"role": "user", "content": "Hi"}]})
        handler.do_POST()
        data = get_response_json(handler)
        self.assertIn("choices", data)

    def test_api_chat_streaming(self):
        """POST /api/chat with stream=True should return SSE."""
        self.mock_engine.generate.return_value = iter(["Hello"])
        handler = make_mock_handler(path="/api/chat", method="POST",
                                     body={"messages": [{"role": "user", "content": "Hi"}],
                                           "stream": True})
        handler.do_POST()
        body = get_response_body(handler)
        text = body.decode("utf-8")
        self.assertIn("data:", text)

    def test_api_chat_invalid_body(self):
        """POST /api/chat with invalid body should return 400."""
        handler = make_mock_handler(path="/api/chat", method="POST")
        handler.headers.get.return_value = str(MAX_BODY_SIZE + 1)
        handler.do_POST()
        self.assertIn(400, handler._response_status)

    def test_api_settings(self):
        """POST /api/settings should update engine params."""
        handler = make_mock_handler(path="/api/settings", method="POST",
                                     body={"max_tokens": 1024, "temperature": 0.5})
        handler.do_POST()
        self.mock_engine.set_params.assert_called_once_with(
            max_new_tokens=1024, temperature=0.5
        )

    def test_api_settings_invalid_body(self):
        """POST /api/settings with invalid body should return 400."""
        handler = make_mock_handler(path="/api/settings", method="POST")
        handler.headers.get.return_value = str(MAX_BODY_SIZE + 1)
        handler.do_POST()
        self.assertIn(400, handler._response_status)

    def test_api_stop(self):
        """POST /api/stop should cancel generation."""
        handler = make_mock_handler(path="/api/stop", method="POST", body={})
        handler.do_POST()
        self.mock_engine.cancel_generation.assert_called_once()

    def test_api_unload(self):
        """POST /api/unload should unload model."""
        handler = make_mock_handler(path="/api/unload", method="POST", body={})
        handler.do_POST()
        self.mock_engine.unload_model.assert_called_once()

    def test_api_ollama_pull(self):
        """POST /api/ollama/pull should stream progress."""
        self.mock_engine.pull_ollama_model.return_value = iter([
            {"status": "success", "message": "Pulled."}
        ])
        handler = make_mock_handler(path="/api/ollama/pull", method="POST",
                                     body={"model": "llama2"})
        handler.do_POST()
        body = get_response_body(handler)
        text = body.decode("utf-8")
        self.assertIn("data:", text)

    def test_api_ollama_pull_missing_model(self):
        """POST /api/ollama/pull without model should return 400."""
        handler = make_mock_handler(path="/api/ollama/pull", method="POST",
                                     body={})
        handler.do_POST()
        self.assertIn(400, handler._response_status)

    def test_api_ollama_pull_invalid_body(self):
        """POST /api/ollama/pull with invalid body should return 400."""
        handler = make_mock_handler(path="/api/ollama/pull", method="POST")
        handler.headers.get.return_value = str(MAX_BODY_SIZE + 1)
        handler.do_POST()
        self.assertIn(400, handler._response_status)

    def test_api_ollama_delete(self):
        """POST /api/ollama/delete should delete model."""
        handler = make_mock_handler(path="/api/ollama/delete", method="POST",
                                     body={"model": "llama2"})
        handler.do_POST()
        self.mock_engine.delete_ollama_model.assert_called_once_with("llama2")

    def test_api_ollama_delete_missing_model(self):
        """POST /api/ollama/delete without model should return 400."""
        handler = make_mock_handler(path="/api/ollama/delete", method="POST",
                                     body={})
        handler.do_POST()
        self.assertIn(400, handler._response_status)

    def test_api_ollama_load_success(self):
        """POST /api/ollama/load should resolve path and load model."""
        handler = make_mock_handler(path="/api/ollama/load", method="POST",
                                     body={"model": "llama2"})
        handler.do_POST()
        self.mock_engine.get_ollama_model_path.assert_called_once_with("llama2")
        self.mock_engine.set_params.assert_called_once_with(model_path="/path/to/blob")

    def test_api_ollama_load_failure(self):
        """POST /api/ollama/load with failed path resolution should return 400."""
        self.mock_engine.get_ollama_model_path.return_value = {
            "status": "error", "message": "not found"
        }
        handler = make_mock_handler(path="/api/ollama/load", method="POST",
                                     body={"model": "nonexistent"})
        handler.do_POST()
        self.assertIn(400, handler._response_status)

    def test_api_ollama_load_missing_model(self):
        """POST /api/ollama/load without model should return 400."""
        handler = make_mock_handler(path="/api/ollama/load", method="POST",
                                     body={})
        handler.do_POST()
        self.assertIn(400, handler._response_status)

    def test_api_ollama_path_valid(self):
        """POST /api/ollama/path with valid directory should update path."""
        with patch("os.path.isdir", return_value=True):
            with patch("os.path.expanduser", return_value="/home/user/.ollama/models"):
                handler = make_mock_handler(path="/api/ollama/path", method="POST",
                                             body={"path": "~/.ollama/models"})
                handler.do_POST()
        self.assertEqual(self.mock_engine._ollama_models_dir, "/home/user/.ollama/models")

    def test_api_ollama_path_invalid_dir(self):
        """POST /api/ollama/path with non-existent directory should return 400."""
        with patch("os.path.isdir", return_value=False):
            with patch("os.path.expanduser", return_value="/nonexistent"):
                handler = make_mock_handler(path="/api/ollama/path", method="POST",
                                             body={"path": "/nonexistent"})
                handler.do_POST()
        self.assertIn(400, handler._response_status)

    def test_api_ollama_path_traversal(self):
        """POST /api/ollama/path with '..' should be blocked."""
        handler = make_mock_handler(path="/api/ollama/path", method="POST",
                                     body={"path": "../../etc"})
        handler.do_POST()
        self.assertIn(400, handler._response_status)

    def test_api_ollama_path_missing_path(self):
        """POST /api/ollama/path without path should return 400."""
        handler = make_mock_handler(path="/api/ollama/path", method="POST",
                                     body={})
        handler.do_POST()
        self.assertIn(400, handler._response_status)

    def test_api_ollama_path_invalid_body(self):
        """POST /api/ollama/path with invalid body should return 400."""
        handler = make_mock_handler(path="/api/ollama/path", method="POST")
        handler.headers.get.return_value = str(MAX_BODY_SIZE + 1)
        handler.do_POST()
        self.assertIn(400, handler._response_status)

    def test_unknown_post_404(self):
        """POST /unknown/path should return 404."""
        handler = make_mock_handler(path="/unknown/path", method="POST", body={})
        handler.do_POST()
        self.assertIn(404, handler._response_status)


# ---------------------------------------------------------------------------
#  ChatHandler.do_OPTIONS tests
# ---------------------------------------------------------------------------

class TestDoOptions(unittest.TestCase):
    """Tests for ChatHandler.do_OPTIONS()."""

    def test_options_returns_200(self):
        """OPTIONS should return 200."""
        handler = make_mock_handler(method="OPTIONS")
        handler.do_OPTIONS()
        self.assertIn(200, handler._response_status)

    def test_options_cors_headers(self):
        """OPTIONS should set CORS headers."""
        handler = make_mock_handler(method="OPTIONS")
        handler.do_OPTIONS()
        header_dict = dict(handler._response_headers)
        self.assertEqual(header_dict.get("Access-Control-Allow-Origin"), "*")
        self.assertIn("GET", header_dict.get("Access-Control-Allow-Methods", ""))
        self.assertIn("POST", header_dict.get("Access-Control-Allow-Methods", ""))

    def test_options_allow_headers(self):
        """OPTIONS should set Access-Control-Allow-Headers."""
        handler = make_mock_handler(method="OPTIONS")
        handler.do_OPTIONS()
        header_dict = dict(handler._response_headers)
        self.assertIn("Content-Type", header_dict.get("Access-Control-Allow-Headers", ""))


# ---------------------------------------------------------------------------
#  _build_openai_response tests
# ---------------------------------------------------------------------------

class TestBuildOpenAIResponse(unittest.TestCase):
    """Tests for ChatHandler._build_openai_response()."""

    def test_response_structure(self):
        """_build_openai_response should have OpenAI-compatible structure."""
        handler = make_mock_handler()
        with patch("app.engine", MagicMock(model_path="test/model")):
            response = handler._build_openai_response("Hello world")
        self.assertIn("id", response)
        self.assertTrue(response["id"].startswith("chatcmpl-"))
        self.assertEqual(response["object"], "chat.completion")
        self.assertIn("created", response)
        self.assertIn("model", response)
        self.assertEqual(len(response["choices"]), 1)
        self.assertEqual(response["choices"][0]["message"]["content"], "Hello world")
        self.assertEqual(response["choices"][0]["finish_reason"], "stop")
        self.assertIn("usage", response)

    def test_response_custom_finish_reason(self):
        """_build_openai_response should support custom finish_reason."""
        handler = make_mock_handler()
        with patch("app.engine", MagicMock(model_path="test/model")):
            response = handler._build_openai_response("test", finish_reason="length")
        self.assertEqual(response["choices"][0]["finish_reason"], "length")


# ---------------------------------------------------------------------------
#  SSE streaming format tests
# ---------------------------------------------------------------------------

class TestSSEStreaming(unittest.TestCase):
    """Tests for SSE streaming format in both OpenAI and internal APIs."""

    def setUp(self):
        self.engine_patcher = patch("app.engine")
        self.mock_engine = self.engine_patcher.start()
        self.mock_engine.model_path = "test/model"
        self.mock_engine._cancel_requested = False
        self.mock_engine.generate.return_value = iter(["Hello", " world"])

    def tearDown(self):
        self.engine_patcher.stop()

    def test_openai_stream_format(self):
        """OpenAI SSE stream should have proper format with chunks."""
        handler = make_mock_handler(path="/v1/chat/completions", method="POST",
                                     body={"messages": [{"role": "user", "content": "Hi"}],
                                           "stream": True})
        handler.do_POST()
        body = get_response_body(handler)
        text = body.decode("utf-8")
        self.assertIn("data: ", text)
        self.assertIn("[DONE]", text)

    def test_internal_stream_format(self):
        """Internal SSE stream should have proper format."""
        handler = make_mock_handler(path="/api/chat", method="POST",
                                     body={"messages": [{"role": "user", "content": "Hi"}],
                                           "stream": True})
        handler.do_POST()
        body = get_response_body(handler)
        text = body.decode("utf-8")
        self.assertIn("data: ", text)
        self.assertIn("[DONE]", text)

    def test_ollama_pull_stream_format(self):
        """Ollama pull SSE stream should have proper format."""
        self.mock_engine.pull_ollama_model.return_value = iter([
            {"status": "pulling", "digest": "abc123"},
            {"status": "success", "message": "Done"},
        ])
        handler = make_mock_handler(path="/api/ollama/pull", method="POST",
                                     body={"model": "llama2"})
        handler.do_POST()
        body = get_response_body(handler)
        text = body.decode("utf-8")
        self.assertIn("data: ", text)
        self.assertIn("[DONE]", text)


# ---------------------------------------------------------------------------
#  ThreadedHTTPServer tests
# ---------------------------------------------------------------------------

class TestThreadedHTTPServer(unittest.TestCase):
    """Tests for ThreadedHTTPServer class."""

    def test_is_threaded(self):
        """ThreadedHTTPServer should be a subclass of ThreadingMixIn."""
        from socketserver import ThreadingMixIn
        from http.server import HTTPServer
        self.assertTrue(issubclass(ThreadedHTTPServer, ThreadingMixIn))
        self.assertTrue(issubclass(ThreadedHTTPServer, HTTPServer))

    def test_daemon_threads(self):
        """ThreadedHTTPServer should have daemon_threads=True."""
        self.assertTrue(ThreadedHTTPServer.daemon_threads)

    def test_allow_reuse_address(self):
        """ThreadedHTTPServer should have allow_reuse_address=True."""
        self.assertTrue(ThreadedHTTPServer.allow_reuse_address)


# ---------------------------------------------------------------------------
#  open_browser tests
# ---------------------------------------------------------------------------

class TestOpenBrowser(unittest.TestCase):
    """Tests for open_browser()."""

    @patch("webbrowser.open")
    @patch("time.sleep")
    def test_opens_browser(self, mock_sleep, mock_open):
        """open_browser should open the correct URL."""
        with patch("app.HOST", "127.0.0.1"), patch("app.PORT", 7860):
            app.open_browser()
        mock_sleep.assert_called_once_with(1.5)
        mock_open.assert_called_once_with("http://127.0.0.1:7860")


# ---------------------------------------------------------------------------
#  main() tests
# ---------------------------------------------------------------------------

class TestMain(unittest.TestCase):
    """Tests for main() function."""

    @patch("app.ThreadedHTTPServer")
    @patch("app.AirLLMEngine")
    @patch("app.argparse.ArgumentParser.parse_args")
    def test_default_args(self, mock_parse_args, mock_engine_cls, mock_server):
        """main() with default args should create engine with defaults."""
        mock_parse_args.return_value = MagicMock(
            port=7860, host="127.0.0.1", model="", max_tokens=512,
            temperature=0.7, no_browser=True
        )
        mock_engine_instance = MagicMock()
        mock_engine_cls.return_value = mock_engine_instance
        mock_server_instance = MagicMock()
        mock_server.return_value = mock_server_instance
        mock_server_instance.serve_forever.side_effect = KeyboardInterrupt()

        with patch("app.engine", None):
            app.main()

        mock_engine_cls.assert_called_once_with(
            model_path="", max_new_tokens=512, temperature=0.7
        )

    @patch("app.ThreadedHTTPServer")
    @patch("app.AirLLMEngine")
    @patch("app.argparse.ArgumentParser.parse_args")
    def test_custom_args(self, mock_parse_args, mock_engine_cls, mock_server):
        """main() with custom args should pass them to engine."""
        mock_parse_args.return_value = MagicMock(
            port=9000, host="0.0.0.0", model="my-model", max_tokens=2048,
            temperature=0.3, no_browser=True
        )
        mock_engine_instance = MagicMock()
        mock_engine_cls.return_value = mock_engine_instance
        mock_server_instance = MagicMock()
        mock_server.return_value = mock_server_instance
        mock_server_instance.serve_forever.side_effect = KeyboardInterrupt()

        with patch("app.engine", None):
            app.main()

        mock_engine_cls.assert_called_once_with(
            model_path="my-model", max_new_tokens=2048, temperature=0.3
        )

    @patch("app.ThreadedHTTPServer")
    @patch("app.AirLLMEngine")
    @patch("app.argparse.ArgumentParser.parse_args")
    def test_loads_model_when_specified(self, mock_parse_args, mock_engine_cls, mock_server):
        """main() should load model when --model is specified."""
        mock_parse_args.return_value = MagicMock(
            port=7860, host="127.0.0.1", model="my-model", max_tokens=512,
            temperature=0.7, no_browser=True
        )
        mock_engine_instance = MagicMock()
        mock_engine_instance.load_model.return_value = {"status": "ok", "message": "loaded"}
        mock_engine_cls.return_value = mock_engine_instance
        mock_server_instance = MagicMock()
        mock_server.return_value = mock_server_instance
        mock_server_instance.serve_forever.side_effect = KeyboardInterrupt()

        with patch("app.engine", None):
            app.main()

        mock_engine_instance.load_model.assert_called_once()

    @patch("app.ThreadedHTTPServer")
    @patch("app.AirLLMEngine")
    @patch("app.argparse.ArgumentParser.parse_args")
    def test_keyboard_interrupt_shutdown(self, mock_parse_args, mock_engine_cls, mock_server):
        """main() should handle KeyboardInterrupt gracefully."""
        mock_parse_args.return_value = MagicMock(
            port=7860, host="127.0.0.1", model="", max_tokens=512,
            temperature=0.7, no_browser=True
        )
        mock_engine_instance = MagicMock()
        mock_engine_cls.return_value = mock_engine_instance
        mock_server_instance = MagicMock()
        mock_server.return_value = mock_server_instance
        mock_server_instance.serve_forever.side_effect = KeyboardInterrupt()

        with patch("app.engine", None):
            app.main()

        mock_engine_instance.unload_model.assert_called_once()
        mock_server_instance.server_close.assert_called_once()

    @patch("app.ThreadedHTTPServer")
    @patch("app.AirLLMEngine")
    @patch("app.argparse.ArgumentParser.parse_args")
    @patch("threading.Thread")
    def test_browser_opened_by_default(self, mock_thread, mock_parse_args, mock_engine_cls, mock_server):
        """main() should open browser when --no-browser is not set."""
        mock_parse_args.return_value = MagicMock(
            port=7860, host="127.0.0.1", model="", max_tokens=512,
            temperature=0.7, no_browser=False
        )
        mock_engine_instance = MagicMock()
        mock_engine_cls.return_value = mock_engine_instance
        mock_server_instance = MagicMock()
        mock_server.return_value = mock_server_instance
        mock_server_instance.serve_forever.side_effect = KeyboardInterrupt()

        with patch("app.engine", None):
            app.main()

        mock_thread.assert_called_once()


# ---------------------------------------------------------------------------
#  Integration-style tests
# ---------------------------------------------------------------------------

class TestIntegration(unittest.TestCase):
    """Integration-style tests for full request flows."""

    def setUp(self):
        self.engine_patcher = patch("app.engine")
        self.mock_engine = self.engine_patcher.start()
        self.mock_engine.model_path = "test/model"
        self.mock_engine.is_loaded = True
        self.mock_engine.max_new_tokens = 512
        self.mock_engine.temperature = 0.7
        self.mock_engine._cancel_requested = False
        self.mock_engine.generate.return_value = "AI response"
        self.mock_engine.load_model.return_value = {"status": "ok", "message": "loaded"}
        self.mock_engine.set_params.return_value = {"status": "ok", "message": "updated"}
        self.mock_engine.unload_model.return_value = {"status": "ok", "message": "unloaded"}
        self.mock_engine.info = {
            "model_path": "test/model", "loaded": True,
            "max_new_tokens": 512, "temperature": 0.7,
            "error": None, "version": "1.2.2",
        }

    def tearDown(self):
        self.engine_patcher.stop()

    def test_full_chat_flow(self):
        """Complete chat flow: load -> chat -> unload."""
        # Load
        handler = make_mock_handler(path="/api/load", method="POST", body={"model": "test/model"})
        handler.do_POST()
        data = get_response_json(handler)
        self.assertEqual(data["status"], "ok")

        # Chat
        handler = make_mock_handler(path="/api/chat", method="POST",
                                     body={"messages": [{"role": "user", "content": "Hello"}]})
        handler.do_POST()
        data = get_response_json(handler)
        self.assertIn("choices", data)

        # Unload
        handler = make_mock_handler(path="/api/unload", method="POST", body={})
        handler.do_POST()
        data = get_response_json(handler)
        self.assertEqual(data["status"], "ok")

    def test_settings_then_chat(self):
        """Settings update then chat should use new parameters."""
        # Settings
        handler = make_mock_handler(path="/api/settings", method="POST",
                                     body={"max_tokens": 2048, "temperature": 0.5})
        handler.do_POST()
        self.mock_engine.set_params.assert_called_with(
            max_new_tokens=2048, temperature=0.5
        )

        # Chat
        handler = make_mock_handler(path="/v1/chat/completions", method="POST",
                                     body={"messages": [{"role": "user", "content": "Hi"}]})
        handler.do_POST()

    def test_openai_compatible_response_format(self):
        """OpenAI API should return fully compatible response format."""
        handler = make_mock_handler(path="/v1/chat/completions", method="POST",
                                     body={"messages": [{"role": "user", "content": "Hi"}]})
        handler.do_POST()
        data = get_response_json(handler)
        # Verify OpenAI response structure
        self.assertTrue(data["id"].startswith("chatcmpl-"))
        self.assertEqual(data["object"], "chat.completion")
        self.assertIsInstance(data["created"], int)
        self.assertIn("model", data)
        self.assertIsInstance(data["choices"], list)
        self.assertEqual(data["choices"][0]["message"]["role"], "assistant")
        self.assertIn("usage", data)
        self.assertIn("prompt_tokens", data["usage"])
        self.assertIn("completion_tokens", data["usage"])
        self.assertIn("total_tokens", data["usage"])


if __name__ == "__main__":
    unittest.main()
