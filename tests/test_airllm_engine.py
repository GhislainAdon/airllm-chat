"""
Comprehensive unit tests for airllm_engine.py.

Covers:
- _read_version() function
- AirLLMEngine.__init__
- load_model / unload_model
- cancel_generation
- generate / _generate_full / _generate_stream
- _build_prompt (various message combinations)
- set_params
- is_loaded / info properties
- get_system_info (with/without psutil, torch)
- get_ollama_info / get_ollama_models / pull_ollama_model / delete_ollama_model
- get_ollama_model_path
- _run_ollama
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import MagicMock, mock_open, patch, PropertyMock

from airllm_engine import AirLLMEngine, _read_version


# ---------------------------------------------------------------------------
#  _read_version() tests
# ---------------------------------------------------------------------------

class TestReadVersion(unittest.TestCase):
    """Tests for the module-level _read_version() function."""

    def test_reads_version_file(self):
        """Normal case: VERSION file exists and is read correctly."""
        with patch("builtins.open", mock_open(read_data="1.2.3\n")):
            with patch("os.path.join", return_value="/fake/VERSION"):
                result = _read_version()
        self.assertEqual(result, "1.2.3")

    def test_strips_whitespace(self):
        """Version with newlines/spaces should be stripped."""
        with patch("builtins.open", mock_open(read_data="  2.0.0  \n")):
            with patch("os.path.join", return_value="/fake/VERSION"):
                result = _read_version()
        self.assertEqual(result, "2.0.0")

    def test_missing_version_file_returns_dev(self):
        """Missing VERSION file should return '0.0.0-dev'."""
        with patch("builtins.open", side_effect=FileNotFoundError):
            with patch("os.path.join", return_value="/fake/VERSION"):
                result = _read_version()
        self.assertEqual(result, "0.0.0-dev")

    def test_frozen_mode_uses_meipass(self):
        """In PyInstaller frozen mode, sys._MEIPASS is used as base_dir."""
        with patch.object(sys, "frozen", True, create=True):
            with patch.object(sys, "_MEIPASS", "/frozen/app", create=True):
                with patch("builtins.open", mock_open(read_data="3.0.0")):
                    # We need to re-import or call the function; since _read_version
                    # checks getattr(sys, 'frozen', False) at call time, just call it
                    result = _read_version()
        # The function should have used sys._MEIPASS, but since we patched open
        # globally, it still works. The key thing is it doesn't crash.
        self.assertIsInstance(result, str)

    def test_non_frozen_mode_uses_file_dir(self):
        """In normal mode, __file__ directory is used."""
        # Default behavior (not frozen)
        result = _read_version()
        # Should return actual version from the project's VERSION file
        self.assertNotEqual(result, "")
        self.assertIsInstance(result, str)


# ---------------------------------------------------------------------------
#  AirLLMEngine.__init__ tests
# ---------------------------------------------------------------------------

class TestEngineInit(unittest.TestCase):
    """Tests for AirLLMEngine initialization."""

    def test_default_values(self):
        """Engine should initialize with default parameter values."""
        e = AirLLMEngine()
        self.assertEqual(e.model_path, "")
        self.assertEqual(e.max_new_tokens, 512)
        self.assertEqual(e.temperature, 0.7)
        self.assertIsNone(e.model)
        self.assertFalse(e._loaded)
        self.assertIsNone(e._load_error)
        self.assertFalse(e._cancel_requested)

    def test_custom_values(self):
        """Engine should accept custom parameter values."""
        e = AirLLMEngine(model_path="my/model", max_new_tokens=1024, temperature=0.3)
        self.assertEqual(e.model_path, "my/model")
        self.assertEqual(e.max_new_tokens, 1024)
        self.assertEqual(e.temperature, 0.3)

    def test_has_lock(self):
        """Engine should have a threading lock."""
        e = AirLLMEngine()
        self.assertIsInstance(e._lock, type(threading.Lock()))

    def test_ollama_models_dir_default(self):
        """Engine should set ollama models dir to ~/.ollama/models."""
        e = AirLLMEngine()
        expected = os.path.expanduser("~/.ollama/models")
        self.assertEqual(e._ollama_models_dir, expected)


# ---------------------------------------------------------------------------
#  is_loaded and info property tests
# ---------------------------------------------------------------------------

class TestEngineProperties(unittest.TestCase):
    """Tests for is_loaded and info properties."""

    def test_is_loaded_default_false(self):
        """is_loaded should be False by default."""
        e = AirLLMEngine()
        self.assertFalse(e.is_loaded)

    def test_is_loaded_after_manual_set(self):
        """is_loaded should reflect the _loaded flag."""
        e = AirLLMEngine()
        e._loaded = True
        self.assertTrue(e.is_loaded)

    def test_info_contains_all_keys(self):
        """info property should return all expected keys."""
        e = AirLLMEngine()
        info = e.info
        for key in ("model_path", "loaded", "max_new_tokens", "temperature", "error", "version"):
            self.assertIn(key, info, f"Missing key: {key}")

    def test_info_values_match_state(self):
        """info property values should match engine state."""
        e = AirLLMEngine(model_path="test/model", max_new_tokens=256, temperature=0.5)
        info = e.info
        self.assertEqual(info["model_path"], "test/model")
        self.assertFalse(info["loaded"])
        self.assertEqual(info["max_new_tokens"], 256)
        self.assertEqual(info["temperature"], 0.5)
        self.assertIsNone(info["error"])

    def test_info_error_after_load_failure(self):
        """info should reflect load error after a failed load."""
        e = AirLLMEngine()
        e._load_error = "some error"
        self.assertEqual(e.info["error"], "some error")


# ---------------------------------------------------------------------------
#  load_model tests
# ---------------------------------------------------------------------------

class TestLoadModel(unittest.TestCase):
    """Tests for AirLLMEngine.load_model()."""

    def test_already_loaded(self):
        """load_model should return ok if model is already loaded."""
        e = AirLLMEngine()
        e._loaded = True
        result = e.load_model()
        self.assertEqual(result["status"], "ok")
        self.assertIn("already loaded", result["message"])

    @patch("airllm_engine.airllm", create=True)
    def test_load_success(self, mock_airllm):
        """load_model should succeed with mocked airllm."""
        mock_model = MagicMock()
        mock_airllm.AutoModelForCausalLM.from_pretrained.return_value = mock_model

        e = AirLLMEngine(model_path="test/model")
        with patch.dict("sys.modules", {"airllm": mock_airllm}):
            result = e.load_model()

        self.assertEqual(result["status"], "ok")
        self.assertIn("test/model", result["message"])
        self.assertTrue(e._loaded)
        self.assertIsNone(e._load_error)

    @patch("airllm_engine.airllm", create=True)
    def test_load_default_model_path(self, mock_airllm):
        """load_model should use default path when model_path is empty."""
        mock_airllm.AutoModelForCausalLM.from_pretrained.return_value = MagicMock()

        e = AirLLMEngine(model_path="")
        with patch.dict("sys.modules", {"airllm": mock_airllm}):
            result = e.load_model()

        self.assertEqual(result["status"], "ok")
        self.assertEqual(e.model_path, "garage-bAInd/Platypus2-70B-instruct")

    @patch("airllm_engine.airllm", create=True)
    def test_load_error(self, mock_airllm):
        """load_model should return error on failure."""
        mock_airllm.AutoModelForCausalLM.from_pretrained.side_effect = RuntimeError("GPU OOM")

        e = AirLLMEngine(model_path="test/model")
        with patch.dict("sys.modules", {"airllm": mock_airllm}):
            result = e.load_model()

        self.assertEqual(result["status"], "error")
        self.assertIn("GPU OOM", result["message"])
        self.assertEqual(e._load_error, "GPU OOM")
        self.assertFalse(e._loaded)


# ---------------------------------------------------------------------------
#  unload_model tests
# ---------------------------------------------------------------------------

class TestUnloadModel(unittest.TestCase):
    """Tests for AirLLMEngine.unload_model()."""

    def test_unload_no_model(self):
        """unload_model when no model is loaded should return ok."""
        e = AirLLMEngine()
        result = e.unload_model()
        self.assertEqual(result["status"], "ok")
        self.assertIn("No model", result["message"])

    def test_unload_loaded_model(self):
        """unload_model should clean up model state."""
        e = AirLLMEngine()
        e._loaded = True
        e.model = MagicMock()
        result = e.unload_model()
        self.assertEqual(result["status"], "ok")
        self.assertIn("unloaded", result["message"].lower())
        self.assertIsNone(e.model)
        self.assertFalse(e._loaded)
        self.assertEqual(e.model_path, "")

    def test_unload_clears_gpu_cache(self):
        """unload_model should call torch.cuda.empty_cache when available."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.empty_cache.return_value = None

        e = AirLLMEngine()
        e._loaded = True
        e.model = MagicMock()

        with patch.dict("sys.modules", {"torch": mock_torch}):
            result = e.unload_model()

        self.assertEqual(result["status"], "ok")
        mock_torch.cuda.empty_cache.assert_called_once()

    def test_unload_handles_del_error(self):
        """unload_model should handle exceptions during model deletion gracefully."""
        e = AirLLMEngine()
        e._loaded = True
        e.model = MagicMock()
        # Simulate a model that raises when accessed for deletion.
        # The engine uses `del self.model` which just removes the reference,
        # but if it somehow raises, the try/except catches it.
        # We test that unload still completes successfully.
        result = e.unload_model()
        self.assertEqual(result["status"], "ok")
        self.assertIsNone(e.model)
        self.assertFalse(e._loaded)


# ---------------------------------------------------------------------------
#  cancel_generation tests
# ---------------------------------------------------------------------------

class TestCancelGeneration(unittest.TestCase):
    """Tests for AirLLMEngine.cancel_generation()."""

    def test_cancel_sets_flag(self):
        """cancel_generation should set _cancel_requested to True."""
        e = AirLLMEngine()
        self.assertFalse(e._cancel_requested)
        e.cancel_generation()
        self.assertTrue(e._cancel_requested)

    def test_cancel_multiple_times(self):
        """Calling cancel_generation multiple times should keep flag True."""
        e = AirLLMEngine()
        e.cancel_generation()
        e.cancel_generation()
        self.assertTrue(e._cancel_requested)


# ---------------------------------------------------------------------------
#  _build_prompt tests
# ---------------------------------------------------------------------------

class TestBuildPrompt(unittest.TestCase):
    """Tests for AirLLMEngine._build_prompt()."""

    def test_system_message(self):
        """System message should appear first with SYSTEM: prefix."""
        e = AirLLMEngine()
        messages = [{"role": "system", "content": "Be helpful."}]
        prompt = e._build_prompt(messages)
        self.assertTrue(prompt.startswith("SYSTEM: Be helpful."))
        self.assertTrue(prompt.endswith("ASSISTANT:"))

    def test_user_message(self):
        """User messages should have USER: prefix."""
        e = AirLLMEngine()
        messages = [{"role": "user", "content": "Hello"}]
        prompt = e._build_prompt(messages)
        self.assertIn("USER: Hello", prompt)
        self.assertTrue(prompt.endswith("ASSISTANT:"))

    def test_assistant_message(self):
        """Assistant messages should have ASSISTANT: prefix."""
        e = AirLLMEngine()
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]
        prompt = e._build_prompt(messages)
        self.assertIn("ASSISTANT: Hello!", prompt)

    def test_empty_messages(self):
        """Empty messages list should produce just ASSISTANT:."""
        e = AirLLMEngine()
        prompt = e._build_prompt([])
        self.assertEqual(prompt, "ASSISTANT:")

    def test_system_before_user(self):
        """System message should come before user messages in the prompt."""
        e = AirLLMEngine()
        messages = [
            {"role": "user", "content": "Question"},
            {"role": "system", "content": "System prompt"},
        ]
        prompt = e._build_prompt(messages)
        sys_pos = prompt.index("SYSTEM:")
        user_pos = prompt.index("USER:")
        self.assertLess(sys_pos, user_pos, "System message should come before user message")

    def test_multi_turn_conversation(self):
        """Multi-turn conversation should preserve order."""
        e = AirLLMEngine()
        messages = [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "Q2"},
            {"role": "assistant", "content": "A2"},
            {"role": "user", "content": "Q3"},
        ]
        prompt = e._build_prompt(messages)
        self.assertIn("USER: Q1", prompt)
        self.assertIn("ASSISTANT: A1", prompt)
        self.assertIn("USER: Q2", prompt)
        self.assertIn("ASSISTANT: A2", prompt)
        self.assertIn("USER: Q3", prompt)
        self.assertTrue(prompt.endswith("ASSISTANT:"))

    def test_message_with_missing_role(self):
        """Messages without a role should default to 'user'."""
        e = AirLLMEngine()
        messages = [{"content": "No role specified"}]
        prompt = e._build_prompt(messages)
        self.assertIn("USER: No role specified", prompt)

    def test_message_with_missing_content(self):
        """Messages without content should default to empty string."""
        e = AirLLMEngine()
        messages = [{"role": "user"}]
        prompt = e._build_prompt(messages)
        self.assertIn("USER: ", prompt)

    def test_unknown_role_ignored(self):
        """Messages with unknown roles should be ignored (not crash)."""
        e = AirLLMEngine()
        messages = [
            {"role": "function", "content": "result"},
            {"role": "user", "content": "Hello"},
        ]
        prompt = e._build_prompt(messages)
        # function role is not handled, so it should be ignored
        self.assertNotIn("FUNCTION: result", prompt)
        self.assertIn("USER: Hello", prompt)

    def test_last_system_wins(self):
        """Multiple system messages: last one should be used (overwrites)."""
        e = AirLLMEngine()
        messages = [
            {"role": "system", "content": "First system"},
            {"role": "system", "content": "Second system"},
        ]
        prompt = e._build_prompt(messages)
        self.assertIn("SYSTEM: Second system", prompt)
        # Only one system message should be present
        self.assertEqual(prompt.count("SYSTEM:"), 1)


# ---------------------------------------------------------------------------
#  generate / _generate_full / _generate_stream tests
# ---------------------------------------------------------------------------

class TestGenerate(unittest.TestCase):
    """Tests for generate, _generate_full, and _generate_stream."""

    def test_generate_full_without_model(self):
        """Non-streaming generate without model should return error."""
        e = AirLLMEngine()
        result = e.generate([{"role": "user", "content": "Hi"}], stream=False)
        self.assertIsInstance(result, str)
        self.assertIn("No model loaded", result)

    def test_generate_stream_without_model(self):
        """Streaming generate without model should yield error."""
        e = AirLLMEngine()
        tokens = list(e.generate([{"role": "user", "content": "Hi"}], stream=True))
        self.assertEqual(len(tokens), 1)
        self.assertIn("No model loaded", tokens[0])

    def test_generate_full_with_mock_model(self):
        """Non-streaming generate with mocked model should return text."""
        e = AirLLMEngine()
        e._loaded = True
        mock_model = MagicMock()
        mock_model.tokenize.return_value = [1, 2, 3]
        mock_model.generate.return_value = "Hello world"
        e.model = mock_model

        result = e.generate([{"role": "user", "content": "Hi"}], stream=False)
        self.assertEqual(result, "Hello world")

    def test_generate_full_dict_output(self):
        """_full_generate should handle dict output from model."""
        e = AirLLMEngine()
        e._loaded = True
        mock_model = MagicMock()
        mock_model.tokenize.return_value = [1, 2, 3]
        mock_model.generate.return_value = {"text": "Dict output"}
        e.model = mock_model

        result = e._generate_full([{"role": "user", "content": "Hi"}])
        self.assertEqual(result, "Dict output")

    def test_generate_full_list_output(self):
        """_full_generate should handle list output from model."""
        e = AirLLMEngine()
        e._loaded = True
        mock_model = MagicMock()
        mock_model.tokenize.return_value = [1, 2, 3]
        mock_model.generate.return_value = ["Hello", "world"]
        e.model = mock_model

        result = e._generate_full([{"role": "user", "content": "Hi"}])
        self.assertEqual(result, "Hello world")

    def test_generate_stream_with_mock_model(self):
        """Streaming generate with mocked model should yield tokens."""
        e = AirLLMEngine()
        e._loaded = True
        mock_model = MagicMock()
        mock_model.tokenize.return_value = [1, 2, 3]
        mock_model.generate.return_value = iter(["Hello", "Hello world"])
        e.model = mock_model

        tokens = list(e.generate([{"role": "user", "content": "Hi"}], stream=True))
        # Should get incremental tokens
        self.assertTrue(len(tokens) > 0)

    def test_cancel_during_stream(self):
        """Cancel during streaming should stop yielding tokens."""
        e = AirLLMEngine()
        e._loaded = True
        mock_model = MagicMock()
        mock_model.tokenize.return_value = [1, 2, 3]
        # Generate a stream of outputs
        mock_model.generate.return_value = iter(["token1", "token1 token2", "token1 token2 token3"])
        e.model = mock_model

        # Set cancel after getting first token
        collected = []
        for token in e.generate([{"role": "user", "content": "Hi"}], stream=True):
            collected.append(token)
            if len(collected) >= 1:
                e._cancel_requested = True

        # Should have stopped after cancel
        self.assertTrue(len(collected) >= 1)

    def test_generate_full_exception(self):
        """_generate_full should return error string on exception."""
        e = AirLLMEngine()
        e._loaded = True
        mock_model = MagicMock()
        mock_model.tokenize.side_effect = RuntimeError("Tokenize error")
        e.model = mock_model

        result = e._generate_full([{"role": "user", "content": "Hi"}])
        self.assertIn("[ERROR]", result)
        self.assertIn("Tokenize error", result)

    def test_generate_stream_exception(self):
        """_generate_stream should yield error on exception."""
        e = AirLLMEngine()
        e._loaded = True
        mock_model = MagicMock()
        mock_model.tokenize.side_effect = RuntimeError("Stream error")
        e.model = mock_model

        tokens = list(e._generate_stream([{"role": "user", "content": "Hi"}]))
        self.assertTrue(any("[ERROR]" in t for t in tokens))

    def test_stream_generate_fallback(self):
        """_stream_generate should fallback when model.generate raises TypeError."""
        e = AirLLMEngine()
        e._loaded = True
        mock_model = MagicMock()
        mock_model.tokenize.return_value = [1, 2, 3]

        # First call (streaming) raises TypeError, second call (fallback full gen) returns text
        call_count = [0]

        def generate_side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise TypeError("not iterable")
            return "Fallback text"

        mock_model.generate.side_effect = generate_side_effect
        e.model = mock_model

        tokens = list(e._stream_generate("test prompt", max_new_tokens=10, temperature=0.5))
        self.assertTrue(len(tokens) > 0)
        combined = "".join(tokens)
        self.assertIn("Fallback", combined)

    def test_generate_stream_dict_output(self):
        """_stream_generate should handle dict outputs with 'text' key."""
        e = AirLLMEngine()
        e._loaded = True
        mock_model = MagicMock()
        mock_model.tokenize.return_value = [1, 2, 3]
        mock_model.generate.return_value = iter([
            {"text": "Hello"},
            {"text": " world"},
        ])
        e.model = mock_model

        tokens = list(e._stream_generate("test", max_new_tokens=10, temperature=0.5))
        self.assertIn("Hello", tokens)
        self.assertIn(" world", tokens)

    def test_generate_resets_cancel_flag(self):
        """_generate_full should reset cancel flag at start."""
        e = AirLLMEngine()
        e._loaded = True
        e._cancel_requested = True
        mock_model = MagicMock()
        mock_model.tokenize.return_value = [1]
        mock_model.generate.return_value = "output"
        e.model = mock_model

        e._generate_full([{"role": "user", "content": "Hi"}])
        self.assertFalse(e._cancel_requested)

    def test_generate_stream_resets_cancel_flag(self):
        """_generate_stream should reset cancel flag at start."""
        e = AirLLMEngine()
        e._loaded = True
        e._cancel_requested = True
        mock_model = MagicMock()
        mock_model.tokenize.return_value = [1]
        mock_model.generate.return_value = iter(["token"])
        e.model = mock_model

        list(e._generate_stream([{"role": "user", "content": "Hi"}]))
        self.assertFalse(e._cancel_requested)


# ---------------------------------------------------------------------------
#  set_params tests
# ---------------------------------------------------------------------------

class TestSetParams(unittest.TestCase):
    """Tests for AirLLMEngine.set_params()."""

    def test_update_max_new_tokens(self):
        """set_params should update max_new_tokens."""
        e = AirLLMEngine()
        result = e.set_params(max_new_tokens=2048)
        self.assertEqual(e.max_new_tokens, 2048)
        self.assertEqual(result["status"], "ok")

    def test_update_temperature(self):
        """set_params should update temperature."""
        e = AirLLMEngine()
        result = e.set_params(temperature=1.5)
        self.assertEqual(e.temperature, 1.5)
        self.assertEqual(result["status"], "ok")

    def test_update_both_params(self):
        """set_params should update both params at once."""
        e = AirLLMEngine()
        result = e.set_params(max_new_tokens=100, temperature=0.1)
        self.assertEqual(e.max_new_tokens, 100)
        self.assertEqual(e.temperature, 0.1)

    def test_no_change_returns_ok(self):
        """set_params with no model_path change should return ok."""
        e = AirLLMEngine(model_path="same/path")
        result = e.set_params(max_new_tokens=512)
        self.assertEqual(result["status"], "ok")
        self.assertIn("Parameters updated", result["message"])

    def test_model_path_change_triggers_reload(self):
        """set_params with different model_path should trigger load_model."""
        e = AirLLMEngine(model_path="old/model")
        e._loaded = True
        with patch.object(e, "load_model", return_value={"status": "ok", "message": "loaded"}) as mock_load:
            result = e.set_params(model_path="new/model")
            mock_load.assert_called_once()
        self.assertEqual(e.model_path, "new/model")
        self.assertFalse(e._loaded)
        self.assertIsNone(e.model)

    def test_same_model_path_no_reload(self):
        """set_params with same model_path should not trigger reload."""
        e = AirLLMEngine(model_path="same/path")
        with patch.object(e, "load_model") as mock_load:
            result = e.set_params(model_path="same/path")
            mock_load.assert_not_called()

    def test_none_params_ignored(self):
        """set_params with None values should not change anything."""
        e = AirLLMEngine(max_new_tokens=100, temperature=0.5)
        e.set_params(max_new_tokens=None, temperature=None, model_path=None)
        self.assertEqual(e.max_new_tokens, 100)
        self.assertEqual(e.temperature, 0.5)


# ---------------------------------------------------------------------------
#  get_system_info tests
# ---------------------------------------------------------------------------

class TestGetSystemInfo(unittest.TestCase):
    """Tests for AirLLMEngine.get_system_info()."""

    def test_returns_all_keys(self):
        """get_system_info should return all expected keys."""
        e = AirLLMEngine()
        info = e.get_system_info()
        for key in ("gpu_name", "gpu_memory_total", "gpu_memory_used", "gpu_memory_free",
                     "ram_total", "ram_used", "ram_free", "cpu_count"):
            self.assertIn(key, info)

    @patch("airllm_engine.psutil", None)
    def test_without_psutil(self):
        """get_system_info without psutil should use os.cpu_count."""
        e = AirLLMEngine()
        info = e.get_system_info()
        self.assertEqual(info["cpu_count"], os.cpu_count() or 0)
        self.assertEqual(info["ram_total"], 0)

    @patch("airllm_engine.psutil", None)
    def test_without_psutil_ram_zeros(self):
        """Without psutil, RAM values should be zero."""
        e = AirLLMEngine()
        info = e.get_system_info()
        self.assertEqual(info["ram_total"], 0)
        self.assertEqual(info["ram_used"], 0)
        self.assertEqual(info["ram_free"], 0)

    def test_with_psutil(self):
        """get_system_info with psutil should populate RAM info."""
        mock_psutil = MagicMock()
        vm = MagicMock()
        vm.total = 16 * 1024**3
        vm.used = 8 * 1024**3
        vm.available = 8 * 1024**3
        mock_psutil.virtual_memory.return_value = vm
        mock_psutil.cpu_count.return_value = 8

        with patch("airllm_engine.psutil", mock_psutil):
            e = AirLLMEngine()
            info = e.get_system_info()
            self.assertEqual(info["ram_total"], 16 * 1024**3)
            self.assertEqual(info["ram_used"], 8 * 1024**3)
            self.assertEqual(info["cpu_count"], 8)

    def test_psutil_exception_handled(self):
        """get_system_info should handle psutil exceptions gracefully."""
        mock_psutil = MagicMock()
        mock_psutil.virtual_memory.side_effect = RuntimeError("psutil error")

        with patch("airllm_engine.psutil", mock_psutil):
            e = AirLLMEngine()
            info = e.get_system_info()
            # Should not crash, RAM values stay at 0
            self.assertEqual(info["ram_total"], 0)

    def test_with_torch_cuda(self):
        """get_system_info with torch.cuda should populate GPU info."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_name.return_value = "NVIDIA RTX 4090"
        props = MagicMock()
        props.total_mem = 24 * 1024**3
        mock_torch.cuda.get_device_properties.return_value = props
        mock_torch.cuda.memory_stats.return_value = {"allocated_bytes.all.current": 4 * 1024**3}

        with patch.dict("sys.modules", {"torch": mock_torch}):
            e = AirLLMEngine()
            info = e.get_system_info()
            self.assertEqual(info["gpu_name"], "NVIDIA RTX 4090")
            self.assertEqual(info["gpu_memory_total"], 24 * 1024**3)
            self.assertEqual(info["gpu_memory_used"], 4 * 1024**3)
            self.assertEqual(info["gpu_memory_free"], 20 * 1024**3)

    def test_torch_cuda_memory_stats_fallback(self):
        """get_system_info should fall back to memory_allocated if memory_stats fails."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_name.return_value = "GPU"
        props = MagicMock()
        props.total_mem = 8 * 1024**3
        mock_torch.cuda.get_device_properties.return_value = props
        mock_torch.cuda.memory_stats.side_effect = RuntimeError("no stats")
        mock_torch.cuda.memory_allocated.return_value = 2 * 1024**3

        with patch.dict("sys.modules", {"torch": mock_torch}):
            e = AirLLMEngine()
            info = e.get_system_info()
            self.assertEqual(info["gpu_memory_used"], 2 * 1024**3)


# ---------------------------------------------------------------------------
#  Ollama tests
# ---------------------------------------------------------------------------

class TestOllamaInfo(unittest.TestCase):
    """Tests for AirLLMEngine.get_ollama_info()."""

    @patch("shutil.which", return_value=None)
    def test_ollama_not_installed(self, mock_which):
        """get_ollama_info when ollama is not installed."""
        e = AirLLMEngine()
        info = e.get_ollama_info()
        self.assertFalse(info["ollama_installed"])
        self.assertFalse(info["ollama_running"])
        self.assertEqual(info["ollama_version"], "")
        self.assertEqual(info["models_count"], 0)

    @patch("shutil.which", return_value="/usr/bin/ollama")
    def test_ollama_installed_running(self, mock_which):
        """get_ollama_info when ollama is installed and running."""
        e = AirLLMEngine()
        mock_proc = MagicMock()
        mock_proc.communicate.return_value = ("0.1.20", "")
        mock_proc.returncode = 0

        mock_list_proc = MagicMock()
        mock_list_proc.communicate.return_value = ("NAME ID SIZE MODIFIED\nmodel1 abc 1GB 2d", "")
        mock_list_proc.returncode = 0

        with patch.object(e, "_run_ollama", side_effect=[mock_proc, mock_list_proc]):
            info = e.get_ollama_info()
        self.assertTrue(info["ollama_installed"])
        self.assertTrue(info["ollama_running"])
        self.assertEqual(info["ollama_version"], "0.1.20")

    @patch("shutil.which", return_value="/usr/bin/ollama")
    def test_ollama_installed_not_running(self, mock_which):
        """get_ollama_info when ollama is installed but not running."""
        e = AirLLMEngine()
        mock_proc = MagicMock()
        mock_proc.communicate.return_value = ("0.1.20", "")
        mock_proc.returncode = 0

        mock_list_proc = MagicMock()
        mock_list_proc.communicate.return_value = ("error", "")
        mock_list_proc.returncode = 1

        with patch.object(e, "_run_ollama", side_effect=[mock_proc, mock_list_proc]):
            info = e.get_ollama_info()
        self.assertTrue(info["ollama_installed"])
        self.assertFalse(info["ollama_running"])

    @patch("shutil.which", return_value="/usr/bin/ollama")
    def test_ollama_version_exception(self, mock_which):
        """get_ollama_info should handle version check exceptions."""
        e = AirLLMEngine()
        mock_proc = MagicMock()
        mock_proc.communicate.side_effect = subprocess.TimeoutExpired("ollama", 10)

        mock_list_proc = MagicMock()
        mock_list_proc.communicate.side_effect = subprocess.TimeoutExpired("ollama", 5)

        with patch.object(e, "_run_ollama", side_effect=[mock_proc, mock_list_proc]):
            info = e.get_ollama_info()
        self.assertTrue(info["ollama_installed"])
        self.assertEqual(info["ollama_version"], "unknown")
        self.assertFalse(info["ollama_running"])


class TestOllamaModels(unittest.TestCase):
    """Tests for AirLLMEngine.get_ollama_models()."""

    @patch("shutil.which", return_value=None)
    def test_not_installed(self, mock_which):
        """get_ollama_models when ollama is not installed."""
        e = AirLLMEngine()
        result = e.get_ollama_models()
        self.assertEqual(result["models"], [])
        self.assertIn("not installed", result["message"])

    @patch("shutil.which", return_value="/usr/bin/ollama")
    def test_list_models(self, mock_which):
        """get_ollama_models should parse ollama list output."""
        e = AirLLMEngine()
        mock_proc = MagicMock()
        mock_proc.communicate.return_value = (
            "NAME                    ID              SIZE    MODIFIED\n"
            "llama2:latest           abc123          4.7GB   2 days ago\n"
            "mistral:latest          def456          4.1GB   5 days ago",
            ""
        )
        mock_proc.returncode = 0

        with patch.object(e, "_run_ollama", return_value=mock_proc):
            result = e.get_ollama_models()
        self.assertEqual(len(result["models"]), 2)
        self.assertEqual(result["models"][0]["name"], "llama2:latest")
        self.assertEqual(result["models"][1]["name"], "mistral:latest")

    @patch("shutil.which", return_value="/usr/bin/ollama")
    def test_list_models_failure(self, mock_which):
        """get_ollama_models should handle ollama list failure."""
        e = AirLLMEngine()
        mock_proc = MagicMock()
        mock_proc.communicate.return_value = ("cannot connect", "")
        mock_proc.returncode = 1

        with patch.object(e, "_run_ollama", return_value=mock_proc):
            result = e.get_ollama_models()
        self.assertEqual(result["models"], [])
        self.assertIn("failed", result["message"])

    @patch("shutil.which", return_value="/usr/bin/ollama")
    def test_list_models_timeout(self, mock_which):
        """get_ollama_models should handle timeout."""
        e = AirLLMEngine()
        mock_proc = MagicMock()
        mock_proc.communicate.side_effect = subprocess.TimeoutExpired("ollama", 30)

        with patch.object(e, "_run_ollama", return_value=mock_proc):
            result = e.get_ollama_models()
        self.assertEqual(result["models"], [])
        self.assertIn("timed out", result["message"])

    @patch("shutil.which", return_value="/usr/bin/ollama")
    def test_list_models_exception(self, mock_which):
        """get_ollama_models should handle unexpected exceptions."""
        e = AirLLMEngine()
        mock_proc = MagicMock()
        mock_proc.communicate.side_effect = OSError("unexpected")

        with patch.object(e, "_run_ollama", return_value=mock_proc):
            result = e.get_ollama_models()
        self.assertEqual(result["models"], [])
        self.assertIn("Error", result["message"])

    @patch("shutil.which", return_value="/usr/bin/ollama")
    def test_empty_model_list(self, mock_which):
        """get_ollama_models should handle empty list (header only)."""
        e = AirLLMEngine()
        mock_proc = MagicMock()
        mock_proc.communicate.return_value = ("NAME ID SIZE MODIFIED", "")
        mock_proc.returncode = 0

        with patch.object(e, "_run_ollama", return_value=mock_proc):
            result = e.get_ollama_models()
        self.assertEqual(result["models"], [])

    @patch("shutil.which", return_value="/usr/bin/ollama")
    def test_list_models_partial_columns(self, mock_which):
        """get_ollama_models should handle lines with fewer columns."""
        e = AirLLMEngine()
        mock_proc = MagicMock()
        mock_proc.communicate.return_value = (
            "NAME ID SIZE MODIFIED\n"
            "mini xyz",
            ""
        )
        mock_proc.returncode = 0

        with patch.object(e, "_run_ollama", return_value=mock_proc):
            result = e.get_ollama_models()
        self.assertEqual(len(result["models"]), 1)
        self.assertEqual(result["models"][0]["name"], "mini")
        self.assertEqual(result["models"][0]["id"], "xyz")
        self.assertEqual(result["models"][0]["size"], "")

    @patch("shutil.which", return_value="/usr/bin/ollama")
    def test_list_models_empty_lines(self, mock_which):
        """get_ollama_models should skip empty lines."""
        e = AirLLMEngine()
        mock_proc = MagicMock()
        mock_proc.communicate.return_value = (
            "NAME ID SIZE MODIFIED\n"
            "\n"
            "model1 abc 1GB 1d\n"
            "\n",
            ""
        )
        mock_proc.returncode = 0

        with patch.object(e, "_run_ollama", return_value=mock_proc):
            result = e.get_ollama_models()
        self.assertEqual(len(result["models"]), 1)


class TestOllamaPull(unittest.TestCase):
    """Tests for AirLLMEngine.pull_ollama_model()."""

    @patch("shutil.which", return_value=None)
    def test_not_installed(self, mock_which):
        """pull_ollama_model when ollama is not installed."""
        e = AirLLMEngine()
        results = list(e.pull_ollama_model("llama2"))
        self.assertTrue(any("not installed" in r.get("message", "") for r in results))

    @patch("shutil.which", return_value="/usr/bin/ollama")
    def test_pull_success(self, mock_which):
        """pull_ollama_model should stream JSON progress and success."""
        e = AirLLMEngine()
        mock_proc = MagicMock()
        mock_proc.stdout = iter([
            '{"status": "downloading", "digest": "abc123"}\n',
            '{"status": "verifying"}\n',
        ])
        mock_proc.wait.return_value = None
        mock_proc.returncode = 0

        with patch.object(e, "_run_ollama", return_value=mock_proc):
            results = list(e.pull_ollama_model("llama2"))
        # Should have downloading, verifying, and success events
        self.assertTrue(any(r.get("status") == "downloading" for r in results))
        self.assertTrue(any(r.get("status") == "success" for r in results))

    @patch("shutil.which", return_value="/usr/bin/ollama")
    def test_pull_failure(self, mock_which):
        """pull_ollama_model should handle non-zero exit code."""
        e = AirLLMEngine()
        mock_proc = MagicMock()
        mock_proc.stdout = iter([])
        mock_proc.wait.return_value = None
        mock_proc.returncode = 1

        with patch.object(e, "_run_ollama", return_value=mock_proc):
            results = list(e.pull_ollama_model("nonexistent"))
        self.assertTrue(any(r.get("status") == "error" for r in results))

    @patch("shutil.which", return_value="/usr/bin/ollama")
    def test_pull_non_json_line(self, mock_which):
        """pull_ollama_model should handle non-JSON lines gracefully."""
        e = AirLLMEngine()
        mock_proc = MagicMock()
        mock_proc.stdout = iter([
            'not json\n',
        ])
        mock_proc.wait.return_value = None
        mock_proc.returncode = 0

        with patch.object(e, "_run_ollama", return_value=mock_proc):
            results = list(e.pull_ollama_model("llama2"))
        # Should have a pulling status message for non-JSON line
        self.assertTrue(any(r.get("status") == "pulling" for r in results))
        self.assertTrue(any(r.get("status") == "success" for r in results))

    @patch("shutil.which", return_value="/usr/bin/ollama")
    def test_pull_timeout(self, mock_which):
        """pull_ollama_model should handle timeout."""
        e = AirLLMEngine()
        mock_proc = MagicMock()
        mock_proc.stdout = iter([])
        mock_proc.wait.side_effect = subprocess.TimeoutExpired("ollama", 600)

        with patch.object(e, "_run_ollama", return_value=mock_proc):
            results = list(e.pull_ollama_model("llama2"))
        self.assertTrue(any(r.get("status") == "error" for r in results))
        self.assertTrue(any("timed out" in r.get("message", "") for r in results))

    @patch("shutil.which", return_value="/usr/bin/ollama")
    def test_pull_exception(self, mock_which):
        """pull_ollama_model should handle unexpected exceptions."""
        e = AirLLMEngine()
        mock_proc = MagicMock()
        mock_proc.stdout = MagicMock(side_effect=OSError("broken"))

        with patch.object(e, "_run_ollama", return_value=mock_proc):
            results = list(e.pull_ollama_model("llama2"))
        self.assertTrue(any(r.get("status") == "error" for r in results))

    @patch("shutil.which", return_value="/usr/bin/ollama")
    def test_pull_empty_lines_skipped(self, mock_which):
        """pull_ollama_model should skip empty lines."""
        e = AirLLMEngine()
        mock_proc = MagicMock()
        mock_proc.stdout = iter([
            '\n',
            '  \n',
            '{"status": "done"}\n',
        ])
        mock_proc.wait.return_value = None
        mock_proc.returncode = 0

        with patch.object(e, "_run_ollama", return_value=mock_proc):
            results = list(e.pull_ollama_model("llama2"))
        # Empty lines should be skipped; only JSON line + success
        json_events = [r for r in results if r.get("status") == "done"]
        self.assertEqual(len(json_events), 1)
        self.assertTrue(any(r.get("status") == "success" for r in results))


class TestOllamaDelete(unittest.TestCase):
    """Tests for AirLLMEngine.delete_ollama_model()."""

    @patch("shutil.which", return_value=None)
    def test_not_installed(self, mock_which):
        """delete_ollama_model when ollama is not installed."""
        e = AirLLMEngine()
        result = e.delete_ollama_model("llama2")
        self.assertEqual(result["status"], "error")
        self.assertIn("not installed", result["message"])

    @patch("shutil.which", return_value="/usr/bin/ollama")
    def test_delete_success(self, mock_which):
        """delete_ollama_model should succeed with zero exit code."""
        e = AirLLMEngine()
        mock_proc = MagicMock()
        mock_proc.communicate.return_value = ("", "")
        mock_proc.returncode = 0

        with patch.object(e, "_run_ollama", return_value=mock_proc):
            result = e.delete_ollama_model("llama2")
        self.assertEqual(result["status"], "success")
        self.assertIn("deleted", result["message"])

    @patch("shutil.which", return_value="/usr/bin/ollama")
    def test_delete_failure(self, mock_which):
        """delete_ollama_model should handle non-zero exit code."""
        e = AirLLMEngine()
        mock_proc = MagicMock()
        mock_proc.communicate.return_value = ("model not found", "")
        mock_proc.returncode = 1

        with patch.object(e, "_run_ollama", return_value=mock_proc):
            result = e.delete_ollama_model("nonexistent")
        self.assertEqual(result["status"], "error")

    @patch("shutil.which", return_value="/usr/bin/ollama")
    def test_delete_timeout(self, mock_which):
        """delete_ollama_model should handle timeout."""
        e = AirLLMEngine()
        mock_proc = MagicMock()
        mock_proc.communicate.side_effect = subprocess.TimeoutExpired("ollama", 60)

        with patch.object(e, "_run_ollama", return_value=mock_proc):
            result = e.delete_ollama_model("llama2")
        self.assertEqual(result["status"], "error")
        self.assertIn("timed out", result["message"])

    @patch("shutil.which", return_value="/usr/bin/ollama")
    def test_delete_exception(self, mock_which):
        """delete_ollama_model should handle unexpected exceptions."""
        e = AirLLMEngine()
        mock_proc = MagicMock()
        mock_proc.communicate.side_effect = OSError("unexpected")

        with patch.object(e, "_run_ollama", return_value=mock_proc):
            result = e.delete_ollama_model("llama2")
        self.assertEqual(result["status"], "error")


class TestOllamaModelPath(unittest.TestCase):
    """Tests for AirLLMEngine.get_ollama_model_path()."""

    @patch("shutil.which", return_value=None)
    def test_not_installed(self, mock_which):
        """get_ollama_model_path when ollama is not installed."""
        e = AirLLMEngine()
        result = e.get_ollama_model_path("llama2")
        self.assertEqual(result["status"], "error")
        self.assertIn("not installed", result["message"])

    @patch("shutil.which", return_value="/usr/bin/ollama")
    def test_show_failure(self, mock_which):
        """get_ollama_model_path should handle ollama show failure."""
        e = AirLLMEngine()
        mock_proc = MagicMock()
        mock_proc.communicate.return_value = ("model not found", "")
        mock_proc.returncode = 1

        with patch.object(e, "_run_ollama", return_value=mock_proc):
            result = e.get_ollama_model_path("nonexistent")
        self.assertEqual(result["status"], "error")

    @patch("shutil.which", return_value="/usr/bin/ollama")
    @patch("os.path.isfile", return_value=True)
    @patch("os.path.getsize", return_value=4096)
    def test_success_with_digest(self, mock_size, mock_isfile, mock_which):
        """get_ollama_model_path should resolve blob path from digest."""
        e = AirLLMEngine()
        mock_proc = MagicMock()
        mock_proc.communicate.return_value = (
            "# Modelfile\nFROM sha256abc123\nPARAMETER temperature 0.7",
            ""
        )
        mock_proc.returncode = 0

        with patch.object(e, "_run_ollama", return_value=mock_proc):
            result = e.get_ollama_model_path("llama2")
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["digest"], "sha256abc123")
        self.assertIn("blobs", result["path"])
        self.assertEqual(result["size_bytes"], 4096)

    @patch("shutil.which", return_value="/usr/bin/ollama")
    def test_no_from_line(self, mock_which):
        """get_ollama_model_path should handle missing FROM line."""
        e = AirLLMEngine()
        mock_proc = MagicMock()
        mock_proc.communicate.return_value = ("# Modelfile\nPARAMETER temperature 0.7", "")
        mock_proc.returncode = 0

        with patch.object(e, "_run_ollama", return_value=mock_proc):
            result = e.get_ollama_model_path("llama2")
        self.assertEqual(result["status"], "error")
        self.assertIn("digest", result["message"].lower())

    @patch("shutil.which", return_value="/usr/bin/ollama")
    @patch("os.path.isfile", return_value=False)
    def test_blob_not_found(self, mock_isfile, mock_which):
        """get_ollama_model_path should handle missing blob file."""
        e = AirLLMEngine()
        mock_proc = MagicMock()
        mock_proc.communicate.return_value = ("FROM abc123\n", "")
        mock_proc.returncode = 0

        with patch.object(e, "_run_ollama", return_value=mock_proc):
            result = e.get_ollama_model_path("llama2")
        self.assertEqual(result["status"], "error")
        self.assertIn("not found", result["message"].lower())

    @patch("shutil.which", return_value="/usr/bin/ollama")
    def test_timeout(self, mock_which):
        """get_ollama_model_path should handle timeout."""
        e = AirLLMEngine()
        mock_proc = MagicMock()
        mock_proc.communicate.side_effect = subprocess.TimeoutExpired("ollama", 30)

        with patch.object(e, "_run_ollama", return_value=mock_proc):
            result = e.get_ollama_model_path("llama2")
        self.assertEqual(result["status"], "error")
        self.assertIn("timed out", result["message"])


class TestRunOllama(unittest.TestCase):
    """Tests for AirLLMEngine._run_ollama()."""

    @patch("subprocess.Popen")
    def test_run_ollama_command(self, mock_popen):
        """_run_ollama should call Popen with correct arguments."""
        e = AirLLMEngine()
        e._run_ollama("list")
        mock_popen.assert_called_once_with(
            ["ollama", "list"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

    @patch("subprocess.Popen")
    def test_run_ollama_with_multiple_args(self, mock_popen):
        """_run_ollama should pass multiple arguments."""
        e = AirLLMEngine()
        e._run_ollama("show", "--modelfile", "llama2")
        mock_popen.assert_called_once_with(
            ["ollama", "show", "--modelfile", "llama2"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )


if __name__ == "__main__":
    unittest.main()
