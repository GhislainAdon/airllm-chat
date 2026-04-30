"""
Smoke tests for AirLLM Chat engine.
Run with: python -m pytest tests/ -v
"""

from airllm_engine import AirLLMEngine


def test_engine_initial_state():
    """Engine should start unloaded with defaults."""
    e = AirLLMEngine()
    assert e.is_loaded is False


def test_engine_info():
    """Engine info should contain required keys."""
    e = AirLLMEngine()
    info = e.info
    assert "model_path" in info
    assert "loaded" in info
    assert "max_new_tokens" in info
    assert "temperature" in info
    assert "version" in info
    assert info["loaded"] is False


def test_engine_info_version():
    """Version should be read from VERSION file, not empty or dev fallback."""
    e = AirLLMEngine()
    info = e.info
    version = info["version"]
    assert version != "", "Version should not be empty"
    assert version != "0.0.0-dev", "VERSION file should exist and contain a real version"
    # Version should be semver-like (X.Y.Z)
    parts = version.split(".")
    assert len(parts) >= 2, f"Version '{version}' should have at least major.minor"
    assert all(p.isdigit() for p in parts), f"Version '{version}' should be numeric segments"


def test_set_params():
    """Setting parameters should update engine state."""
    e = AirLLMEngine()
    e.set_params(max_new_tokens=1024, temperature=0.8)
    assert e.max_new_tokens == 1024
    assert e.temperature == 0.8


def test_set_params_reset():
    """Resetting to original values should work."""
    e = AirLLMEngine(max_new_tokens=256, temperature=0.5)
    assert e.max_new_tokens == 256
    assert e.temperature == 0.5
    e.set_params(max_new_tokens=512, temperature=1.0)
    assert e.max_new_tokens == 512
    assert e.temperature == 1.0


def test_build_prompt():
    """Prompt builder should format messages correctly."""
    e = AirLLMEngine()
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello"},
    ]
    prompt = e._build_prompt(messages)
    assert "SYSTEM: You are helpful." in prompt
    assert "USER: Hello" in prompt
    assert prompt.endswith("ASSISTANT:")


def test_build_prompt_multi_turn():
    """Multi-turn conversations should preserve history."""
    e = AirLLMEngine()
    messages = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
        {"role": "user", "content": "How are you?"},
    ]
    prompt = e._build_prompt(messages)
    assert "USER: Hi" in prompt
    assert "ASSISTANT: Hello!" in prompt
    assert "USER: How are you?" in prompt


def test_generate_without_model():
    """Generate without model should return error string."""
    e = AirLLMEngine()
    result = e.generate([{"role": "user", "content": "test"}], stream=False)
    assert isinstance(result, str), f"Expected str, got {type(result).__name__}"
    assert "No model loaded" in result


def test_stream_without_model():
    """Stream without model should yield error string."""
    e = AirLLMEngine()
    tokens = list(e.generate([{"role": "user", "content": "test"}], stream=True))
    assert len(tokens) == 1
    assert "No model loaded" in tokens[0]


def test_cancel_generation():
    """Cancel flag should be settable."""
    e = AirLLMEngine()
    assert e._cancel_requested is False
    e.cancel_generation()
    assert e._cancel_requested is True


def test_unload_no_model():
    """Unloading when no model is loaded should be safe."""
    e = AirLLMEngine()
    result = e.unload_model()
    assert result["status"] == "ok"
