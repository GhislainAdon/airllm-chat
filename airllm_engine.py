"""
airllm Engine Wrapper
Handles model loading and text generation using the airllm library.
"""

import threading
import time
from typing import Generator, Optional


class AirLLMEngine:
    """Wraps airllm for chat-style inference with streaming support."""

    def __init__(self, model_path: str = "", max_new_tokens: int = 512, temperature: float = 0.7):
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.model = None
        self._lock = threading.Lock()
        self._loaded = False
        self._load_error: Optional[str] = None

    def load_model(self) -> dict:
        """Load the airllm model. Returns status dict."""
        if self._loaded:
            return {"status": "ok", "message": "Model already loaded."}
        try:
            import airllm
            self.model_path = self.model_path or "garage-bAInd/Platypus2-70B-instruct"
            self.model = airllm.AutoModelForCausalLM.from_pretrained(self.model_path)
            self._loaded = True
            return {"status": "ok", "message": f"Model loaded: {self.model_path}"}
        except Exception as e:
            self._load_error = str(e)
            return {"status": "error", "message": f"Failed to load model: {e}"}

    def generate(self, messages: list[dict], stream: bool = False) -> Generator[str, None, None] | str:
        """
        Generate a response from the model.
        messages: list of {"role": "user"|"assistant", "content": "..."}
        stream: if True, yields tokens as they are generated.
        """
        if not self._loaded or self.model is None:
            if stream:
                yield "[ERROR] No model loaded. Please load a model first."
                return
            return "[ERROR] No model loaded. Please load a model first."

        with self._lock:
            try:
                prompt = self._build_prompt(messages)
                generation_kwargs = {
                    "max_new_tokens": self.max_new_tokens,
                    "temperature": self.temperature,
                }

                if stream:
                    for token in self._stream_generate(prompt, **generation_kwargs):
                        yield token
                else:
                    result = self._full_generate(prompt, **generation_kwargs)
                    return result

            except Exception as e:
                error_msg = f"[ERROR] Generation failed: {e}"
                if stream:
                    yield error_msg
                    return
                return error_msg

    def _build_prompt(self, messages: list[dict]) -> str:
        """Build a chat prompt from message history."""
        prompt_parts = []
        system_msg = None
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                system_msg = content
            elif role == "user":
                prompt_parts.append(f"USER: {content}")
            elif role == "assistant":
                prompt_parts.append(f"ASSISTANT: {content}")

        if system_msg:
            prompt_parts.insert(0, f"SYSTEM: {system_msg}")

        prompt_parts.append("ASSISTANT:")
        return "\n".join(prompt_parts)

    def _stream_generate(self, prompt: str, **kwargs) -> Generator[str, None, None]:
        """Stream tokens from the model."""
        # airllm uses a generator-style API
        try:
            # airllm v0.2+ supports streaming via generate()
            input_ids = self.model.tokenize(prompt)
            prev_output = ""
            for output in self.model.generate(
                input_ids,
                max_new_tokens=kwargs.get("max_new_tokens", 512),
                temperature=kwargs.get("temperature", 0.7),
            ):
                if isinstance(output, str):
                    new_text = output[len(prev_output):] if output.startswith(prev_output) else output
                    if new_text:
                        yield new_text
                        prev_output = output
                elif isinstance(output, dict):
                    text = output.get("text", "")
                    if text:
                        yield text
        except TypeError:
            # Fallback for older airllm: non-streaming then chunk
            result = self._full_generate(prompt, **kwargs)
            # Simulate streaming by yielding word-by-word
            words = result.split(" ")
            for i, word in enumerate(words):
                token = word if i == 0 else " " + word
                yield token
                time.sleep(0.01)

    def _full_generate(self, prompt: str, **kwargs) -> str:
        """Full (non-streaming) generation."""
        try:
            input_ids = self.model.tokenize(prompt)
            output = self.model.generate(
                input_ids,
                max_new_tokens=kwargs.get("max_new_tokens", 512),
                temperature=kwargs.get("temperature", 0.7),
            )
            if isinstance(output, str):
                return output
            elif isinstance(output, dict):
                return output.get("text", str(output))
            elif isinstance(output, list):
                return " ".join(str(o) for o in output)
            return str(output)
        except Exception as e:
            return f"[ERROR] Generation exception: {e}"

    def set_params(self, max_new_tokens: int = None, temperature: float = None, model_path: str = None):
        """Update generation parameters."""
        if max_new_tokens is not None:
            self.max_new_tokens = max_new_tokens
        if temperature is not None:
            self.temperature = temperature
        if model_path is not None and model_path != self.model_path:
            self.model_path = model_path
            self._loaded = False
            self.model = None
            return self.load_model()
        return {"status": "ok", "message": "Parameters updated."}

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def info(self) -> dict:
        return {
            "model_path": self.model_path,
            "loaded": self._loaded,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "error": self._load_error,
        }
