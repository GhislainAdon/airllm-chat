"""
AirLLM Engine Wrapper
Handles model loading and text generation using the airllm library.
"""

from __future__ import annotations

import threading
import time
from typing import Generator, Optional, Union


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
        self._cancel_requested = False

    def load_model(self) -> dict:
        """Load the airllm model. Returns status dict."""
        if self._loaded:
            return {"status": "ok", "message": "Model already loaded."}
        try:
            import airllm
            self.model_path = self.model_path or "garage-bAInd/Platypus2-70B-instruct"
            print(f"  [airllm] Loading model: {self.model_path}")
            self.model = airllm.AutoModelForCausalLM.from_pretrained(self.model_path)
            self._loaded = True
            self._load_error = None
            return {"status": "ok", "message": f"Model loaded: {self.model_path}"}
        except Exception as e:
            self._load_error = str(e)
            return {"status": "error", "message": f"Failed to load model: {e}"}

    def unload_model(self) -> dict:
        """Unload the model and free GPU memory."""
        with self._lock:
            if self.model is not None:
                try:
                    del self.model
                except Exception:
                    pass
                self.model = None
                self._loaded = False
                self.model_path = ""
                # Force GPU memory cleanup
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:
                    pass
                return {"status": "ok", "message": "Model unloaded."}
            return {"status": "ok", "message": "No model to unload."}

    def cancel_generation(self) -> None:
        """Request cancellation of the current generation."""
        self._cancel_requested = True

    def generate(
        self, messages: list[dict], stream: bool = False
    ) -> Union[Generator[str, None, None], str]:
        """
        Generate a response from the model.
        messages: list of {"role": "user"|"assistant", "content": "..."}
        stream: if True, yields tokens as they are generated.
        """
        if stream:
            return self._generate_stream(messages)
        return self._generate_full(messages)

    def _generate_full(self, messages: list[dict]) -> str:
        """Non-streaming generation — returns a plain string."""
        if not self._loaded or self.model is None:
            return "[ERROR] No model loaded. Please load a model first."

        self._cancel_requested = False

        with self._lock:
            try:
                prompt = self._build_prompt(messages)
                kwargs = {
                    "max_new_tokens": self.max_new_tokens,
                    "temperature": self.temperature,
                }
                return self._full_generate(prompt, **kwargs)
            except Exception as e:
                return f"[ERROR] Generation failed: {e}"

    def _generate_stream(self, messages: list[dict]) -> Generator[str, None, None]:
        """Streaming generation — yields tokens one by one."""
        if not self._loaded or self.model is None:
            yield "[ERROR] No model loaded. Please load a model first."
            return

        self._cancel_requested = False

        with self._lock:
            try:
                prompt = self._build_prompt(messages)
                kwargs = {
                    "max_new_tokens": self.max_new_tokens,
                    "temperature": self.temperature,
                }
                for token in self._stream_generate(prompt, **kwargs):
                    if self._cancel_requested:
                        break
                    yield token
            except Exception as e:
                yield f"[ERROR] Generation failed: {e}"

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
        try:
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
        except (TypeError, AttributeError):
            # Fallback for older airllm: non-streaming then chunk
            result = self._full_generate(prompt, **kwargs)
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

    def set_params(
        self,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        model_path: Optional[str] = None,
    ) -> dict:
        """Update generation parameters. If model_path changes, reloads the model."""
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
            "version": "1.0.0",
        }
