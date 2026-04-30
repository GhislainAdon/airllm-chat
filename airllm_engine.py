"""
AirLLM Engine Wrapper
Handles model loading and text generation using the airllm library.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import threading
import time
from typing import Generator, Optional, Union

try:
    import psutil
except ImportError:
    psutil = None


def _read_version() -> str:
    """Read version from VERSION file next to this script."""
    version_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "VERSION")
    try:
        with open(version_file, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        return "0.0.0-dev"


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
        self._ollama_models_dir = os.path.expanduser("~/.ollama/models")

    # ------------------------------------------------------------------ #
    #  Ollama model management
    # ------------------------------------------------------------------ #

    def _run_ollama(self, *args, timeout: int = 120) -> subprocess.Popen:
        """Run an ``ollama`` CLI command and return the Popen object."""
        return subprocess.Popen(
            ["ollama", *args],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

    def get_ollama_models(self) -> dict:
        """List models installed in Ollama.

        Returns a dict with ``models`` (list[dict]) and an optional ``message``.
        Each model dict contains: name, size, modified, family, quantization.
        """
        if not shutil.which("ollama"):
            return {"models": [], "message": "Ollama is not installed or not on PATH."}

        try:
            proc = self._run_ollama("list")
            stdout, _ = proc.communicate(timeout=30)
            if proc.returncode != 0:
                return {"models": [], "message": f"ollama list failed: {stdout.strip()}"}

            models: list[dict] = []
            lines = stdout.strip().splitlines()
            # First line is the header: "NAME ID SIZE MODIFIED ..."
            for line in lines[1:]:
                parts = line.split()
                if not parts:
                    continue
                models.append({
                    "name": parts[0],
                    "id": parts[1] if len(parts) > 1 else "",
                    "size": parts[2] if len(parts) > 2 else "",
                    "modified": parts[3] if len(parts) > 3 else "",
                    "family": parts[4] if len(parts) > 4 else "",
                    "quantization": parts[5] if len(parts) > 5 else "",
                })
            return {"models": models}
        except subprocess.TimeoutExpired:
            return {"models": [], "message": "ollama list timed out."}
        except Exception as exc:
            return {"models": [], "message": f"Error listing models: {exc}"}

    def pull_ollama_model(self, model_name: str) -> Generator[dict, None, None]:
        """Pull a model from the Ollama registry.

        Yields progress dicts (``{"status": "...", "digest": ...}``) as the
        download proceeds so the caller can stream them to the client.
        """
        if not shutil.which("ollama"):
            yield {"status": "error", "message": "Ollama is not installed or not on PATH."}
            return

        try:
            proc = self._run_ollama("pull", model_name)
            for raw_line in proc.stdout:
                line = raw_line.strip()
                if not line:
                    continue
                # Each line from ``ollama pull`` is a JSON object
                try:
                    import json
                    data = json.loads(line)
                    yield data
                except (json.JSONDecodeError, ValueError):
                    yield {"status": "pulling", "message": line}

            proc.wait(timeout=600)
            if proc.returncode == 0:
                yield {"status": "success", "message": f"Model '{model_name}' pulled successfully."}
            else:
                yield {"status": "error", "message": f"ollama pull failed with exit code {proc.returncode}."}
        except subprocess.TimeoutExpired:
            yield {"status": "error", "message": "ollama pull timed out."}
        except Exception as exc:
            yield {"status": "error", "message": str(exc)}

    def delete_ollama_model(self, model_name: str) -> dict:
        """Delete a model from Ollama via ``ollama rm``."""
        if not shutil.which("ollama"):
            return {"status": "error", "message": "Ollama is not installed or not on PATH."}

        try:
            proc = self._run_ollama("rm", model_name)
            stdout, _ = proc.communicate(timeout=60)
            if proc.returncode == 0:
                return {"status": "success", "message": f"Model '{model_name}' deleted."}
            return {"status": "error", "message": f"Failed to delete model: {stdout.strip()}"}
        except subprocess.TimeoutExpired:
            return {"status": "error", "message": "ollama rm timed out."}
        except Exception as exc:
            return {"status": "error", "message": str(exc)}

    def get_ollama_model_path(self, model_name: str) -> dict:
        """Return the on-disk GGUF blob path for an Ollama model.

        Uses ``ollama show --modelfile <model>`` and parses the ``FROM`` line
        which points to the digest-based blob inside ``~/.ollama/models/``.
        """
        if not shutil.which("ollama"):
            return {"status": "error", "message": "Ollama is not installed or not on PATH."}

        try:
            # ``ollama show`` outputs the Modelfile which includes a FROM <sha256>
            proc = self._run_ollama("show", "--modelfile", model_name)
            stdout, _ = proc.communicate(timeout=30)
            if proc.returncode != 0:
                return {"status": "error", "message": f"ollama show failed: {stdout.strip()}"}

            blob_digest = None
            for line in stdout.splitlines():
                stripped = line.strip()
                if stripped.upper().startswith("FROM "):
                    blob_digest = stripped[5:].strip()
                    break

            if not blob_digest:
                return {"status": "error", "message": "Could not determine model digest from ollama show output."}

            # The blob lives at ~/.ollama/models/manifests/<registry>/<model>/<digest>
            # but the actual GGUF blob is in ~/.ollama/models/blobs/sha256/<digest>
            models_dir = self._ollama_models_dir
            blob_path = os.path.join(models_dir, "blobs", "sha256", blob_digest)

            if not os.path.isfile(blob_path):
                # Fallback: also check with the full sha256 prefix
                if not blob_digest.startswith("sha256:"):
                    blob_path2 = os.path.join(models_dir, "blobs", "sha256", blob_digest)
                else:
                    blob_path2 = os.path.join(models_dir, "blobs", blob_digest)

                if os.path.isfile(blob_path2):
                    blob_path = blob_path2
                else:
                    return {
                        "status": "error",
                        "message": f"Blob file not found. Expected at: {blob_path}",
                        "digest": blob_digest,
                    }

            return {
                "status": "success",
                "model": model_name,
                "digest": blob_digest,
                "path": blob_path,
                "size_bytes": os.path.getsize(blob_path),
            }
        except subprocess.TimeoutExpired:
            return {"status": "error", "message": "ollama show timed out."}
        except Exception as exc:
            return {"status": "error", "message": str(exc)}

    def get_ollama_info(self) -> dict:
        """Return general Ollama status information."""
        ollama_installed = shutil.which("ollama") is not None
        ollama_version = ""
        ollama_running = False

        if ollama_installed:
            # Version
            try:
                proc = self._run_ollama("--version")
                stdout, _ = proc.communicate(timeout=10)
                ollama_version = stdout.strip()
            except Exception:
                ollama_version = "unknown"

            # Running check: try a quick ``ollama list`` (non-blocking)
            try:
                proc = self._run_ollama("list")
                stdout, _ = proc.communicate(timeout=5)
                ollama_running = proc.returncode == 0
            except Exception:
                ollama_running = False

        models_result = self.get_ollama_models() if ollama_running else {"models": []}
        models_count = len(models_result.get("models", []))

        return {
            "ollama_installed": ollama_installed,
            "ollama_running": ollama_running,
            "ollama_version": ollama_version,
            "models_dir": self._ollama_models_dir,
            "models_count": models_count,
        }

    # ------------------------------------------------------------------ #
    #  System information
    # ------------------------------------------------------------------ #

    def get_system_info(self) -> dict:
        """Return hardware / system information (GPU, RAM, CPU)."""
        info: dict = {
            "gpu_name": "",
            "gpu_memory_total": 0,
            "gpu_memory_used": 0,
            "gpu_memory_free": 0,
            "ram_total": 0,
            "ram_used": 0,
            "ram_free": 0,
            "cpu_count": 0,
        }

        # --- GPU via torch ---
        try:
            import torch
            if torch.cuda.is_available():
                info["gpu_name"] = torch.cuda.get_device_name(0)
                total = torch.cuda.get_device_properties(0).total_mem
                # Use memory_stats() when available for more accurate usage
                try:
                    stats = torch.cuda.memory_stats(0)
                    info["gpu_memory_used"] = stats.get("allocated_bytes.all.current", 0)
                    info["gpu_memory_free"] = total - info["gpu_memory_used"]
                except Exception:
                    info["gpu_memory_used"] = torch.cuda.memory_allocated(0)
                    info["gpu_memory_free"] = total - info["gpu_memory_used"]
                info["gpu_memory_total"] = total
        except ImportError:
            pass

        # --- RAM & CPU via psutil ---
        if psutil is not None:
            try:
                vm = psutil.virtual_memory()
                info["ram_total"] = vm.total
                info["ram_used"] = vm.used
                info["ram_free"] = vm.available
                info["cpu_count"] = psutil.cpu_count(logical=True) or 0
            except Exception:
                pass
        else:
            info["cpu_count"] = os.cpu_count() or 0

        return info

    # ------------------------------------------------------------------ #
    #  Original airllm inference methods (unchanged)
    # ------------------------------------------------------------------ #

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
            "version": _read_version(),
        }
