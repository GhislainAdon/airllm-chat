---
Task ID: 1
Agent: Main Agent
Task: Design review, QA review, and comprehensive unit tests for airllm-chat

Work Log:
- Explored all source files in the airllm-chat project (app.py, airllm_engine.py, static files, templates, workflows, build scripts)
- Studied Ollama's GitHub repo architecture for inspiration on patterns
- Performed design review identifying 10 architectural improvements inspired by Ollama
- Performed QA review identifying bugs, security issues, and edge cases
- Wrote 197 comprehensive unit tests (93 for engine, 93 for app, 11 existing smoke tests)
- All tests pass without GPU/airllm/torch/ollama dependencies
- Committed and pushed to GitHub (GhislainAdon/airllm-chat)

Stage Summary:
- tests/test_airllm_engine.py: 93 tests (14 test classes)
- tests/test_app.py: 93 tests (17 test classes)
- 197 total tests passing
- Key findings documented below

---
## Design Review Summary (Inspired by Ollama Architecture)

### Top 10 Architectural Improvements Recommended:

1. **Middleware adapter for OpenAI compat** - Create middleware that transforms OpenAI requests → internal format. Currently OpenAI logic is baked into route handlers (violates SRP).

2. **Scheduler with ref-counted models** - Implement model registry with reference counting and auto-unload after idle timeout. Currently only one model can be loaded at a time.

3. **Structured error types** - Define error classes carrying HTTP status codes. Currently errors are plain strings or dicts without status differentiation.

4. **NDJSON streaming for native API** - Use newline-delimited JSON for /api/ endpoints, SSE only for /v1/ endpoints (Ollama pattern).

5. **Environment variable config** - Lazy-evaluated config functions (AIRLLM_HOST, AIRLLM_PORT, AIRLLM_KEEP_ALIVE, etc.) with sensible defaults.

6. **Host-based access control** - Default to localhost-only access. Validate request Host headers to prevent DNS rebinding.

7. **Abstract LLM server interface** - Define Python ABC/Protocol for inference backend to enable swapping between llama.cpp, MLX, etc.

8. **Keep-alive / explicit unload** - Support keep_alive parameter to control model memory lifetime.

9. **Table-driven API tests** - Already implemented in the new test suite.

10. **Multi-API compatibility via middleware** - Same handler serves multiple API formats through middleware transformation.

---
## QA Review Summary

### Critical Issues:
1. **Race condition in _cancel_requested**: The cancel flag is read/written from multiple threads without synchronization. Should use threading.Event or Lock.
2. **Streaming lock held during generation**: _generate_stream holds self._lock while yielding tokens, blocking all other operations for the entire generation duration.

### Security Issues:
3. **Path traversal incomplete**: '..' check is too simplistic. Can be bypassed with URL encoding or symlinks. Should use os.path.realpath() + prefix check.
4. **CORS allows all origins**: Access-Control-Allow-Origin: * is fine for localhost but dangerous if bound to 0.0.0.0.
5. **No request rate limiting**: Server is vulnerable to DoS.
6. **Error messages leak internal details**: Exception messages may contain file paths, stack info.

### Medium Issues:
7. **No timeout on _generate_full**: If model hangs, server is blocked forever (lock is held).
8. **Duplicate _read_version()**: Two copies in app.py and airllm_engine.py - should be a shared utility.
9. **Hardcoded model list**: /api/models and /v1/models have hardcoded model IDs that don't match actual available models.
10. **No request queue limit**: Unlimited concurrent requests can exhaust memory.
11. **Ollama path API allows directory override**: /api/ollama/path sets _ollama_models_dir without validation of what's inside.

### Minor Issues:
12. **Global mutable state**: `engine` is a module-level global - makes testing and multi-instance impossible.
13. **Missing input validation on temperature/max_tokens in /api/settings**: Only validated in /v1/chat/completions.
14. **SSE finish_reason "cancelled" is non-standard**: OpenAI spec only defines "stop", "length", "content_filter".
15. **Unused `import json` inside pull_ollama_model**: Line 122 imports json inside the method (already imported at top level).
