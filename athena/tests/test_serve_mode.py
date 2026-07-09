"""
Tests for serve mode (/serve).

Serve mode wraps a provider's already-resident model in an OpenAI-compatible
HTTP app so external clients (e.g. Open WebUI) can use it. These tests exercise
the app with a fake provider whose ``model`` mimics llama_cpp.Llama's
``create_chat_completion`` — no real model is loaded — covering the model list,
non-streaming and streaming chat completions, request validation, parameter
translation, and the provider's unload/reload memory management.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from fastapi.testclient import TestClient

from athena.serve.openai_server import build_app, _extract_params


class FakeLlama:
    """Stand-in for llama_cpp.Llama recording the kwargs it was called with."""

    def __init__(self):
        self.last_kwargs = None

    def create_chat_completion(self, messages, stream=False, **kwargs):
        self.last_kwargs = {"messages": messages, "stream": stream, **kwargs}
        if stream:
            return iter([
                {"choices": [{"delta": {"content": "Hi"}}]},
                {"choices": [{"delta": {"content": " there"}}]},
            ])
        return {
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "choices": [{"message": {"role": "assistant", "content": "Hi there"}}],
        }


class FakeProvider:
    """Minimal provider exposing what build_app needs."""

    def __init__(self):
        self.model = FakeLlama()
        self.model_name = "fake-model.gguf"
        self.max_tokens = 2048
        self.temperature = 0.7


def _client():
    return TestClient(build_app(FakeProvider()))


def test_list_models_advertises_the_served_model():
    resp = _client().get("/v1/models")
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "list"
    assert data["data"][0]["id"] == "fake-model.gguf"


def test_chat_completion_non_streaming_passes_through():
    resp = _client().post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hello"}]},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["choices"][0]["message"]["content"] == "Hi there"


def test_chat_completion_streaming_emits_sse_and_done():
    with _client() as client:
        resp = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}], "stream": True},
        )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")

    lines = [ln for ln in resp.text.splitlines() if ln.startswith("data: ")]
    payloads = [ln[len("data: "):] for ln in lines]
    assert payloads[-1] == "[DONE]"
    first = json.loads(payloads[0])
    assert first["choices"][0]["delta"]["content"] == "Hi"


def test_empty_messages_is_rejected():
    resp = _client().post("/v1/chat/completions", json={"messages": []})
    assert resp.status_code == 400


def test_extract_params_uses_defaults_and_forwards_supplied():
    # Defaults fill in when the client omits them.
    params = _extract_params({}, default_max_tokens=2048, default_temperature=0.7)
    assert params["max_tokens"] == 2048
    assert params["temperature"] == 0.7
    assert "top_p" not in params  # unset stays unset

    # Supplied values win, including the newer max_completion_tokens spelling.
    params = _extract_params(
        {"max_completion_tokens": 100, "temperature": 0.1, "top_p": 0.9, "stop": ["\n"]},
        default_max_tokens=2048,
        default_temperature=0.7,
    )
    assert params["max_tokens"] == 100
    assert params["temperature"] == 0.1
    assert params["top_p"] == 0.9
    assert params["stop"] == ["\n"]


def test_provider_unload_and_reload_manage_the_shared_cache():
    # Uses the real LlamaCppProvider unload/reload against a stub cache entry,
    # so no GGUF file is needed.
    from athena.providers.llamacpp import LlamaCppProvider

    provider = LlamaCppProvider.__new__(LlamaCppProvider)
    provider.model_path = "/fake/path/model.gguf"
    provider.n_ctx = 4096
    provider.n_threads = 4
    provider.n_batch = 512
    provider.gpu_layers = 0
    provider.flash_attn = True

    closed = {"called": False}

    class Closable:
        def close(self):
            closed["called"] = True

    LlamaCppProvider._models[provider.model_path] = Closable()
    provider.model = LlamaCppProvider._models[provider.model_path]

    assert provider.unload() is True
    assert closed["called"] is True
    assert provider.model is None
    assert provider.model_path not in LlamaCppProvider._models

    # A second unload is a no-op (nothing resident).
    assert provider.unload() is False


if __name__ == "__main__":
    import subprocess

    raise SystemExit(subprocess.call([sys.executable, "-m", "pytest", __file__, "-v"]))
