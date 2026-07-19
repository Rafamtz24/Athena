"""OpenAI-compatible HTTP server backed by Athena's resident reasoning model.

Either way, external clients share the model the terminal is already using — no
second copy is loaded into memory.

Both local providers are supported, because they hold a model in different
shapes:

``llamaserver`` (the default)
    Already runs llama.cpp's own OpenAI-compatible server, on a random loopback
    port behind a per-session API key. Serving is therefore a thin proxy: it
    republishes that endpoint on a stable, key-free port a client can be pointed
    at. Nothing is re-implemented; requests and SSE frames are passed through.

``llamacpp`` (in-process)
    Holds a ``llama_cpp.Llama`` object with no HTTP surface of its own, so the
    endpoints are implemented here on top of ``create_chat_completion``. That
    object is single-threaded and not re-entrant, so generation is serialized
    behind a lock; requests arriving mid-generation wait their turn.

Only the subset of the OpenAI API that chat frontends need is implemented:

    GET  /v1/models            — advertise the single served model
    POST /v1/chat/completions  — chat completion (streaming and non-streaming)
"""

import json
import threading
import time


def _extract_params(body: dict, default_max_tokens: int, default_temperature: float) -> dict:
    """Translate an OpenAI request body into create_chat_completion kwargs.

    Only forwards parameters the client actually supplied (falling back to the
    provider's defaults for the essentials), so unset fields keep llama.cpp's
    own defaults instead of being clobbered with ``None``.
    """
    params: dict = {}

    # max_completion_tokens is the newer spelling; accept both.
    max_tokens = body.get("max_tokens", body.get("max_completion_tokens"))
    params["max_tokens"] = max_tokens if max_tokens is not None else default_max_tokens

    temperature = body.get("temperature")
    params["temperature"] = temperature if temperature is not None else default_temperature

    for key in ("top_p", "top_k", "presence_penalty", "frequency_penalty", "seed", "stop"):
        if body.get(key) is not None:
            params[key] = body[key]

    return params


def can_serve(provider) -> bool:
    """Whether ``provider`` exposes a model serve mode can publish.

    True for both local providers; false for ones that are already a remote
    endpoint (e.g. LM Studio), where the client should be pointed at that
    endpoint directly instead of chaining a proxy through Athena.
    """
    return hasattr(provider, "upstream") or hasattr(provider, "model")


def build_app(provider):
    """Build a FastAPI app serving ``provider``'s resident model.

    Args:
        provider: A local provider — either one exposing ``.upstream`` (an
            already-running OpenAI-compatible server to proxy) or one exposing
            ``.model`` as a ``llama_cpp.Llama`` to wrap. Both must also provide
            ``.model_name`` / ``.max_tokens`` / ``.temperature``.

    Returns:
        A configured FastAPI application.

    Raises:
        TypeError: If the provider offers neither shape.
    """
    if hasattr(provider, "upstream"):
        return _build_proxy_app(provider)
    if hasattr(provider, "model"):
        return _build_inprocess_app(provider)
    raise TypeError(
        f"{type(provider).__name__} exposes no local model to serve."
    )


def _build_proxy_app(provider):
    """Republish a provider's existing OpenAI endpoint on a public port.

    llama-server binds to a random loopback port and requires a per-session key,
    neither of which an external client can be expected to know. This forwards
    to it, injecting the key, so the user only ever sees the stable port.
    """
    import requests
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, StreamingResponse
    from starlette.concurrency import run_in_threadpool

    app = FastAPI(title="Athena", docs_url=None, redoc_url=None)

    base_url, auth_headers = provider.upstream
    model_id = provider.model_name

    # Generation is slow on a partially-offloaded model and has no meaningful
    # upper bound, so only the connection gets a timeout, never the read.
    _CONNECT_TIMEOUT = (10, None)

    def _model_card() -> dict:
        return {
            "id": model_id,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "athena",
        }

    @app.get("/v1/models")
    async def list_models():
        return {"object": "list", "data": [_model_card()]}

    @app.get("/")
    async def root():
        return {"status": "ok", "service": "athena", "model": model_id}

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(
                {"error": {"message": "Request body must be valid JSON.", "type": "invalid_request_error"}},
                status_code=400,
            )

        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            return JSONResponse(
                {"error": {"message": "'messages' must be a non-empty array.", "type": "invalid_request_error"}},
                status_code=400,
            )

        # Clients echo back whatever id they were given, but some send their own
        # label. Only one model is loaded, so pin it to the served alias rather
        # than letting a mismatch surface as an upstream error.
        body["model"] = model_id

        url = f"{base_url}/v1/chat/completions"

        if bool(body.get("stream", False)):
            def event_stream():
                # Pass SSE frames through byte-for-byte: llama-server already
                # emits OpenAI-shaped chunks and its own "[DONE]" sentinel, so
                # re-framing them here would only risk corrupting them.
                with requests.post(
                    url, json=body, headers=auth_headers,
                    stream=True, timeout=_CONNECT_TIMEOUT,
                ) as upstream_response:
                    for chunk in upstream_response.iter_content(chunk_size=None):
                        yield chunk

            return StreamingResponse(event_stream(), media_type="text/event-stream")

        def forward():
            return requests.post(
                url, json=body, headers=auth_headers, timeout=_CONNECT_TIMEOUT
            )

        # Keep the blocking call off the event loop so the server stays responsive.
        try:
            upstream_response = await run_in_threadpool(forward)
        except requests.RequestException as exc:
            return JSONResponse(
                {"error": {"message": f"Upstream model server unreachable: {exc}",
                           "type": "upstream_error"}},
                status_code=502,
            )

        return JSONResponse(
            upstream_response.json(), status_code=upstream_response.status_code
        )

    return app


def _build_inprocess_app(provider):
    """Implement the endpoints directly over an in-process ``llama_cpp.Llama``."""
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, StreamingResponse
    from starlette.concurrency import run_in_threadpool

    app = FastAPI(title="Athena", docs_url=None, redoc_url=None)

    llm = provider.model
    model_id = provider.model_name
    # Serializes access to the single, non-thread-safe Llama instance.
    lock = threading.Lock()

    def _model_card() -> dict:
        return {
            "id": model_id,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "athena",
        }

    @app.get("/v1/models")
    async def list_models():
        return {"object": "list", "data": [_model_card()]}

    @app.get("/")
    async def root():
        return {"status": "ok", "service": "athena", "model": model_id}

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(
                {"error": {"message": "Request body must be valid JSON.", "type": "invalid_request_error"}},
                status_code=400,
            )

        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            return JSONResponse(
                {"error": {"message": "'messages' must be a non-empty array.", "type": "invalid_request_error"}},
                status_code=400,
            )

        params = _extract_params(body, provider.max_tokens, provider.temperature)
        stream = bool(body.get("stream", False))

        if stream:
            def event_stream():
                # Hold the lock for the whole generation so concurrent requests
                # wait rather than corrupting the shared model. The `with` block
                # also releases on client disconnect (GeneratorExit at yield).
                with lock:
                    for chunk in llm.create_chat_completion(
                        messages=messages, stream=True, **params
                    ):
                        yield f"data: {json.dumps(chunk)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(event_stream(), media_type="text/event-stream")

        def generate():
            with lock:
                return llm.create_chat_completion(messages=messages, stream=False, **params)

        # Run the blocking call off the event loop so the server stays responsive.
        result = await run_in_threadpool(generate)
        return JSONResponse(result)

    return app
