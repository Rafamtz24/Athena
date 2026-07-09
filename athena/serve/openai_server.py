"""OpenAI-compatible HTTP server backed by Athena's resident reasoning model.

Builds a small FastAPI app that wraps the already-loaded ``llama_cpp.Llama``
instance held by a :class:`LlamaCppProvider`. Because that instance is loaded
once and reused, external clients share the same model the terminal uses — no
second copy is loaded into memory.

Concurrency: ``llama_cpp.Llama`` is single-threaded and not re-entrant, so all
generation is serialized behind a lock. Requests that arrive while another is
generating simply wait their turn.

Only the subset of the OpenAI API that chat frontends need is implemented:

    GET  /v1/models            — advertise the single served model
    POST /v1/chat/completions  — chat completion (streaming and non-streaming)

llama-cpp-python already returns OpenAI-shaped payloads from
``create_chat_completion``, so both handlers are mostly pass-throughs.
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


def build_app(provider):
    """Build a FastAPI app serving ``provider``'s resident model.

    Args:
        provider: A LlamaCppProvider (or any provider exposing ``.model`` as a
            ``llama_cpp.Llama`` and ``.model_name`` / ``.max_tokens`` /
            ``.temperature``).

    Returns:
        A configured FastAPI application.
    """
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
