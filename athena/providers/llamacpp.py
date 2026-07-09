"""
LlamaCpp Provider

Provides local GGUF model inference using llama-cpp-python.
No HTTP, no external server, no LM Studio required.
"""

import re

from athena.config.settings import get_settings


def _strip_reasoning(text: str) -> str:
    """Remove <think>…</think> reasoning traces emitted by thinking models.

    Thinking models (e.g. Qwen3) wrap their internal chain-of-thought in
    <think>…</think>. Some chat templates open the tag for the model, so the
    output may contain only a trailing </think>. This handles all cases:

        - complete <think>…</think> blocks are removed
        - a stray leading </think> (opening tag suppressed by the template)
          drops everything up to and including it
        - an unterminated <think> (generation truncated mid-thought) drops
          everything from the tag onward

    Args:
        text: Raw model output.

    Returns:
        The user-facing answer with reasoning traces removed.
    """
    if not text:
        return text

    # Remove complete <think>…</think> blocks anywhere in the text.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)

    # A stray closing tag means the opening was emitted by the template;
    # everything before it is reasoning.
    lowered = text.lower()
    if "</think>" in lowered:
        text = text[lowered.rindex("</think>") + len("</think>"):]

    # An unterminated opening tag means thinking was cut off before an answer.
    lowered = text.lower()
    if "<think>" in lowered:
        text = text[: lowered.index("<think>")]

    return text.strip()


class LlamaCppProvider:
    """
    A provider for local GGUF model inference via llama-cpp-python.

    Loads a GGUF model directly into memory and keeps it resident
    for the lifetime of the application. The model is never reloaded
    between prompts.

    Constructor accepts:
        model_path (str | None): Path to the GGUF model file.
            Defaults to {model_directory}/{reasoning_model} from settings.
        inference_config (InferenceConfiguration | None): Pre-computed
            inference parameters.  If None, defaults are used (CPU, 4 threads,
            512 batch, 4096 context).  The provider does NOT perform hardware
            detection itself — that is the caller's responsibility.
        temperature (float): Sampling temperature. Defaults to value from settings.
    """

    # Loaded models keyed by absolute model path. Sharing this cache across
    # instances means two providers pointed at the same file reuse one resident
    # model (e.g. when learning falls back to the reasoning model), while
    # distinct files each load once.
    _models: dict = {}

    def __init__(
        self,
        model_path: str | None = None,
        inference_config: "InferenceConfiguration | None" = None,
        temperature: float | None = None,
        label: str = "reasoning",
    ):
        settings = get_settings()

        if model_path:
            self.model_path = model_path
        else:
            from athena.providers.model_selector import resolve_model_path

            self.model_path = resolve_model_path(
                settings.provider.reason_model_directory
            )

        self.temperature = (
            temperature if temperature is not None else settings.provider.temperature
        )
        self.max_tokens = getattr(settings.provider, "max_tokens", 2048)

        # Inference configuration — supplied externally, never self-detected
        if inference_config is not None:
            self.n_ctx = inference_config.n_ctx
            self.n_threads = inference_config.n_threads
            self.n_batch = inference_config.n_batch
            self.gpu_layers = inference_config.gpu_layers
            self.backend = inference_config.backend
            self.flash_attn = getattr(inference_config, "flash_attn", True)
        else:
            self.n_ctx = 4096
            self.n_threads = 4
            self.n_batch = 512
            self.gpu_layers = 0
            self.backend = "CPU"
            self.flash_attn = True

        # Load model if not already resident (one load per distinct path).
        if self.model_path not in LlamaCppProvider._models:
            from llama_cpp import Llama

            print(f"Loading {label} model...")
            LlamaCppProvider._models[self.model_path] = Llama(
                model_path=self.model_path,
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
                n_batch=self.n_batch,
                n_gpu_layers=self.gpu_layers,
                flash_attn=self.flash_attn,
                verbose=False,
            )
            print("Model loaded.\n")

        self.model = LlamaCppProvider._models[self.model_path]

    @property
    def model_name(self) -> str:
        """The filename of the loaded GGUF model (without directory)."""
        from pathlib import Path

        return Path(self.model_path).name

    def generate(self, prompt: str, system: str | None = None) -> str:
        """
        Generate a response from the local GGUF model.

        Args:
            prompt: The input prompt string (user content).
            system: Optional system prompt. Delivered in the `system` role so
                the model treats it as its own identity/instructions rather
                than as a user claim. Without this, the base model's built-in
                system identity (e.g. "You are Qwen") stays in force.

        Returns:
            The LLM response text.

        Raises:
            RuntimeError: If the model fails to generate a response.
        """
        # Read the thinking toggle live so `/think on|off` takes effect on the
        # next call without reloading the model.
        thinking_enabled = getattr(get_settings().provider, "thinking_enabled", True)
        user_content = prompt if thinking_enabled else f"{prompt} /no_think"

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user_content})

        response = self.model.create_chat_completion(
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        try:
            content = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            raise RuntimeError(
                f"LlamaCpp returned unexpected response format: {response}"
            ) from e

        return _strip_reasoning(content)

    def call(self, prompt: str) -> str:
        """Alias for generate() to match KnowledgeManager expectations."""
        return self.generate(prompt)

    def unload(self) -> bool:
        """Free this provider's resident GGUF model from memory.

        Drops the model from the shared cache, closes the underlying llama.cpp
        context (releasing VRAM/RAM), and clears this instance's reference. Used
        by serve mode to reclaim memory held by a dedicated learning model that
        is idle while only the reasoning model is being served.

        Returns:
            True if a model was unloaded; False if none was resident.
        """
        import gc

        model = LlamaCppProvider._models.pop(self.model_path, None)
        self.model = None
        if model is None:
            return False

        close = getattr(model, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                # Best-effort: even if close() fails, dropping references and
                # collecting still frees the bulk of the memory.
                pass
        del model
        gc.collect()
        return True

    def reload(self) -> None:
        """Reload this provider's model if it was previously unloaded.

        Reuses an already-resident model at the same path when present;
        otherwise loads it fresh with this instance's inference settings. This
        mutates ``self.model`` in place, so any component holding a reference to
        this provider transparently sees the restored model.
        """
        if self.model is not None:
            return

        if self.model_path in LlamaCppProvider._models:
            self.model = LlamaCppProvider._models[self.model_path]
            return

        from llama_cpp import Llama

        print("Reloading learning model...")
        LlamaCppProvider._models[self.model_path] = Llama(
            model_path=self.model_path,
            n_ctx=self.n_ctx,
            n_threads=self.n_threads,
            n_batch=self.n_batch,
            n_gpu_layers=self.gpu_layers,
            flash_attn=self.flash_attn,
            verbose=False,
        )
        self.model = LlamaCppProvider._models[self.model_path]
        print("Learning model reloaded.\n")

    def count_tokens(self, text: str) -> int:
        """Count tokens using the model's native tokenizer.

        Args:
            text: The text to tokenize and count.

        Returns:
            The number of tokens in the text.
        """
        tokens = self.model.tokenize(text.encode("utf-8"))
        return len(tokens)

    def get_context_window(self) -> int:
        """Get the configured context window size.

        Returns:
            The maximum context window in tokens (self.n_ctx).
        """
        return self.n_ctx
