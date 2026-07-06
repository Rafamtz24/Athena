"""
LlamaCpp Provider

Provides local GGUF model inference using llama-cpp-python.
No HTTP, no external server, no LM Studio required.
"""

from pathlib import Path

from athena.config.settings import get_settings


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

    _model = None
    _model_path = None

    def __init__(
        self,
        model_path: str | None = None,
        inference_config: "InferenceConfiguration | None" = None,
        temperature: float | None = None,
    ):
        settings = get_settings()

        if model_path:
            self.model_path = model_path
        else:
            self.model_path = str(
                Path(settings.provider.model_directory)
                / settings.provider.reasoning_model
            )

        self.temperature = (
            temperature if temperature is not None else settings.provider.temperature
        )

        # Inference configuration — supplied externally, never self-detected
        if inference_config is not None:
            self.n_ctx = inference_config.n_ctx
            self.n_threads = inference_config.n_threads
            self.n_batch = inference_config.n_batch
            self.gpu_layers = inference_config.gpu_layers
            self.backend = inference_config.backend
        else:
            self.n_ctx = 4096
            self.n_threads = 4
            self.n_batch = 512
            self.gpu_layers = 0
            self.backend = "CPU"

        # Load model if not already resident
        if LlamaCppProvider._model is None or LlamaCppProvider._model_path != self.model_path:
            from llama_cpp import Llama

            print("Loading reasoning model...")
            LlamaCppProvider._model = Llama(
                model_path=self.model_path,
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
                n_batch=self.n_batch,
                n_gpu_layers=self.gpu_layers,
                verbose=False,
            )
            LlamaCppProvider._model_path = self.model_path
            print("Model loaded.\n")

        self.model = LlamaCppProvider._model

    def generate(self, prompt: str) -> str:
        """
        Generate a response from the local GGUF model.

        Args:
            prompt: The input prompt string.

        Returns:
            The LLM response text.

        Raises:
            RuntimeError: If the model fails to generate a response.
        """
        response = self.model.create_chat_completion(
            messages=[
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
        )

        try:
            return response["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            raise RuntimeError(
                f"LlamaCpp returned unexpected response format: {response}"
            ) from e

    def call(self, prompt: str) -> str:
        """Alias for generate() to match KnowledgeManager expectations."""
        return self.generate(prompt)

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
