"""
LlamaServer Provider

Runs local GGUF models through llama.cpp's ``llama-server`` binary, which
Athena starts and stops as a child process. Nothing about it is visible to the
user: it looks and behaves exactly like in-process inference.

Why a subprocess instead of the llama-cpp-python bindings
---------------------------------------------------------
llama-cpp-python publishes no prebuilt Windows wheels, so installing it means
compiling llama.cpp locally — several GB of Visual Studio build tools and a
20-30 minute build. That is an unreasonable barrier for someone who just
cloned the repository.

llama.cpp itself, by contrast, ships prebuilt binaries for every backend,
including Vulkan for AMD and Intel GPUs. Setup downloads the right one (about
32 MB) and GPU acceleration works with no compiler involved. The trade-off is
talking to the model over local HTTP rather than in-process, which costs a
negligible amount of latency on a loopback connection.

The in-process :mod:`athena.providers.llamacpp` provider is still available for
anyone who prefers to compile; this one is the default because it works out of
the box.
"""

import atexit
import json
import os
import secrets
import socket
import subprocess
import tempfile
import time
from pathlib import Path

import requests

from athena.config.settings import get_settings
from athena.providers import reasoning_trace, streaming
from athena.providers.gguf import read_model_info
from athena.providers.llamacpp import ReasoningStream, _strip_reasoning


# Fraction of VRAM available for weights, leaving room for the KV cache and
# compute buffers. Matches the margin the in-process provider uses.
_VRAM_USABLE_FRACTION = 0.85


def _should_keep_experts_on_cpu(model_path: str, config) -> bool:
    """Whether to hand this model's expert tensors to the CPU (``--cpu-moe``).

    A Mixture-of-Experts model activates only a few experts per token, but the
    experts are nearly all of its weight. When the model is too big for VRAM,
    letting llama.cpp split it by layer puts a mix of attention and expert
    tensors on each side of the bus, and the GPU ends up waiting on expert
    weights streaming from system memory.

    Pinning every expert to the CPU instead keeps the dense, latency-critical
    attention path entirely on the GPU. Measured on an 8 GB RX 9060 XT with a
    21 GB 35B-A3B model: ~9-12 tok/s by layer split, ~26 tok/s this way.

    Only worth it when all three hold, hence the guards below:
      - there is a GPU at all,
      - the model actually has experts (on a dense model the flag does nothing
        useful and misrepresents intent),
      - the model does not already fit in VRAM — when it fits, forcing experts
        out to the CPU would give away the speed this is meant to gain.
    """
    if getattr(config, "backend", "CPU") == "CPU":
        return False

    vram_bytes = getattr(config, "vram_bytes", None)
    if not vram_bytes:
        return False

    if not read_model_info(model_path).get("expert_count"):
        return False

    try:
        model_bytes = Path(model_path).stat().st_size
    except OSError:
        return False

    return model_bytes > vram_bytes * _VRAM_USABLE_FRACTION


# How long to wait for a model to finish loading before giving up. Large models
# on slow disks genuinely take minutes on a cold cache, so this is generous.
_STARTUP_TIMEOUT_SECONDS = 600


def _kill_on_exit_job():
    """Return a Windows job object that kills its members when Athena dies.

    ``atexit`` only runs on a clean interpreter exit. Closing the console
    window terminates Python outright, so the ``llama-server`` children survive
    with the whole model still resident — gigabytes of RAM and VRAM held by a
    program the user has closed, until they notice and kill it by hand.

    A job object with ``JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`` fixes that at the
    OS level: the handle is owned by this process, so however Athena exits —
    clean quit, window close, or crash — Windows closes the handle and
    terminates every process in the job.

    Created once and cached; the handle is deliberately never closed, since
    holding it open for the process lifetime is the entire mechanism.

    Returns:
        A job handle, or None when unavailable (non-Windows, or the API failed
        — in which case ``atexit`` remains as the best-effort fallback).
    """
    global _JOB_HANDLE

    if _JOB_HANDLE is not _JOB_UNSET:
        return _JOB_HANDLE

    _JOB_HANDLE = None
    if os.name != "nt":
        return _JOB_HANDLE

    import ctypes
    from ctypes import wintypes

    class _IoCounters(ctypes.Structure):
        _fields_ = [(name, ctypes.c_ulonglong) for name in (
            "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
            "ReadTransferCount", "WriteTransferCount", "OtherTransferCount",
        )]

    class _BasicLimits(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_int64),
            ("PerJobUserTimeLimit", ctypes.c_int64),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.POINTER(ctypes.c_ulong)),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class _ExtendedLimits(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", _BasicLimits),
            ("IoInfo", _IoCounters),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
    _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9

    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

        handle = kernel32.CreateJobObjectW(None, None)
        if not handle:
            return _JOB_HANDLE

        limits = _ExtendedLimits()
        limits.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE

        if not kernel32.SetInformationJobObject(
            handle,
            _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
            ctypes.byref(limits),
            ctypes.sizeof(limits),
        ):
            kernel32.CloseHandle(handle)
            return _JOB_HANDLE

        _JOB_HANDLE = handle
    except (OSError, AttributeError):
        # Job objects are unavailable on this system; atexit still covers the
        # normal exit path, so degrade quietly rather than blocking startup.
        pass

    return _JOB_HANDLE


# Sentinel distinguishing "not yet attempted" from "attempted and unavailable",
# so a failed setup is not retried for every model that loads.
_JOB_UNSET = object()
_JOB_HANDLE = _JOB_UNSET


def _bind_to_athena_lifetime(process) -> None:
    """Tie ``process`` to Athena's own lifetime, best-effort.

    Failure is never fatal: the process still stops normally via ``stop()`` and
    ``atexit``. This only covers the abrupt-exit paths those miss.
    """
    job = _kill_on_exit_job()
    if job is None:
        return

    import ctypes

    _PROCESS_SET_QUOTA, _PROCESS_TERMINATE = 0x0100, 0x0001

    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        handle = kernel32.OpenProcess(
            _PROCESS_SET_QUOTA | _PROCESS_TERMINATE, False, process.pid
        )
        if not handle:
            return
        try:
            kernel32.AssignProcessToJobObject(job, handle)
        finally:
            kernel32.CloseHandle(handle)
    except (OSError, AttributeError):
        pass


def find_server_binary() -> Path | None:
    """Locate the bundled ``llama-server`` executable.

    Returns:
        Path to the binary, or None when the runtime has not been downloaded.
    """
    exe = "llama-server.exe" if os.name == "nt" else "llama-server"
    root = Path(get_settings().provider.runtime_directory)

    if not root.is_dir():
        return None

    direct = root / exe
    if direct.is_file():
        return direct

    # Some release archives nest the binaries one level down.
    for candidate in root.rglob(exe):
        return candidate

    return None


def _free_port() -> int:
    """Reserve an unused localhost port and return it.

    Binding to port 0 lets the OS pick a free one. There is a small race
    between closing the socket here and llama-server binding it, but the window
    is microseconds and the alternative — a fixed port — collides with any
    other Athena instance or leftover server.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def build_server_command(binary, model_path: str, port: int, api_key: str, config) -> list:
    """Assemble the ``llama-server`` argument list for one model.

    Separate from the process that runs it so the flags can be asserted on
    without starting a server or loading several gigabytes of weights.

    Args:
        binary: Path to the llama-server executable.
        model_path: GGUF file to serve.
        port: Loopback port to bind.
        api_key: Per-session key; the server allows all CORS origins, so
            without one any web page could talk to this port.
        config: InferenceConfiguration with the tuned hardware settings.

    Returns:
        The full command list, ready for subprocess.
    """
    command = [
        str(binary),
        "--model", model_path,
        "--host", "127.0.0.1",
        "--port", str(port),
        "--ctx-size", str(config.n_ctx),
        "--batch-size", str(config.n_batch),
        "--threads", str(config.n_threads),
        # AutoConfigurator only enables Flash Attention on CUDA: llama.cpp's
        # Vulkan implementation can hang the GPU and trip a driver timeout.
        # Pass the decision through explicitly rather than leaving it 'auto'.
        "--flash-attn", "on" if config.flash_attn else "off",
        # The model is addressed by this name in API calls, so pin it to
        # something stable rather than the full path.
        "--alias", Path(model_path).name,
        "--api-key", api_key,
        # Athena is the only client; the bundled browser UI is one more
        # thing listening that nobody asked for.
        "--no-webui",
    ]

    # Cap the thinking phase so it cannot consume the whole token budget and
    # leave nothing for the answer. llama-server injects the message below in
    # place of the model's next thought and then closes the think tag, so the
    # model wraps up and answers instead of being cut mid-sentence. Measured
    # with a 64-token budget: reasoning stopped at the cap, and a full answer
    # still followed. No-op on models that do not think.
    reasoning_budget = getattr(get_settings().provider, "reasoning_budget", -1)
    if reasoning_budget and reasoning_budget > 0:
        command += [
            "--reasoning-budget", str(reasoning_budget),
            "--reasoning-budget-message",
            "I have thought about this enough. Time to give the answer.",
        ]

    # A negative gpu_layers means "offload as much as fits". Pinning that to
    # a large number turns it into "offload everything", which fails outright
    # on any model bigger than VRAM. Omitting -ngl instead hands the decision
    # to llama.cpp's own fitter, which measures free device memory and keeps
    # the remaining layers on the CPU. An explicit count is still honoured.
    if config.gpu_layers >= 0:
        command += ["--n-gpu-layers", str(config.gpu_layers)]
    elif _should_keep_experts_on_cpu(model_path, config):
        # Deliberately left with -ngl unset: llama.cpp then fits the
        # remaining (non-expert) layers itself. Measured identical to
        # forcing every layer onto the GPU, and it cannot overcommit VRAM
        # on a model whose attention path alone exceeds it.
        command.append("--cpu-moe")
        print("Offloading expert layers to CPU for speed.")

    return command


class _Server:
    """A running ``llama-server`` process serving one model.

    Instances are cached per model path in :attr:`LlamaServerProvider._servers`
    so that two providers pointed at the same model share one process (and one
    copy of the weights in VRAM).
    """

    def __init__(self, model_path: str, config, label: str):
        binary = find_server_binary()
        if binary is None:
            raise RuntimeError(
                "The llama.cpp runtime is missing, so local models cannot be "
                "loaded.\n\n"
                "Run setup.bat to download it (about 32 MB, no compiler "
                "needed)."
            )

        self.model_path = model_path
        self.port = _free_port()
        self.base_url = f"http://127.0.0.1:{self.port}"

        # llama-server allows all CORS origins by default, so without a key any
        # web page the user visits could talk to this port. The server is bound
        # to loopback and the port is random, but a per-session key costs
        # nothing and closes the hole properly.
        self.api_key = secrets.token_urlsafe(32)

        command = build_server_command(
            binary, model_path, self.port, self.api_key, config
        )

        # Keep the server's own logging out of Athena's console, but retain it
        # so a startup failure can be explained.
        self._log = tempfile.NamedTemporaryFile(
            mode="w+", suffix=".log", prefix=f"athena-{label}-", delete=False
        )

        # CREATE_NO_WINDOW stops a console window flashing up on Windows.
        creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0

        print(f"Loading {label} model...")

        self.process = subprocess.Popen(
            command,
            stdout=self._log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            creationflags=creation_flags,
        )

        # Stop the child process from outliving Athena. Two layers, because
        # each covers what the other misses: atexit handles the clean exit and
        # runs the graceful terminate/kill sequence, while the job object is the
        # backstop for exits that never reach Python (window close, crash).
        _bind_to_athena_lifetime(self.process)
        atexit.register(self.stop)

        try:
            self._wait_until_ready()
        except Exception:
            self.stop()
            raise

        print("Model loaded.\n")

    def _wait_until_ready(self) -> None:
        """Block until the server answers /health, or fail with its log.

        llama-server returns 503 while the model is still loading and 200 once
        it can serve requests.
        """
        deadline = time.monotonic() + _STARTUP_TIMEOUT_SECONDS
        health_url = f"{self.base_url}/health"

        while time.monotonic() < deadline:
            # A dead process will never become healthy — fail immediately
            # rather than waiting out the full timeout.
            if self.process.poll() is not None:
                raise RuntimeError(
                    "The llama.cpp server stopped while loading the model.\n\n"
                    f"{self._log_tail()}"
                )

            try:
                response = requests.get(
                    health_url,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    timeout=2,
                )
                if response.status_code == 200:
                    return
            except requests.RequestException:
                # Not listening yet; that is normal during startup.
                pass

            time.sleep(0.5)

        raise RuntimeError(
            f"The llama.cpp server did not become ready within "
            f"{_STARTUP_TIMEOUT_SECONDS} seconds.\n\n{self._log_tail()}"
        )

    def _log_tail(self, lines: int = 15) -> str:
        """Return the last few lines of the server log, for error messages."""
        try:
            self._log.flush()
            with open(self._log.name, "r", encoding="utf-8", errors="replace") as handle:
                tail = handle.readlines()[-lines:]
            if tail:
                return "Server output:\n" + "".join(tail)
        except OSError:
            pass
        return f"See the server log at: {self._log.name}"

    def stop(self) -> None:
        """Terminate the server process. Safe to call more than once."""
        process = getattr(self, "process", None)
        if process is None or process.poll() is not None:
            return

        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()

        try:
            self._log.close()
        except OSError:
            pass


class LlamaServerProvider:
    """
    A provider for local GGUF inference via a bundled ``llama-server`` process.

    Presents the same interface as :class:`~athena.providers.llamacpp.LlamaCppProvider`
    so the two are interchangeable behind the provider factory.

    Constructor accepts:
        model_path: Path to the .gguf file. Resolved from settings when omitted.
        inference_config: An InferenceConfiguration. Required in practice —
            the factory always supplies one.
        temperature: Sampling temperature. Defaults to the configured value.
        label: Human-readable role ("reasoning" / "learning"), used in
            progress messages only.
    """

    # One server per distinct model path, shared across provider instances.
    _servers: dict[str, _Server] = {}

    def __init__(
        self,
        model_path: str | None = None,
        inference_config=None,
        temperature: float | None = None,
        label: str = "reasoning",
    ):
        settings = get_settings()

        if model_path is not None:
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

        if inference_config is not None:
            self.n_ctx = inference_config.n_ctx
            self.backend = inference_config.backend
        else:
            self.n_ctx = 4096
            self.backend = "CPU"

        if self.model_path not in LlamaServerProvider._servers:
            if inference_config is None:
                from athena.config.inference import InferenceConfiguration

                inference_config = InferenceConfiguration()

            LlamaServerProvider._servers[self.model_path] = _Server(
                self.model_path, inference_config, label
            )

        self._server = LlamaServerProvider._servers[self.model_path]

    @property
    def base_url(self) -> str:
        """The local URL this provider's model is served on."""
        return self._server.base_url

    @property
    def _auth_headers(self) -> dict[str, str]:
        """Authorization header for this session's model server."""
        return {"Authorization": f"Bearer {self._server.api_key}"}

    @property
    def upstream(self) -> tuple[str, dict[str, str]]:
        """Where this model is reachable, and the headers needed to reach it.

        Public counterpart to :attr:`base_url` / :attr:`_auth_headers`, for
        serve mode: llama-server already exposes an OpenAI-compatible API, so
        /serve proxies to it rather than wrapping a model object. The per-session
        API key is an internal detail, so callers get it bundled here instead of
        reaching into the provider.
        """
        return self.base_url, dict(self._auth_headers)

    @property
    def model_name(self) -> str:
        """The filename of the loaded GGUF model (without directory)."""
        return Path(self.model_path).name

    supports_streaming = True

    def generate(
        self, prompt: str, system: str | None = None, stream: bool = False
    ) -> str:
        """Generate a response from the model.

        Args:
            prompt: The input prompt string (user content).
            system: Optional system prompt, delivered in the `system` role so
                the model treats it as its own instructions rather than as a
                user claim.
            stream: Echo tokens to the registered streaming sink as they are
                produced. Only the answer call sets this; the planner and
                learning calls stay silent.

        Returns:
            The model's answer, with any <think>…</think> reasoning stripped.

        Raises:
            RuntimeError: If the server returns an error or an unexpected
                response shape.
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }

        if stream and streaming.active():
            return self._generate_streaming(payload)

        try:
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                headers=self._auth_headers,
                timeout=_STARTUP_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as exc:
            raise RuntimeError(f"The local model server failed to respond: {exc}") from exc

        try:
            message = data["choices"][0]["message"]
            content = message["content"]
        except (KeyError, IndexError) as exc:
            raise RuntimeError(
                f"The local model server returned an unexpected response: {data}"
            ) from exc

        # llama-server parses reasoning out of the completion itself and returns
        # it in its own field, so for most thinking models `content` arrives
        # already clean and there are no tags left for _strip_reasoning to find.
        # Without reading this field the trace is simply lost.
        reasoning_trace.record(message.get("reasoning_content") or "")

        return _strip_reasoning(content or "")

    def _generate_streaming(self, payload: dict) -> str:
        """Run a completion over SSE, echoing tokens to the streaming sink.

        llama-server sends reasoning in a separate `reasoning_content` delta,
        but not every build does, so content deltas still go through
        :class:`ReasoningStream` in case the tags arrive inline instead.

        Returns:
            The complete answer, reasoning excluded.
        """
        # return_progress makes the server report prompt-evaluation progress
        # before any token exists. That phase is the bulk of the wait on a
        # large model — 38s of a 45s turn, measured on a 35B with experts on
        # the CPU — and without it the user watches an unchanging spinner.
        # Builds that do not support the field ignore it.
        payload = {**payload, "stream": True, "return_progress": True}
        parser = ReasoningStream(streaming.emit)
        reasoning_parts: list[str] = []

        try:
            with requests.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                headers=self._auth_headers,
                timeout=_STARTUP_TIMEOUT_SECONDS,
                stream=True,
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines(decode_unicode=True):
                    if not line or not line.startswith("data:"):
                        continue
                    chunk = line[len("data:"):].strip()
                    if chunk == "[DONE]":
                        break
                    try:
                        frame = json.loads(chunk)
                        delta = frame["choices"][0].get("delta") or {}
                    except (ValueError, KeyError, IndexError):
                        # A malformed frame costs one token, not the answer.
                        continue

                    progress = frame.get("prompt_progress")
                    if progress:
                        total = progress.get("total") or 0
                        if total:
                            # Truncated, not rounded: the last batch reports
                            # e.g. 1028/1029, and showing 100% while the user
                            # still waits is the one thing a progress figure
                            # must never do. It sits at 99% instead.
                            percent = int(progress.get("processed", 0) / total * 100)
                            streaming.emit("progress", str(percent))
                        continue

                    thinking = delta.get("reasoning_content")
                    if thinking:
                        reasoning_parts.append(thinking)
                        streaming.emit("reasoning", thinking)
                    parser.feed(delta.get("content") or "")
        except requests.RequestException as exc:
            raise RuntimeError(f"The local model server failed to respond: {exc}") from exc

        answer, inline_reasoning = parser.finish()
        reasoning_trace.record("".join(reasoning_parts) or inline_reasoning)
        return answer

    def call(self, prompt: str) -> str:
        """Alias for generate() to match KnowledgeManager expectations."""
        return self.generate(prompt)

    def count_tokens(self, text: str) -> int:
        """Count tokens using the model's own tokenizer.

        Args:
            text: The text to tokenize and count.

        Returns:
            The number of tokens in the text. Falls back to a
            characters-over-four estimate if the server cannot be reached, so
            that context budgeting degrades rather than crashes.
        """
        try:
            response = requests.post(
                f"{self.base_url}/tokenize",
                json={"content": text},
                headers=self._auth_headers,
                timeout=30,
            )
            response.raise_for_status()
            return len(response.json()["tokens"])
        except (requests.RequestException, KeyError, ValueError):
            return len(text) // 4

    def get_context_window(self) -> int:
        """Get the configured context window size in tokens."""
        return self.n_ctx

    def unload(self) -> bool:
        """Stop this model's server, freeing its VRAM.

        Mirrors LlamaCppProvider.unload(): serve mode uses it to reclaim memory
        held by a dedicated learning model that is idle while only the
        reasoning model is being served.

        Returns:
            True if a server was stopped; False if none was running.
        """
        server = LlamaServerProvider._servers.pop(self.model_path, None)
        self._server = None
        if server is None:
            return False

        server.stop()
        return True

    def reload(self) -> None:
        """Restart this model's server after :meth:`unload`."""
        if self._server is not None:
            return

        from athena.config.inference import InferenceConfiguration

        existing = LlamaServerProvider._servers.get(self.model_path)
        if existing is not None:
            self._server = existing
            return

        print("Reloading learning model...")
        config = InferenceConfiguration(n_ctx=self.n_ctx, backend=self.backend)
        LlamaServerProvider._servers[self.model_path] = _Server(
            self.model_path, config, "learning"
        )
        self._server = LlamaServerProvider._servers[self.model_path]
        print("Learning model reloaded.\n")

    def health_check(self) -> bool:
        """True when the model server is up and able to serve requests."""
        if self._server is None:
            return False

        try:
            response = requests.get(
                f"{self.base_url}/health", headers=self._auth_headers, timeout=5
            )
            return response.status_code == 200
        except requests.RequestException:
            return False
