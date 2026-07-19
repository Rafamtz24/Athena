"""
Inference Configuration

Defines the InferenceConfiguration dataclass and the AutoConfigurator
that produces optimal inference parameters based on detected hardware.

The AutoConfigurator uses generic heuristics — no values are hardcoded
for any specific machine.  The same logic runs everywhere and adapts
to the hardware it discovers.

Usage:
    config = AutoConfigurator().configure(hardware_info)
    provider = LlamaCppProvider(inference_config=config)
"""

from dataclasses import dataclass, field
from typing import Optional

from athena.config.settings import get_settings


@dataclass(frozen=True)
class InferenceConfiguration:
    """Runtime inference parameters for llama.cpp.

    Attributes:
        gpu_layers:  Number of layers to offload to GPU.
                     0 = CPU only, -1 = as many as fit in VRAM.
                     -1 is a deferred decision, not "all layers": the number
                     that fits depends on the model file, and one
                     InferenceConfiguration is shared by the reasoning and
                     learning models, which are different sizes. Each provider
                     resolves it against the model it is actually loading.
        n_threads:   Number of CPU threads for inference.
        n_batch:     Batch size for prompt processing.
        n_ctx:       Maximum context window size in tokens.
        backend:     Inference backend string (CPU / CUDA / HIP / Vulkan).
        vram_bytes:  Detected VRAM on the selected GPU, or None when unknown.
                     Passed through so a provider resolving gpu_layers=-1 can
                     size the offload against the model it is loading.
        flash_attn:  Enable Flash Attention (faster, lower KV-cache memory).
                     Defaults OFF: it is only reliable on CUDA. On the Vulkan
                     backend (AMD / Intel) llama.cpp's flash-attention can hang
                     the GPU and trigger a driver timeout, so AutoConfigurator
                     enables it only when the backend is CUDA.
    """
    gpu_layers: int = 0
    n_threads: int = 4
    n_batch: int = 512
    n_ctx: int = 4096
    backend: str = "CPU"
    flash_attn: bool = False
    vram_bytes: int | None = None


# ---------------------------------------------------------------------------
# Constants used by heuristics (tunable but machine-independent)
# ---------------------------------------------------------------------------

# Safety margin fraction kept free in VRAM for KV cache, buffers, etc.
_VRAM_SAFETY_MARGIN = 0.85

# Safety margin fraction kept free in system RAM for OS and other processes
_RAM_SAFETY_MARGIN = 0.75


class AutoConfigurator:
    """Produces an InferenceConfiguration from detected hardware.

    Heuristics are based on logical, scalable rules that generalise across
    different hardware profiles.  No machine-specific constants.
    """

    def configure(self, hardware: "HardwareInfo") -> InferenceConfiguration:
        """Derive inference parameters from the given hardware information.

        Args:
            hardware: A HardwareInfo instance from HardwareDetector.detect().

        Returns:
            An InferenceConfiguration with values suited to the host.
        """
        settings = get_settings()
        mode = settings.performance.performance_mode

        if mode == "cpu_only":
            return self._configure_cpu_only(hardware)

        # "auto", "balanced", and "maximum" all use GPU if available.
        return self._configure_auto(hardware, mode)

    # ------------------------------------------------------------------
    # Heuristic: auto / balanced / maximum
    # ------------------------------------------------------------------

    @staticmethod
    def _configure_auto(hardware: "HardwareInfo", mode: str) -> InferenceConfiguration:
        cpu = hardware.cpu
        ram = hardware.ram
        gpu = hardware.gpu
        backend = hardware.backend

        # --- Threads ---------------------------------------------------
        # CPU-only: use all physical cores.
        # GPU-offloaded: inference is mostly on GPU; use half the physical
        # cores for prompt pre-processing, clamped to [2, 8].
        if backend == "CPU":
            n_threads = cpu.physical_cores
        else:
            n_threads = max(2, min(cpu.physical_cores // 2, 8))

        # --- GPU layers ------------------------------------------------
        gpu_layers = AutoConfigurator._pick_gpu_layers(backend)

        # --- Batch size ------------------------------------------------
        n_batch = AutoConfigurator._pick_batch_size(gpu, backend, mode)

        # --- Context size ----------------------------------------------
        n_ctx = AutoConfigurator._pick_context_size(ram, mode)

        # --- Flash attention -------------------------------------------
        # Only enable on CUDA, where it is stable. On Vulkan (AMD / Intel) it
        # can hang the GPU and cause a driver timeout, so leave it off there.
        flash_attn = backend == "CUDA"

        return InferenceConfiguration(
            gpu_layers=gpu_layers,
            n_threads=n_threads,
            n_batch=n_batch,
            n_ctx=n_ctx,
            backend=backend,
            flash_attn=flash_attn,
            vram_bytes=gpu.vram_bytes,
        )

    # ------------------------------------------------------------------
    # Heuristic: cpu_only
    # ------------------------------------------------------------------

    @staticmethod
    def _configure_cpu_only(hardware: "HardwareInfo") -> InferenceConfiguration:
        cpu = hardware.cpu
        ram = hardware.ram

        n_threads = cpu.physical_cores
        n_ctx = AutoConfigurator._pick_context_size(ram, "cpu_only")

        return InferenceConfiguration(
            gpu_layers=0,
            n_threads=n_threads,
            n_batch=256,
            n_ctx=n_ctx,
            backend="CPU",
        )

    # ------------------------------------------------------------------
    # Sub-heuristics
    # ------------------------------------------------------------------

    @staticmethod
    def _pick_gpu_layers(backend: str) -> int:
        """Decide whether to offload to the GPU at all.

        Deliberately does not return a layer count. A count is only meaningful
        against a specific model file, and this configuration is built once from
        hardware alone and then shared by every model Athena loads. Returning
        -1 ("as many as fit") defers the arithmetic to the provider, which knows
        the model and can measure free VRAM at load time.

        The previous implementation estimated a count here from VRAM and a
        fixed 35 MB-per-layer constant borrowed from a 7B model. Because the
        model size never entered the calculation, any GPU with more than ~1.3 GB
        of VRAM was told every layer fit — so a 21 GB model on an 8 GB card
        produced "offload everything" and llama.cpp died allocating VRAM.
        """
        if backend == "CPU":
            return 0

        # Offload as much as fits. This is correct whether the model is far
        # smaller than VRAM (everything lands on the GPU) or far larger (the
        # remainder stays on the CPU), and needs no guess about either.
        return -1

    @staticmethod
    def _pick_batch_size(
        gpu: "GpuInfo",
        backend: str,
        mode: str,
    ) -> int:
        """Determine batch size for prompt processing."""
        if backend == "CPU":
            return 256

        vram = gpu.vram_bytes
        if vram is None:
            return 512

        usable_vram = vram * _VRAM_SAFETY_MARGIN

        if mode == "maximum" and usable_vram >= 8 * (1024 ** 3):
            return 1024
        if usable_vram >= 6 * (1024 ** 3):
            return 512
        if usable_vram >= 3 * (1024 ** 3):
            return 512
        return 256

    @staticmethod
    def _pick_context_size(ram: "RamInfo", mode: str) -> int:
        """Determine safe context window size based on available RAM."""
        total_gb = ram.total_gb

        if mode == "maximum":
            return 8192

        if mode == "cpu_only":
            if total_gb >= 32:
                return 4096
            if total_gb >= 16:
                return 2048
            return 1024

        # "auto" or "balanced"
        if total_gb >= 32:
            return 8192
        if total_gb >= 16:
            return 4096
        if total_gb >= 8:
            return 2048
        return 1024