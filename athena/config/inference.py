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
                     0 = CPU only, -1 = all layers.
        n_threads:   Number of CPU threads for inference.
        n_batch:     Batch size for prompt processing.
        n_ctx:       Maximum context window size in tokens.
        backend:     Inference backend string (CPU / CUDA / HIP / Vulkan).
    """
    gpu_layers: int = 0
    n_threads: int = 4
    n_batch: int = 512
    n_ctx: int = 4096
    backend: str = "CPU"


# ---------------------------------------------------------------------------
# Constants used by heuristics (tunable but machine-independent)
# ---------------------------------------------------------------------------

# Average VRAM consumed per offloaded layer for a ~7B Q4_K_M model (~35 MB)
_ESTIMATED_BYTES_PER_LAYER = 35 * 1024 * 1024

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
        gpu_layers = AutoConfigurator._pick_gpu_layers(gpu, backend, mode)

        # --- Batch size ------------------------------------------------
        n_batch = AutoConfigurator._pick_batch_size(gpu, backend, mode)

        # --- Context size ----------------------------------------------
        n_ctx = AutoConfigurator._pick_context_size(ram, mode)

        return InferenceConfiguration(
            gpu_layers=gpu_layers,
            n_threads=n_threads,
            n_batch=n_batch,
            n_ctx=n_ctx,
            backend=backend,
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
    def _pick_gpu_layers(
        gpu: "GpuInfo",
        backend: str,
        mode: str,
    ) -> int:
        """Determine how many layers to offload to the GPU."""
        if backend == "CPU":
            return 0

        vram = gpu.vram_bytes
        if vram is None:
            # VRAM unknown — conservative: offload a small amount
            return 10

        usable_vram = vram * _VRAM_SAFETY_MARGIN

        if mode == "maximum":
            # Offload everything
            return -1

        # Estimate how many layers fit in usable VRAM
        max_layer_count = 999  # effectively all layers
        estimated_layers = int(usable_vram // _ESTIMATED_BYTES_PER_LAYER)

        if estimated_layers >= 32:
            return -1  # all layers fit comfortably
        if estimated_layers >= 20:
            return -1 if mode == "balanced" else estimated_layers
        if estimated_layers >= 10:
            return estimated_layers

        # Very little VRAM — offload whatever little fits
        return max(estimated_layers, 0)

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