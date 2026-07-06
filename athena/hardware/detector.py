"""
Hardware Detector

Determines the hardware capabilities of the host machine for the purpose
of automatically configuring llama.cpp inference parameters.

Detection methods:
  - CPU: py-cpuinfo + psutil
  - RAM: psutil
  - GPU (NVIDIA): nvidia-smi CLI, fallback pynvml
  - GPU (generic): WMI on Windows
  - Backend: CUDA / HIP / Vulkan / CPU based on GPU vendor and runtime presence

This module has no dependencies beyond psutil and py-cpuinfo (already in
the project) and standard library modules.
"""

import platform
import subprocess
from dataclasses import dataclass
from typing import Optional

import cpuinfo
import psutil


@dataclass(frozen=True)
class CpuInfo:
    """Detected CPU capabilities."""
    model: str
    physical_cores: int
    logical_threads: int


@dataclass(frozen=True)
class RamInfo:
    """Detected system memory."""
    total_bytes: int

    @property
    def total_gb(self) -> float:
        return round(self.total_bytes / (1024 ** 3), 2)


@dataclass(frozen=True)
class GpuInfo:
    """Detected GPU capabilities."""
    vendor: str                # "NVIDIA" | "AMD" | "Intel" | "Unknown"
    model: str                 # Human-readable GPU name
    vram_bytes: Optional[int]  # None when VRAM cannot be determined

    @property
    def vram_gb(self) -> Optional[float]:
        if self.vram_bytes is not None:
            return round(self.vram_bytes / (1024 ** 3), 2)
        return None


@dataclass(frozen=True)
class HardwareInfo:
    """Aggregated hardware information."""
    cpu: CpuInfo
    ram: RamInfo
    gpu: GpuInfo
    backend: str   # "CPU" | "CUDA" | "HIP" | "Vulkan"


class HardwareDetector:
    """Detects host hardware capabilities for inference tuning."""

    def detect(self) -> HardwareInfo:
        """Run full hardware detection and return aggregated result."""
        cpu = self._detect_cpu()
        ram = self._detect_ram()
        gpu = self._detect_gpu()
        backend = self._detect_backend(gpu)
        return HardwareInfo(cpu=cpu, ram=ram, gpu=gpu, backend=backend)

    # ----------------------------------------------------------------
    # CPU
    # ----------------------------------------------------------------

    @staticmethod
    def _detect_cpu() -> CpuInfo:
        """Detect CPU model, physical cores, and logical threads."""
        raw = cpuinfo.get_cpu_info()
        return CpuInfo(
            model=raw.get("brand_raw", "Unknown"),
            physical_cores=psutil.cpu_count(logical=False) or 1,
            logical_threads=psutil.cpu_count(logical=True) or 1,
        )

    # ----------------------------------------------------------------
    # RAM
    # ----------------------------------------------------------------

    @staticmethod
    def _detect_ram() -> RamInfo:
        """Detect total system RAM."""
        return RamInfo(total_bytes=psutil.virtual_memory().total)

    # ----------------------------------------------------------------
    # GPU
    # ----------------------------------------------------------------

    def _detect_gpu(self) -> GpuInfo:
        """Detect primary GPU vendor, model, and VRAM."""
        # 1) Try NVIDIA via nvidia-smi
        gpu = self._detect_nvidia_smi()
        if gpu is not None:
            return gpu

        # 2) Try NVIDIA via pynvml (if installed)
        gpu = self._detect_nvidia_pynvml()
        if gpu is not None:
            return gpu

        # 3) Fall back to WMI on Windows
        if platform.system() == "Windows":
            gpu = self._detect_gpu_wmi()
            if gpu is not None:
                return gpu

        return GpuInfo(vendor="Unknown", model="Unknown", vram_bytes=None)

    @staticmethod
    def _detect_nvidia_smi() -> Optional[GpuInfo]:
        """Query NVIDIA GPU via nvidia-smi CLI."""
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode != 0 or not result.stdout.strip():
                return None

            line = result.stdout.strip().split("\n")[0]
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 2 and parts[1].isdigit():
                vram_mib = int(parts[1])
                vram_bytes = vram_mib * 1024 * 1024
                return GpuInfo(
                    vendor="NVIDIA",
                    model=parts[0],
                    vram_bytes=vram_bytes,
                )
        except (FileNotFoundError, subprocess.TimeoutExpired, ValueError, IndexError):
            pass
        return None

    @staticmethod
    def _detect_nvidia_pynvml() -> Optional[GpuInfo]:
        """Query NVIDIA GPU via pynvml Python package."""
        try:
            import pynvml  # type: ignore[import-untyped]
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            name = pynvml.nvmlDeviceGetName(handle)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return GpuInfo(
                vendor="NVIDIA",
                model=name,
                vram_bytes=mem_info.total,
            )
        except (ImportError, Exception):
            pass
        return None

    @staticmethod
    def _detect_gpu_wmi() -> Optional[GpuInfo]:
        """Query GPU on Windows via PowerShell Get-CimInstance.

        Uses PowerShell rather than the deprecated wmic.exe.
        Iterates over all adapters and returns the first real (non-virtual)
        GPU found.  Virtual adapters (Virtual Desktop, Basic Display,
        Microsoft Hyper-V, Remote Desktop, VMware, VirtualBox, etc.) are
        skipped.

        AdapterRAM is reported if present and non-zero; when absent,
        zero, or otherwise unreliable it is reported as Unknown.
        """
        try:
            ps_command = (
                'Get-CimInstance Win32_VideoController '
                '| Select-Object Name, AdapterRAM '
                '| ConvertTo-Csv -NoTypeInformation'
            )
            result = subprocess.run(
                ["powershell", "-NoProfile", "-Command", ps_command],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0 or not result.stdout.strip():
                return None

            lines = [l.strip() for l in result.stdout.strip().split("\n") if l.strip()]

            # Find header row
            header_idx = None
            for i, line in enumerate(lines):
                if 'Name' in line and 'AdapterRAM' in line:
                    header_idx = i
                    break
            if header_idx is None or header_idx + 1 >= len(lines):
                return None

            # Iterate data rows — skip virtual adapters, pick first real GPU
            for data_line in lines[header_idx + 1:]:
                if not data_line:
                    continue

                name: str = ""
                vram_raw: str = ""

                if data_line.startswith('"'):
                    # CSV format: "Name","AdapterRAM"
                    quote_parts = data_line.split('","')
                    if len(quote_parts) == 2:
                        name = quote_parts[0].lstrip('"').strip()
                        vram_raw = quote_parts[1].rstrip('"').strip()
                    elif len(quote_parts) == 1:
                        # Single column (AdapterRAM may be empty/null)
                        name = quote_parts[0].strip('"').strip()
                        vram_raw = ""
                    else:
                        continue
                else:
                    parts = [p.strip() for p in data_line.split(",")]
                    name = parts[0] if len(parts) >= 1 else ""
                    vram_raw = parts[1] if len(parts) >= 2 else ""

                # Skip virtual / display-only adapters
                if HardwareDetector._is_virtual_adapter(name):
                    continue

                vram_bytes: Optional[int] = None
                if vram_raw and vram_raw.isdigit():
                    val = int(vram_raw)
                    if val > 0:
                        vram_bytes = val  # AdapterRAM is in bytes (caps at ~4 GB)

                # AdapterRAM is a 32-bit WMI field and caps at ~4 GB, so any
                # GPU with >=4 GB is misreported. Prefer the accurate 64-bit
                # value from the registry when it is available.
                registry_vram = HardwareDetector._detect_vram_registry(name)
                if registry_vram is not None:
                    vram_bytes = registry_vram

                vendor = HardwareDetector._classify_gpu_vendor(name)
                return GpuInfo(vendor=vendor, model=name, vram_bytes=vram_bytes)

        except (FileNotFoundError, subprocess.TimeoutExpired, ValueError, IndexError):
            pass
        return None

    @staticmethod
    def _detect_vram_registry(gpu_name: Optional[str] = None) -> Optional[int]:
        """Read true dedicated VRAM (bytes) from the Windows registry.

        ``Win32_VideoController.AdapterRAM`` is a 32-bit field that caps at
        ~4 GB, so any GPU with >=4 GB of VRAM is misreported. The display
        adapter registry key stores the real size as a 64-bit QWORD under
        ``HardwareInformation.qwMemorySize``.

        When ``gpu_name`` is provided, the value for the adapter whose
        ``DriverDesc`` matches it is returned; otherwise the largest value
        among physical adapters is used. Returns None when unavailable
        (non-Windows, key/value missing, or registry access denied).
        """
        if platform.system() != "Windows":
            return None
        try:
            import winreg
        except ImportError:
            return None

        base = (
            r"SYSTEM\CurrentControlSet\Control\Class"
            r"\{4d36e968-e325-11ce-bfc1-08002be10318}"
        )
        candidates: list[tuple[str, int]] = []  # (adapter description, vram bytes)
        try:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, base) as root:
                index = 0
                while True:
                    try:
                        subkey_name = winreg.EnumKey(root, index)
                    except OSError:
                        break
                    index += 1
                    if not subkey_name.isdigit():
                        continue  # skip "Properties", "Configuration", etc.
                    try:
                        with winreg.OpenKey(root, subkey_name) as subkey:
                            try:
                                size, _ = winreg.QueryValueEx(
                                    subkey, "HardwareInformation.qwMemorySize"
                                )
                            except FileNotFoundError:
                                continue
                            if not isinstance(size, int) or size <= 0:
                                continue
                            try:
                                desc, _ = winreg.QueryValueEx(subkey, "DriverDesc")
                            except FileNotFoundError:
                                desc = ""
                            if HardwareDetector._is_virtual_adapter(desc):
                                continue
                            candidates.append((desc, size))
                    except OSError:
                        continue
        except OSError:
            return None

        if not candidates:
            return None

        # Prefer the adapter whose description matches the detected GPU name.
        if gpu_name:
            target = gpu_name.lower()
            for desc, size in candidates:
                lower_desc = desc.lower()
                if lower_desc and (lower_desc in target or target in lower_desc):
                    return size

        # Otherwise fall back to the largest physical adapter's VRAM.
        return max(size for _, size in candidates)

    @staticmethod
    def _is_virtual_adapter(name: str) -> bool:
        """Return True if the adapter name indicates a virtual/display-only device."""
        lower = name.lower()
        virtual_keywords = [
            "virtual", "vGPU", "basic display", "microsoft",
            "remote", "vmware", "virtualbox", "paravirtual",
            "hyper-v", "hyperv", "synth", "render-only",
        ]
        return any(kw in lower for kw in virtual_keywords)

    @staticmethod
    def _classify_gpu_vendor(name: str) -> str:
        """Classify GPU vendor from the adapter name string."""
        lower = name.lower()
        if "nvidia" in lower:
            return "NVIDIA"
        if "amd" in lower or "radeon" in lower or "firepro" in lower:
            return "AMD"
        if "intel" in lower or "arc" in lower:
            return "Intel"
        return "Unknown"

    # ----------------------------------------------------------------
    # Backend
    # ----------------------------------------------------------------

    @staticmethod
    def _detect_backend(gpu: GpuInfo) -> str:
        """Determine the best available inference backend based on GPU.

        Backend selection rules:
          - NVIDIA → CUDA
          - AMD    → Vulkan (Windows); HIP would be Linux-only with ROCm
          - Intel  → Vulkan (if runtime present) or CPU
          - Unknown → CPU
        """
        vendor = gpu.vendor
        if vendor == "NVIDIA":
            return "CUDA"
        if vendor == "AMD":
            return "Vulkan"
        if vendor == "Intel":
            if HardwareDetector._check_vulkan_present():
                return "Vulkan"
            return "CPU"
        # Unknown vendor — check Vulkan as a fallback
        if HardwareDetector._check_vulkan_present():
            return "Vulkan"
        return "CPU"

    @staticmethod
    def _check_vulkan_present() -> bool:
        """Check whether the Vulkan runtime is available on the system."""
        try:
            if platform.system() == "Windows":
                cmd = ["where", "vulkaninfo"]
            else:
                cmd = ["which", "vulkaninfo"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3)
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False