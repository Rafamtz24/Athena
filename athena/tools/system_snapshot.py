"""System Snapshot generator.

Produces a comprehensive snapshot of the user's computer for the /system tool.
The snapshot is TEMPORARY context for a single request and is NOT stored in
any memory system.
"""

import datetime
import platform
import socket
import subprocess
import sys
from typing import Optional

import psutil
import cpuinfo


def _run_wmic(query: str) -> str:
    """Run a WMIC query and return stdout, or empty string on failure.

    WMIC arguments must be split individually (not passed as one string).
    """
    try:
        args = query.split()
        result = subprocess.run(
            ["wmic"] + args,
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            out = result.stdout.strip().replace('\x00', '').replace('\ufeff', '')
            lines = [l for l in out.splitlines() if l.strip() and 'wmic' not in l.lower()[:10]]
            return '\n'.join(lines)
        return ""
    except Exception:
        return ""


def _run_powershell(script: str) -> str:
    """Run a PowerShell command and return stdout, or empty string on failure."""
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", script],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            out = result.stdout.strip().replace('\x00', '').replace('\ufeff', '')
            return out
        return ""
    except Exception:
        return ""


def _parse_wmic_table(text: str) -> list[dict[str, str]]:
    """Parse a WMIC/PowerShell CSV-like table into a list of dicts.

    Handles:
    - WMIC format: Node,prop1,prop2... \\n NODE,val1,val2...
    - PowerShell ConvertTo-Csv format: "prop1","prop2"... \\n "val1","val2"...
    """
    if not text:
        return []
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 2:
        return []

    # Find the header line — look for a line with commas and no leading "Node" prefix
    header_idx = 0
    for i, line in enumerate(lines):
        parts = [p.strip().strip('"') for p in line.split(",")]
        if len(parts) >= 2:
            # Check if this looks like a header (all text, no data with commas)
            header_idx = i
            break

    headers = [h.strip().strip('"') for h in lines[header_idx].split(",")]

    # Parse data lines
    data_lines = []
    for i in range(header_idx + 1, len(lines)):
        line = lines[i]
        if not line or "," not in line:
            continue
        parts = [p.strip().strip('"') for p in line.split(",")]
        # Skip the first column if it's the WMIC node name
        if len(parts) > len(headers) and not parts[0].startswith('"'):
            parts = parts[1:]
        # Only add if we have the right number of columns
        relevant = parts[:len(headers)]
        row = {}
        for j, h in enumerate(headers):
            val = relevant[j] if j < len(relevant) else ""
            row[h] = val
        data_lines.append(row)

    return data_lines


# ─────────────────────────────────────────
# OS
# ─────────────────────────────────────────

def _get_os_info() -> str:
    lines = []
    try:
        system = platform.system()
        release = platform.release()
        build = platform.win32_ver()[2] if hasattr(platform, "win32_ver") else "N/A"
        arch = platform.machine()

        lines.append(f"OS: {system}")
        lines.append(f"Version: {release}")
        lines.append(f"Build: {build}")
        lines.append(f"Architecture: {arch}")

        # Uptime
        boot = psutil.boot_time()
        uptime = datetime.datetime.now() - datetime.datetime.fromtimestamp(boot)
        days = uptime.days
        hours, remainder = divmod(uptime.seconds, 3600)
        minutes, _ = divmod(remainder, 60)
        uptime_str = f"{days}d {hours}h {minutes}m"
        lines.append(f"Uptime: {uptime_str}")
    except Exception:
        lines.append("OS: Unable to determine")
    return "\n".join(f"  {l}" for l in lines)


# ─────────────────────────────────────────
# CPU
# ─────────────────────────────────────────

def _get_cpu_info() -> str:
    lines = []
    try:
        info = cpuinfo.get_cpu_info()
        brand = info.get("brand_raw", "Unknown")
        hz_advertised = info.get("hz_advertised_friendly", "N/A")
        lines.append(f"CPU: {brand}")
    except Exception:
        lines.append("CPU: Unable to determine")
        return "\n".join(f"  {l}" for l in lines)

    try:
        physical = psutil.cpu_count(logical=False) or 0
        logical = psutil.cpu_count(logical=True) or 0
        lines.append(f"Physical cores: {physical}")
        lines.append(f"Logical threads: {logical}")
    except Exception:
        pass

    try:
        lines.append(f"Base frequency: {hz_advertised}")
    except Exception:
        pass

    try:
        freq = psutil.cpu_freq()
        if freq and freq.current:
            lines.append(f"Current frequency: {freq.current:.0f} MHz")
    except Exception:
        pass

    try:
        utilization = psutil.cpu_percent(interval=0.5)
        lines.append(f"Current utilization: {utilization}%")
    except Exception:
        pass

    return "\n".join(f"  {l}" for l in lines)


# ─────────────────────────────────────────
# RAM
# ─────────────────────────────────────────

def _get_ram_info() -> str:
    lines = []
    try:
        mem = psutil.virtual_memory()
        total_gb = mem.total / (1024 ** 3)
        available_gb = mem.available / (1024 ** 3)
        used_gb = mem.used / (1024 ** 3)

        lines.append(f"Installed: {total_gb:.1f} GB")
        lines.append(f"Available: {available_gb:.1f} GB")
        lines.append(f"Used: {used_gb:.1f} GB")
        lines.append(f"Utilization: {mem.percent}%")
    except Exception:
        lines.append("RAM: Unable to determine")
    return "\n".join(f"  {l}" for l in lines)


# ─────────────────────────────────────────
# GPU
# ─────────────────────────────────────────

def _get_gpu_info() -> str:
    """Gather GPU information via PowerShell/WMI."""
    lines = []
    try:
        # Use PowerShell to get GPU info (more reliable than wmic)
        ps_cmd = (
            'Get-CimInstance Win32_VideoController | '
            'Select-Object Name, AdapterRAM, DriverVersion, VideoProcessor | '
            'ConvertTo-Csv -NoTypeInformation'
        )
        raw = _run_powershell(ps_cmd)
        gpus = _parse_wmic_table(raw) if raw else []

        if not gpus:
            # Fallback: try wmic
            raw = _run_wmic("path Win32_VideoController get Name,AdapterRAM,DriverVersion /format:csv")
            gpus = _parse_wmic_table(raw) if raw else []

        if not gpus:
            lines.append("No GPU detected")
            return "\n".join(f"  {l}" for l in lines)

        # Filter out virtual/minimal GPUs (0 VRAM, or names like "Virtual Desktop Monitor")
        real_gpus = []
        for gpu in gpus:
            name = gpu.get("Name", "").lower()
            ram_raw = gpu.get("AdapterRAM", "0")
            try:
                vram_bytes = int(ram_raw)
            except (ValueError, TypeError):
                vram_bytes = 0
            # Skip virtual GPUs with no VRAM or obvious software displays
            if vram_bytes == 0 or "virtual" in name or "remote" in name:
                continue
            real_gpus.append(gpu)

        if not real_gpus:
            real_gpus = gpus  # fallback: show all if all were filtered

        for idx, gpu in enumerate(real_gpus):
            name = gpu.get("Name", "Unknown")
            ram_raw = gpu.get("AdapterRAM", "0")
            driver = gpu.get("DriverVersion", "N/A")
            processor = gpu.get("VideoProcessor", "")

            # AdapterRAM is a 32-bit WMI field that caps at ~4 GB, so any GPU
            # with >=4 GB is misreported. Prefer the accurate 64-bit VRAM from
            # the registry (HardwareInformation.qwMemorySize), falling back to
            # AdapterRAM only when the registry value is unavailable.
            from athena.hardware.detector import HardwareDetector
            vram_bytes = HardwareDetector._detect_vram_registry(name)
            if vram_bytes is None:
                try:
                    vram_bytes = int(ram_raw)
                except (ValueError, TypeError):
                    vram_bytes = None
            if vram_bytes:
                vram_str = f"{vram_bytes / (1024 ** 3):.1f} GB"
            else:
                vram_str = "N/A"

            # Try to extract vendor from name
            name_lower = name.lower()
            if "nvidia" in name_lower:
                vendor = "NVIDIA"
            elif "amd" in name_lower or "radeon" in name_lower:
                vendor = "AMD"
            elif "intel" in name_lower:
                vendor = "Intel"
            else:
                vendor = processor if processor else "Unknown"

            if idx > 0:
                lines.append("")
            lines.append(f"GPU {idx + 1}:")
            lines.append(f"  Vendor: {vendor}")
            lines.append(f"  Model: {name}")
            lines.append(f"  Dedicated VRAM: {vram_str}")
            lines.append(f"  Driver version: {driver}")

            # Current utilization via PowerShell
            try:
                ut_cmd = (
                    '(Get-Counter "\\GPU(*)\\% GPU Time" -ErrorAction SilentlyContinue).CounterSamples | '
                    'Select-Object -First 1 | ForEach-Object { $_.CookedValue }'
                )
                gpu_util = _run_powershell(ut_cmd)
                if gpu_util:
                    try:
                        lines.append(f"  Current utilization: {float(gpu_util):.0f}%")
                    except (ValueError, TypeError):
                        pass
            except Exception:
                pass

            # Current VRAM usage via PowerShell
            try:
                vr_cmd = (
                    '(Get-Counter "\\GPU(*)\\Dedicated Used Memory" -ErrorAction SilentlyContinue).CounterSamples | '
                    'Select-Object -First 1 | ForEach-Object { $_.CookedValue }'
                )
                vram_used = _run_powershell(vr_cmd)
                if vram_used:
                    try:
                        vram_used_gb = float(vram_used) / (1024 ** 3)
                        lines.append(f"  Current VRAM usage: {vram_used_gb:.1f} GB")
                    except (ValueError, TypeError):
                        pass
            except Exception:
                pass

            # Vulkan — most modern GPUs support it
            lines.append(f"  Vulkan support: Yes (modern driver)")

            # DirectX version from registry on Windows
            try:
                dx_cmd = (
                    '(Get-ItemProperty "HKLM:\\SOFTWARE\\Microsoft\\DirectX" -ErrorAction SilentlyContinue).Version'
                )
                dx_ver = _run_powershell(dx_cmd)
                if dx_ver:
                    v = dx_ver.strip()
                    if v:
                        lines.append(f"  DirectX support: {v}")
                    else:
                        lines.append(f"  DirectX support: Yes (modern Windows)")
                else:
                    lines.append(f"  DirectX support: Yes (modern Windows)")
            except Exception:
                lines.append(f"  DirectX: Unable to determine")
    except Exception:
        lines.append("GPU: Unable to determine")
    return "\n".join(f"  {l}" for l in lines)


# ─────────────────────────────────────────
# Storage
# ─────────────────────────────────────────

def _get_storage_info() -> str:
    lines = []
    try:
        partitions = psutil.disk_partitions()
        for part in partitions:
            try:
                usage = psutil.disk_usage(part.mountpoint)
                total_gb = usage.total / (1024 ** 3)
                free_gb = usage.free / (1024 ** 3)
                used_gb = usage.used / (1024 ** 3)

                drive = part.device
                fstype = part.fstype

                # Check if SSD/HDD via PowerShell
                drive_type = ""
                try:
                    ps_out = _run_powershell(
                        '(Get-PhysicalDisk -ErrorAction SilentlyContinue | '
                        'Where-Object { $_.MediaType -ne "Unspecified" } | '
                        'Select-Object -First 1).MediaType'
                    )
                    if ps_out and "SSD" in ps_out:
                        drive_type = "SSD"
                    elif ps_out and "HDD" in ps_out:
                        drive_type = "HDD"
                except Exception:
                    pass

                type_str = f" ({drive_type})" if drive_type else ""
                lines.append(f"{drive} ({fstype}){type_str}")
                lines.append(f"  Capacity: {total_gb:.1f} GB")
                lines.append(f"  Free: {free_gb:.1f} GB")
                lines.append(f"  Used: {used_gb:.1f} GB")
            except PermissionError:
                continue
    except Exception:
        lines.append("Storage: Unable to determine")
    return "\n".join(f"  {l}" for l in lines)


# ─────────────────────────────────────────
# Displays
# ─────────────────────────────────────────

def _get_display_info() -> str:
    """Gather display information using Windows API and PowerShell."""
    lines = []

    # Use PowerShell to get all displayed resolutions via .NET
    try:
        ps_cmd = (
            'Add-Type -AssemblyName System.Windows.Forms; '
            '[System.Windows.Forms.Screen]::AllScreens | '
            'ForEach-Object { '
            '$b = $_.Bounds; '
            '$p = if($_.Primary){" (Primary)"}else{""}; '
            '"Monitor: $($_.DeviceName)${p}: $($b.Width)x$($b.Height)" '
            '}'
        )
        ps_out = _run_powershell(ps_cmd)
        if ps_out:
            for line in ps_out.splitlines():
                line = line.strip()
                if line:
                    lines.append(line.replace("Monitor: \\\\.\\DISPLAY", "Monitor "))
    except Exception:
        pass

    # Get refresh rates via ctypes API
    if lines:
        try:
            import ctypes
            from ctypes import wintypes

            gdi32 = ctypes.windll.gdi32

            class DEVMODE(ctypes.Structure):
                _fields_ = [
                    ("dmDeviceName", wintypes.CHAR * 32),
                    ("dmSpecVersion", wintypes.WORD),
                    ("dmDriverVersion", wintypes.WORD),
                    ("dmSize", wintypes.WORD),
                    ("dmDriverExtra", wintypes.WORD),
                    ("dmFields", wintypes.DWORD),
                    ("dmPositionX", ctypes.c_long),
                    ("dmPositionY", ctypes.c_long),
                    ("dmDisplayOrientation", wintypes.DWORD),
                    ("dmDisplayFixedOutput", wintypes.DWORD),
                    ("dmColor", ctypes.c_short),
                    ("dmDuplex", ctypes.c_short),
                    ("dmYResolution", ctypes.c_short),
                    ("dmTTOption", ctypes.c_short),
                    ("dmCollate", ctypes.c_short),
                    ("dmFormName", wintypes.CHAR * 32),
                    ("dmLogPixels", wintypes.WORD),
                    ("dmBitsPerPel", wintypes.DWORD),
                    ("dmPelsWidth", wintypes.DWORD),
                    ("dmPelsHeight", wintypes.DWORD),
                    ("dmDisplayFlags", wintypes.DWORD),
                    ("dmDisplayFrequency", wintypes.DWORD),
                    ("dmICMMethod", wintypes.DWORD),
                    ("dmICMIntent", wintypes.DWORD),
                    ("dmMediaType", wintypes.DWORD),
                    ("dmDitherType", wintypes.DWORD),
                    ("dmReserved1", wintypes.DWORD),
                    ("dmReserved2", wintypes.DWORD),
                    ("dmPanningWidth", wintypes.DWORD),
                    ("dmPanningHeight", wintypes.DWORD),
                ]

            dm = DEVMODE()
            dm.dmSize = ctypes.sizeof(DEVMODE)
            monitor_idx = 0
            while gdi32.EnumDisplaySettingsW(None, monitor_idx, ctypes.byref(dm)):
                refresh = dm.dmDisplayFrequency
                if monitor_idx < len(lines):
                    lines[monitor_idx] += f" @ {dm.dmDisplayFrequency} Hz"
                monitor_idx += 1
                dm = DEVMODE()
                dm.dmSize = ctypes.sizeof(DEVMODE)
                if monitor_idx >= 4:
                    break
        except Exception:
            pass

    if not lines:
        # Fallback: basic resolution via ctypes
        try:
            import ctypes
            user32 = ctypes.windll.user32
            width = user32.GetSystemMetrics(0)
            height = user32.GetSystemMetrics(1)
            lines.append(f"Primary Monitor: {width}x{height}")
        except Exception:
            lines.append("Displays: Unable to determine")

    return "\n".join(f"  {l}" for l in lines)


# ─────────────────────────────────────────
# Motherboard
# ─────────────────────────────────────────

def _get_motherboard_info() -> str:
    lines = []
    try:
        # Query via PowerShell
        mb_cmd = (
            'Get-CimInstance Win32_BaseBoard | '
            'Select-Object Manufacturer, Product, Version | '
            'ConvertTo-Csv -NoTypeInformation'
        )
        mb_raw = _run_powershell(mb_cmd)
        boards = _parse_wmic_table(mb_raw) if mb_raw else []

        if boards:
            board = boards[0]
            manufacturer = board.get("Manufacturer", "N/A")
            product = board.get("Product", "N/A")
            version = board.get("Version", "N/A")
            lines.append(f"Manufacturer: {manufacturer}")
            lines.append(f"Model: {product}")
            if version and version != "N/A":
                lines.append(f"Version: {version}")
        else:
            lines.append("Motherboard: Unable to determine")

        # BIOS version via PowerShell
        bios_cmd = (
            'Get-CimInstance Win32_BIOS | '
            'Select-Object Manufacturer, SMBIOSBIOSVersion | '
            'ConvertTo-Csv -NoTypeInformation'
        )
        bios_raw = _run_powershell(bios_cmd)
        bioses = _parse_wmic_table(bios_raw) if bios_raw else []

        if bioses:
            bios = bioses[0]
            bios_ver = bios.get("SMBIOSBIOSVersion", "N/A")
            bios_man = bios.get("Manufacturer", "")
            lines.append(f"BIOS: {bios_ver} ({bios_man})")
    except Exception:
        lines.append("Motherboard: Unable to determine")
    return "\n".join(f"  {l}" for l in lines)


# ─────────────────────────────────────────
# Power
# ─────────────────────────────────────────

def _get_power_info() -> str:
    lines = []
    try:
        # Current power plan via PowerShell
        ps_cmd = (
            'powercfg /GETACTIVESCHEME'
        )
        ps_out = _run_powershell(ps_cmd)
        if ps_out:
            # Output format: "GUID (Name)" - extract the name
            ps_clean = ps_out.splitlines()[0] if ps_out.splitlines() else ""
            if "(" in ps_clean:
                name = ps_clean.split("(")[1].rstrip(")")
                lines.append(f"Power plan: {name.strip()}")
            else:
                lines.append(f"Power plan: {ps_clean.strip()}")
    except Exception:
        pass

    if not any("Power plan" in l for l in lines):
        lines.append("Power plan: Unable to determine")

    try:
        battery = psutil.sensors_battery()
        if battery:
            pct = battery.percent
            charging = "Charging" if battery.power_plugged else "On battery"
            secs = battery.secsleft
            if secs == psutil.POWER_TIME_UNLIMITED:
                time_left = "Unlimited (plugged in)"
            elif secs == psutil.POWER_TIME_UNKNOWN:
                time_left = "Unknown"
            else:
                hours, remainder = divmod(secs, 3600)
                minutes, _ = divmod(remainder, 60)
                time_left = f"{int(hours)}h {int(minutes)}m"
            lines.append(f"Battery: {pct}% ({charging})")
            if time_left:
                lines.append(f"  Time left: {time_left}")
    except Exception:
        pass

    return "\n".join(f"  {l}" for l in lines)


# ─────────────────────────────────────────
# Network
# ─────────────────────────────────────────

def _get_network_info() -> str:
    lines = []
    try:
        stats = psutil.net_if_stats()
        addrs = psutil.net_if_addrs()
        has_active = False
        for name, stat in stats.items():
            if stat.isup:
                has_active = True
                speed = stat.speed
                duplex = stat.duplex
                interface_name_lower = name.lower()
                if "wi" in interface_name_lower or "wlan" in interface_name_lower or "wireless" in interface_name_lower:
                    iface_type = "WiFi"
                elif "eth" in interface_name_lower or "ethernet" in interface_name_lower:
                    iface_type = "Ethernet"
                else:
                    iface_type = "Network"

                speed_str = f"{speed} Mbps" if speed and speed > 0 else "Unknown speed"
                lines.append(f"{iface_type}: {name} ({speed_str})")

                # Get IP address
                if name in addrs:
                    for addr in addrs[name]:
                        if addr.family == socket.AF_INET:
                            lines.append(f"  IP: {addr.address}")
                            break

        if not has_active:
            lines.append("No active network interfaces detected")
    except Exception:
        lines.append("Network: Unable to determine")

    # Internet availability
    try:
        socket.setdefaulttimeout(3)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect(("8.8.8.8", 53))
        lines.append("Internet: Available")
    except (OSError, socket.error):
        lines.append("Internet: Not available")
    finally:
        try:
            socket.setdefaulttimeout(None)
        except Exception:
            pass

    return "\n".join(f"  {l}" for l in lines)


# ─────────────────────────────────────────
# Athena Runtime
# ─────────────────────────────────────────

def _get_athena_runtime_info(
    provider_info: Optional[dict] = None,
    memory_info: Optional[dict] = None,
) -> str:
    """Gather Athena's own runtime information."""
    lines = []

    if provider_info:
        provider_name = provider_info.get("provider", "N/A")
        model = provider_info.get("reasoning_model", "N/A")
        backend = provider_info.get("backend", "N/A")
        gpu_layers = provider_info.get("gpu_layers", "N/A")
        context_size = provider_info.get("context_size", "N/A")
        threads = provider_info.get("threads", "N/A")
        batch_size = provider_info.get("batch_size", "N/A")
        lines.append(f"Provider: {provider_name}")
        lines.append(f"Reasoning model: {model}")
        lines.append(f"Backend: {backend}")
        lines.append(f"GPU layers: {gpu_layers}")
        lines.append(f"Context size: {context_size}")
        lines.append(f"Inference threads: {threads}")
        lines.append(f"Batch size: {batch_size}")
    else:
        lines.append("Provider: Not configured")

    if memory_info:
        wm_size = memory_info.get("working_memory_size", "N/A")
        sm_count = memory_info.get("semantic_memory_count", "N/A")
        ch_size = memory_info.get("chat_history_size", "N/A")
        lines.append(f"Working Memory size: {wm_size}")
        lines.append(f"Semantic Memory entries: {sm_count}")
        lines.append(f"Chat History entries: {ch_size}")

    if not lines:
        lines.append("Athena Runtime: N/A")
    return "\n".join(f"  {l}" for l in lines)


# ─────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────

def generate_system_snapshot(
    tool_prompt: str = "",
    provider_info: Optional[dict] = None,
    memory_info: Optional[dict] = None,
) -> str:
    """Generate a comprehensive system snapshot string.

    Each section is independently gathered so a failure in one section
    does not affect the others.

    Args:
        tool_prompt: The user's prompt/query for the /system command.
        provider_info: Dict with keys like provider, reasoning_model, etc.
        memory_info: Dict with keys like working_memory_size, etc.

    Returns:
        A formatted string containing the complete system snapshot.
    """
    sections = []
    sections.append(("Operating System", _get_os_info()))
    sections.append(("CPU", _get_cpu_info()))
    sections.append(("RAM", _get_ram_info()))
    sections.append(("GPU", _get_gpu_info()))
    sections.append(("Storage", _get_storage_info()))
    sections.append(("Displays", _get_display_info()))
    sections.append(("Motherboard", _get_motherboard_info()))
    sections.append(("Power", _get_power_info()))
    sections.append(("Network", _get_network_info()))
    sections.append(("Athena Runtime", _get_athena_runtime_info(provider_info, memory_info)))

    if tool_prompt:
        sections.insert(0, ("Query", f"  {tool_prompt}"))

    output_parts = []
    for title, content in sections:
        output_parts.append(f"-- {title} --")
        output_parts.append(content)
        output_parts.append("")

    return "\n".join(output_parts).rstrip()
