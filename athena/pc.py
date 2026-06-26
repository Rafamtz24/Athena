"""
Athena PC Module
Version: 0.1.0
"""

import platform
import psutil
import cpuinfo


def get_system_info():
    """Return basic live system information."""

    memory = psutil.virtual_memory()

    return {
        "computer": platform.node(),
        "windows": platform.platform(),
        "cpu": {
            "name": cpuinfo.get_cpu_info()["brand_raw"],
            "physical_cores": psutil.cpu_count(logical=False),
            "logical_cores": psutil.cpu_count(logical=True),
            "usage_percent": psutil.cpu_percent(interval=1),
        },
        "ram": {
            "total_gb": round(memory.total / (1024 ** 3), 2),
            "used_gb": round(memory.used / (1024 ** 3), 2),
            "available_gb": round(memory.available / (1024 ** 3), 2),
            "percent": memory.percent,
        },
        "disks": [
            {
                "drive": p.device,
                "mount": p.mountpoint,
                "filesystem": p.fstype,
            }
            for p in psutil.disk_partitions()
        ],
    }