"""
Tests for first-run onboarding (athena/onboarding.py).

Onboarding runs when no reasoning model is installed: it creates the model
folders, prints a consumer-friendly guide (what a .gguf is, where to get one,
the optional learning model), and recommends model sizes for the detected
hardware. These tests exercise folder creation, model detection, the memory
budget, the recommendation tiers, and the guide's content — using temp
directories and fake hardware so no real model or detection is needed.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from athena import onboarding
from athena.config.settings import get_settings
from athena.hardware.detector import CpuInfo, GpuInfo, HardwareInfo, RamInfo


def _fake_hardware(vram_gb=None, ram_gb=16.0):
    """Build a HardwareInfo with the given VRAM (None = CPU-only) and RAM."""
    vram_bytes = int(vram_gb * 1024 ** 3) if vram_gb is not None else None
    return HardwareInfo(
        cpu=CpuInfo(model="Test CPU", physical_cores=8, logical_threads=16),
        ram=RamInfo(total_bytes=int(ram_gb * 1024 ** 3)),
        gpu=GpuInfo(vendor="NVIDIA" if vram_gb else "Unknown",
                    model="Test GPU" if vram_gb else "Unknown",
                    vram_bytes=vram_bytes),
        backend="CUDA" if vram_gb else "CPU",
    )


def _point_model_dirs_at(tmp_path, monkeypatch):
    """Redirect the model folders into a temp dir; return (reason, learning).

    Uses monkeypatch so the global settings singleton is restored after the
    test — other tests in the same session see the original directories.
    """
    provider = get_settings().provider
    reason = tmp_path / "models" / "reason"
    learning = tmp_path / "models" / "learning"
    monkeypatch.setattr(provider, "reason_model_directory", str(reason))
    monkeypatch.setattr(provider, "learning_model_directory", str(learning))
    return reason, learning


def test_ensure_model_folders_creates_both(tmp_path, monkeypatch):
    reason, learning = _point_model_dirs_at(tmp_path, monkeypatch)
    assert not reason.exists() and not learning.exists()
    onboarding.ensure_model_folders()
    assert reason.is_dir() and learning.is_dir()


def test_has_reasoning_model_detects_gguf(tmp_path, monkeypatch):
    reason, _ = _point_model_dirs_at(tmp_path, monkeypatch)
    onboarding.ensure_model_folders()
    assert onboarding.has_reasoning_model() is False

    # A non-gguf file doesn't count; a .gguf (even in a subfolder) does.
    (reason / "readme.txt").write_text("not a model")
    assert onboarding.has_reasoning_model() is False
    sub = reason / "qwen"
    sub.mkdir()
    (sub / "model.gguf").write_bytes(b"\x00")
    assert onboarding.has_reasoning_model() is True


def test_usable_memory_prefers_vram():
    gb, basis = onboarding.usable_memory_gb(_fake_hardware(vram_gb=12.0))
    assert gb == 12.0
    assert "video memory" in basis


def test_usable_memory_falls_back_to_half_ram():
    gb, basis = onboarding.usable_memory_gb(_fake_hardware(vram_gb=None, ram_gb=16.0))
    assert gb == 8.0
    assert "system memory" in basis


def test_recommendation_tiers_scale_with_memory():
    # Every tier returns both roles, and bigger budgets get bigger models.
    tiny = onboarding.recommend_models(2.0)
    common = onboarding.recommend_models(8.0)   # a typical 8 GB gaming GPU
    mid = onboarding.recommend_models(10.0)
    big = onboarding.recommend_models(24.0)
    for recs in (tiny, common, mid, big):
        assert recs["reasoning"] and recs["learning"]
    assert "1B" in tiny["reasoning"][0]
    # An 8 GB card comfortably runs an 8B Q4 (~5 GB file) — don't undersell it.
    assert "8B" in common["reasoning"][0]
    assert "8B" in mid["reasoning"][0]
    assert "32B" in big["reasoning"][0]
    # The learning model is always small (3B or under).
    for recs in (tiny, mid, big):
        assert any(size in recs["learning"][0] for size in ("1B", "1.5B", "3B"))


def test_first_run_guide_mentions_the_essentials(tmp_path, capsys, monkeypatch):
    _point_model_dirs_at(tmp_path, monkeypatch)
    onboarding.print_first_run_guide(hardware=_fake_hardware(vram_gb=8.0))
    out = capsys.readouterr().out

    assert "Welcome to Athena" in out
    assert ".gguf" in out
    assert "huggingface.co" in out
    assert "reason" in out          # tells the user where to put the file
    assert "learning" in out       # explains the optional learning model
    assert "Q4_K_M" in out          # tells the user which file to pick
    assert "Test GPU" in out        # detected hardware is shown
    assert "8B" in out              # 8 GB VRAM tier recommendation


def test_first_run_guide_survives_failed_detection(tmp_path, capsys, monkeypatch):
    _point_model_dirs_at(tmp_path, monkeypatch)

    class ExplodingDetector:
        def detect(self):
            raise RuntimeError("no hardware info")

    import athena.hardware as hw
    monkeypatch.setattr(hw, "HardwareDetector", ExplodingDetector)
    onboarding.print_first_run_guide()

    out = capsys.readouterr().out
    assert "Welcome to Athena" in out
    assert "rule of thumb" in out   # generic fallback advice shown instead


if __name__ == "__main__":
    import subprocess

    raise SystemExit(subprocess.call([sys.executable, "-m", "pytest", __file__, "-v"]))
