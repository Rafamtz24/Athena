"""
First-Run Onboarding

Greets users who start Athena before installing a model. Instead of a crash
(the provider raises FileNotFoundError when the reasoning folder is empty),
they get a friendly guide: what a .gguf model is, where to download one, why a
second small "learning" model makes Athena feel faster, and — based on their
detected hardware — which model sizes will actually run well on their machine.

Everything here is best-effort: hardware detection failures degrade to generic
advice, never to a crash.
"""

from pathlib import Path

from athena.config.settings import get_settings


def ensure_model_folders() -> None:
    """Create the model folders if missing so users have a place to drop files."""
    provider = get_settings().provider
    Path(provider.reason_model_directory).mkdir(parents=True, exist_ok=True)
    Path(provider.learning_model_directory).mkdir(parents=True, exist_ok=True)


def has_reasoning_model() -> bool:
    """True when at least one .gguf file exists under the reasoning folder."""
    directory = Path(get_settings().provider.reason_model_directory)
    return directory.is_dir() and any(directory.rglob("*.gguf"))


def usable_memory_gb(hardware) -> tuple[float, str]:
    """Estimate how much memory is realistically available for a model.

    GPU with known VRAM: the model must fit in VRAM (plus context headroom),
    so VRAM is the budget. CPU-only or unknown VRAM: models run from system
    RAM, but the OS and apps need room, so only about half of RAM is a safe
    budget.

    Returns:
        (gigabytes, human-readable description of what the number is).
    """
    vram = hardware.gpu.vram_gb if hardware.gpu is not None else None
    if vram is not None and vram > 0:
        return vram, f"{vram:g} GB of video memory on your {hardware.gpu.model}"
    ram = hardware.ram.total_gb
    return round(ram * 0.5, 1), (
        f"about half of your {ram:g} GB of system memory (no dedicated "
        f"graphics memory was detected, so the model runs on the CPU)"
    )


def recommend_models(usable_gb: float) -> dict:
    """Pick reasoning / learning model recommendations for a memory budget.

    Sizes are Q4_K_M quantizations (the sweet spot of quality vs. size) with
    approximate download sizes, leaving headroom in the budget for the
    context window. Users can substitute any model of similar size.

    Returns:
        {"reasoning": [str, ...], "learning": [str, ...]}
    """
    # Thresholds leave ~2.5-3 GB of the budget for the context window and
    # runtime overhead beyond the model file itself.
    if usable_gb >= 22:
        reasoning = [
            "Qwen3 32B (Q4_K_M, ~19 GB download) - top-tier local reasoning",
            "Gemma 3 27B (Q4_K_M, ~16 GB download)",
        ]
    elif usable_gb >= 13:
        reasoning = [
            "Qwen3 14B (Q4_K_M, ~9 GB download) - excellent all-rounder",
            "Gemma 3 12B (Q4_K_M, ~7 GB download)",
        ]
    elif usable_gb >= 7.5:
        reasoning = [
            "Qwen3 8B (Q4_K_M, ~5 GB download) - great quality for its size",
            "Llama 3.1 8B Instruct (Q4_K_M, ~5 GB download)",
        ]
    elif usable_gb >= 5:
        reasoning = [
            "Qwen3 4B (Q4_K_M, ~2.5 GB download) - fast and surprisingly capable",
            "Gemma 3 4B (Q4_K_M, ~2.5 GB download)",
        ]
    elif usable_gb >= 3:
        reasoning = [
            "Qwen2.5 3B Instruct (Q4_K_M, ~2 GB download)",
            "Llama 3.2 3B Instruct (Q4_K_M, ~2 GB download)",
        ]
    else:
        reasoning = [
            "Llama 3.2 1B Instruct (Q4_K_M, ~0.8 GB download)",
            "Qwen2.5 1.5B Instruct (Q4_K_M, ~1 GB download)",
        ]

    # The learning model should be much smaller than the reasoning model —
    # it runs quietly after every answer, so speed matters more than depth.
    # It shares memory with the resident reasoning model, so it stays modest.
    if usable_gb >= 13:
        learning = ["Qwen2.5 3B Instruct (Q4_K_M, ~2 GB download)"]
    elif usable_gb >= 5:
        learning = ["Qwen2.5 1.5B Instruct (Q4_K_M, ~1 GB download)"]
    else:
        learning = ["Llama 3.2 1B Instruct (Q4_K_M, ~0.8 GB download)"]

    return {"reasoning": reasoning, "learning": learning}


def print_first_run_guide(hardware=None) -> None:
    """Print the consumer-friendly no-model-installed guide.

    Args:
        hardware: Optional pre-detected HardwareInfo (mainly for tests).
            When None, detection runs here; failures fall back to generic
            size advice.
    """
    provider = get_settings().provider
    reason_dir = str(Path(provider.reason_model_directory))
    learning_dir = str(Path(provider.learning_model_directory))

    print("=" * 64)
    print("  Welcome to Athena!")
    print("=" * 64)
    print(f"""
Athena runs AI entirely on your own computer - private, offline,
and free. But it looks like no AI model is installed yet:

    there is no .gguf file in the '{reason_dir}' folder.

WHAT IS A .GGUF FILE?
    A .gguf file is an AI model - the "brain" Athena thinks with -
    packaged in a format your computer can run locally. You download
    it once (like an app), drop it in a folder, and it's yours: no
    subscription, no internet needed afterwards.

HOW TO GET ONE
    1. Go to https://huggingface.co (the main site for AI models)
    2. Search for a model name below followed by the word GGUF
       (for example: "Qwen3 8B GGUF")
    3. On the model page, open "Files" and download the file whose
       name contains Q4_K_M (best balance of quality and size)
    4. Put the downloaded .gguf file in:  {reason_dir}
    5. Start Athena again - that's it!""")

    # ── Hardware-tailored recommendations (best-effort) ──
    detected = hardware
    if detected is None:
        try:
            from athena.hardware import HardwareDetector

            print("\nChecking your hardware to recommend the right model size...")
            detected = HardwareDetector().detect()
        except Exception:
            detected = None

    if detected is not None:
        budget, basis = usable_memory_gb(detected)
        recs = recommend_models(budget)
        print(f"""
YOUR COMPUTER
    Processor:  {detected.cpu.model}
    Memory:     {detected.ram.total_gb:g} GB RAM
    Graphics:   {detected.gpu.model}"""
              + (f" ({detected.gpu.vram_gb:g} GB video memory)"
                 if detected.gpu.vram_gb else ""))
        print(f"""
RECOMMENDED FOR YOUR MACHINE
    Based on {basis}, these will run well
    (any model of a similar size works too):

    Reasoning model  ->  put in '{reason_dir}':""")
        for item in recs["reasoning"]:
            print(f"        * {item}")
        print(f"""
    Learning model (optional)  ->  put in '{learning_dir}':""")
        for item in recs["learning"]:
            print(f"        * {item}")
    else:
        print(f"""
WHICH SIZE TO CHOOSE?
    As a rule of thumb: pick a model whose download size is smaller
    than your graphics card's video memory (or half your RAM if you
    don't have a dedicated graphics card).""")

    print(f"""
OPTIONAL, BUT RECOMMENDED: A SECOND "LEARNING" MODEL
    After each answer, Athena quietly studies the conversation to
    remember useful facts about you. Out of the box it uses the big
    reasoning model for this, which makes replies feel slower.

    Drop a second, SMALL .gguf into '{learning_dir}' and
    Athena will use it for learning instead - same memory quality,
    much snappier conversations.
""")
    print("=" * 64)
