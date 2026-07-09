"""
GGUF Model Discovery and Selection

Scans the configured model directory for GGUF model files so Athena works
with whatever model the user drops into that folder, instead of a hardcoded
filename.

Behaviour:
    - Zero models found  -> raise FileNotFoundError with a helpful message.
    - Exactly one model  -> use it automatically.
    - Multiple models    -> prompt the user to pick one by number.

The search is recursive, so models organised into sub-folders are also found.
"""

from pathlib import Path


def discover_models(model_directory: str | Path) -> list[Path]:
    """Return all GGUF model files under ``model_directory``, sorted by name.

    Args:
        model_directory: Directory to search (searched recursively).

    Returns:
        A sorted list of paths to ``.gguf`` files. Empty if none are found.

    Raises:
        FileNotFoundError: If the directory itself does not exist.
    """
    directory = Path(model_directory)
    if not directory.is_dir():
        raise FileNotFoundError(
            f"Model directory does not exist: {directory}\n"
            f"Create it and place a .gguf model file inside."
        )

    return sorted(directory.rglob("*.gguf"), key=lambda p: str(p).lower())


def resolve_model_path(model_directory: str | Path) -> str:
    """Resolve which GGUF model to load from ``model_directory``.

    Auto-selects when there is a single model; otherwise prompts the user to
    choose one by number.

    Args:
        model_directory: Directory containing GGUF model files.

    Returns:
        The path to the chosen model file, as a string.

    Raises:
        FileNotFoundError: If the directory or any GGUF model is missing.
    """
    models = discover_models(model_directory)

    if not models:
        raise FileNotFoundError(
            f"No .gguf model files found in: {Path(model_directory)}\n"
            f"Place a GGUF model file in that folder and try again."
        )

    if len(models) == 1:
        return str(models[0])

    return str(_prompt_for_model(models, Path(model_directory)))


def resolve_model_path_optional(model_directory: str | Path) -> str | None:
    """Like resolve_model_path, but returns None instead of raising.

    Used for the optional learning-model folder: a missing directory or an
    empty one simply means "no dedicated model here" so the caller can fall
    back to the reasoning model.

    Args:
        model_directory: Directory that may contain GGUF model files.

    Returns:
        The chosen model path, or None if the directory is missing or empty.
    """
    try:
        models = discover_models(model_directory)
    except FileNotFoundError:
        return None

    if not models:
        return None
    if len(models) == 1:
        return str(models[0])
    return str(_prompt_for_model(models, Path(model_directory)))


def _prompt_for_model(models: list[Path], base_directory: Path) -> Path:
    """Ask the user to select a model from a numbered list.

    Args:
        models: Discovered model files (at least two).
        base_directory: Directory used to display concise relative names.

    Returns:
        The selected model path.
    """
    print("Multiple models found. Select one to load:\n")
    for index, model in enumerate(models, start=1):
        try:
            display = model.relative_to(base_directory)
        except ValueError:
            display = model.name
        print(f"  [{index}] {display}")
    print()

    while True:
        choice = input(f"Enter model number (1-{len(models)}): ").strip()
        if choice.isdigit():
            number = int(choice)
            if 1 <= number <= len(models):
                selected = models[number - 1]
                print(f"\nSelected: {selected.name}\n")
                return selected
        print(f"Invalid selection. Enter a number between 1 and {len(models)}.")
