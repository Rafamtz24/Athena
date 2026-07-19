"""
Minimal GGUF metadata reader

Reads just enough of a GGUF file's header to answer "how many layers does this
model have?" — the number needed to turn "offload as much as fits" into a
concrete ``n_gpu_layers`` for the in-process llama-cpp-python provider, which
has no layer-fitting logic of its own.

Only the header is read: metadata sits at the front of the file, before any
tensor data, so this costs a few KB regardless of whether the model is 500 MB
or 50 GB.
"""

import struct
from pathlib import Path


_MAGIC = b"GGUF"

# GGUF metadata value type tags, and the struct format for the fixed-width ones.
_UINT8, _INT8, _UINT16, _INT16, _UINT32, _INT32 = 0, 1, 2, 3, 4, 5
_FLOAT32, _BOOL, _STRING, _ARRAY, _UINT64, _INT64, _FLOAT64 = 6, 7, 8, 9, 10, 11, 12

_FIXED_FORMATS = {
    _UINT8: "<B", _INT8: "<b",
    _UINT16: "<H", _INT16: "<h",
    _UINT32: "<I", _INT32: "<i",
    _FLOAT32: "<f", _BOOL: "<?",
    _UINT64: "<Q", _INT64: "<q",
    _FLOAT64: "<d",
}

# Metadata is small; refuse to scan a file that claims otherwise rather than
# reading an arbitrary amount of a corrupt or hostile file into memory.
_MAX_HEADER_BYTES = 16 * 1024 * 1024


class _Reader:
    """Sequential reader over the header bytes."""

    def __init__(self, data: bytes):
        self._data = data
        self._pos = 0

    def take(self, count: int) -> bytes:
        end = self._pos + count
        if end > len(self._data):
            raise ValueError("GGUF header ended unexpectedly")
        chunk = self._data[self._pos:end]
        self._pos = end
        return chunk

    def scalar(self, fmt: str):
        return struct.unpack(fmt, self.take(struct.calcsize(fmt)))[0]

    def string(self) -> str:
        length = self.scalar("<Q")
        return self.take(length).decode("utf-8", errors="replace")

    def value(self, value_type: int):
        """Read one metadata value, discarding anything we do not need."""
        if value_type == _STRING:
            return self.string()

        if value_type == _ARRAY:
            element_type = self.scalar("<I")
            count = self.scalar("<Q")
            return [self.value(element_type) for _ in range(count)]

        fmt = _FIXED_FORMATS.get(value_type)
        if fmt is None:
            raise ValueError(f"unknown GGUF value type {value_type}")
        return self.scalar(fmt)


def read_model_info(model_path: str | Path) -> dict:
    """Return what the header says about the model's shape.

    Keys (each None when the file does not state it):
        layer_count:  transformer block count
        expert_count: number of Mixture-of-Experts experts; None or 0 means the
                      model is dense

    Both live under architecture-namespaced keys (``qwen3moe.block_count``,
    ``llama.block_count``, …), so ``general.architecture`` is read first to know
    which prefix applies.

    Never raises for a malformed or unreadable file — every caller's fallback is
    the safe conservative path, and a metadata quirk must not stop a model from
    loading.
    """
    unknown = {"layer_count": None, "expert_count": None}

    try:
        with open(model_path, "rb") as handle:
            data = handle.read(_MAX_HEADER_BYTES)
    except OSError:
        return unknown

    if not data.startswith(_MAGIC):
        return unknown

    reader = _Reader(data)
    collected: dict[str, int] = {}
    architecture = None

    try:
        reader.take(4)              # magic
        reader.scalar("<I")         # format version
        reader.scalar("<Q")         # tensor count
        kv_count = reader.scalar("<Q")

        for _ in range(kv_count):
            key = reader.string()
            value = reader.value(reader.scalar("<I"))

            if key == "general.architecture":
                architecture = value
            elif key.endswith((".block_count", ".expert_count")):
                collected[key] = value
    except (ValueError, struct.error, UnicodeDecodeError):
        # Keep whatever was parsed before the file went strange; a partial
        # header still commonly contains both keys, which appear early.
        pass

    def pick(suffix: str) -> int | None:
        if architecture:
            namespaced = collected.get(f"{architecture}{suffix}")
            if namespaced is not None:
                return int(namespaced)
        # Some files namespace these unusually; a single unambiguous match is
        # still safe to use.
        matches = [v for k, v in collected.items() if k.endswith(suffix)]
        return int(matches[0]) if len(matches) == 1 else None

    return {
        "layer_count": pick(".block_count"),
        "expert_count": pick(".expert_count"),
    }


def read_layer_count(model_path: str | Path) -> int | None:
    """Return the model's transformer block count, or None if undetermined."""
    return read_model_info(model_path)["layer_count"]
