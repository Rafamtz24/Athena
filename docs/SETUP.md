# Setup Guide

Everything needed to take a freshly cloned Athena to a working install, plus
what to do when a step goes wrong.

For the short version, see the [README](../README.md).

---

## The one-step install

Double-click **`setup.bat`**.

It performs, in order:

1. **Finds a supported Python** (3.10–3.13), installing 3.12 via `winget` if
   none is present.
2. **Creates `.venv`** — a project-local virtual environment, so Athena never
   touches your system Python packages.
3. **Detects your GPU** and picks a runtime.
4. **Installs the dependencies** from `requirements.txt`.
5. **Downloads a prebuilt llama.cpp runtime** into `runtime\llama\`.
6. **Creates `models\reason\` and `models\learning\`.**

Every step is idempotent. If setup fails partway, fix the cause and run it
again — it picks up where it left off rather than redoing everything.

---

## How Athena runs models

Athena's default provider (`llamaserver`) runs your model through llama.cpp's
`llama-server` binary, started and stopped automatically as a background
process. You never see it: it binds to loopback on a random port, is protected
by a per-session API key, and shuts down with Athena.

**Why not the llama-cpp-python bindings?** That package publishes no prebuilt
Windows wheels — only a source archive. Installing it means compiling llama.cpp
locally: several GB of Visual Studio build tools and a 20–30 minute build.
llama.cpp itself ships ready-made binaries for every backend, so Athena
downloads one (32 MB for Vulkan) and talks to it over local HTTP instead. Same
models, same GPU acceleration, no compiler. The loopback round-trip costs a
negligible amount of latency.

The in-process route is still available if you prefer it — see
[In-process inference](#in-process-inference-advanced).

---

## Backends

Choose with `setup.bat -Backend <name>`. All are prebuilt downloads.

| Backend | Best for | Size | Notes |
|---|---|---|---|
| `auto` | Everyone (default) | — | Picks from the detected GPU |
| `vulkan` | AMD, Intel, NVIDIA | ~32 MB | Widest hardware support |
| `cuda` | NVIDIA | ~370 MB | Includes the CUDA runtime DLLs |
| `cpu` | No GPU / fallback | ~17 MB | Always works, slowest |
| `none` | External providers | — | Pair with LM Studio |

By default setup installs a specific llama.cpp build that has been verified
against Athena. To take the newest release instead:

```
setup.bat -LlamaBuild latest
```

### What `auto` decides

| Detected GPU | Backend |
|---|---|
| NVIDIA | `cuda` |
| AMD | `vulkan` |
| Intel | `vulkan` |
| None / unknown | `cpu` |

**Why Vulkan for AMD?** ROCm — AMD's compute stack — has limited Windows
support and does not cover most consumer Radeon cards. Vulkan is the backend
llama.cpp offers that works reliably across the whole AMD range on Windows.
Athena also disables Flash Attention automatically on Vulkan, because
llama.cpp's Vulkan implementation of it can hang the GPU and trip a driver
timeout.

Vulkan works on NVIDIA too, and is a tenth the download of the CUDA build. CUDA
is usually a little faster on NVIDIA hardware, but Vulkan is a perfectly good
choice if you would rather not fetch 370 MB.

---

## Using LM Studio

[LM Studio](https://lmstudio.ai) ships its own accelerated llama.cpp runtimes —
including Vulkan and ROCm for AMD — so nothing has to be compiled. Athena
already speaks its API.

1. Install Athena without a local runtime:

   ```
   setup.bat -Backend none
   ```

2. Install LM Studio, download a model inside it, and start its local server
   (**Developer** tab → **Start Server**). It listens on port `1234` by default.

3. Point Athena at it. In `athena/config/settings.py`, under `ProviderSettings`:

   ```python
   provider: str = "lmstudio"
   base_url: str = "http://127.0.0.1:1234"
   model: str = "your-model-name-in-lm-studio"
   ```

4. Start Athena with `Athena.bat`.

The trade-off: LM Studio has to be running whenever you use Athena, and model
loading is managed there rather than by Athena.

---

## Manual installation

If you prefer to do it yourself, or `setup.bat` cannot run:

```powershell
# 1. Install Python 3.12 from https://www.python.org/downloads/
#    Tick "Add python.exe to PATH".

# 2. Create the virtual environment
python -m venv .venv

# 3. Install the dependencies
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r requirements.txt

# 4. Create the folders Athena expects
mkdir models\reason, models\learning, runtime\llama
```

5. Download a llama.cpp Windows build from
   [the releases page](https://github.com/ggml-org/llama.cpp/releases) and
   extract it into `runtime\llama\` so that `runtime\llama\llama-server.exe`
   exists. Pick the asset matching your GPU:

   | GPU | Asset |
   |---|---|
   | AMD / Intel | `llama-<build>-bin-win-vulkan-x64.zip` |
   | NVIDIA | `llama-<build>-bin-win-cuda-<ver>-x64.zip` **and** `cudart-llama-bin-win-cuda-<ver>-x64.zip` |
   | None | `llama-<build>-bin-win-cpu-x64.zip` |

This is also the route to take on a machine with no internet access to GitHub:
copy the archive across by hand and extract it to the same place.

---

## In-process inference (advanced)

If you would rather run the model inside the Python process than as a
subprocess, install `llama-cpp-python` yourself and switch providers. This
requires [Visual Studio Build Tools](https://visualstudio.microsoft.com/downloads/)
with the **Desktop development with C++** workload, plus the
[Vulkan SDK](https://vulkan.lunarg.com/sdk/home) for Vulkan builds.

```powershell
# Vulkan (AMD / Intel)
$env:CMAKE_ARGS = "-DGGML_VULKAN=ON"
.venv\Scripts\python -m pip install llama-cpp-python==0.3.32 --no-cache-dir

# CUDA (NVIDIA)
$env:CMAKE_ARGS = "-DGGML_CUDA=ON"
.venv\Scripts\python -m pip install llama-cpp-python==0.3.32 --no-cache-dir

# CPU only
.venv\Scripts\python -m pip install llama-cpp-python==0.3.32 --no-cache-dir
```

Then set `provider` to `"llamacpp"` in `athena/config/settings.py`. Expect the
build to take 20–30 minutes.

---

## Troubleshooting

### "Python was not found" when you already installed it

Windows ships a stub `python.exe` that opens the Microsoft Store instead of
running Python. Setup ignores that stub deliberately.

If you installed Python but setup cannot see it, open a **new** terminal — a
running process does not pick up `PATH` changes — and try again. Failing that,
reinstall Python with **"Add python.exe to PATH"** ticked.

### "The llama.cpp runtime is missing"

Athena could not find `runtime\llama\llama-server.exe`. Run `setup.bat` — or
`setup.bat -Force` to re-download a runtime that was interrupted or deleted.

### The download fails

Setup fetches the runtime from GitHub. On a restricted network this may be
blocked. Either download the archive by hand (see
[Manual installation](#manual-installation)), or skip local inference with
`setup.bat -Backend none` and use LM Studio.

### The GPU is not being used

Ask the runtime directly what it can see:

```
.\runtime\llama\llama-bench.exe --list-devices
```

A healthy Vulkan install prints something like:

```
Available devices:
  Vulkan0: AMD Radeon RX 9060 XT (8144 MiB, 7272 MiB free)
```

If no device is listed, update your graphics drivers — Vulkan support comes
from the driver, not from Athena. If a device *is* listed but Athena still feels
slow, check what it detected:

```
.venv\Scripts\python -c "from athena.hardware import HardwareDetector; print(HardwareDetector().detect())"
```

You can also confirm which runtime is installed by reading
`runtime\llama\.athena-runtime.json`, and switch with e.g.
`setup.bat -Backend vulkan -Force`.

### The model takes a long time to load

The first load of a large model reads several GB from disk. Subsequent starts
are much faster thanks to the OS file cache. If loading never finishes, Athena
reports the server's own log — that usually names the cause (not enough VRAM,
a corrupt .gguf, an unsupported quantisation).

### The virtual environment is broken

Delete it and re-run setup:

```
rmdir /s /q .venv
setup.bat
```

### Tests fail

```
setup.bat -Dev
.venv\Scripts\python -m pytest athena
```

Tests that construct a real `AthenaBrain` need a model in `models\reason\` and
fail without one. That is expected on a model-less install.

---

## What setup installs, and where

| Item | Location | Notes |
|---|---|---|
| Python 3.12 | `%LOCALAPPDATA%\Programs\Python\Python312` | Only if absent |
| Virtual environment | `.venv\` | Project-local, gitignored |
| llama.cpp runtime | `runtime\llama\` | Gitignored; re-run setup to restore |
| Model folders | `models\reason\`, `models\learning\` | Gitignored; you fill these |

Python is the only thing installed system-wide, and only when you do not
already have a supported version. Everything else lives inside the project
folder: deleting `.venv\` and `runtime\` returns the machine to a clean state,
and re-running `setup.bat` rebuilds both.
