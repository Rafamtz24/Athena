# Athena

A local-first AI assistant. Athena runs a language model entirely on your own
computer — private, offline, and free — and builds a persistent memory of your
conversations over time.

## Quick start

1. **Download or clone this repository.**
2. **Run `setup.bat`** (double-click it).
   It installs Python if needed, creates a virtual environment, installs the
   dependencies, and builds the inference backend for your GPU.
3. **Download a model.** Setup creates a `models\reason\` folder. Put a `.gguf`
   model file in it — Athena recommends sizes that suit your hardware the first
   time you start it.
4. **Run `Athena.bat`** to start chatting.

Setup is safe to re-run; it repairs a partial install rather than starting over.

## Requirements

| | |
|---|---|
| OS | Windows 10 / 11 |
| Python | 3.10 – 3.13 (setup installs 3.12 if you have none) |
| Disk | ~500 MB for Athena, plus the size of your model |
| GPU | Optional. NVIDIA, AMD and Intel are all supported |

Athena runs on CPU alone, just more slowly.

**Nothing is compiled and no build tools are required.** Setup downloads a
prebuilt llama.cpp runtime (about 32 MB for Vulkan) matched to your GPU, and
Athena runs it as a background process. Total setup time is a couple of
minutes.

## Getting a model

Athena uses GGUF model files:

1. Go to [huggingface.co](https://huggingface.co)
2. Search for a model name followed by `GGUF` — for example `Qwen3 8B GGUF`
3. Under **Files**, download the one with `Q4_K_M` in the name (the best
   quality-to-size balance)
4. Put it in `models\reason\`

Optionally, drop a second, much smaller model into `models\learning\`. Athena
uses it for the background memory-extraction pass that runs after each answer,
which keeps replies snappy. Without one, the main model does that work too.

## Commands

Inside the chat:

| Command | Description |
|---|---|
| `/help` | List available commands |
| `/context size [value]` | Show or set the conversation context size |
| `/think [on\|off]` | Toggle the model's reasoning trace |
| `/learn [on\|off]` | Toggle post-answer memory learning |
| `/system` | Report a system snapshot (CPU, memory, GPU) |
| `/book` | Reading mode — answer from a selected PDF |
| `/tarot` | Tarot mode |
| `/serve [port]` | Serve the model over an OpenAI-compatible API |
| `exit` / `quit` | Leave the chat |

## Setup options

Run these from a terminal:

```
setup.bat                    install, auto-detecting your GPU
setup.bat -Backend vulkan    force Vulkan (AMD / Intel / NVIDIA)
setup.bat -Backend cuda      force CUDA (NVIDIA, much larger download)
setup.bat -Backend cpu       no GPU offload
setup.bat -Backend none      skip the runtime; use LM Studio instead
setup.bat -Dev               also install the test dependencies
setup.bat -Force             re-download the llama.cpp runtime
setup.bat -Yes               never pause for confirmation
```

Setup picks Vulkan for AMD and Intel GPUs, CUDA for NVIDIA, and CPU otherwise.

## Development

```
setup.bat -Dev
.venv\Scripts\python -m pytest athena
```

Some tests construct a real `AthenaBrain` and need a model in `models\reason\`;
they fail without one.

## Documentation

| Document | Contents |
|---|---|
| [docs/SETUP.md](docs/SETUP.md) | Detailed setup, backends, troubleshooting |
| [ARCHITECTURE.md](ARCHITECTURE.md) | How Athena is put together |
| [VISION.md](VISION.md) | What Athena is for |
| [ATHENA_CORE_PRINCIPLES.md](ATHENA_CORE_PRINCIPLES.md) | Design principles |
| [ENGINEERING.md](ENGINEERING.md) | Engineering standards |
| [ROADMAP.md](ROADMAP.md) | Planned work |
| [pipelineexplanation.md](pipelineexplanation.md) | The reasoning pipeline |
