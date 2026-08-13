# Fattervoice

`fattervoice` is a wrapper around **OmniVoice** for use as a fast and accurate voice cloning TTS server with OpenAPI/Wyoming endpoints.

## Features
- OpenAI-compatible endpoint `POST /v1/audio/speech`
- Wyoming Protocol support for Home Assistant and Streaming support for low latency voice assistants
- Offline-oriented model prefetch for Docker/runtime use
- Docker first implimentation for ease of deployment
- Optimized for low VRAM environments, generally consuming between 2.5-3GB of VRAM on an NVIDIA interface

## Table Of Contents
1. [Voices Configuration](#voices-configuration)
2. [Server Configuration](#server-configuration)
3. [Docker Deployment](#docker)

## Voices Configuration

### Reference Voice Audio
For best results, reference audio should be clearly spoken with minimal background noise. The reference audio clips should ideally be between 10-15 seconds in length.

### Voices directory layout

Fattervoice will search the given `voices` for an audio file. It then uses the name of that audio file to search for an `<name>.ref.txt` and `<name>.instruct.txt` file.

`<name>.ref.txt` is REQUIRED, and should be a direct transcript of the matched audio file
`<name>.instruct.txt` is OPTIONAL, it is guidance on voice synthesis, see the [Omnivoice Documentation](https://github.com/k2-fsa/OmniVoice/blob/master/docs/voice-design.md) for more details.

Examples:

```
voices/
├── hank.wav
├── hank.ref.txt
├── hank.instruct.txt          # optional: voice-design/style guidance
├── jane.flac
├── jane.ref.txt
├── narrator.mp3
├── narrator.ref.txt
├── robot.aac
└── robot.ref.txt
```

The basename becomes the public `voice` identifier exposed through both the OpenAI-compatible API and Wyoming.

Because `fattervoice` requires reference transcripts for every voice, OmniVoice auto-ASR is intentionally **not** part of the normal server flow. This keeps voice cloning faster, more stable, and easier to run offline.

### Supported audio formats

| Format | Extension(s) | Decoder |
| --- | --- | --- |
| WAV | `.wav` | stdlib `wave` |
| AIFF | `.aif`, `.aiff` | `soundfile` (libsndfile) |
| FLAC | `.flac` | `soundfile` (libsndfile) |
| Ogg Vorbis | `.ogg` | `soundfile` (libsndfile) |
| MP3 | `.mp3` | `pydub` + `ffmpeg` |
| Opus | `.opus` | `pydub` + `ffmpeg` |
| AAC | `.aac` | `pydub` + `ffmpeg` |
| ALAC | `.alac` | `pydub` + `ffmpeg` |

All formats are validated at startup so broken or unreadable reference files fail fast before any traffic is served. Formats decoded through `pydub` require `ffmpeg` on `PATH` (included in the official Docker image).

## Server configuration 

### Standard Configuration Reference

| CLI flag | ENV fallback | Default | Description |
| --- | --- | --- | --- |
| `--voices-dir` | `FATTERVOICE_VOICES_DIR` | `voices` | Directory containing `<voice>.<audio>` + `<voice>.ref.txt` pairs. Change when your voice files live outside the project root (e.g. a Docker volume mount). |
| `--preload-voice` | `FATTERVOICE_PRELOAD_VOICE` | *(none)* | Voice ID whose clone prompt should be built at startup instead of lazily on first request. The value must match a voice basename in `--voices-dir` (e.g. `hank` for `hank.wav` + `hank.ref.txt`). Useful when a single voice handles most traffic and you want to avoid the first-request cold-start penalty. |
| `--openapi-host` | `FATTERVOICE_OPENAPI_HOST` | `0.0.0.0` | Bind address for the OpenAI-compatible HTTP server. |
| `--openapi-port` | `FATTERVOICE_OPENAPI_PORT` | `8000` | Port for the OpenAI-compatible HTTP server. |
| `--wyoming-host` | `FATTERVOICE_WYOMING_HOST` | `0.0.0.0` | Bind address for the Wyoming TCP protocol endpoint (Home Assistant). |
| `--wyoming-port` | `FATTERVOICE_WYOMING_PORT` | `10300` | Port for the Wyoming TCP protocol endpoint. |
| `--device` | `FATTERVOICE_DEVICE` | `cuda:0` | Torch device passed to OmniVoice (e.g. `cuda:0`, `cpu`, `mps`). Use `cpu` only for debugging — GPU is required for practical latency. |
| `--log-level` | `FATTERVOICE_LOG_LEVEL` | `INFO` | Python logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`). Use `DEBUG` for troubleshooting model loading and voice resolution. |

### Tuning Configuration Reference
| CLI flag | ENV fallback | Default | Description |
| --- | --- | --- | --- |
| `--dtype` | `FATTERVOICE_DTYPE` | `bfloat16` | Torch precision (`float16`, `bfloat16`, `float32`). `bfloat16` is the default balanced setting; `float16` saves VRAM but can reduce quality on some models; `float32` maximizes quality at the cost of speed and memory. |
| `--default-language` | `FATTERVOICE_DEFAULT_LANGUAGE` | `auto` | Fallback language when the client does not specify one. `auto` lets OmniVoice detect the language from the input text. Set to a concrete code (e.g. `en`, `zh`) to force a specific language. |
| `--num-step` | `FATTERVOICE_NUM_STEP` | `32` | OmniVoice diffusion decoding steps. Lower values are faster but can reduce quality; `16` is the practical minimum, `32` is the default balanced setting. |
| `--guidance-scale` | `FATTERVOICE_GUIDANCE_SCALE` | `2.0` | Classifier-free guidance scale. Higher values push output closer to the text prompt but can introduce artifacts; lower values produce more natural but less controlled speech. |
| `--denoise` / `--no-denoise` | `FATTERVOICE_DENOISE` | `true` | Enables denoise prompting for cleaner output. Disable only if you notice unwanted artifacts and want raw model output. |
| `--t-shift` | `FATTERVOICE_T_SHIFT` | `0.1` | Diffusion time-shift parameter. Adjust only if tuning diffusion scheduling for specific quality/latency trade-offs. |
| `--position-temperature` | `FATTERVOICE_POSITION_TEMPERATURE` | `5.0` | Position-sampling temperature. Higher values increase variation in prosody; lower values produce more consistent but potentially monotonous speech. |
| `--class-temperature` | `FATTERVOICE_CLASS_TEMPERATURE` | `0.0` | Class-sampling temperature. Non-zero values add stochasticity to phoneme-level choices; `0.0` (default) is deterministic. |
| `--layer-penalty-factor` | `FATTERVOICE_LAYER_PENALTY_FACTOR` | `5.0` | Penalty that biases OmniVoice toward lower codebook layers first. Higher values can improve quality by reducing reliance on higher layers; lower values allow more layer diversity. |
| `--preprocess-voice-clone-prompt` / `--no-preprocess-voice-clone-prompt` | `FATTERVOICE_PREPROCESS_VOICE_CLONE_PROMPT` | `true` | Preprocesses reference audio and transcript before caching the voice-clone prompt. Disable only if preprocessing causes issues with specific reference files. |
| `--postprocess-output-audio` / `--no-postprocess-output-audio` | `FATTERVOICE_POSTPROCESS_OUTPUT_AUDIO` | `true` | Lets OmniVoice remove excess silence and apply fade-in/fade-out to output audio. Disable if you need raw unmodified model output. |
| `--max-sentence-length` | `FATTERVOICE_MAX_SENTENCE_LENGTH` | `400` | Maximum character length of a single synthesis segment. All text is split into sentence-sized segments first; only segments exceeding this cap are broken further on word boundaries. Default 400 (~25s of speech). Setting this too high may induce higher peak VRAM usage during generation. Setting it too low may cause unnatural-sounding pauses from extra synthesis boundaries. |
| `--break-point-lookback` | `FATTERVOICE_BREAKPOINT_WINDOW` | `100` | Number of characters before the max sentence length to search for a natural break point (comma, conjunction, etc.) when splitting oversized segments. If no suitable break is found in this window, a hard word-boundary split is used. If you increase `--max-sentence-length` past its default, you should probably increase this as well to prevent breaks in unsuitable locations. |
| `--flashinfer` | `FATTERVOICE_FLASHINFER` | `auto` | FlashInfer acceleration mode: `auto` (default), `on`, or `off`. `auto` enables FlashInfer on sm_80+ NVIDIA GPUs (Ampere RTX 30 / Ada RTX 40 / Hopper / Blackwell RTX 50, A100/H100/B200) and falls back to the standard path on Turing (RTX 20) or non-CUDA devices; when active it implies float16 and coerces the dtype automatically (logged). `on` forces FlashInfer and requires `--device cuda:*`. `off` always uses the standard path. Measured on an RTX 3090: ~1.9x faster model compute, ~1.5x faster end-to-end (upstream H100 benchmarks report 2–2.9x). See [FlashInfer acceleration](#flashinfer-acceleration). |
| `--flashinfer-cuda-graph` / `--no-flashinfer-cuda-graph` | `FATTERVOICE_FLASHINFER_CUDA_GRAPH` | `false` | Replays CUDA graphs for FlashInfer generation. Only meaningful when FlashInfer is active. Recommended only for fixed-shape workloads: each distinct sequence shape pays a one-time graph capture, and this server splits text into variable-length sentence segments, so per-shape captures can exceed the replay savings (measured slower than plain FlashInfer on mixed-length streaming traffic). |

Boolean environment variables accept `1`, `true`, `yes`, or `on` for true, and `0`, `false`, `no`, or `off` for false.

At startup, `fattervoice` logs the fully resolved runtime configuration in a boxed summary so you can confirm the exact effective settings before it begins serving traffic.

The runtime model is now hardcoded to `omnivoice`, Wyoming support is always enabled, Wyoming audio chunking is fixed at `4096` mono PCM samples per emitted `audio-chunk`, and model-cache / prefetch-manifest selection is no longer exposed through wrapper-specific server flags.

## FlashInfer acceleration

[FlashInfer](https://github.com/flashinfer-ai/flashinfer) kernels accelerate OmniVoice inference by ~2–2.9x (upstream H100 benchmark) with no measurable quality change. The integration is a vendored copy of the upstream `omnivoice_flashinfer.py` patch (`fattervoice/omnivoice_flashinfer.py`) that replaces OmniVoice's iterative decoder with a packed-sequence implementation (cond+uncond CFG pair in one ragged-attention row, fused RMSNorm/RoPE/MLP kernels, optional CUDA graphs). It is not yet shipped in any released `omnivoice` PyPI package, which is why the patch lives in this repo.

FlashInfer is **enabled by default in `auto` mode** — no flags needed. The server probes the GPU at startup:

- **sm_80+ (Ampere and newer)** — FlashInfer is applied automatically and dtype is coerced to `float16` (a startup warning is logged if you configured something else).
- **Turing (sm_75, RTX 20 series / T4)** — FlashInfer is skipped with a log message (the upstream ragged-prefill kernel has a known launch bug on Turing); the standard path is used instead.
- **CPU / MPS** — FlashInfer is skipped; the standard path is used.

### GPU compatibility

| GPU family | Compute capability | FlashInfer | Notes |
| --- | --- | --- | --- |
| RTX 20 series, T4 | 7.5 (Turing) | ❌ auto-skips | Standard path only (known upstream prefill bug on Turing) |
| RTX 30 series, A100, A10 | 8.0 / 8.6 (Ampere) | ✅ | RTX 3090 verified |
| RTX 40 series, L4/L40 | 8.9 (Ada) | ✅ | |
| H100 / H200 | 9.0 (Hopper) | ✅ | |
| B100 / B200 | 10.0 (Blackwell) | ✅ | |
| RTX 50 series | 12.0 (Blackwell) | ✅ | Needs a recent driver (≥ 570) |

**Driver requirement:** the image is built with CUDA 13.0 (torch cu130), so the host NVIDIA driver must be **≥ 580** (driver 595+ is current). Older drivers on 30/40-series machines will need an update. All GPUs from Turing (sm_75) through Blackwell (sm_120) are covered by the single image.

### Notes

- The prebuilt kernel cache (`flashinfer-jit-cache==0.6.15.post1+cu130`) is fetched from the FlashInfer index during install; there is no runtime JIT compilation, so the offline Docker image keeps working offline.
- CUDA graphs (`--flashinfer-cuda-graph`) are exposed but default off: with this server's sentence-split streaming, request shapes vary constantly, so graph captures can cost more than they save. Leave it off unless your traffic is fixed-shape.
- `--flashinfer on` fails fast at startup if the device is not CUDA; `--flashinfer off` disables acceleration entirely.

## Interfaces

### OpenAI-compatible API

Generate speech:

```bash
curl http://localhost:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
        "model": "tts-1",
        "input": "Hello from fattervoice.",
        "voice": "hank",
        "response_format": "wav",
        "speed": 1.0
      }' \
  --output speech.wav
```

`speed` is forwarded to OmniVoice.

### Wyoming support

By default the server also exposes a Wyoming TCP endpoint at `tcp://0.0.0.0:10300`, configured through `--wyoming-host` plus `--wyoming-port`.

The Wyoming handler:

- answers `describe` with `info`
- exposes the same voice registry used by HTTP
- handles `synthesize`
- handles `synthesize-start` / `synthesize-chunk` / `synthesize-stop`
- emits `audio-start` / `audio-chunk` / `audio-stop`
- buffers incoming streaming text on sentence boundaries before synthesis

## Docker

### Premade Image
The hosted image contains the entire omnivoice model baked in for offline functionality, however this means the image is large, and may take a while to download. 
The current image is `15.4 GB`. The Premade image can be found [here](https://hub.docker.com/r/justinlime/fattervoice).

## Docker Run 
```bash
docker run --name fattervoice --replace -p 10300:10300 -p 8000:8000 -v /path/to/fatterbox/voices:/opt/fattervoice/voices --device=nvidia.com/gpu=all docker.io/justinlime/fattervoice:latest
```

The image is built with CUDA 13.0 (torch cu130) and ships FlashInfer preinstalled, so a single image covers Turing (RTX 20) through Blackwell (RTX 50). FlashInfer enables itself automatically on sm_80+ GPUs and stays off on Turing/non-CUDA — no environment variables needed to use it. The host NVIDIA driver must be **≥ 580**.

Build it yourself:

```bash
docker build -t fattervoice:latest .
```

## Docker Compose
```yaml
services:
  fattervoice:
    image: docker.io/justinlime/fattervoice:latest
    container_name: fattervoice
    ports:
      - "10200:10300"
      - "5002:8000"
    volumes:
      - /path/to/fatterbox/voices:/opt/fattervoice/voices
    devices:
      - nvidia.com/gpu=all
    restart: unless-stopped
```

## Local development with `uv`

Install dependencies:

```bash
uv sync --extra mp3 --extra dev
```

Run the server:

```bash
uv run fattervoice \
  --voices-dir ./voices \
  --openapi-host 0.0.0.0 \
  --openapi-port 8000 \
  --wyoming-host 0.0.0.0 \
  --wyoming-port 10300
```

CLI arguments take precedence, and environment variables act as fallbacks.
