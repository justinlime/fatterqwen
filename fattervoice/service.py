"""Shared synthesis service used by both the HTTP and Wyoming adapters."""

from __future__ import annotations

import asyncio
import gc
import logging
import os
import re
import threading
import time
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, AsyncIterator

import numpy as np

from .audio import audio_to_pcm16_bytes, iter_byte_chunks
from .config import ServerConfig
from .hf_cache import is_huggingface_offline_mode_enabled
from .model_catalog import resolve_model_id
from .prefetch_manifest import resolve_cached_model_snapshot_path
from .voice_registry import VoiceEntry, VoiceRegistry

LOGGER = logging.getLogger(__name__)
_HARDCODED_MODEL_ALIAS = "omnivoice"
_STREAMING_PCM_CHUNK_BYTES = 8192
# OmniVoice native long-form chunking activation threshold in seconds. Set to an
# effectively unreachable value so the model never switches to its internal
# chunked generation path — the wrapper owns all text segmentation via
# split_text_for_streaming, and native chunking would silently change the
# audio-join behavior (cross-faded seams) for very long single sentences.
_AUDIO_CHUNK_THRESHOLD_NEVER = 999999.0


@dataclass(frozen=True)
class CachedVoiceClonePrompt:
    """CPU-resident OmniVoice prompt object cached for one validated voice entry."""

    voice_id: str
    prompt: object


@dataclass(frozen=True)
class SynthesisRequest:
    """Normalized synthesis request shared across protocol adapters."""

    text: str
    voice_id: str | None = None
    language: str | None = None
    speed: float | None = None



def move_prompt_value_to_cpu(value: object) -> object:
    """Recursively copy tensor-bearing prompt values onto CPU memory.

    Usage:
        OmniVoice prompt objects may contain GPU tensors even though only one
        request can actively generate at a time. This helper converts cached
        prompt payloads into CPU-resident structures so previously used voices
        can be reused later without continuing to pin VRAM between requests.

    Parameters:
        value: Any prompt payload object, including dataclasses, dictionaries,
            lists, tuples, tensors, or primitive values.

    Returns:
        A deep-copied prompt payload where any tensor leaves have been detached
        and moved to CPU memory. Non-tensor values are preserved.
    """
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a runtime dependency in production.
        torch = None

    if torch is not None and torch.is_tensor(value):
        return value.detach().cpu()

    if is_dataclass(value) and not isinstance(value, type):
        return value.__class__(
            **{
                field.name: move_prompt_value_to_cpu(getattr(value, field.name))
                for field in fields(value)
            }
        )

    if isinstance(value, dict):
        return {
            key: move_prompt_value_to_cpu(nested_value)
            for key, nested_value in value.items()
        }

    if isinstance(value, list):
        return [move_prompt_value_to_cpu(item) for item in value]

    if isinstance(value, tuple):
        return tuple(move_prompt_value_to_cpu(item) for item in value)

    return value



def normalize_omnivoice_language(language: str | None) -> str | None:
    """Normalize user-supplied language values into OmniVoice-friendly identifiers.

    Usage:
        The HTTP and Wyoming adapters may receive human-readable names, BCP47
        tags, or special auto-detection markers. This helper keeps that cleanup
        logic in one place before values are passed into OmniVoice generation.

    Parameters:
        language: The raw optional language value supplied by a client or config.

    Returns:
        `None` when OmniVoice should auto-detect the language, otherwise a
        cleaned language code or language name string.
    """
    if language is None:
        return None

    normalized_language = language.strip()
    if not normalized_language:
        return None

    normalized_key = normalized_language.replace("_", "-").lower()
    if normalized_key in {"*", "any", "auto", "automatic", "default", "mul"}:
        return None

    if "-" in normalized_key:
        primary_subtag = normalized_key.split("-", 1)[0]
        if primary_subtag:
            return primary_subtag

    return normalized_language



def normalize_omnivoice_device_map(device: str) -> str:
    """Normalize configured device strings into values accepted by OmniVoice.

    Usage:
        Existing deployments often use the shorthand `cuda`, while OmniVoice
        examples and Hugging Face device-map handling are more predictable when a
        concrete CUDA ordinal such as `cuda:0` is used. This helper preserves
        existing non-CUDA values and upgrades the common shorthand.

    Parameters:
        device: The raw runtime device string from configuration.

    Returns:
        A normalized device-map string suitable for `OmniVoice.from_pretrained`.
    """
    normalized_device = device.strip()
    if normalized_device == "cuda":
        return "cuda:0"
    return normalized_device



def coerce_waveform_array(waveform: Any) -> np.ndarray:
    """Convert an OmniVoice waveform output into a flat float32 NumPy array.

    Usage:
        OmniVoice is documented to return NumPy arrays, but tests and future
        library versions may hand back tensor-like objects. This helper gives the
        service one place to coerce those values into the stable mono array shape
        expected by the HTTP and Wyoming adapters.

    Parameters:
        waveform: The first audio item returned by `OmniVoice.generate(...)`.

    Returns:
        A one-dimensional float32 NumPy waveform ready for PCM/WAV conversion.
    """
    if hasattr(waveform, "detach"):
        waveform = waveform.detach().cpu().numpy()
    return np.asarray(waveform, dtype=np.float32).flatten()



# Regex-based sentence segmentation, ported from the reference
# omnivoice-server implementation (maemreyo/omnivoice-server) and extended:
# - splits after ANY sentence-ending punctuation plus whitespace (no uppercase
#   lookahead), so lowercase sentence starts are recognized;
# - never splits when the punctuation itself is surrounded by whitespace, so
#   spaced ellipses (``. . .``) stay attached;
# - the false-boundary merge pass protects decimals, version numbers,
#   abbreviations, and URLs, so characters are never detached or dropped.
_SENTENCE_END_RE = re.compile(
    r"(?<=[.!?])(?<!\s[.!?])\s+"
    r"|(?<=[。！？])"
)
_FALSE_END_RE = re.compile(
    r"\d+\.\d+"  # Decimals: 3.14
    r"|v\d+\.\d+"  # Version numbers: v2.1.0
    r"|[A-Z][a-z]{0,3}\."  # Abbreviations: Dr., Inc.
    r"|\w+\.\w{2,6}(?:/|\s|$)"  # URLs: example.com
    r"|e\.g\.|i\.e\.|etc\.|vs\.|a\.m\.|p\.m\.|cf\.|approx\.|pp\.|et al\."  # Lowercase abbreviations
    r"|jan\.|feb\.|mar\.|apr\.|jun\.|jul\.|aug\.|sep\.|oct\.|nov\.|dec\."  # Lowercase months
)


def split_text_into_sentences(text: str) -> list[str]:
    """Split text into sentence-like segments at punctuation boundaries.

    Usage:
        ``split_text_for_streaming`` calls this helper as its first pass so every
        valid sentence becomes its own synthesis segment. The splitter breaks
        after sentence-ending punctuation (``.``/``!``/``?`` followed by
        whitespace, or ``。``/``！``/``？``) regardless of whether the next
        sentence starts uppercase or lowercase, then merges back any pieces that
        were split inside a false boundary (decimals, version numbers, URLs,
        uppercase abbreviations, and common lowercase abbreviations such as
        ``e.g.``, ``i.e.``, ``etc.``, ``vs.``, ``a.m.``/``p.m.``, ``cf.``,
        ``approx.``, ``pp.``, ``et al.``, and lowercase month names) so no
        characters are ever dropped from the text the model receives. Punctuation
        surrounded by whitespace (spaced ellipses like ``. . .``) is never
        treated as a boundary.

    Parameters:
        text: The raw synthesis text to segment.

    Returns:
        A non-empty ordered list of stripped sentence segments, or an empty
        list when the input is blank.
    """
    normalized_text = text.strip()
    if not normalized_text:
        return []

    raw_segments = [
        segment.strip()
        for segment in _SENTENCE_END_RE.split(normalized_text)
        if segment.strip()
    ]
    if not raw_segments:
        return [normalized_text]

    merged_segments: list[str] = []
    index = 0
    while index < len(raw_segments):
        current_segment = raw_segments[index]

        # Merge back pieces split at false boundaries (e.g. "Dr." or "3.14"
        # followed by more of the same token). The -2 tolerance accounts for
        # trailing punctuation such as "v2.1." where the period after the
        # false-end pattern should still trigger a merge.
        while index + 1 < len(raw_segments):
            last_false_end = None
            for false_end_match in _FALSE_END_RE.finditer(current_segment):
                last_false_end = false_end_match
            if last_false_end and last_false_end.end() >= len(current_segment) - 2:
                current_segment = f"{current_segment} {raw_segments[index + 1]}"
                index += 1
            else:
                break

        merged_segments.append(current_segment)
        index += 1

    return merged_segments


def split_text_for_streaming(text: str, max_length: int = 400, break_point_lookback: int = 100) -> list[str]:
    """Split request text into sentence-first synthesis segments with a length cap.

    Usage:
        OmniVoice currently exposes buffered generation rather than a documented
        model-incremental audio streaming API. The wrapper splits text into
        sentence-like segments so each synthesis call stays bounded in memory
        and time. Every valid sentence (detected by punctuation-aware regex)
        becomes its own segment and resets the character cap; any single
        segment that still exceeds ``max_length`` characters is broken further
        on word boundaries, preferring natural pause markers inside the
        lookback window, to prevent runaway resource usage.

    Parameters:
        text: The already-validated request text that should be segmented.
        max_length: Maximum character count for any single segment. Sentence
            boundaries are respected first; only oversized segments are split
            further on spaces.

    Returns:
        A non-empty ordered list of stripped text segments suitable for
        sequential synthesis.
    """
    raw_segments = split_text_into_sentences(text)

    if not raw_segments:
        return [text.strip()]

    # Enforce a minimum of 100 characters so segments stay pronounceable
    # and the break-point lookback window has room to operate.
    effective_max = max(max_length, 100)

    capped: list[str] = []
    for segment in raw_segments:
        if len(segment) <= effective_max:
            capped.append(segment)
        else:
            capped.extend(_split_segment_on_words(segment, effective_max, break_point_lookback))
    return capped


def _split_segment_on_words(segment: str, max_length: int, break_point_lookback: int = 100) -> list[str]:
    """Break an oversized segment into smaller chunks, preferring natural pause points.

    Usage:
        ``split_text_for_streaming`` calls this helper when a single sentence
        exceeds the configured character cap so it can still be synthesized
        in manageable pieces. The algorithm searches for a natural break point
        (comma, conjunction, etc.) within the **last 100 characters before
        ``max_length``** is reached. If one is found, the segment splits there
        (still within the limit). If none exists in that lookback window, it
        falls back to a hard word-boundary split at ``max_length`` so the
        limit is never exceeded.

    Parameters:
        segment: A stripped text segment that is longer than ``max_length``.
        max_length: The hard maximum character count for each produced chunk.

    Returns:
        An ordered list of stripped sub-segments, each at most ``max_length``
        characters long.
    """
    # Natural break markers ordered by preference (earlier = preferred on tie).
    # Commas give the shortest natural pause; conjunctions give clause boundaries.
    _PREFERRED_BREAKS: tuple[str, ...] = (
        ", ",
        "; ",
        ". ",
        " but ",
        " and ",
        " because ",
        " so ",
        " yet ",
        " or ",
        " while ",
        " although ",
        " however, ",
        " therefore, ",
        " meanwhile, ",
        " furthermore, ",
    )
    _LOOKBACK_WINDOW = break_point_lookback  # characters to scan before max_length for a break point

    def _find_break_in_window(text: str, limit: int) -> int | None:
        """Find the best natural break point in the last _LOOKBACK_WINDOW chars before ``limit``.

        Returns the character position to split at (after the break marker),
        or ``None`` if no preferred break exists in the window.
        """
        window_start = max(0, limit - _LOOKBACK_WINDOW)
        best_split: int | None = None
        best_marker_idx: int = len(_PREFERRED_BREAKS)

        for marker_idx, break_marker in enumerate(_PREFERRED_BREAKS):
            pos = 0
            while True:
                pos = text.find(break_marker, pos)
                if pos == -1:
                    break
                split_after = pos + len(break_marker)
                # Must be inside the lookback window [window_start, limit]
                if window_start <= split_after <= limit:
                    if (best_split is None
                            or split_after > best_split
                            or (split_after == best_split and marker_idx < best_marker_idx)):
                        best_split = split_after
                        best_marker_idx = marker_idx
                pos += 1

        return best_split

    chunks: list[str] = []
    words = segment.split()
    current: list[str] = []
    current_len = 0

    for word in words:
        word_len = len(word)
        added_len = word_len + (1 if current else 0)  # +1 for the space

        # Force-split a single word that exceeds the hard limit
        if word_len > max_length:
            if current:
                chunks.append(" ".join(current))
                current = []
                current_len = 0
            for i in range(0, word_len, max_length):
                chunks.append(word[i : i + max_length])
            continue

        if current_len + added_len > max_length:
            # We've exceeded the limit — try to find a break point in the
            # lookback window of the text accumulated so far.
            accumulated = " ".join(current)
            split_at = _find_break_in_window(accumulated, max_length)

            if split_at is not None:
                # Split at the natural break point (guaranteed <= max_length)
                chunks.append(accumulated[:split_at].rstrip())
                # Remaining words: everything after the break + current word
                leftover_words = accumulated[split_at:].split()
                current = leftover_words + [word]
                current_len = sum(len(w) for w in leftover_words) + word_len
                if len(leftover_words) > 1:
                    current_len += len(leftover_words) - 1  # spaces
                elif leftover_words:
                    current_len += 1  # space between leftover and word
            else:
                # No break point in window — emit what we have, start fresh
                chunks.append(" ".join(current))
                current = [word]
                current_len = word_len
        else:
            current.append(word)
            current_len += added_len

    if current:
        chunks.append(" ".join(current))

    return [chunk for chunk in chunks if chunk]


class TtsService:
    """Long-lived model wrapper that serializes GPU inference and voice resolution."""

    def __init__(self, config: ServerConfig, voice_registry: VoiceRegistry) -> None:
        """Initialize the shared synthesis service without loading the model yet.

        Usage:
            Construct the service once during application startup, then call
            `await start()` before serving any traffic.

        Parameters:
            config: Immutable runtime configuration for the server.
            voice_registry: The validated voice registry shared by all protocols.

        Returns:
            None. A new `TtsService` instance is initialized in place.
        """
        self.config = config
        self.voice_registry = voice_registry
        self.model_id = resolve_model_id(_HARDCODED_MODEL_ALIAS)
        self.device_map = normalize_omnivoice_device_map(config.device)
        self._model = None
        self._model_lock = threading.Lock()
        self._prepared_voice_lock = threading.Lock()
        self._prepared_voice_cache: dict[str, CachedVoiceClonePrompt] = {}
        self._last_generated_voice_id: str | None = None

    def close(self) -> None:
        """Release cached prompt state and unused accelerator memory.

        Usage:
            Tests or future shutdown hooks can call this method during teardown
            so cached prompt objects are dropped and any unused accelerator cache
            is released before the process exits.

        Parameters:
            None.

        Returns:
            None. Cached prompt objects are cleared in place and best-effort
            accelerator cleanup is performed.
        """
        with self._prepared_voice_lock:
            self._prepared_voice_cache.clear()

        with self._model_lock:
            self._last_generated_voice_id = None
            self._release_unused_accelerator_memory()

    @property
    def sample_rate(self) -> int:
        """Return the model sample rate after startup has loaded the model.

        Usage:
            Protocol adapters use this property to advertise audio metadata and
            construct WAV headers without duplicating model-specific knowledge.

        Parameters:
            None.

        Returns:
            The integer waveform sample rate reported by the loaded model.

        Raises:
            RuntimeError: If startup has not loaded the model yet.
        """
        return int(self._require_model().sampling_rate)

    async def start(self) -> None:
        """Load the OmniVoice model before either server adapter starts serving.

        Usage:
            Call this exactly once during process startup before either server
            adapter begins accepting requests. Voice-clone prompts are built
            lazily on first use so large voice directories do not eagerly occupy
            CPU or GPU memory at startup, unless a specific voice is configured
            for pre-loading via ``--preload-voice`` / ``FATTERVOICE_PRELOAD_VOICE``.

            After the model and optional preload voice are ready, a short warmup
            synthesis is performed to pre-compile CUDA kernels and initialize
            lazy internal state so the first real request is not penalized.

        Parameters:
            None.

        Returns:
            None. The method completes when the model is ready to serve requests.
        """
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._load_model)

        if self.config.preload_voice:
            voice = self.voice_registry.get(self.config.preload_voice)
            await loop.run_in_executor(
                None, lambda: self._resolve_cached_voice_clone_prompt(voice)
            )
            LOGGER.info(
                "Pre-loaded voice clone prompt for voice %s", self.config.preload_voice
            )

        await self._warmup(loop)

    async def _warmup(self, loop: asyncio.AbstractEventLoop) -> None:
        """Run a short synthesis to pre-compile CUDA kernels and warm internal caches.

        Usage:
            Startup calls this after the model is loaded so the first real
            request benefits from already-compiled kernels and initialized
            memory pools rather than paying the one-time compilation cost.

        Parameters:
            loop: The running asyncio event loop for executor dispatch.
        """
        voice = self._pick_warmup_voice()
        if voice is None:
            LOGGER.info("Skipping warmup synthesis (no voices available)")
            return

        LOGGER.info("Running warmup synthesis for voice %s", voice.voice_id)
        warmup_request = SynthesisRequest(
            text="This is a warmup phrase to pre-compile kernels.",
            voice_id=voice.voice_id,
        )

        start_time = time.monotonic()
        await loop.run_in_executor(
            None, lambda: self._generate_waveform(warmup_request, voice)
        )
        elapsed = time.monotonic() - start_time
        LOGGER.info("Warmup synthesis complete in %.2fs", elapsed)

    def _pick_warmup_voice(self) -> VoiceEntry | None:
        """Pick a voice to use for the warmup synthesis.

        Returns the preloaded voice if configured, otherwise the first voice
        in the registry, or None if the registry is empty.
        """
        if self.config.preload_voice:
            return self.voice_registry.get(self.config.preload_voice)
        voice_ids = self.voice_registry.list_voice_ids()
        if voice_ids:
            return self.voice_registry.get(voice_ids[0])
        return None

    def validate_request(self, request: SynthesisRequest) -> VoiceEntry:
        """Validate request text and resolve the referenced voice before synthesis.

        Usage:
            Protocol adapters call this method before opening a streaming response
            so client-facing validation errors are raised early and consistently.

        Parameters:
            request: The normalized request that should be validated.

        Returns:
            The resolved `VoiceEntry` that should be used for synthesis.

        Raises:
            ValueError: If the request text is empty after surrounding whitespace is removed.
            VoiceRegistryError: If the requested voice does not exist.
        """
        self._validate_request_text(request.text)
        return self.voice_registry.get(request.voice_id)

    async def synthesize(self, request: SynthesisRequest) -> tuple[np.ndarray, int]:
        """Generate a complete waveform for a single synthesis request.

        Usage:
            Non-streaming HTTP responses and Wyoming non-streaming synthesis use
            this method when the entire waveform is needed before returning a
            response. The text is split into sentence-sized segments internally
            so no single OmniVoice call exceeds the configured segment cap.

        Parameters:
            request: A normalized request describing the text, voice, and runtime
                options for the generation.

        Returns:
            A tuple of `(waveform, sample_rate)` where the waveform is a mono
            float32 NumPy array containing all synthesized segments concatenated.
        """
        voice = self.validate_request(request)
        segments = split_text_for_streaming(
            request.text, self.config.max_sentence_length, self.config.break_point_lookback
        )
        loop = asyncio.get_running_loop()
        waveforms: list[np.ndarray] = []
        for i, segment_text in enumerate(segments):
            seg_request = SynthesisRequest(
                text=segment_text,
                voice_id=voice.voice_id,
                language=request.language,
                speed=request.speed,
            )
            waveform, _ = await loop.run_in_executor(
                None, lambda sr=seg_request, v=voice: self._generate_waveform(sr, v)
            )
            waveforms.append(waveform)
        combined = np.concatenate(waveforms) if waveforms else np.array([], dtype=np.float32)
        return combined, self.sample_rate

    def stream_pcm_chunks(self, request: SynthesisRequest) -> AsyncIterator[bytes]:
        """Yield PCM chunks by synthesizing sentence-sized segments sequentially.

        Usage:
            All streaming paths (HTTP and Wyoming) use this method. It splits
            validated text into sentence-like segments, synthesizes them one at
            a time, and emits PCM bytes as each segment completes.

        Parameters:
            request: A normalized request describing the text, voice, and runtime
                options for the generation.

        Returns:
            An async iterator that yields raw 16-bit PCM audio chunks as each
            synthesis segment completes.
        """
        resolved_voice = self.validate_request(request)
        segments = split_text_for_streaming(
            request.text, self.config.max_sentence_length, self.config.break_point_lookback
        )

        async def emit_pcm_chunks() -> AsyncIterator[bytes]:
            """Synthesize segments sequentially and yield PCM chunks.

            Usage:
                The outer method performs eager validation and segment planning
                so this nested generator focuses on sequential synthesis and
                byte emission once the response body has started.

            Parameters:
                None. It closes over the validated request state.

            Returns:
                An async iterator that yields fixed-size PCM byte chunks.
            """
            loop = asyncio.get_running_loop()
            for segment_text in segments:
                seg_request = SynthesisRequest(
                    text=segment_text,
                    voice_id=resolved_voice.voice_id,
                    language=request.language,
                    speed=request.speed,
                )
                waveform, _ = await loop.run_in_executor(
                    None,
                    lambda sr=seg_request, v=resolved_voice: self._generate_waveform(sr, v),
                )
                pcm_payload = audio_to_pcm16_bytes(waveform)
                for pcm_chunk in iter_byte_chunks(pcm_payload, _STREAMING_PCM_CHUNK_BYTES):
                    yield pcm_chunk

        return emit_pcm_chunks()

    def stream_low_latency_pcm_chunks(self, request: SynthesisRequest) -> AsyncIterator[bytes]:
        """Alias for ``stream_pcm_chunks`` — all paths now use sentence-based splitting.

        Usage:
            The HTTP adapter still calls this name for explicit ``stream=true``
            requests, but the underlying behavior is identical to the default
            streaming path.

        Parameters:
            request: A normalized request describing the text, voice, and runtime
                options for the generation.

        Returns:
            An async iterator that yields raw 16-bit PCM audio chunks.
        """
        return self.stream_pcm_chunks(request)

    def _load_model(self) -> None:
        """Load the OmniVoice model using the configured device and dtype.

        Usage:
            Startup calls this method through `start()`. Repeated calls are cheap
            because the method exits immediately once the model is already loaded.

        Parameters:
            None.

        Returns:
            None. The loaded model is stored on the service instance.
        """
        if self._model is not None:
            return

        import torch
        from omnivoice import OmniVoice

        flashinfer_active = self._resolve_flashinfer_active(torch)

        dtype_name = self.config.dtype
        if flashinfer_active and dtype_name.strip().lower() != "float16":
            LOGGER.warning(
                "FlashInfer kernels are hard-coded for float16; coercing dtype from %s to float16",
                dtype_name,
            )
            dtype_name = "float16"

        try:
            dtype = getattr(torch, dtype_name)
        except AttributeError as exc:
            raise ValueError(
                f"Unsupported torch dtype {dtype_name!r}."
            ) from exc

        model_source = resolve_cached_model_snapshot_path(self.model_id, None)
        resolved_model_path = Path(model_source).expanduser()

        if is_huggingface_offline_mode_enabled() and not resolved_model_path.exists():
            build_model_selection_hint = os.environ.get("MODEL_SELECTION_HINT", "").strip()
            if build_model_selection_hint and build_model_selection_hint.lower() != "all":
                try:
                    built_model_id_hint = resolve_model_id(build_model_selection_hint)
                except ValueError:
                    built_model_id_hint = build_model_selection_hint
                if built_model_id_hint != self.model_id:
                    raise FileNotFoundError(
                        "Offline mode is enabled, but this container image was built without the requested OmniVoice model. "
                        f"The image build hint is {build_model_selection_hint!r}, while runtime requested {_HARDCODED_MODEL_ALIAS!r} "
                        f"({self.model_id}). Rebuild the image with --build-arg MODEL_SELECTION=all or --build-arg MODEL_SELECTION={_HARDCODED_MODEL_ALIAS}, "
                        "or run an image that already contains the requested snapshot."
                    )
            raise FileNotFoundError(
                "Offline mode is enabled, but no local snapshot path could be resolved for "
                f"{self.model_id!r}. Check the Hugging Face hub cache contents."
            )

        LOGGER.info(
            "Loading OmniVoice model %s from %s on device %s with dtype %s",
            _HARDCODED_MODEL_ALIAS,
            model_source,
            self.device_map,
            dtype_name,
        )
        self._model = OmniVoice.from_pretrained(
            model_source,
            device_map=self.device_map,
            dtype=dtype,
            load_asr=False,
        )
        LOGGER.info("Model ready with sample rate %s Hz", self._model.sampling_rate)

        if flashinfer_active:
            self._apply_flashinfer_acceleration()

    def _resolve_flashinfer_active(self, torch) -> bool:
        """Decide whether FlashInfer acceleration should be applied for this run.

        Usage:
            `_load_model` calls this helper before loading the model so the
            effective dtype (float16 coercion) and the FlashInfer patch decision
            are made in one place. The configured mode is one of ``auto``, ``on``,
            or ``off``:

            - ``off``: never apply FlashInfer.
            - ``on``: always apply FlashInfer; fails fast when the device is not
              CUDA (the CLI already validates this, but programmatic
              configuration is re-checked here).
            - ``auto`` (default): apply FlashInfer on sm_80+ NVIDIA GPUs
              (Ampere/Ada/Hopper/Blackwell) and skip it on Turing (sm_75, e.g.
              RTX 20 series) or non-CUDA devices, logging the reason. This is
              what makes a single image safe across the full RTX 20-50 range:
              RTX 30/40/50 and datacenter GPUs get the fast path, while RTX 20
              (where the upstream ragged-prefill kernel has a known launch bug)
              transparently runs the standard path.

        Parameters:
            torch: The imported torch module used for CUDA capability queries.

        Returns:
            ``True`` when FlashInfer should be applied and dtype coerced to
            float16, otherwise ``False``.
        """
        mode = self.config.flashinfer
        if mode == "off":
            return False
        if mode == "on":
            if not self.device_map.startswith("cuda"):
                raise ValueError(
                    "FlashInfer acceleration requires a CUDA device (got "
                    f"{self.device_map!r}). Use --flashinfer auto or --flashinfer off."
                )
            LOGGER.info("FlashInfer mode=on: acceleration enabled")
            return True

        # auto mode
        if not self.device_map.startswith("cuda"):
            LOGGER.info(
                "FlashInfer auto mode: skipped (device %s is not CUDA)",
                self.device_map,
            )
            return False
        compute_capability = self._cuda_device_capability(torch)
        if compute_capability is None or compute_capability[0] < 8:
            LOGGER.warning(
                "FlashInfer auto mode: skipped (GPU compute capability %s is below "
                "sm_80; FlashInfer kernels require Ampere or newer, and the upstream "
                "prefill path is unreliable on Turing). Running the standard path.",
                ".".join(str(part) for part in compute_capability)
                if compute_capability is not None
                else "unknown",
            )
            return False
        LOGGER.info(
            "FlashInfer auto mode: enabled on compute capability %s (sm_%s+)",
            ".".join(str(part) for part in compute_capability),
            compute_capability[0],
        )
        return True

    def _cuda_device_capability(self, torch) -> tuple[int, int] | None:
        """Return the (major, minor) compute capability of the configured CUDA device.

        Usage:
            ``_resolve_flashinfer_active`` calls this helper in auto mode to
            decide whether the installed GPU can run FlashInfer kernels. The
            configured ``device_map`` may be ``cuda`` or ``cuda:N``; the first
            explicit device index (or 0) is queried.

        Parameters:
            torch: The imported torch module used for CUDA queries.

        Returns:
            A ``(major, minor)`` tuple for the device, or ``None`` when CUDA is
            unavailable or the device cannot be queried.
        """
        try:
            if not torch.cuda.is_available():
                return None
            device_index = 0
            device_match = re.match(r"cuda(?::(\d+))?", self.device_map)
            if device_match is not None and device_match.group(1) is not None:
                device_index = int(device_match.group(1))
            return tuple(int(part) for part in torch.cuda.get_device_capability(device_index))
        except Exception:  # pragma: no cover - defensive; capability queries are stable in practice.
            return None

    def _apply_flashinfer_acceleration(self) -> None:
        """Patch the loaded OmniVoice model to use FlashInfer-accelerated generation.

        Usage:
            `_load_model` calls this helper right after the model is loaded when
            FlashInfer is active. The vendored patch (see
            `fattervoice/omnivoice_flashinfer.py`) replaces `_generate_iterative`
            with a packed-sequence implementation that fuses the cond/uncond CFG
            pair into one ragged-attention row, replaces RMSNorm/RoPE/MLP with
            fused FlashInfer kernels, and optionally replays CUDA graphs.

            The model was already loaded in float16 (coerced in `_load_model`)
            and on a CUDA device, but the device check is repeated here so
            programmatic configuration (e.g. tests) fails fast with a clear error
            instead of a cryptic kernel mismatch.

        Parameters:
            None.

        Returns:
            None. The loaded model is patched in place.
        """
        if not self.device_map.startswith("cuda"):
            raise ValueError(
                "FlashInfer acceleration requires a CUDA device (got "
                f"{self.device_map!r})."
            )
        try:
            import flashinfer  # noqa: F401 - presence check; kernels used via the patch
        except ImportError as exc:
            raise ImportError(
                "FlashInfer acceleration is enabled but the flashinfer packages are not installed. "
                "Install the optional dependencies (uv sync --extra flashinfer) or run with "
                "--flashinfer off."
            ) from exc

        from .omnivoice_flashinfer import apply_flashinfer

        apply_flashinfer(self._model)
        LOGGER.info("FlashInfer acceleration enabled")

    def _require_model(self):
        """Return the loaded OmniVoice model instance or fail with a clear error.

        Usage:
            Internal helper methods call this before attempting prompt creation or
            generation so the error surface stays consistent if startup ordering is
            incorrect.

        Parameters:
            None.

        Returns:
            The loaded `OmniVoice` model instance.

        Raises:
            RuntimeError: If the model has not been loaded yet.
        """
        if self._model is None:
            raise RuntimeError("The TTS model has not been loaded yet.")
        return self._model

    def _release_unused_accelerator_memory(self) -> None:
        """Release unused accelerator cache after prompt or voice transitions.

        Usage:
            The synthesis service calls this helper when it has already dropped
            Python references to GPU-backed prompt data or when it is switching
            away from a previously generated voice. This keeps large prompt-driven
            VRAM high-water marks from lingering longer than necessary.

        Parameters:
            None.

        Returns:
            None. The helper performs best-effort garbage collection and, when
            available, releases unused accelerator cache blocks back to the
            runtime.
        """
        gc.collect()

        try:
            import torch
        except ImportError:  # pragma: no cover - torch is a runtime dependency in production.
            return

        if hasattr(torch, "cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
            return

        if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache") and torch.mps.is_available():
            torch.mps.empty_cache()

    def _resolve_cached_voice_clone_prompt(
        self,
        voice: VoiceEntry,
    ) -> CachedVoiceClonePrompt:
        """Create or retrieve the cached OmniVoice prompt for one validated voice.

        Usage:
            Buffered and streaming synthesis both call this helper so a voice
            prompt is built lazily on first use and then reused for later
            requests. Cached prompts are stored in CPU memory so previously used
            voices do not keep VRAM pinned between requests, while still avoiding
            repeated reference-audio tokenization when users swap back to an
            earlier voice.

        Parameters:
            voice: The validated voice entry selected for the current request.

        Returns:
            A cached `CachedVoiceClonePrompt` object ready for OmniVoice
            generation calls.
        """
        with self._prepared_voice_lock:
            cached_voice = self._prepared_voice_cache.get(voice.voice_id)
            if cached_voice is not None:
                return cached_voice

            LOGGER.info("Creating lazy OmniVoice prompt cache entry for voice %s", voice.voice_id)
            with self._model_lock:
                prompt = self._require_model().create_voice_clone_prompt(
                    ref_audio=str(voice.audio_path),
                    ref_text=voice.transcript,
                    preprocess_prompt=self.config.preprocess_voice_clone_prompt,
                )

            cpu_prompt = move_prompt_value_to_cpu(prompt)
            del prompt
            self._release_unused_accelerator_memory()
            cached_voice = CachedVoiceClonePrompt(
                voice_id=voice.voice_id,
                prompt=cpu_prompt,
            )
            self._prepared_voice_cache[voice.voice_id] = cached_voice
            return cached_voice

    def _validate_request_text(self, text: str) -> None:
        """Validate request text before it reaches the heavy model inference path.

        Usage:
            Both streaming and non-streaming synthesis call this helper to reject
            empty requests consistently across protocol adapters before any model
            work or voice resolution begins.

        Parameters:
            text: The request text that will be synthesized.

        Returns:
            None. The function succeeds silently when the text is acceptable.

        Raises:
            ValueError: If the text is empty after surrounding whitespace is removed.
        """
        normalized_text = text.strip()
        if not normalized_text:
            raise ValueError("Synthesis text cannot be empty.")

    def _generate_waveform(
        self,
        request: SynthesisRequest,
        voice: VoiceEntry,
    ) -> tuple[np.ndarray, int]:
        """Generate one full OmniVoice waveform for a validated synthesis request.

        Usage:
            This helper centralizes the per-call interaction with OmniVoice so
            the HTTP adapter and Wyoming adapter both share the same prompt
            caching and generation-parameter logic.

        Parameters:
            request: The normalized request supplying text, language, and speed.
            voice: The validated voice entry selected for the request.

        Returns:
            A tuple of `(waveform, sample_rate)` where the waveform is ready for
            WAV/PCM encoding.
        """
        cached_voice = self._resolve_cached_voice_clone_prompt(voice)
        generation_kwargs = self._build_generation_kwargs(request, voice, cached_voice)
        with self._model_lock:
            if (
                self._last_generated_voice_id is not None
                and self._last_generated_voice_id != voice.voice_id
            ):
                self._release_unused_accelerator_memory()

            audio_list = self._require_model().generate(**generation_kwargs)
            self._last_generated_voice_id = voice.voice_id

        waveform = audio_list[0] if audio_list else np.zeros(0, dtype=np.float32)
        normalized_waveform = coerce_waveform_array(waveform)
        del audio_list
        self._release_unused_accelerator_memory()
        return normalized_waveform, self.sample_rate

    def _build_generation_kwargs(
        self,
        request: SynthesisRequest,
        voice: VoiceEntry,
        cached_voice: CachedVoiceClonePrompt,
    ) -> dict[str, object]:
        """Build the OmniVoice keyword arguments for one synthesis call.

        Usage:
            Internal generation methods call this helper so every request uses
            the same cached voice-clone prompt, the selected voice's optional
            instruct text, and the same server-level OmniVoice tuning defaults
            unless a client overrides request-level speed.

        Parameters:
            request: The normalized request supplying text, language, and speed.
            voice: The validated voice entry selected for this request, including
                any optional instruct text loaded from `<voice>.instruct.txt`.
            cached_voice: The tokenized OmniVoice voice-clone prompt selected for
                this request.

        Returns:
            A dictionary of keyword arguments accepted by `OmniVoice.generate`.
        """
        generation_language = normalize_omnivoice_language(
            request.language or self.config.default_language
        )
        generation_kwargs: dict[str, object] = {
            "text": request.text.strip(),
            "language": generation_language,
            "voice_clone_prompt": cached_voice.prompt,
            "num_step": self.config.num_step,
            "guidance_scale": self.config.guidance_scale,
            "denoise": self.config.denoise,
            "t_shift": self.config.t_shift,
            "position_temperature": self.config.position_temperature,
            "class_temperature": self.config.class_temperature,
            "layer_penalty_factor": self.config.layer_penalty_factor,
            "postprocess_output": self.config.postprocess_output_audio,
            "audio_chunk_threshold": _AUDIO_CHUNK_THRESHOLD_NEVER,
        }
        if voice.instruct is not None:
            generation_kwargs["instruct"] = voice.instruct
        if request.speed is not None:
            generation_kwargs["speed"] = request.speed
        return generation_kwargs
