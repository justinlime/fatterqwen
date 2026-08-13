I now have all the evidence needed from primary sources (README, docs, source code at both master and tag 0.1.5, the arXiv paper, PyPI, and the HF Space demo code). Here is the research report.

---

# OmniVoice TTS — Research Findings

**Scope note:** The task names pip package `omnivoice` 0.1.5 (released 2026-04-28; latest is 0.2.1, 2026-07-16 — PyPI release history). I verified both the `0.1.5` git tag and `master`; the generation-config defaults, chunking, and cross-fade logic are identical between them. Differences are flagged where they exist (`pad_duration`/`fade_duration` became configurable only in 0.2.x). All citations are to file paths in `k2-fsa/OmniVoice` (raw GitHub URLs) or the paper.

---

## 1. Generation parameters — semantics, defaults, and effect on prosody vs. text adherence

**Defaults confirmed in source** — `OmniVoiceGenerationConfig` dataclass, `omnivoice/models/omnivoice.py` (identical at tag `0.1.5` and `master`):

```python
@dataclass
class OmniVoiceGenerationConfig:
    num_step: int = 32
    guidance_scale: float = 2.0
    t_shift: float = 0.1
    layer_penalty_factor: float = 5.0
    position_temperature: float = 5.0
    class_temperature: float = 0.0
    denoise: bool = True
    preprocess_prompt: bool = True
    postprocess_output: bool = True
    audio_chunk_duration: float = 15.0
    audio_chunk_threshold: float = 30.0
    # 0.2.x only: pad_duration: float = 0.1, fade_duration: float = 0.1
```
Source: https://raw.githubusercontent.com/k2-fsa/OmniVoice/master/omnivoice/models/omnivoice.py (and `/0.1.5/...`)

**Official descriptions** — `docs/generation-parameters.md` (identical at 0.1.5 and master):

| Param | Default | Doc description |
|---|---|---|
| `num_step` | 32 | "Number of iterative unmasking steps. Higher values improve quality but slow down generation. Use 16 for faster inference." |
| `guidance_scale` | 2.0 | "Classifier-free guidance scale." |
| `t_shift` | 0.1 | "Time-step shift for the noise schedule. Smaller values emphasise earlier steps in decoding." |
| `position_temperature` | 5.0 | "Temperature for mask-position selection. 0 = greedy (deterministic). Higher values increase randomness." |
| `class_temperature` | 0.0 | "Temperature for token sampling at each step. 0 = greedy (deterministic). Higher values increase randomness." |
| `layer_penalty_factor` | 5.0 | "Penalty applied to deeper codebook layers, encouraging earlier (lower) layers to unmask first." |

**How they actually work in code** (`omnivoice/models/omnivoice.py`, `_generate_iterative` / `_predict_tokens_with_scoring` / `_get_time_steps`):
- `t_shift` shapes the unmasking schedule: `timesteps = t_shift * t / (1 + (t_shift - 1) * t)` applied to `linspace(0, 1, num_step+1)`; the per-step unmask counts are `ceil(total_mask * (timesteps[step+1] - timesteps[step]))`. This is exactly Eq. (3) of the paper with N=32, τ=0.1.
- `guidance_scale`: CFG in log-softmax space — `log_probs = softmax(logP_cond + g * (logP_cond − logP_uncond))`; `0` disables CFG.
- `position_temperature`: applied to per-position confidence scores (max log-prob) via Gumbel sampling (`scores/temperature + gumbel_noise`) before selecting which masked positions to unmask.
- `class_temperature`: if > 0, top-k (10% ratio) filtering + Gumbel sampling for **token identity**; at default 0.0, token identity is deterministic `argmax`.
- `layer_penalty_factor`: `scores = scores − (layer_id * layer_penalty_factor)` — penalizes the 8 codebook layers by depth so lower layers unmask first.
- `num_step`: number of iterative unmasking iterations; masks are all filled by the last step.

**Paper's corroboration** (arXiv 2604.00688, §3.4 Inference): "we perform a 32-steps iterative unmasking process… τ=0.1 is the shift parameter… we apply a temperature T=5 to the confidence scores before sampling the indices. Token Assignment: …deterministically by taking the argmax… We also apply a layer penalty on the confidence scores to encourage unmasking lower-layer tokens first. Furthermore, classifier-free guidance … with a guidance scale of 2. **All aforementioned strategies are verified to effectively improve model performance and generation stability.**" Appendix B (steps 8/16/32/64): 32 ≈ 64 in SIM-o/WER/UTMOS; quality clearly degrades at 8 steps (e.g., LibriSpeech-PC WER 1.30→2.02, UTMOS 4.28→4.02).

**Prosody vs. text adherence — what the docs actually support:**
- The docs **do not** make an explicit "prosody vs. text adherence" parameter taxonomy. Stated explicitly: the docs are sparse on tuning guidance beyond defaults.
- What can be inferred from documented semantics: `guidance_scale` is the adherence/conditioning knob (CFG strength vs. the uncond path; demo exposes 0–4); `num_step` is the quality knob ("Higher values improve quality… Use 16 for faster inference"); `position_temperature` (default 5.0) and `class_temperature` (default 0.0) control sampling randomness — the paper deliberately keeps **token identity greedy** (argmax) while adding stochasticity only in position selection; `layer_penalty_factor`/`t_shift` are schedule/stability knobs.
- Official recommendation for natural output = **keep the shipped defaults** (they are the paper's verified settings). The only officially suggested deviation anywhere is `num_step=16` "for faster inference" (docs/generation-parameters.md) and the demo slider's "Lower = faster, higher = better quality" (omnivoice/cli/demo.py). There is no official recommendation to raise `class_temperature` or lower `position_temperature` for prosody.

## 2. Explicit `language` vs. language-agnostic mode

**Yes — the exact wording is in the source docstring of `generate()`** (`omnivoice/models/omnivoice.py`, both 0.1.5 and master):

> `language: Language name (e.g. "English") or code (e.g. "en"). None for language-agnostic mode. Performance is slightly better if you specify the language.`

- The README does **not** repeat this; it appears only in the API docstring (the sentence is absent from README.md, docs/generation-parameters.md, docs/tips.md, and the HF model card). I searched all of these; the quote is uniquely in the `generate()` docstring.
- Behavior in code: `_resolve_language()` accepts an ID (`"en"`) or full name (`"English"`); `None` (or `"None"`/unrecognized) → language-agnostic mode, where the style token becomes `<|lang_start|>None<|lang_end|>` (`_prepare_inference_inputs`). Unrecognized languages log a warning and fall back to `None`.
- Nuance: the **official Gradio demo defaults to "Auto"** (= `language=None`), with info text "Keep as Auto to auto-detect the language." (`omnivoice/cli/demo.py`; HF Space `app.py`). So the demo does not force an explicit language, but the API docstring recommends it for slightly better performance.

## 3. Long text handling: built-in chunking + cross-fade

**Yes, both exist and are automatic.** Pipeline in `generate()` (`omnivoice/models/omnivoice.py`):

1. `GenerationTask.get_indices()` splits items by estimated audio length vs. `audio_chunk_threshold * frame_rate` (30 s): shorter items → single-shot `_generate_iterative`; longer → `_generate_chunked`.
2. `_generate_chunked()`: derives a text chunk length from `audio_chunk_duration` (15 s): `text_chunk_len = int(audio_chunk_duration * frame_rate / avg_tokens_per_char)`, then splits with `chunk_text_punctuation(text, chunk_len, min_chunk_len=3)` — sentence-boundary splitting at `.,;:!?。，；：！？` that is abbreviation-aware (Mr., Dr., e.g., …) and merges undersized chunks (`omnivoice/utils/text.py`).
3. Each chunk's target token count is re-estimated (with `speed` applied per chunk); in **auto-voice (no reference) mode, chunk 0's generated audio tokens are used as the reference for all later chunks** (keeps voice consistent); in voice-clone mode the original ref audio is reused for every chunk. Chunks are batched by chunk index for VRAM efficiency.
4. Chunk waveforms are merged with **`cross_fade_chunks()`** (`omnivoice/utils/audio.py`, present in 0.1.5): inserts 0.3 s of silence per boundary broken into fade-out (0.1 s) + gap (0.1 s) + fade-in (0.1 s). This is **not a config parameter** — it is applied automatically whenever chunking occurs (`_decode_and_post_process`, `if isinstance(tokens, list)`).

**Recommended way to synthesize multi-sentence text:** per `docs/generation-parameters.md` ("Long-Form Generation"), there is no manual sentence-by-sentence workflow recommended — "the text is automatically split into smaller segments when the estimated duration of the generated speech exceeds `audio_chunk_duration`, with each segment producing approximately `audio_chunk_duration` seconds of audio… allows the model to accept arbitrarily long text and generate arbitrarily long speech with near-constant VRAM consumption." I.e., pass the whole text to `generate()` and let the built-in chunking handle it (note: the doc's wording blurs threshold vs. chunk size; per code, chunking is *activated* above `audio_chunk_threshold`=30 s and chunk *target size* is `audio_chunk_duration`=15 s).

## 4. `speed` — duration estimation only, plus documented caveats

- **Code:** `speed` only scales the **duration estimate** (target token count). `_estimate_target_tokens()`: `est = duration_estimator.estimate_duration(...); if speed > 0 and speed != 1.0: est = est / speed`. The duration estimator is a character-weight rule (`RuleDurationEstimator`, `omnivoice/utils/duration.py`: phonetic weights per script, e.g. Latin 1.0 ≈ 40–50 ms/char, CJK 3.0). No audio resampling or prosody post-processing — speaking rate changes only because the model generates a shorter/longer token sequence. **No documented claim that `speed` preserves or alters prosody**; docs are silent on prosody effects.
- **Documented caveats:**
  - `docs/generation-parameters.md`: "`speed` … Values > 1.0 produce shorter audio (faster); values < 1.0 produce longer audio (slower). **Ignored when `duration` is set. Defaults to 1.0 when both are None.** Priority: `duration` > `speed`." Also: "When using `duration`, the default post-processing step may trim trailing silence, causing the actual output to be slightly shorter than the requested duration. If you need the output duration to **exactly** match the specified value, set `postprocess_output=False` to disable silence removal."
  - README.md (Batch Inference): "`duration` (in seconds) fixes the output length; `speed` controls the speaking rate. If `duration` and `speed` are both provided, `speed` will be ignored."
  - Demo UI: Speed slider 0.5–1.5, default 1.0, "1.0 = normal. >1 faster, <1 slower. Ignored if Duration is set." (`omnivoice/cli/demo.py`).
- Code detail: when `duration` is set, the estimate is computed at speed=1.0 and `target_lens` are overwritten with exact frame counts; a derived `est/target_tokens` ratio is stored in `speed_list` and used to scale chunked generation (`_preprocess_all`). Note: `speed` is passed as `speed` to `generate()` and the README example uses `speed=1.0`; docs table lists default None (→ 1.0).

## 5. `remove_silence` / `postprocess_output` and boundary artifacts

- **Post-processing pipeline** (`_decode_and_post_process` → `_post_process_audio`, `omnivoice/models/omnivoice.py`):
  1. If `postprocess_output=True` (default): `remove_silence(wav, sr, mid_sil=500, lead_sil=100, trail_sil=100)` — removes mid-signal silences > 500 ms (keeping 500 ms), trims edges keeping 100 ms lead/trail (`omnivoice/utils/audio.py`, pydub `split_on_silence`, −50 dB threshold).
  2. Volume normalization (ref RMS scaling, or peak→0.5 for auto voice).
  3. `fade_and_pad_audio(pad_duration=0.1, fade_duration=0.1)` — 0.1 s silence pad + 0.1 s fade-in/out per edge "to prevent clicks". **This step runs unconditionally**, even when `postprocess_output=False`; in 0.1.5 the 0.1 s/0.1 s values are hardcoded function defaults (not configurable), while 0.2.x exposes `pad_duration`/`fade_duration` in the config and docs.
- **Chunk boundaries:** cross-fade (0.1 s out + 0.1 s gap + 0.1 s in, §3) is applied **before** post-processing, over the whole merged waveform. The resulting ~0.1 s inter-chunk gap is below the 500 ms `mid_sil` threshold, so it survives `remove_silence`; only edges get trimmed (100 ms) and padded/faded (0.1 s). There is no documented user-facing statement about boundary artifacts — docs only mention the trailing-silence/duration interaction (see Q4). This part of the docs is sparse; the behavior above is from source reading.
- **Official demo:** yes, the demo synthesizes the full text in a single `model.generate(**kw)` call per submission — chunking and post-processing happen inside `generate()` over the full-text result (HF Space `app.py` and `omnivoice/cli/demo.py`, both 0.1.5 and master). The demo exposes a "Postprocess Output" checkbox, default **True**, info "Remove long silences from generated audio." There is no per-chunk postprocessing in the demo.

---

## Sources (primary)

- README: https://github.com/k2-fsa/OmniVoice (raw: `raw.githubusercontent.com/k2-fsa/OmniVoice/master/README.md`)
- `docs/generation-parameters.md` (master & tag 0.1.5) — parameter tables, duration/speed notes, long-form chunking
- `docs/tips.md` (master & 0.1.5) — ref_audio+instruct, short-clip tip
- `omnivoice/models/omnivoice.py` (master & 0.1.5) — `OmniVoiceGenerationConfig`, `generate()` docstring (language quote), `_generate_iterative`, `_predict_tokens_with_scoring`, `_get_time_steps`, `_generate_chunked`, `_preprocess_all`, `_estimate_target_tokens`, `_post_process_audio`, `_resolve_language`
- `omnivoice/utils/audio.py` (master & 0.1.5) — `cross_fade_chunks`, `remove_silence`, `fade_and_pad_audio`
- `omnivoice/utils/duration.py` — `RuleDurationEstimator`
- `omnivoice/utils/text.py` — `chunk_text_punctuation`
- `omnivoice/cli/demo.py` (master & 0.1.5); HF Space `app.py` (https://huggingface.co/spaces/k2-fsa/OmniVoice/raw/main/app.py)
- Paper: https://arxiv.org/html/2604.00688v1 (§3.4 Inference, Appendix B)
- PyPI: https://pypi.org/project/omnivoice/ (0.1.5 released 2026-04-28; 0.2.1 latest)