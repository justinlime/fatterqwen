# Task for researcher

Research the OmniVoice TTS model (k2-fsa/OmniVoice, pip package omnivoice 0.1.5, diffusion language model TTS) and answer these specific questions with citations to its README/docs/paper/source:

1. What do the generation parameters `position_temperature` (default 5.0), `class_temperature` (default 0.0), `layer_penalty_factor` (default 5.0), `t_shift` (default 0.1), `num_step` (default 32), `guidance_scale` (default 2.0) do? Which ones most affect prosody naturalness vs. text adherence, and what do the official docs recommend for natural-sounding output?

2. Does OmniVoice recommend passing an explicit `language` (e.g. "en") rather than None/language-agnostic mode for better quality? What does the official documentation say ("Performance is slightly better if you specify the language" — find the exact wording)?

3. How does OmniVoice handle long text internally? Does it have built-in chunking (audio_chunk_duration=15.0, audio_chunk_threshold=30.0) with cross-fading between chunks (cross_fade_chunks)? What is the recommended way to synthesize multi-sentence text with natural prosody?

4. Does OmniVoice's `speed` parameter (default 1.0) affect prosody or only duration estimation? Any documented caveats about `speed` or `duration`?

5. Anything documented about `remove_silence`/`postprocess_output` (fade/pad) artifacts at audio boundaries — does the official demo apply postprocess per full-text synthesis?

Report concisely with concrete findings and source citations (file/line or URL). If the OmniVoice docs are sparse on some points, say so explicitly rather than speculating.

## Acceptance Contract
Acceptance level: attested
Completion is not accepted from prose alone. End with a structured acceptance report.

Criteria:
- criterion-1: Return concrete findings with file paths and severity when applicable

Required evidence: review-findings, residual-risks

Finish with a fenced JSON block tagged `acceptance-report` in this shape:
Use empty arrays when no items apply; array fields contain strings unless object entries are shown.
`criteriaSatisfied[].status` must be exactly one of: satisfied, not-satisfied, not-applicable.
`commandsRun[].result` must be exactly one of: passed, failed, not-run.
`manualNotes` and `notes` are optional strings; an empty string means no note and does not satisfy `manual-notes` evidence.
```acceptance-report
{
  "criteriaSatisfied": [
    {
      "id": "criterion-1",
      "status": "satisfied",
      "evidence": "specific proof"
    }
  ],
  "changedFiles": [
    "src/file.ts"
  ],
  "testsAddedOrUpdated": [
    "test/file.test.ts"
  ],
  "commandsRun": [
    {
      "command": "command",
      "result": "passed",
      "summary": "short result"
    }
  ],
  "validationOutput": [
    "validation output or concise summary"
  ],
  "residualRisks": [
    "none"
  ],
  "noStagedFiles": true,
  "diffSummary": "short description of the diff",
  "reviewFindings": [
    "blocker: file.ts:12 - issue found, or no blockers"
  ],
  "manualNotes": "anything else the parent should know"
}
```