# Task for researcher

Research PyTorch CUDA wheel architecture coverage to decide the right CUDA runtime version for a Docker TTS image that must run on NVIDIA GPUs from RTX 20 series (Turing sm_75) through RTX 50 series (Blackwell sm_120). Be precise and sourced.

Context: our image currently uses torch 2.11.0+cu126, whose shipped arch list is ['sm_50','sm_60','sm_70','sm_75','sm_80','sm_86','sm_90'] — NO sm_89, sm_100, or sm_120. RTX 40 works today only because sm_86 cubins are binary-compatible with sm_89. RTX 50 (sm_120) cannot run on cu126 at all.

Questions:

1. What is the exact TORCH_CUDA_ARCH_LIST for the official PyTorch 2.11.0+cu128 Linux x86_64 wheel? Specifically: does it include sm_75 (Turing, RTX 20 series) and sm_120 (Blackwell, RTX 50)? Also does it include sm_89/sm_100? Check the pytorch/builder GitHub repo build configs for v2.11.0 / release/2.11 (files like manywheel/build.sh or .ci/... scripts) and any PyTorch 2.11 release notes about dropping or keeping Turing/Ampere archs.

2. Same question for torch 2.11.0+cu130 (or any cu130 torch build): does the cu130 build include sm_75? IMPORTANT: CUDA 13.0 may have dropped or deprecated Turing (sm_75) support — verify whether PyTorch cu130 wheels still ship sm_75 cubins, or whether moving to cu130 would lose RTX 20 series support. Also check whether CUDA 13.0 dropped sm_70 (Volta) and what that means for sm_75.

3. Minimum host NVIDIA driver required for: (a) CUDA 12.8 (torch cu128) — confirm >= 570.26 on Linux; (b) CUDA 13.0 (torch cu130) — what is the minimum driver (580.x?) on Linux? Cite NVIDIA CUDA release notes tables.

4. Does PyTorch 2.11.0+cu128 run on sm_75 (Turing) in practice — any known regressions/dropped Turing support in PyTorch 2.11? And does PyTorch 2.11+cu128 run on sm_120 (RTX 50) — any known issues (e.g. needing specific env vars like PYTORCH_CUDA_ALLOC_CONF, or cuDNN issues on Blackwell)?

5. torchaudio: does torchaudio 2.11.0 exist for cu128 and cu130 (matching torch 2.11.0)? Any version pairing constraints?

Return a concise factual summary with source URLs. Explicitly flag anything unverified.

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