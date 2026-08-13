# Task for researcher

Research FlashInfer GPU architecture support and runtime requirements, to answer deployment questions about a TTS server using flashinfer-python==0.6.15.post1 with the prebuilt flashinfer-jit-cache==0.6.15.post1+cu128 wheel (AOT precompiled kernels). I need precise, sourced answers to:

1. Which NVIDIA GPU compute capabilities (sm_XX) does FlashInfer 0.6.x actually SUPPORT for the attention/prefill kernels (e.g. batch_prefill_with_ragged_kv_cache, BatchPrefillWithRaggedKVCacheWrapper)? Specifically: is Turing sm_75 (RTX 20 series like 2080 Ti, T4) genuinely supported for these kernels in 0.6.x, or do the prefill/attention kernels require sm_80+ (Ampere)? Check the FlashInfer GitHub repo (flashinfer-ai/flashinfer), docs at docs.flashinfer.ai, and any release notes for 0.6.15. Look for statements like "requires Ampere or newer" vs "sm75 and later", and note if prefill kernels specifically list sm80+ requirements while only some kernels (e.g. page/decoding) support sm75.

2. For the prebuilt AOT jit-cache wheels (flashinfer-jit-cache +cu128): do they cover all archs from sm_80 through sm_120 (RTX 30/40/50 series, A100, H100, B200), or are some archs excluded (needing runtime JIT with nvcc)? Is there any per-arch wheel variant, or is one manylinux x86_64 wheel universal across archs? Check what's published on the flashinfer.ai wheel index for cu128 and the GitHub releases.

3. What host NVIDIA driver version is required to load kernels from the +cu128 (CUDA 12.8) prebuilt cache? (cubin/PTX from CUDA 12.8 requires driver >= 570.x typically — confirm). Does running the cu128 cache inside a container whose CUDA runtime libs are 12.6 (torch built for cu126) but host driver is newer (e.g. 595) work, and any caveats?

4. RTX 40 series (sm_89) and RTX 50 series (sm_120): any known issues with flashinfer 0.6.15 prefill kernels on those archs?

Return a concise factual summary with source URLs for each claim, and explicitly flag anything you could not verify. Do not write code.

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