I have all the evidence needed. Let me compile the final research report.

---

# FlashInfer 0.6.15 GPU Architecture Support & Runtime Requirements — Research Findings

**Scope verified against:** `flashinfer-ai/flashinfer` tag `v0.6.15` (source), release `v0.6.15.post1` assets, the wheel index at `flashinfer.ai/whl/cu128`, GitHub issues/PRs, and NVIDIA CUDA 12.8 release notes.

---

## Q1. Which compute capabilities does FlashInfer 0.6.x actually support for prefill/attention kernels?

**Bottom line: SM75 (Turing) is *nominally* supported for the FA2-based prefill/attention kernels in 0.6.x — fp16 only — but with serious caveats. SM80+ is the well-tested path. bf16 prefill on sm75 is explicitly rejected at runtime.**

Evidence:

1. **Official support statement (README @ v0.6.15):** "Modern Architecture Support: **Support for SM75 (Turing) and later (through Blackwell)**", with a GPU table listing Turing SM7.5 (T4, RTX 20 series), Ampere SM8.0/8.6, Ada SM8.9, Hopper SM9.0, Blackwell SM10.0/10.3 and SM12.0/12.1.
   Source: https://github.com/flashinfer-ai/flashinfer/blob/v0.6.15/README.md

2. **Framework-level minimum check (`flashinfer/jit/core.py` @ v0.6.15):** `check_cuda_arch()` raises `RuntimeError("FlashInfer requires GPUs with sm75 or higher")` only if all detected archs are < sm75. sm75 is explicitly accepted (`major == 7 and minor >= 5`).
   Source: https://github.com/flashinfer-ai/flashinfer/blob/v0.6.15/flashinfer/jit/core.py (lines ~96–108)

3. **Prefill kernels have real SM75 code paths with a bf16 exclusion (`include/flashinfer/attention/prefill.cuh` @ v0.6.15):** The device functions `SinglePrefillWithKVCacheDevice` (~line 1983), `BatchPrefillWithRaggedKVCacheKernel` (~line 2596) and `BatchPrefillWithPagedKVCacheDevice` (~line 3365) wrap their bodies in `#if (__CUDA_ARCH__ < 800)`, and inside that guard:
   ```cpp
   if constexpr (std::is_same_v<DTypeQ, nv_bfloat16>) {
     FLASHINFER_RUNTIME_ASSERT("Prefill kernels do not support bf16 on sm75.");
   } else { ... }
   ```
   So on sm75: **fp16 (and fp32) prefill compiles and runs; bf16 prefill compiles to a kernel that asserts at runtime.** Same guard repeats at lines ~2356, 2599–2601, 3066, 3370–3372, 3961.
   Source: https://github.com/flashinfer-ai/flashinfer/blob/v0.6.15/include/flashinfer/attention/prefill.cuh

4. **Decode kernels (`include/flashinfer/attention/decode.cuh` @ v0.6.15):** no `__CUDA_ARCH__ < 800` guards found; the only arch guards are `__CUDA_ARCH__ >= 900` for Hopper `griddepcontrol` asm. Decode is not sm80-gated. (bf16 decode on sm75 uses the same MMA path question as prefill; I did **not** verify a bf16-on-sm75 decode guard — flagged as unverified.)

5. **Known open SM75 bug — severity: high for Turing deployment (issue #3620, still open):** "BatchPrefillWithPagedKVCache fails on SM75 (Turing / Tesla T4) with CUDA 'invalid argument'". Root cause per the issue: the prefill launcher's shared-memory budget (`max_num_mma_kv_smem` in `prefill.cuh`) is a q-only estimate that omits the K/V scale-factor buffers and padding; on sm75 the 64 KB per-block opt-in limit leaves no headroom, so `cudaFuncSetAttribute(..., cudaFuncAttributeMaxDynamicSharedMemorySize, ...)` rejects the launch. "The same code path runs fine on SM80+." Fix PR #3621 is **not merged** as of v0.6.15, and the same estimation pattern (`max_num_mma_kv_smem`) is present in the v0.6.15 tag in all three prefill launchers: `SinglePrefillWithKVCacheDispatched` (line ~2472), `BatchPrefillWithRaggedKVCacheDispatched` (line ~4075), and the paged launcher (line ~4179+). I verified the pattern exists at v0.6.15 by reading the tagged file, but did **not** execute on a T4, so "affects 0.6.15" is code-inspection-based, not run-verified.
   Sources: https://github.com/flashinfer-ai/flashinfer/issues/3620 , https://github.com/flashinfer-ai/flashinfer/pull/3621

6. **Historical context:** issue #1648 "Please don't give up on SM75" (open, Sept 2025) documents community concern that sm75 keeps losing support; closed issue #421 shows prefill didn't work on sm75 in the early 0.1.x line.
   Sources: https://github.com/flashinfer-ai/flashinfer/issues/1648 , https://github.com/flashinfer-ai/flashinfer/issues/421

7. **Some attention kernels are arch-gated above sm75** (so "attention kernels" is not monolithic): FP8 prefill kernels are SM90-only (`csrc/batch_prefill_fp8_sm90.cu`), FA3 prefill is SM90a (`prefill_sm90.cu`), FMHAv2-TRTLLM has an SM120 module, XQA is gated SM90+ (`aot.py` `gen_xqa_modules`: "XQA requires SM90+"), NVFP4 attention is SM120-only, MLA-decode has an SM80+ CuTe path. These all live under `flashinfer/jit/attention/` and `include/flashinfer/attention/` at the v0.6.15 tag.

---

## Q2. Does the prebuilt cu128 AOT jit-cache wheel cover sm80–sm120? Per-arch wheels?

**Bottom line: one universal `manylinux_2_28_x86_64` wheel (plus one `aarch64`), not per-GPU-arch. For cu128 it is built for sm75, sm80, sm89, sm90a, sm100a, sm120a. Some archs (sm103a B300, sm110a GB10, sm121a DGX Spark) are excluded and need other CUDA-version wheels or runtime JIT with nvcc.**

Evidence:

1. **Wheel index (https://flashinfer.ai/whl/cu128/flashinfer-jit-cache/):** only two variants per version — `...-cp39-abi3-manylinux_2_28_x86_64.whl` and `...-manylinux_2_28_aarch64.whl`. No per-GPU-arch wheels. `flashinfer_jit_cache-0.6.15.post1+cu128-cp39-abi3-manylinux_2_28_x86_64.whl` exists (sha256 `1af314fd...`), served from GitHub release assets of `v0.6.15.post1` (1.32 GB). (Release API shows cu129 = 2.05 GB, cu130 = 1.59 GB x86_64 — the size differences reflect the different arch lists below.)

2. **Exact arch list used to build the wheels (`.github/workflows/release.yml` @ v0.6.15, line ~185):**
   ```
   FLASHINFER_CUDA_ARCH_LIST: cuda<12.9 → "7.5 8.0 8.9 9.0a 10.0a 12.0a"
                            cuda<13.0 → "7.5 8.0 8.9 9.0a 10.0a 10.3a 12.0f"
                            aarch64   → "7.5 8.0 8.9 9.0a 10.0a 10.3a 11.0a 12.0f"
                            x86_64    → "7.5 8.0 8.9 9.0a 10.0a 10.3a 12.0f"
   ```
   So **cu128 (both platforms): `7.5 8.0 8.9 9.0a 10.0a 12.0a`**. Each module `.so` is a fatbin containing cubins for all 6 targets (one `.so` per module; `JitSpec.aot_path = FLASHINFER_AOT_DIR / name / name.so`).
   Sources: https://github.com/flashinfer-ai/flashinfer/blob/v0.6.15/.github/workflows/release.yml

3. **Coverage interpretation (per NVIDIA binary compatibility + FlashInfer's own arch handling in `flashinfer/compilation_context.py` @ v0.6.15):**
   - sm75 (T4, RTX 20): included ✓ (fp16-only prefill caveats, see Q1)
   - sm80 (A100) ✓; **sm86 (RTX 30 series / A10) runs via the sm80 cubin** (binary compat within the 8.x major; sm86 is not compiled separately)
   - sm89 (RTX 40, L4, L40) ✓
   - sm90 (H100/H200) ✓ via sm90a
   - sm100 (B100/B200) ✓ via sm100a
   - sm103 (B300/GB300): ✗ **not in the cu128 wheel** (cu129+ only)
   - sm110 (GB10/Grace Blackwell): ✗ not in any cu128 wheel; cu130 aarch64 only
   - sm120 (RTX 50): ✓ via sm120a
   - sm121 (DGX Spark): ✗ in **no** official wheel per the v0.6.15 workflow; the docs (0.6.18) say to add `12.1a` manually when building your own jit-cache
   - Missing cubins for the device → the AOT `.so` load fails at the driver level (no fallback to JIT in `JitSpec.build_and_load()` — if `is_aot` is true it loads the `.so` and returns), or you must JIT-compile at runtime, which requires **nvcc** in the container.
   Sources: https://github.com/flashinfer-ai/flashinfer/blob/v0.6.15/flashinfer/compilation_context.py , https://github.com/flashinfer-ai/flashinfer/blob/v0.6.15/flashinfer/jit/core.py , https://docs.flashinfer.ai/installation.html

4. **AOT build machinery (`flashinfer/aot.py` @ v0.6.15):** build requires `FLASHINFER_CUDA_ARCH_LIST` explicitly; `detect_sm_capabilities()` has **no sm75 key** (sm80/90/100/100f/103/110/120/120f/121 only), i.e., the AOT system treats sm75 as part of the generic (non-gated) module set. Default AOT config also excludes `head_dim=512` FA2 prefill/decode modules from the wheel to save space ("Note: head_dim=512 (FA2 prefill/decode, SM100+) excluded to reduce space in the jit-cache wheel"), matching release note "fix(aot): exclude head_dim=512 FA2 modules from AOT jit-cache wheel (#3769)".
   Sources: https://github.com/flashinfer-ai/flashinfer/blob/v0.6.15/flashinfer/aot.py , release notes https://github.com/flashinfer-ai/flashinfer/releases/tag/v0.6.15

5. **Runtime version-match check (`flashinfer/jit/env.py` @ v0.6.15):** the installed `flashinfer-jit-cache` version must `startswith` the `flashinfer` version → `0.6.15.post1+cu128` vs `flashinfer-python==0.6.15.post1` **passes**. Mismatched pairs (e.g., 0.6.15.post1 python + 0.6.16 cache) raise at import unless `FLASHINFER_DISABLE_VERSION_CHECK=1`.

6. **glibc requirement:** wheels are `manylinux_2_28` → host/container glibc ≥ 2.28 (Ubuntu 18.04+/20.04+, RHEL 8+).

---

## Q3. Host driver requirement for the +cu128 AOT cache, and the container (cu126 runtime / driver 595) question

**Bottom line: host driver ≥ 570.x (Linux) is required for the CUDA 12.8-built cubins (570.26 minimum for CUDA 12.8 GA; NVIDIA shipped 570.86.10). A driver of 595 is more than sufficient. Running the cu128 cache inside a torch-cu126 container works in the common case, with caveats.**

Evidence:

1. **NVIDIA CUDA 12.8 Release Notes, Table 3:** CUDA 12.8 GA → Linux x86_64 toolkit driver ≥ **570.26** (shipped driver 570.86.10). Table 2 ("minor version compatibility" floor for the whole CUDA 12.x family): ≥ 525.60.13 — that floor applies to apps *built with* ≤12.x toolchains, **not** to 12.8-built artifacts. The cu128 cubins (incl. arch-specific `sm_90a`/`sm_100a`/`sm_120a` targets) need the CUDA-12.8-level driver → **≥ 570.26**.
   Source: https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/index.html (Tables 2–3)
   (The task's "570.x typically" guess is confirmed; note the exact floor is 570.26, not 570.00.)

2. **Container with CUDA 12.6 runtime libs (torch cu126) + host driver 595:** works, because:
   - FlashInfer loads the AOT `.so` through the CUDA **driver API** (`tvm_ffi.load_module` → `cuModuleLoad`; `JitSpec.load()` in `flashinfer/jit/core.py`). Cubin loading is done by the host driver, and driver 595 ≥ 570 supports all CUDA 12.8 features including `sm_120a`.
   - The container's libcudart 12.6 only needs driver ≥ 525.60.13 (backward compatible). Driver 595 satisfies both sides.
   Caveats (verified in code / NVIDIA docs):
   - **NVIDIA minor-version compatibility is forward-only** (12.6-built code → 12.8+ runtime). 12.8-*built* host code running against a 12.6 libcudart is outside NVIDIA's compatibility matrix. In practice flashinfer's kernel path is driver-API driven (low risk), but any host-side symbol resolved from the container's 12.6 cudart is unsupported territory. Recommend keeping torch cu126 (runtime) but ensuring the host driver is ≥ 570 — 595 is fine.
   - **CUDA version detection matters:** with no nvcc in the container, `flashinfer/jit/cpp_ext.py::get_cuda_version()` falls back to `torch.version.cuda` → **12.6**. This feeds `CompilationContext._normalize_cuda_arch()`, which raises `RuntimeError("SM 12.x requires CUDA >= 12.9")` for sm120 devices (see Q4). It also gates sm90/sm100/sm103/sm110 feature flags in `flashinfer/utils.py` (e.g., `has_sm100` requires torch cuda ≥ 12.8). For sm80/sm89 (RTX 30/40) and sm90 (H100) these gates don't bite; for RTX 50 they do.
   - **No nvcc in torch cu126 containers:** if an arch missing from the wheel (sm103a/sm110a/sm121a) is needed, runtime JIT (`ninja` + `nvcc`) will fail — the container must have a CUDA toolkit (devel image) installed.
   Sources: https://github.com/flashinfer-ai/flashinfer/blob/v0.6.15/flashinfer/jit/cpp_ext.py , https://github.com/flashinfer-ai/flashinfer/blob/v0.6.15/flashinfer/jit/env.py , https://github.com/flashinfer-ai/flashinfer/blob/v0.6.15/flashinfer/compilation_context.py

3. **flashinfer-python 0.6.15.post1 exists on PyPI** (`py3-none-any`, requires-python ≥3.10,<4.0) — verified: https://pypi.org/pypi/flashinfer-python/0.6.15.post1/json

---

## Q4. Known issues on sm89 (RTX 40) and sm120 (RTX 50) with 0.6.15 prefill

**RTX 40 (sm89):** no open sm89-specific prefill correctness bugs found in the 0.6.15 timeframe; sm89 is fully covered by the wheel (dedicated `sm_89` cubins; `sm89_nvcc_flags` enable FP8). Cross-arch edge cases that also affect sm89 (all open): FA2 prefill never validates `head_dim` (#4299 — non-multiple-of-64 head dims → illegal memory access / wrong results); FA2 attention silently outputs zeros when every logit in a row is below −5e4 (#4267, filed on 0.6.12 **after** the 0.6.15 release — **unverified for 0.6.15**); bf16 decode precision for GQA group_size=7 (#2896). #4090 (nvfp4 GEMM test failure on 4090) is closed (test-only).

**RTX 50 (sm120):** the cu128 wheel contains `sm_120a` cubins, so AOT prefill is available — but:
- **Runtime JIT arch-detection bug (severity: high, open, PR #3633 unmerged):** with an sm120 device and CUDA toolkit/torch < 12.9 and no `FLASHINFER_CUDA_ARCH_LIST`, `_normalize_cuda_arch(12,0)` raises "SM 12.x requires CUDA >= 12.9", the caught exception empties `TARGET_CUDA_ARCHS`, and kernel use then fails with the misleading "FlashInfer requires GPUs with sm75 or higher". PR #3633 ("fix(jit): refresh stale arch detection and allow SM120 on CUDA 12.8") documents exactly this on RTX 5090 with CUDA 12.8 and is **not merged** into v0.6.15. Workaround: set `FLASHINFER_CUDA_ARCH_LIST` explicitly (e.g. `"12.0a"`, or the full wheel list).
  Sources: https://github.com/flashinfer-ai/flashinfer/pull/3633 , https://github.com/flashinfer-ai/flashinfer/blob/v0.6.15/flashinfer/compilation_context.py
- **sm120-family cubin isolation is intentional:** the code comments note each SM12.x variant gets its own cubin "to avoid running SM120 code on SM121 (DGX Spark) which can cause cudaErrorIllegalInstruction" — i.e., don't expect sm120a cubins to run on sm121, and vice versa.
- **Open sm120 issues (all still open):** #3935 — ~11% long-prefill regression on RTX 5090 between 0.6.12 and 0.6.13 (GDN hybrid model via vLLM; 0.6.15 contains "Revert GDN prefill regression" #3889 which may address a related regression, but I could not confirm it fixes #3935 — flagged unverified); #3828 — SM120 sparse-MLA DSV4 decode+prefill kernels not instantiated for topk=256 (DeepSeek-style MLA, not plain attention); #2638 — XQA (decode) headdim=256 with pagesize<64 fails on SM120 (decode, edge config); #3628 — RFC for MXFP8 block-scaled prefill on SM120a (feature gap: MXFP8 prefill not yet available on consumer Blackwell).

---

## Things I could NOT verify (flagged)

1. **Whether issue #3620 (sm75 prefill launch failure) reproduces identically on 0.6.15** — reported/verified by the reporter on 0.6.12/T4; the v0.6.15 tag retains the same smem-estimation code and the fix PR is unmerged, but I did not execute on Turing hardware.
2. **bf16 decode behavior on sm75** in 0.6.x — no explicit guard found in `decode.cuh`; not tested.
3. **Whether #3889 (GDN prefill revert) resolves the RTX 5090 perf regression #3935.**
4. **Whether #4267 (extreme-negative-logits FA2 zeros) affects 0.6.15** — filed on 0.6.12 after 0.6.15 shipped.
5. **Actual per-`.so` fatbin contents of the 1.32 GB cu128 wheel** (did not download; arch list derived from the release workflow, `compilation_context.py`, and `aot.py` at the v0.6.15 tag).
6. **Wheel size note:** the 0.6.15 release ships `flashinfer-cubin-0.6.15.post1` (611 MB, "pre-compiled kernel binaries for all supported GPU architectures", per-arch cubin store with a default NVIDIA artifactory URL in `cubin_loader.py`) as a separate path — not analyzed in depth since the deployment uses jit-cache.

---

## Acceptance Report