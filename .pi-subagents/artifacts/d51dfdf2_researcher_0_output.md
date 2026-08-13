All research complete. Let me compile the final report.

---

# PyTorch 2.11 CUDA Wheel Architecture Coverage — Research Report

## Executive Summary

Both **torch 2.11.0+cu128** and **torch 2.11.0+cu130** ship the **identical** arch list: `sm_75 sm_80 sm_86 sm_90 sm_100 sm_120`. This covers **RTX 20 (Turing sm_75) through RTX 50 (Blackwell sm_120)**. Your current cu126 build (`sm_50 sm_60 sm_70 sm_75 sm_80 sm_86 sm_90`) is the only one missing Blackwell; **either cu128 or cu130 solves the RTX 50 problem without losing RTX 20**. Recommended: **cu130** — it is the stable PyPI variant, and CUDA 12.8 is already deprecated for removal in PyTorch 2.12.

All arch lists below were **verified by downloading and installing the actual wheels** (cp311, Linux x86_64) and calling `torch._C._cuda_getArchFlags()` (the API behind `torch.cuda.get_arch_list()`), cross-checked against the official build config and the PyTorch 2.11 RFC support matrix.

---

## Q1 — torch 2.11.0+cu128 arch list

**Answer: `7.5;8.0;8.6;9.0;10.0;12.0` → `sm_75 sm_80 sm_86 sm_90 sm_100 sm_120`**

- **sm_75 (Turing, RTX 20): YES, included.** ✅
- **sm_120 (Blackwell, RTX 50): YES, included.** ✅
- **sm_100 (Blackwell datacenter): YES, included.** ✅
- **sm_89 (Ada): NO — never shipped by PyTorch.** RTX 40 runs via sm_86→sm_89 binary compatibility (CUDA C Programming Guide: "a cubin object generated for compute capability X.y will only execute on devices of compute capability X.z where z≥y").
- **sm_70 (Volta): REMOVED in 2.11** (was present in 2.10 cu128).
- **sm_50/sm_60 (Maxwell/Pascal): REMOVED in 2.11** (kept only in cu126).

**Evidence (3 independent sources, all agree):**

1. **Wheel verification (ground truth)** — installed `torch-2.11.0+cu128-cp311-cp311-manylinux_2_28_x86_64.whl` (sha256 `c9a7ca4c...`, from `https://download.pytorch.org/whl/cu128/torch/`):
   ```
   torch version: 2.11.0+cu128 | cuda: 12.8 | cudnn: 91900 (9.19.0.56)
   arch flags: sm_75 sm_80 sm_86 sm_90 sm_100 sm_120
   ```
   The wheel's `torch/version.py` = commit `70d99e998b4955e0049d13a98d77ae1b14db1f45` (the v2.11.0 release commit). Embedded nvcc string in `libtorch_cuda.so`: `-gencode;arch=compute_75,code=sm_75;-gencode;arch=compute_80,code=sm_80;-gencode;arch=compute_86,code=sm_86;-gencode;arch=compute_90,code=sm_90;-gencode;arch=compute_100,code=sm_100;-gencode;arch=compute_120,code=sm_120` (found at byte offset `0x459c0c0` region of `.nv_fatbin`). 444 fatbin containers × ~6 embedded ELF cubins ≈ 2729 cubins — consistent with 6 archs per kernel.
2. **Build config** — `pytorch/pytorch@v2.11.0/.ci/manywheel/build_cuda.sh` (lines 107–112): base `TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;9.0;10.0"`; `12.8) TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST};12.0"`. (Note: wheel build configs moved from `pytorch/builder` into `pytorch/pytorch/.ci/`; the builder repo's `release_211_changes` branch is stale.)
3. **Official support matrix** — RFC `pytorch/pytorch#172663`: "12.8.1 → Turing(7.5), Ampere(8.0, 8.6), Hopper(9.0), Blackwell(10.0, 12.0)".

**Methodology sanity check:** your stated cu126 list was reproduced exactly from the installed 2.11.0+cu126 wheel: `sm_50 sm_60 sm_70 sm_75 sm_80 sm_86 sm_90` ✅.

---

## Q2 — torch 2.11.0+cu130 arch list

**Answer: identical to cu128 — `sm_75 sm_80 sm_86 sm_90 sm_100 sm_120`.** sm_75 **IS still included** in cu130.

Wheel-verified (`torch-2.11.0+cu130-cp311...whl`, from `https://download.pytorch.org/whl/cu130/torch/`): `arch flags: sm_75 sm_80 sm_86 sm_90 sm_100 sm_120`; cuDNN 9.19.0.56; runtime `cuda-toolkit==13.0.2`. Build config (`v2.11.0/.ci/manywheel/build_cuda.sh` line 118): `13.0) TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST};12.0"` (adds `11.0` on aarch64 only), with `TORCH_NVCC_FLAGS="-compress-mode=size"` and `BUILD_BUNDLE_PTXAS=1`.

**CUDA 13.0 architecture removals (NVIDIA CUDA 13.0 Update 1 release notes, §2.4.2 "Deprecated Architectures"):**
> "Architecture support for Maxwell, Pascal, and Volta is considered feature-complete. **Offline compilation and library support for these architectures have been removed in CUDA Toolkit 13.0 major version release.**" — and cuFFT notes: "**Removed support for Maxwell, Pascal, and Volta GPUs, corresponding to compute capabilities earlier than Turing.**"

So **CUDA 13.0 dropped sm_70 (Volta) — but the floor is Turing (sm_75), which is fully supported.** PyTorch 2.11 release notes (v2.11.0, Backwards-Incompatible, Release Engineering): "CUDA 13.0 only supports Turing (SM 7.5) and newer GPU architectures on Linux x86_64. Maxwell and Pascal GPUs are no longer supported under CUDA 13.0. Users with these older GPUs should use the CUDA 12.6 builds instead."

**Moving to cu130 does NOT lose RTX 20 series support.** It loses Maxwell/Pascal/Volta only.

Related: Volta was dropped from the **cu128/cu129** 2.11 builds too (PR `pytorch/pytorch#172598`, "to enable updating to CuDNN 9.15.1, which is incompatible with Volta"; announcement: dev-discuss "Dropping Volta support from CUDA-12.8 binaries for release 2.11"). cu129 wheels also exist for 2.11 (arch list `7.5;8.0;8.6;9.0;10.0;12.0+PTX`).

---

## Q3 — Minimum host driver (Linux x86_64)

Source: NVIDIA CUDA Toolkit Release Notes, Table 3 "CUDA Toolkit and Corresponding Driver Versions" (`https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html`, also `archive/13.0.1/...`):

| CUDA | Minimum Linux x86_64 driver |
|---|---|
| **CUDA 12.8 GA** | **≥ 570.26** ✅ (your assumption confirmed) |
| CUDA 12.8 Update 1 (what torch cu128 ships: `cuda-toolkit==12.8.1`) | ≥ 570.124.06 |
| **CUDA 13.0 GA** | **≥ 580.65.06** |
| CUDA 13.0 Update 2 (what torch cu130 ships: `cuda-toolkit==13.0.2`) | ≥ 580.95.05 |
| CUDA 13.x minor-version-compat floor | ≥ 580 |

(a) **cu128 → driver ≥ 570.26** (570.124.06+ for the exact 12.8u1 runtime shipped). (b) **cu130 → driver ≥ 580.x** (580.65.06 GA; 580.95.05 for the shipped 13.0.2 runtime). Note NVIDIA drivers are backward compatible, so a single ≥580.95 driver serves both cu128 and cu130 apps. ⚠️ Unverified nuance: these are the toolkit-driver tables; NVIDIA's "13.x ≥ 580" minor-version-compat row is the conservative runtime floor — flagging as slightly conservative.

---

## Q4 — Runtime behavior in practice

### sm_75 (Turing, RTX 20) on 2.11+cu128: **works**
- Real sm_75 cubins ship (not JIT/PTX fallback). No "no kernel image" reports for RTX 20 with cu128/cu130 in the 2.11 era (GitHub issue search: zero hits for Turing + cu128 compatibility failures).
- Turing is the **new floor** for cu128+ ("minimum GPU version supported by CUDA-12.8+ will be Turing" — dev-discuss, Volta-removal thread). No Turing-specific regression or drop was found in the 2.11 release notes.
- cuDNN 9.19.0.56 ships in cu128/cu130 wheels; PyTorch's reason for dropping Volta was cuDNN incompatibility — Turing+ remains cuDNN-supported. ⚠️ Partially inferred: NVIDIA's cuDNN support-matrix page is JS-only and couldn't be directly verified; corroborated by PyTorch's own statements and the shipped-wheel composition.
- Non-arch-specific 2.11 change to watch (not Turing-specific): MAGMA backend deprecated; `torch.linalg.svd`/`solve_triangular`/`lstsq` now dispatch unconditionally to cuSOLVER/cuBLAS (release notes; e.g. `#183806` shows a related cuSOLVER regression, but that one is aarch64/GH200-specific).

### sm_120 (Blackwell, RTX 50) on 2.11+cu128/cu130: **works**, with caveats
- Real sm_120 cubins ship; Blackwell supported since 2.7+cu128 (ptrblck/PyTorch support statements). 2.11 adds FlexAttention FlashAttention-4 backend on Blackwell and cuDNN SDPA Blackwell shape checks (`#172621`).
- **Known open issues (community-reported, 2.11-era, all non-blocking for typical TTS inference):**
  - `pytorch/pytorch#179214` (open, Apr 2026): `cudaErrorLaunchOutOfResources` in **CTCLoss backward** on RTX 5090 (sm_120) when transcript length forces 512 threads (>256); workaround batch ≤3. Relevant if training TTS with CTC alignment on RTX 50.
  - `pytorch/pytorch#181379` (open, Apr 2026): cuDNN SDPA **head_dim limited to 128 on sm_120** in release/2.11 (`sdp_utils.cpp` only lifts to 256 for sm_90); Flash/Efficient/Math backends accept head_dim 192/256. TTS attention (typically head_dim 64) unaffected.
  - `pytorch/pytorch#176426` (open, Mar 2026): **Triton kernels with ≥2 `tl.load()` segfault on sm_120** (RTX PRO 6000) — affects `torch.compile` on Blackwell.
  - `pytorch/pytorch#182286` (open, May 2026): caching-allocator fragmentation on large contiguous allocations (50+ GiB free, 3–18 GiB request fails). Mitigation: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (still opt-in in 2.11; release note `#172056`).
  - Closed/smaller: `#184062` (DataLoader GC segfault on RTX 5080, closed), `#174731` (sm_120 cubin loading, Windows-only, closed), `#178038` (`torch.dot` SIGFPE on RTX 5060 Ti, open), `#192707` (FP8 `_scaled_mm` on Blackwell, closed).
- No env vars are *required* for basic operation; `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is recommended for Blackwell memory-fragmentation avoidance.

---

## Q5 — torchaudio 2.11.0 pairing

- **torchaudio-2.11.0+cu128** exists: cp310, cp311, cp312, cp313 (`https://download.pytorch.org/whl/cu128/torchaudio/`). ✅
- **torchaudio-2.11.0+cu130** exists: cp310–cp313 (`https://download.pytorch.org/whl/cu130/torchaudio/`). ✅
- Also exists for cu126 and cu129, and `torchaudio 2.11.0` (CPU) on PyPI (28 files).
- **Pairing constraint:** the torchaudio 2.11.0+cu128 wheel METADATA declares **zero `Requires-Dist`** (no hard pin on torch; verified also via PyPI `requires_dist: None`). Pairing is by release-version convention: install `torchaudio==2.11.0+cuXXX` together with `torch==2.11.0+cuXXX` from the same index.

---

## Sources
- Wheels + arch flags (primary): `https://download.pytorch.org/whl/cu128/torch/`, `.../whl/cu130/torch/`, `.../whl/cu126/torch/` (verified in local venvs via `torch._C._cuda_getArchFlags()`)
- Build config: `https://github.com/pytorch/pytorch/blob/v2.11.0/.ci/manywheel/build_cuda.sh` (lines 107–120); Volta-removal diff `https://github.com/pytorch/pytorch/pull/172598.diff`
- Release notes: `https://github.com/pytorch/pytorch/releases/tag/v2.11.0` (Volta drop; PyPI→CUDA 13.0; Maxwell/Pascal drop); blog `https://pytorch.org/blog/pytorch-2-11-release-blog/`
- Support matrix RFC: `https://github.com/pytorch/pytorch/issues/172663`; dev-discuss `https://dev-discuss.pytorch.org/t/transitioning-pypi-cuda-wheels-to-cuda-13-0-as-the-stable-release-2-11/3325`; Volta removal `.../t/3290`; CUDA 12.8 deprecation `.../t/3337`
- NVIDIA drivers: `https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html` (Table 2/3); CUDA 13.0 arch removals: `https://docs.nvidia.com/cuda/archive/13.0.1/cuda-toolkit-release-notes/index.html` (§2.4.2)
- Binary compatibility: `https://docs.nvidia.com/cuda/archive/12.8.0/cuda-c-programming-guide/index.html` (§3.1.2)
- Runtime issues: pytorch/pytorch `#179214`, `#181379`, `#176426`, `#182286`, `#184062`, `#174731`, `#178038`, `#183806`, `#172621`; discuss.pytorch.org threads 222207/223536/224288/223683

---