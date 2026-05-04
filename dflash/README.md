# Luce DFlash

Bare runtime for Qwen3.5/3.6 27B GGUF target inference with DFlash speculative decoding and PFlash prefill support.

## 5090 Defaults

- Build arch: `sm_120`.
- Target: `/home/jake/models/Qwen3.6-27B-GGUF/Qwen3.6-27B-UD-Q5_K_XL.gguf`.
- DFlash draft: `/home/jake/models/Qwen3.6-27B-DFlash-safetensors/draft-q8_0.gguf`.
- PFlash drafter: `/home/jake/models/Qwen3-0.6B-GGUF/Qwen3-0.6B-BF16.gguf`.
- KV cache: `DFLASH27B_KV_K=f16`, `DFLASH27B_KV_V=f16`.
- FA window: `DFLASH27B_FA_WINDOW=4096`.
- DDTree budget: `22`.
- GPU DDTree top-K and rollback are default-on.

Disable the new GPU paths only for debugging:

```bash
DFLASH27B_GPU_DDTREE_TOPK=0
DFLASH27B_GPU_DDTREE_ROLLBACK=0
```

`DFLASH27B_GPU_DDTREE_PREP=1` exists, but is not default because it did not win the combined 5090 run.

## Build

```bash
cmake -B build-luce-sm120 -S . \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DDFLASH27B_USER_CUDA_ARCHITECTURES=120 \
  -DDFLASH27B_ENABLE_BSA=ON
cmake --build build-luce-sm120 --target test_dflash test_generate test_flashprefill_kernels -j 8
```

## Quantized DFlash Draft

The runtime accepts only a quantized DFlash GGUF draft. For Qwen3.6, build the
Q8 draft with `scripts/quantize_draft_q8.py`; it writes the required
sliding-window metadata (`2048`, pattern `[S,S,S,S,F]`) while preserving the
local draft weights.
The runtime applies the SWA mask only when the draft context exceeds that
window; short contexts keep the original full-attention draft path, which avoids
unnecessary acceptance regressions on HumanEval-style prompts.
Directory resolution accepts `draft-q8_0.gguf` or another explicit `.gguf`.
There is no runtime safetensors fallback.

Build the local Q8 draft:

```bash
PYTHONPATH=deps/llama.cpp/gguf-py python3 scripts/quantize_draft_q8.py \
  /home/jake/models/Qwen3.6-27B-DFlash-safetensors/model.safetensors \
  /home/jake/models/Qwen3.6-27B-DFlash-safetensors/draft-q8_0.gguf \
```

Smoke-load and run a tiny draft graph:

```bash
./build-luce-sm120/smoke_load_draft \
  /home/jake/models/Qwen3.6-27B-DFlash-safetensors/draft-q8_0.gguf
./build-luce-sm120/smoke_draft_graph \
  /home/jake/models/Qwen3.6-27B-DFlash-safetensors/draft-q8_0.gguf 8
```

## Mega PFlash Server

The default PFlash backend is still the daemon-backed Qwen3-0.6B path. The
native Qwen3.5-0.8B Mega PFlash backend is opt-in and runs inside the
`test_dflash` daemon through the `compress-mega` protocol:

```bash
python scripts/server_tools.py \
  --target /home/jake/models/Qwen3.6-27B-GGUF/Qwen3.6-27B-UD-Q5_K_XL.gguf \
  --draft /home/jake/models/Qwen3.6-27B-DFlash-safetensors/draft-q8_0.gguf \
  --bin /home/jake/lucebox-hub/dflash/build-luce-sm120/test_dflash \
  --budget 22 \
  --max-ctx 8192 \
  --prefill-compression auto \
  --prefill-backend mega-native \
  --prefill-keep-ratio 0.05 \
  --prefill-mega-native-model /home/jake/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17
```

`--prefill-mega-max-seq-len` defaults to `--max-ctx`. Set it explicitly only
when you want a smaller or larger megakernel compression window.

When prefill compression is enabled, the server defaults `DFLASH_FP_USE_BSA=1`
and `DFLASH_FP_ALPHA=0.85`. Native Mega PFlash uses the same FlashPrefill/BSA
bridge for Qwen3.5 full-attention layers (`H=8`, `Hk=2`, `D=256`) and falls
back to dense cuBLAS attention if BSA is unavailable.

Native smoke test:

```bash
cmake --build build-luce-sm120 --target smoke_bsa_hdim256 smoke_mega_pflash_native -j 8
DFLASH_FP_USE_BSA=1 DFLASH_FP_PROFILE=1 ./build-luce-sm120/smoke_bsa_hdim256 512
DFLASH_FP_USE_BSA=1 DFLASH_FP_PROFILE=1 ./build-luce-sm120/smoke_mega_pflash_native \
  /home/jake/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17 \
  512 0.05 1024
```

## Bench

```bash
DFLASH_TARGET=/home/jake/models/Qwen3.6-27B-GGUF/Qwen3.6-27B-UD-Q5_K_XL.gguf \
DFLASH_DRAFT=/home/jake/models/Qwen3.6-27B-DFlash-safetensors/draft-q8_0.gguf \
DFLASH_BIN=/home/jake/lucebox-hub/dflash/build-luce-sm120/test_dflash \
DFLASH_BIN_AR=/home/jake/lucebox-hub/dflash/build-luce-sm120/test_generate \
DFLASH27B_KV_K=f16 \
DFLASH27B_KV_V=f16 \
DFLASH27B_FA_WINDOW=4096 \
python scripts/bench_llm.py
```

Current headline numbers are in `RESULTS.md`.

To compare end-to-end prefill modes through `server_tools.py`:

```bash
python scripts/bench_prefill_backends.py --runs 3 --show-timing
python scripts/bench_prefill_backends.py --runs 1 --backends off,mega-native --show-timing
```

This runs no-compression, daemon PFlash, Python Mega PFlash, and native Mega
PFlash with prefix/full prefill caches disabled so every request exercises the
selected backend.
