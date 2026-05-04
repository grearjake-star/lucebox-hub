# Luce DFlash

Standalone CUDA/ggml runtime for Qwen3.5/3.6 27B GGUF target inference with a
quantized DFlash GGUF draft, DDTree speculative verification, and optional
PFlash prefill.

The production DFlash draft path is GGUF-only. The runtime does not link or
fallback to DFlash safetensors.

## Current 5090 Profile

| Component | Value |
|-----------|-------|
| GPU build arch | `sm_120` |
| Target | `/home/jake/models/Qwen3.6-27B-GGUF/Qwen3.6-27B-UD-Q5_K_XL.gguf` |
| DFlash draft | `/home/jake/models/Qwen3.6-27B-DFlash-safetensors/draft-q8_0.gguf` |
| DDTree budget | `22` |
| Short bench KV | `DFLASH27B_KV_K=f16`, `DFLASH27B_KV_V=f16` |
| Short bench FA window | `DFLASH27B_FA_WINDOW=4096` |
| GPU DDTree top-K | default on |
| GPU DDTree rollback | default on |
| GPU DDTree verify prep | opt-in, `DFLASH27B_GPU_DDTREE_PREP=1` |

Disable GPU DDTree paths only for debugging:

```bash
DFLASH27B_GPU_DDTREE_TOPK=0
DFLASH27B_GPU_DDTREE_ROLLBACK=0
```

## Model Artifacts

Required:

- Target GGUF: `Qwen3.6-27B-UD-Q5_K_XL.gguf`
- Quantized draft GGUF: `draft-q8_0.gguf`

Optional:

- `Qwen3-0.6B-BF16.gguf` for daemon-backed PFlash prefill.
- Qwen3.5-0.8B safetensors snapshot for native Mega PFlash.
- Original DFlash safetensors only when rebuilding the Q8 draft. It is not
  needed for runtime.

The local cleaned model layout is:

```text
/home/jake/models/Qwen3.6-27B-GGUF/Qwen3.6-27B-UD-Q5_K_XL.gguf
/home/jake/models/Qwen3.6-27B-DFlash-safetensors/draft-q8_0.gguf
/home/jake/models/Qwen3-0.6B-GGUF/Qwen3-0.6B-BF16.gguf
```

## Build

```bash
git submodule update --init --recursive

cmake -B build-luce-sm120 -S . \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DDFLASH27B_USER_CUDA_ARCHITECTURES=120 \
  -DDFLASH27B_ENABLE_BSA=ON

cmake --build build-luce-sm120 \
  --target test_dflash test_generate smoke_load_draft smoke_draft_graph \
  -j 8
```

Useful optional targets:

```bash
cmake --build build-luce-sm120 \
  --target smoke_bsa_hdim256 smoke_mega_pflash_native pflash_daemon \
  -j 8
```

## Quant Draft

`draft-q8_0.gguf` is produced from the upstream DFlash draft weights with:

```bash
PYTHONPATH=deps/llama.cpp/gguf-py \
python3 scripts/quantize_draft_q8.py \
  /path/to/model.safetensors \
  /home/jake/models/Qwen3.6-27B-DFlash-safetensors/draft-q8_0.gguf
```

The converter always writes the Qwen3.6 DFlash sliding-window metadata:

- `attention.sliding_window = 2048`
- `attention.sliding_window_pattern = [S,S,S,S,F]`

The GGUF loader hard-rejects missing metadata, alternate arch names, and
alternate tensor names. This keeps the runtime deterministic and prevents
silently loading an incompatible draft.

## Smoke Tests

```bash
./build-luce-sm120/smoke_load_draft \
  /home/jake/models/Qwen3.6-27B-DFlash-safetensors/draft-q8_0.gguf

./build-luce-sm120/smoke_draft_graph \
  /home/jake/models/Qwen3.6-27B-DFlash-safetensors/draft-q8_0.gguf 8
```

Expected draft load summary includes:

```text
loaded 58 tensors, total 1.71 GiB
sliding_window=2048 layer_is_swa=1,1,1,1,0
OK
```

Safetensors should fail intentionally:

```text
draft model must be a quantized DFlash GGUF (.gguf)
```

## Raw Benchmark

Runs AR target-only and DFlash on the same prompts.

```bash
DFLASH_TARGET=/home/jake/models/Qwen3.6-27B-GGUF/Qwen3.6-27B-UD-Q5_K_XL.gguf \
DFLASH_DRAFT=/home/jake/models/Qwen3.6-27B-DFlash-safetensors/draft-q8_0.gguf \
DFLASH_BIN=/home/jake/lucebox-hub/dflash/build-luce-sm120/test_dflash \
DFLASH_BIN_AR=/home/jake/lucebox-hub/dflash/build-luce-sm120/test_generate \
DFLASH27B_KV_K=f16 \
DFLASH27B_KV_V=f16 \
DFLASH27B_FA_WINDOW=4096 \
.venv/bin/python scripts/bench_llm.py \
  --benches HumanEval,GSM8K,Math500 \
  --n-gen 256 \
  --budget 22 \
  --n-sample 10 \
  --out-json /tmp/dflash_bench/bench_llm_gguf_all3.json
```

Current Q8 GGUF draft results on RTX 5090:

| Task | AR tok/s | DFlash tok/s | AL | Speedup |
|------|---------:|-------------:|---:|--------:|
| HumanEval | 58.01 | 273.11 | 7.63 | 4.71x |
| GSM8K | 58.78 | 228.02 | 6.35 | 3.88x |
| Math500 | 58.66 | 257.18 | 7.20 | 4.38x |

See `RESULTS.md` for the safetensors-vs-GGUF A/B and historical sweeps.

## API / HTTPS Latency Benchmark

Start the OpenAI-compatible server:

```bash
DFLASH27B_KV_TQ3=1 .venv/bin/python scripts/server_tools.py \
  --host 127.0.0.1 \
  --port 8012 \
  --target /home/jake/models/Qwen3.6-27B-GGUF/Qwen3.6-27B-UD-Q5_K_XL.gguf \
  --draft /home/jake/models/Qwen3.6-27B-DFlash-safetensors/draft-q8_0.gguf \
  --bin /home/jake/lucebox-hub/dflash/build-luce-sm120/test_dflash \
  --budget 22 \
  --max-ctx 4096 \
  --prefix-cache-slots 0 \
  --prefill-cache-slots 0
```

Measure streaming request latency:

```bash
.venv/bin/python scripts/bench_daemon.py \
  --url http://127.0.0.1:8012 \
  --benches HumanEval,GSM8K,Math500 \
  --n-gen 256 \
  --n-sample 10 \
  --warmup \
  --out-json /tmp/dflash_bench/bench_api_gguf_all3.json
```

Use an `https://...` URL to include TLS, reverse-proxy, or remote network
latency. The benchmark reports wall tok/s and first-token-to-last-token decode
tok/s separately.

Current local HTTP Q8 GGUF results:

| Task | Wall tok/s | Decode tok/s | Aggregate decode |
|------|-----------:|-------------:|-----------------:|
| HumanEval | 166.87 | 227.60 | 224.01 |
| GSM8K | 187.17 | 253.05 | 247.30 |
| Math500 | 194.50 | 264.82 | 259.49 |

## Server

Minimal chat server:

```bash
.venv/bin/python scripts/server_tools.py \
  --target /home/jake/models/Qwen3.6-27B-GGUF/Qwen3.6-27B-UD-Q5_K_XL.gguf \
  --draft /home/jake/models/Qwen3.6-27B-DFlash-safetensors/draft-q8_0.gguf \
  --bin /home/jake/lucebox-hub/dflash/build-luce-sm120/test_dflash \
  --budget 22 \
  --max-ctx 8192
```

Native Mega PFlash is opt-in:

```bash
.venv/bin/python scripts/server_tools.py \
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

When prefill compression is enabled, the server defaults `DFLASH_FP_USE_BSA=1`
and `DFLASH_FP_ALPHA=0.85`.

## Script Index

| Script | Purpose |
|--------|---------|
| `scripts/bench_llm.py` | Raw AR vs DFlash benchmark for HumanEval, GSM8K, and Math500. |
| `scripts/bench_daemon.py` | Streaming API benchmark over HTTP or HTTPS. |
| `scripts/bench_he.py` | Focused HumanEval-style DFlash acceptance harness. |
| `scripts/bench_prefill_backends.py` | End-to-end PFlash backend comparison. |
| `scripts/quantize_draft_q8.py` | Build the production Q8 DFlash draft GGUF. |
| `scripts/server_tools.py` | OpenAI-compatible server with tool and PFlash support. |
| `scripts/run.py` | One-shot streaming generation helper. |

## Runtime Invariants

- DFlash draft path must be `.gguf`.
- GGUF arch must be `qwen35-dflash-draft`.
- Required tensor names must use the local `dflash.*` naming convention.
- Required SWA metadata must be present.
- Short contexts only allocate/apply the draft SWA mask when context exceeds
  the draft sliding window.
