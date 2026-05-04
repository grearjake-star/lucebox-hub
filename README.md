# Lucebox Hub

Local inference systems for Lucebox experiments. The active path is DFlash:
Qwen3.6 27B GGUF target inference with a quantized DFlash GGUF draft, DDTree
speculative decoding, and optional PFlash prefill.

## Projects

| Path | Status | Purpose |
|------|--------|---------|
| `dflash/` | Active | Qwen3.5/3.6 27B target runtime, quantized DFlash draft, DDTree verify, OpenAI-compatible server, PFlash prefill hooks. |
| `megakernel/` | Support | Qwen3.5-0.8B megakernel and Mega PFlash components used by the DFlash server. |

## Current Default

The current 5090 configuration is:

- Target: `/home/jake/models/Qwen3.6-27B-GGUF/Qwen3.6-27B-UD-Q5_K_XL.gguf`
- DFlash draft: `/home/jake/models/Qwen3.6-27B-DFlash-safetensors/draft-q8_0.gguf`
- Runtime draft format: quantized GGUF only. Safetensors is not linked into the runtime.
- KV cache: FP16/FP16 for short benches; server can auto-enable TQ3 for larger contexts.
- FA window: `4096` for current short-context bench defaults.
- DDTree budget: `22`
- GPU DDTree top-K and rollback: default on.
- GPU DDTree verify prep: opt-in only.

## Build

```bash
git submodule update --init --recursive

cd dflash
cmake -B build-luce-sm120 -S . \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DDFLASH27B_USER_CUDA_ARCHITECTURES=120 \
  -DDFLASH27B_ENABLE_BSA=ON

cmake --build build-luce-sm120 \
  --target test_dflash test_generate smoke_load_draft smoke_draft_graph \
  -j 8
```

Use `build-luce-sm120` for RTX 5090 / Blackwell work. Use a separate build
directory for other architectures so CMake does not reuse stale CUDA settings.

## Smoke Test

```bash
cd dflash

./build-luce-sm120/smoke_load_draft \
  /home/jake/models/Qwen3.6-27B-DFlash-safetensors/draft-q8_0.gguf

./build-luce-sm120/smoke_draft_graph \
  /home/jake/models/Qwen3.6-27B-DFlash-safetensors/draft-q8_0.gguf 8
```

## Benchmark

```bash
cd dflash

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
  --n-sample 10
```

Current Q8 GGUF draft results on RTX 5090:

| Task | AR | DFlash | AL | Speedup |
|------|---:|-------:|---:|--------:|
| HumanEval | 58.01 | 273.11 | 7.63 | 4.71x |
| GSM8K | 58.78 | 228.02 | 6.35 | 3.88x |
| Math500 | 58.66 | 257.18 | 7.20 | 4.38x |

Full methodology and A/B notes are in `dflash/RESULTS.md`.

## API Latency Bench

Start the server:

```bash
cd dflash

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

Run the streaming API benchmark over HTTP or HTTPS:

```bash
.venv/bin/python scripts/bench_daemon.py \
  --url http://127.0.0.1:8012 \
  --benches HumanEval,GSM8K,Math500 \
  --n-gen 256 \
  --n-sample 10 \
  --warmup
```

The same command accepts an `https://...` URL for TLS, reverse-proxy, or remote
latency measurements.

## Model Files

Keep model weights outside git under `/home/jake/models`.

Required for the current DFlash path:

- `Qwen3.6-27B-UD-Q5_K_XL.gguf` target GGUF.
- `draft-q8_0.gguf` quantized DFlash draft GGUF.
- `Qwen3-0.6B-BF16.gguf` only if using daemon PFlash prefill.

Optional:

- Qwen3.5-0.8B safetensors snapshot only for native Mega PFlash experiments.
- Original DFlash safetensors only when rebuilding `draft-q8_0.gguf`; it is not
  needed by runtime and is not linked into the DFlash binary.

## Repository Notes

- `dflash/deps/llama.cpp` and `dflash/deps/Block-Sparse-Attention` are submodules.
- Generated build folders, virtualenvs, raw logs, and local launch scripts stay untracked.
- Push review work to the fork remote (`grearjake`); `origin` is fetch-only in this workspace.
