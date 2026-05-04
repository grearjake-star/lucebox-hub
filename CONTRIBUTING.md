# Contributing to Lucebox

Lucebox is a hub of self-contained local-inference optimization projects. Each
project owns its build, benchmark harness, and results. Keep changes scoped and
make performance claims reproducible.

## What We Accept

- Kernel, graph, cache, and scheduling changes that preserve correctness and
  improve `tok/s`, latency, memory footprint, or fit.
- Speculative decoding changes that improve DFlash acceptance length or reduce
  verify overhead on the same target/draft pair.
- Benchmark harness work that makes A/B runs easier to reproduce.
- Documentation that removes stale commands, records methodology, or explains
  non-obvious runtime constraints.

## What We Do Not Accept

- Closed-source runtime dependencies.
- Benchmark-only claims without exact hardware, model files, command lines, and
  before/after numbers.
- New runtime fallbacks that silently change model format, tensor naming, or
  metadata requirements. DFlash production runtime is quantized GGUF-only.

## DFlash Setup

Hardware:

- Recommended: RTX 5090 / Blackwell, CUDA 12.8+.
- Supported development path: NVIDIA sm_86+ with enough VRAM for the chosen
  target/draft/cache configuration.
- Native Mega PFlash uses a Qwen3.5-0.8B safetensors snapshot; this is separate
  from the DFlash draft runtime.

System dependencies:

```bash
sudo dflash/scripts/setup_system.sh
git submodule update --init --recursive
```

Manual minimums:

| Tool | Minimum |
|------|---------|
| GCC / G++ | 11 |
| CMake | 3.18 |
| Git | 2.x |
| git-lfs | any |
| CUDA Toolkit | 12.0+ |
| Python | 3.10+ |

Build the current 5090 target:

```bash
cmake -B dflash/build-luce-sm120 -S dflash \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DDFLASH27B_USER_CUDA_ARCHITECTURES=120 \
  -DDFLASH27B_ENABLE_BSA=ON

cmake --build dflash/build-luce-sm120 \
  --target test_dflash test_generate smoke_load_draft smoke_draft_graph \
  -j 8
```

If CMake was previously run with different CUDA arch settings, use a new build
directory. Do not reuse a stale CUDA build directory for performance numbers.

## Required Checks

For DFlash runtime changes:

```bash
cd dflash

./build-luce-sm120/smoke_load_draft \
  /home/jake/models/Qwen3.6-27B-DFlash-safetensors/draft-q8_0.gguf

./build-luce-sm120/smoke_draft_graph \
  /home/jake/models/Qwen3.6-27B-DFlash-safetensors/draft-q8_0.gguf 8

DFLASH_TARGET=/home/jake/models/Qwen3.6-27B-GGUF/Qwen3.6-27B-UD-Q5_K_XL.gguf \
DFLASH_DRAFT=/home/jake/models/Qwen3.6-27B-DFlash-safetensors/draft-q8_0.gguf \
DFLASH_BIN=/home/jake/lucebox-hub/dflash/build-luce-sm120/test_dflash \
DFLASH_BIN_AR=/home/jake/lucebox-hub/dflash/build-luce-sm120/test_generate \
.venv/bin/python scripts/bench_llm.py \
  --benches HumanEval,GSM8K,Math500 \
  --n-gen 256 \
  --budget 22 \
  --n-sample 10
```

For Python script changes:

```bash
cd dflash
.venv/bin/python -m py_compile \
  scripts/bench_llm.py \
  scripts/bench_daemon.py \
  scripts/bench_he.py \
  scripts/run.py \
  scripts/server.py \
  scripts/server_tools.py
```

## PR Expectations

1. State the exact model files, GPU, CUDA version, build directory, and command line.
2. Report before/after numbers on the same hardware and settings.
3. Keep one concern per PR. Docs, build config, runtime behavior, and benchmark
   methodology should be separable.
4. Include any residual risks, skipped tests, or known cases where a fallback was
   intentionally removed.

## Commit Messages

Use conventional commits:

```text
perf(dflash): harden q8 draft gguf loader
bench(dflash): add api latency harness for all3 datasets
docs(hub): refresh quant dflash build path
```

Allowed types: `feat`, `fix`, `refactor`, `perf`, `docs`, `test`, `bench`,
`chore`, `ci`.

## Hardware Access

If you need benchmark numbers but do not have matching hardware, open an issue
or PR with the exact command. We can run numbered RTX 5090 or RTX 3090 passes
when the branch is reviewable.

## Getting Help

- Discord: https://discord.gg/yHfswqZmJQ
- Issues: https://github.com/Luce-Org/lucebox-hub/issues

## License

By contributing you agree your work is MIT-licensed, same as the rest of the repo.
