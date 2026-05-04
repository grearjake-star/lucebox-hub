"""
10 prompts per dataset, AR + DFlash per prompt.

    python3 scripts/bench_llm.py

Paths resolve from the repo root by default. Override with env vars:
    DFLASH_TARGET   path to target Qwen3.6-27B-Q4_K_M.gguf (or 3.5)
    DFLASH_DRAFT    path to quantized DFlash draft GGUF
    DFLASH_BIN      path to build/test_dflash
    DFLASH_BIN_AR   path to build/test_generate
"""
import json
import os
import re
import struct
import subprocess
import tempfile
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BIN_SUFFIX = ".exe" if os.name == "nt" else ""
TARGET = os.environ.get(
    "DFLASH_TARGET",
    str(ROOT / "models" / "Qwen3.6-27B-Q4_K_M.gguf"),
)
_LOCAL_DRAFT_Q8 = ROOT / "models" / "draft" / "draft-q8_0.gguf"
_LOCAL_DRAFT_ROOT = ROOT / "models" / "draft"
DRAFT = None
TEST_DFLASH = os.environ.get("DFLASH_BIN", str(ROOT / "build" / f"test_dflash{BIN_SUFFIX}"))
TEST_GENERATE = os.environ.get("DFLASH_BIN_AR", str(ROOT / "build" / f"test_generate{BIN_SUFFIX}"))
TMPDIR = Path(tempfile.gettempdir()) / "dflash_bench"
TMPDIR.mkdir(parents=True, exist_ok=True)

N_GEN = 256
BUDGET = 22
N_SAMPLE = 10

BENCHES = [
    ("HumanEval", "openai_humaneval", None, "test", lambda x: x["prompt"]),
    ("GSM8K", "gsm8k", "main", "test", lambda x: f"Question: {x['question']}\nAnswer: "),
    ("Math500", "HuggingFaceH4/MATH-500", None, "test", lambda x: f"Problem: {x['problem']}\nSolution: "),
]


def _find_draft_model(root: Path) -> str | None:
    if root.is_file():
        return str(root) if root.suffix == ".gguf" else None
    if not root.is_dir():
        return None
    preferred = root / "draft-q8_0.gguf"
    if preferred.is_file():
        return str(preferred)
    for gguf in root.rglob("*.gguf"):
        return str(gguf)
    return None


def _resolve_draft() -> str:
    env = os.environ.get("DFLASH_DRAFT")
    if env:
        found = _find_draft_model(Path(env))
        if found:
            return found
        raise FileNotFoundError(f"DFLASH_DRAFT does not point to a draft model: {env}")

    for candidate in (_LOCAL_DRAFT_Q8, _LOCAL_DRAFT_ROOT):
        found = _find_draft_model(candidate)
        if found:
            return found

    raise FileNotFoundError(
        "draft GGUF not found. Expected one of:\n"
        f"  - {_LOCAL_DRAFT_Q8}\n"
        "Build it as documented in the README, or set DFLASH_DRAFT to an explicit .gguf file or directory."
    )


def _require_file(path: str, label: str):
    if not Path(path).is_file():
        raise FileNotFoundError(f"{label} not found: {path}")


def _run_checked(cmd, timeout: int, label: str) -> subprocess.CompletedProcess:
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        tail = (r.stderr or r.stdout or "<no output>").strip()[-2000:]
        raise RuntimeError(f"{label} exited {r.returncode}: {tail}")
    return r


def tokenize(tok, p, path: Path):
    ids = tok.encode(p, add_special_tokens=False)
    with open(path, "wb") as f:
        for t in ids:
            f.write(struct.pack("<i", int(t)))
    return len(ids)


def run_ar(path: Path, n_gen: int):
    out_bin = TMPDIR / "ar_out.bin"
    r = _run_checked(
        [TEST_GENERATE, TARGET, str(path), str(n_gen), str(out_bin)],
        timeout=300,
        label="test_generate",
    )
    m = re.search(r"(\d+\.\d+)\s+tok/s", r.stdout)
    if not m:
        raise RuntimeError(f"test_generate output parse failed: {r.stdout[-1000:]}")
    return float(m.group(1))


def _auto_max_ctx(n_prompt, n_gen: int):
    # Auto-fit attention budget: prompt + gen + small verify pad, aligned to
    # FATTN_KQ_STRIDE=256. Oversizing max_ctx makes attention stride over
    # unused KV and can cost >20× prefill time (32K prompt + --kv-q4 +
    # max_ctx=131072 → 1035s vs 38s at max_ctx=32768). See scripts/run.py.
    pad = 64  # covers q_len=16 + ddtree budget up to 22 with margin
    return ((n_prompt + n_gen + pad + 255) // 256) * 256


def run_df(path: Path, n_prompt: int, n_gen: int, budget: int):
    max_ctx = _auto_max_ctx(n_prompt, n_gen)
    out_bin = TMPDIR / "df_out.bin"
    r = _run_checked(
        [
            TEST_DFLASH,
            TARGET,
            DRAFT,
            str(path),
            str(n_gen),
            str(out_bin),
            "--fast-rollback",
            "--ddtree",
            f"--ddtree-budget={budget}",
            f"--max-ctx={max_ctx}",
        ],
        timeout=300,
        label="test_dflash",
    )
    tps = re.search(r"(\d+(?:\.\d+)?)\s+tok/s", r.stdout)
    al = re.search(r"avg commit/step=(\d+(?:\.\d+)?)", r.stdout)
    if not (tps and al):
        raise RuntimeError(f"test_dflash output parse failed: {r.stdout[-1500:]}")
    return float(tps.group(1)), float(al.group(1))


def main():
    global DRAFT
    ap = argparse.ArgumentParser()
    ap.add_argument("--benches", default=",".join(name for name, *_ in BENCHES),
                    help="Comma-separated benchmark names to run, e.g. GSM8K,Math500")
    ap.add_argument("--n-gen", type=int, default=N_GEN)
    ap.add_argument("--budget", type=int, default=BUDGET)
    ap.add_argument("--n-sample", type=int, default=N_SAMPLE)
    ap.add_argument("--out-json", type=Path, default=TMPDIR / "bench_llm_results.json")
    args = ap.parse_args()

    DRAFT = _resolve_draft()
    _require_file(TARGET, "target GGUF")
    _require_file(TEST_DFLASH, "test_dflash binary")
    _require_file(TEST_GENERATE, "test_generate binary")
    selected = {name.strip().lower() for name in args.benches.split(",") if name.strip()}
    benches = [b for b in BENCHES if b[0].lower() in selected]
    if not benches:
        known = ", ".join(name for name, *_ in BENCHES)
        raise SystemExit(f"no matching benches in {args.benches!r}; known: {known}")

    print(f"[bench] target = {TARGET}", flush=True)
    print(f"[bench] draft  = {DRAFT}", flush=True)
    print(f"[bench] ar bin = {TEST_GENERATE}", flush=True)
    print(f"[bench] df bin = {TEST_DFLASH}", flush=True)
    print(f"[bench] n_gen={args.n_gen}  budget={args.budget}  n_sample={args.n_sample}", flush=True)

    from datasets import load_dataset
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-27B", trust_remote_code=True)

    results = {}
    for name, ds_name, cfg, split, extract in benches:
        print(f"\n[bench] ==== {name} (n={args.n_sample}) ====", flush=True)
        ds = load_dataset(ds_name, cfg, split=split)
        ds = ds.shuffle(seed=42).select(range(args.n_sample))
        ar_tps, df_tps, df_al = [], [], []
        for i, s in enumerate(ds):
            p = extract(s)
            path = TMPDIR / f"b_{name}_{i:02d}.bin"
            n = tokenize(tok, p, path)
            if n == 0 or n > 3500:
                continue
            try:
                ar = run_ar(path, args.n_gen)
                df, al = run_df(path, n, args.n_gen, args.budget)
            except Exception as e:
                print(f"  [{i+1:02d}/{args.n_sample}] n_tok={n:4d}  FAILED: {e}", flush=True)
                continue
            if ar > 0:
                ar_tps.append(ar)
            if df > 0:
                df_tps.append(df)
                df_al.append(al)
            print(f"  [{i+1:02d}/{args.n_sample}] n_tok={n:4d}  AR={ar:6.2f}  DFlash={df:7.2f}  AL={al:5.2f}", flush=True)
        ar_m = sum(ar_tps) / len(ar_tps) if ar_tps else 0
        df_m = sum(df_tps) / len(df_tps) if df_tps else 0
        al_m = sum(df_al) / len(df_al) if df_al else 0
        results[name] = {"ar": ar_m, "dflash": df_m, "al": al_m,
                         "speedup": df_m / ar_m if ar_m else 0}
        print(f"  {name} mean: AR={ar_m:.2f}  DFlash={df_m:.2f}  AL={al_m:.2f}  {results[name]['speedup']:.2f}x", flush=True)

    print("\n[bench] === SUMMARY ===")
    print(f"{'Task':12s}  {'AR':>8s}  {'DFlash':>8s}  {'AL':>6s}  {'Speedup':>8s}")
    for name, r in results.items():
        print(f"{name:12s}  {r['ar']:8.2f}  {r['dflash']:8.2f}  {r['al']:6.2f}  {r['speedup']:7.2f}x")

    out_json = args.out_json
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[bench] wrote {out_json}", flush=True)


if __name__ == "__main__":
    main()
