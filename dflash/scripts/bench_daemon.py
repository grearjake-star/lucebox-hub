"""Daemon/API bench. Hits /v1/chat/completions and reports wall/decode tok/s.

Streams the response and reports two numbers per prompt:

  * wall    — total HTTP time (tokenize + prefill + decode + HTTP / JSON)
  * decode  — first-token → last-token elapsed, matching bench_he.py's
              tok/s (excludes prefill + setup)

Compare `decode` against bench_he.py to verify the C++ decode path is as
fast under the daemon as under a one-shot test_dflash invocation.

Start the server first (same config the published numbers use):
    DFLASH27B_KV_TQ3=1 python3 scripts/server_tools.py \\
        --budget 22 --max-ctx 16384 --port 8000

Then:
    python3 scripts/bench_daemon.py --url http://localhost:8000 --n-gen 256
    python3 scripts/bench_daemon.py --url https://example.com --benches GSM8K,Math500
"""
import argparse
import json
import time
import urllib.request
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_llm import BENCHES


def run(url: str, prompt: str, n_gen: int) -> tuple[int, float, float]:
    """POST to /v1/chat/completions with stream=true. Return (n_tok, wall_secs,
    decode_secs) where decode_secs starts at the first streamed token (after
    prefill) and ends at the last token."""
    body = json.dumps({
        "model": "luce-dflash",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": n_gen,
        "stream": True,
    }).encode()
    req = urllib.request.Request(
        url + "/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json",
                 "Accept": "text/event-stream"},
    )
    t0 = time.perf_counter()
    t_first = 0.0
    t_last = 0.0
    n_tok = 0
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            line = raw.decode("utf-8", errors="replace").rstrip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            choices = chunk.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            # Count tokens by content / reasoning deltas. Tool-call deltas
            # aren't counted — they arrive as a single final chunk.
            if delta.get("content") or delta.get("reasoning_content"):
                if n_tok == 0:
                    t_first = time.perf_counter()
                n_tok += 1
                t_last = time.perf_counter()
    wall = time.perf_counter() - t0
    decode = (t_last - t_first) if n_tok > 1 else 0.0
    return n_tok, wall, decode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000",
                    help="Base URL of the running server (no /v1 suffix)")
    ap.add_argument("--n-gen", type=int, default=256)
    ap.add_argument("--benches", default=",".join(name for name, *_ in BENCHES),
                    help="Comma-separated benchmark names to run, e.g. HumanEval,GSM8K,Math500")
    ap.add_argument("--n-sample", type=int, default=10)
    ap.add_argument("--out-json", type=Path, default=Path("/tmp/dflash_bench/bench_daemon_results.json"))
    ap.add_argument("--warmup", action="store_true",
                    help="Run the first prompt once before timing to discard "
                         "cold-start effects (model is already resident, but "
                         "the first request allocates the decode VMM chunks).")
    args = ap.parse_args()

    selected = {name.strip().lower() for name in args.benches.split(",") if name.strip()}
    benches = [b for b in BENCHES if b[0].lower() in selected]
    if not benches:
        known = ", ".join(name for name, *_ in BENCHES)
        raise SystemExit(f"no matching benches in {args.benches!r}; known: {known}")

    from datasets import load_dataset
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-27B", trust_remote_code=True)

    if args.warmup:
        print("[bench] warmup...", flush=True)
        name, ds_name, cfg, split, extract = benches[0]
        sample = load_dataset(ds_name, cfg, split=split).shuffle(seed=42)[0]
        run(args.url, extract(sample), args.n_gen)

    print(f"[bench] daemon API  n_gen={args.n_gen}  n_sample={args.n_sample}  url={args.url}", flush=True)
    results = {}
    for bench_name, ds_name, cfg, split, extract in benches:
        print(f"\n[bench] ==== {bench_name} (n={args.n_sample}) ====", flush=True)
        print(f"{'idx':>4s} {'in_tok':>6s} {'out':>5s} {'wall_s':>7s} {'dec_s':>7s} "
              f"{'wall_tps':>9s} {'dec_tps':>9s}")
        print("-" * 64)
        ds = load_dataset(ds_name, cfg, split=split).shuffle(seed=42).select(range(args.n_sample))
        wall_tps_list: list[float] = []
        dec_tps_list: list[float] = []
        total_tok = 0
        total_decode = 0.0
        for i, sample in enumerate(ds):
            text = extract(sample)
            in_tok = len(tok.encode(text, add_special_tokens=False))
            if in_tok == 0 or in_tok > 3500:
                continue
            try:
                n_tok, wall, decode = run(args.url, text, args.n_gen)
            except Exception as e:
                print(f"  {i+1:02d} {in_tok:6d}  FAILED: {e}", flush=True)
                continue
            if n_tok == 0:
                print(f"  {i+1:02d} {in_tok:6d} {n_tok:5d} {wall:7.2f}    --         --        --", flush=True)
                continue
            wall_tps = n_tok / wall
            dec_tps = (n_tok - 1) / decode if decode > 0 else 0.0
            wall_tps_list.append(wall_tps)
            if dec_tps > 0:
                dec_tps_list.append(dec_tps)
                total_decode += decode
            total_tok += n_tok
            print(f"  {i+1:02d} {in_tok:6d} {n_tok:5d} {wall:7.2f} {decode:7.2f} "
                  f"{wall_tps:9.2f} {dec_tps:9.2f}", flush=True)

        wall_mean = sum(wall_tps_list) / len(wall_tps_list) if wall_tps_list else 0.0
        dec_mean = sum(dec_tps_list) / len(dec_tps_list) if dec_tps_list else 0.0
        dec_agg = ((total_tok - len(dec_tps_list)) / total_decode) if total_decode > 0 else 0.0
        results[bench_name] = {
            "wall_tps": wall_mean,
            "decode_tps": dec_mean,
            "decode_tps_aggregate": dec_agg,
        }
        print(f"  {bench_name} mean: wall={wall_mean:.2f}  decode={dec_mean:.2f}  aggregate_decode={dec_agg:.2f}", flush=True)

    print("\n[bench] === SUMMARY ===")
    print(f"{'Task':12s}  {'Wall':>8s}  {'Decode':>8s}  {'AggDec':>8s}")
    for name, r in results.items():
        print(f"{name:12s}  {r['wall_tps']:8.2f}  {r['decode_tps']:8.2f}  {r['decode_tps_aggregate']:8.2f}")

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[bench] wrote {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
