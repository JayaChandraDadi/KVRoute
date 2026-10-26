"""Compare routing policies on the same trace.

For each policy: reset the router's index and every replica's prefix cache,
switch the policy, replay the trace, record the summary. Then print a table
and write CSV / JSON (and a PNG chart if matplotlib is installed).

Against real vLLM replicas behind a running router:
    python -m kvroute.bench --router http://localhost:8000 --kind chat \
        --model Qwen/Qwen2.5-0.5B-Instruct

Fully local, with simulated replicas (no GPU):
    python -m kvroute.bench --mock 4 --kind chat --scale 0.3
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import aiohttp

from .replay import replay, write_results
from .traces import generate, load, save

DEFAULT_POLICIES = ["round_robin", "least_load", "prefix", "linear", "lmetric"]
COLUMNS = [
    ("policy", "policy", "{}"),
    ("ttft_p50_s", "TTFT p50 (s)", "{:.3f}"),
    ("ttft_p99_s", "TTFT p99 (s)", "{:.3f}"),
    ("e2e_p99_s", "E2E p99 (s)", "{:.2f}"),
    ("tpot_p50_s", "TPOT p50 (ms)", "ms"),
    ("output_tok_per_s", "out tok/s", "{:.0f}"),
    ("replica_hit_rate", "replica hit rate", "pct"),
    ("router_est_hit_rate", "router est. hit", "pct"),
    ("load_imbalance", "max/mean load", "{:.2f}"),
    ("errors", "errors", "{}"),
]


def _fmt(value, fmt: str) -> str:
    if value is None:
        return "-"
    if fmt == "pct":
        return f"{100 * value:.1f}%"
    if fmt == "ms":
        return f"{1000 * value:.1f}"
    return fmt.format(value)


def markdown_table(summaries: list[dict]) -> str:
    head = "| " + " | ".join(c[1] for c in COLUMNS) + " |"
    sep = "|" + "|".join("---" for _ in COLUMNS) + "|"
    rows = ["| " + " | ".join(_fmt(s.get(k), f) for k, _, f in COLUMNS) + " |" for s in summaries]
    return "\n".join([head, sep, *rows])


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def _wait_http(url: str, timeout: float = 30.0) -> None:
    deadline = time.time() + timeout
    async with aiohttp.ClientSession() as http:
        while time.time() < deadline:
            try:
                async with http.get(url) as resp:
                    if resp.status < 500:
                        return
            except aiohttp.ClientError:
                pass
            await asyncio.sleep(0.2)
    raise TimeoutError(f"{url} did not come up")


def start_mock_stack(n: int, mock_args: list[str]) -> tuple[str, list[subprocess.Popen]]:
    procs = []
    urls = []
    for _ in range(n):
        port = _free_port()
        urls.append(f"http://127.0.0.1:{port}")
        procs.append(subprocess.Popen([sys.executable, "-m", "kvroute.mock_replica", "--port", str(port), *mock_args]))
    rport = _free_port()
    procs.append(
        subprocess.Popen(
            [sys.executable, "-m", "kvroute.router", "--replicas", ",".join(urls), "--tokenizer", "word",
             "--host", "127.0.0.1", "--port", str(rport), "--poll-interval", "0.5"],
            stderr=subprocess.DEVNULL,
        )
    )
    router = f"http://127.0.0.1:{rport}"

    async def wait_all():
        for u in urls:
            await _wait_http(u + "/health")
        await _wait_http(router + "/health")

    asyncio.run(wait_all())
    time.sleep(1.5)  # let the router scrape the replicas once
    return router, procs


async def _configure(router: str, policy: str, params: dict) -> None:
    async with aiohttp.ClientSession() as http:
        async with http.post(router + "/router/reset?replicas=1") as resp:
            resp.raise_for_status()
        async with http.post(router + "/router/policy", json={"policy": policy, "params": params}) as resp:
            resp.raise_for_status()
    await asyncio.sleep(1.0)


def plot(summaries: list[dict], path: Path) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    names = [s["policy"] for s in summaries]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    axes[0].bar(names, [s["ttft_p50_s"] or 0 for s in summaries], label="p50")
    axes[0].bar(names, [(s["ttft_p99_s"] or 0) - (s["ttft_p50_s"] or 0) for s in summaries],
                bottom=[s["ttft_p50_s"] or 0 for s in summaries], alpha=0.5, label="p99")
    axes[0].set_title("Time to first token (s)")
    axes[0].legend()
    axes[1].bar(names, [100 * (s["replica_hit_rate"] or 0) for s in summaries])
    axes[1].set_title("Replica prefix-cache hit rate (%)")
    axes[2].bar(names, [s["output_tok_per_s"] for s in summaries])
    axes[2].set_title("Output tokens / s")
    for ax in axes:
        ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--router", help="URL of a running router")
    src.add_argument("--mock", type=int, metavar="N", help="start N simulated replicas + a router locally")
    ap.add_argument("--trace", help="JSONL trace (otherwise generated from --kind)")
    ap.add_argument("--kind", choices=["chat", "agent", "mixed"], default="chat")
    ap.add_argument("--scale", type=float, default=1.0, help="session count multiplier for generated traces")
    ap.add_argument("--rate-scale", type=float, default=1.0, help="arrival-rate multiplier for generated traces")
    ap.add_argument("--policies", default=",".join(DEFAULT_POLICIES))
    ap.add_argument("--model", default="mock-model")
    ap.add_argument("--speedup", type=float, default=1.0)
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--mock-args", default="", help="extra args for each mock replica, e.g. '--time-scale 0.5'")
    args = ap.parse_args()

    if args.trace:
        records = load(args.trace)
        trace_name = Path(args.trace).stem
    else:
        records = generate(args.kind, scale=args.scale, rate_scale=args.rate_scale)
        trace_name = f"{args.kind}_x{args.scale:g}"
        save(records, Path(args.out_dir) / "traces" / f"{trace_name}.jsonl")

    procs: list[subprocess.Popen] = []
    router = args.router
    if args.mock:
        router, procs = start_mock_stack(args.mock, args.mock_args.split())
        trace_name = f"mock{args.mock}_{trace_name}"

    out_dir = Path(args.out_dir) / trace_name
    summaries = []
    try:
        for policy in [p.strip() for p in args.policies.split(",") if p.strip()]:
            asyncio.run(_configure(router, policy, {}))
            print(f"== {policy}: {len(records)} requests", flush=True)
            results, summary = asyncio.run(replay(router, records, args.model, args.speedup))
            write_results(results, summary, out_dir, policy)
            summaries.append(summary)
            print(markdown_table([summary]).splitlines()[-1], flush=True)
    finally:
        for p in procs:
            p.terminate()
        for p in procs:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()

    table = markdown_table(summaries)
    print("\n" + table)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "comparison.md").write_text(
        f"Trace: {trace_name} ({len(records)} requests), "
        f"{'simulated replicas' if args.mock else 'router ' + router}\n\n{table}\n"
    )
    with (out_dir / "comparison.csv").open("w", newline="") as f:
        keys = [k for k in summaries[0] if k != "requests_per_replica"]
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(summaries)
    (out_dir / "comparison.json").write_text(json.dumps(summaries, indent=2))
    if plot(summaries, out_dir / "comparison.png"):
        print(f"chart: {out_dir / 'comparison.png'}")
    print(f"results: {out_dir}{os.sep}")


if __name__ == "__main__":
    main()
