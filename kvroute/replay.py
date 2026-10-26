"""Replay a trace against the router and measure what users would see.

Each session runs as its own task: turn 0 starts at its trace time, and each
later turn starts after the previous response finishes plus its think time.
Requests stream (``stream_options.include_usage``) so time to first token is
measured at the client.

Reported per run: TTFT p50/p90/p99, end-to-end p50/p99, time per output
token, output tokens/s, requests/s, the router's estimated prefix hit rate,
the replicas' *own* prefix hit rate over the run (from their /metrics), and
how evenly requests spread across replicas.

    python -m kvroute.replay --router http://localhost:8000 --trace traces/chat.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import aiohttp

from .metrics import parse_vllm_metrics
from .traces import load


@dataclass
class Result:
    session: str
    turn: int
    ok: bool
    replica: str | None
    prompt_tokens: int
    matched_tokens: int
    output_tokens: int
    ttft_s: float | None
    e2e_s: float | None
    error: str | None = None


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    vals = sorted(values)
    k = (len(vals) - 1) * q / 100.0
    lo, hi = int(k), min(int(k) + 1, len(vals) - 1)
    return vals[lo] + (vals[hi] - vals[lo]) * (k - lo)


async def _send(session: aiohttp.ClientSession, router: str, rec: dict, model: str) -> Result:
    payload = {
        "model": model,
        "messages": rec["messages"],
        "max_tokens": rec["max_tokens"],
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "ignore_eos": True,  # vLLM extension: always generate max_tokens
    }
    t0 = time.perf_counter()
    ttft = None
    chunks = 0
    usage_tokens = None
    try:
        async with session.post(router + "/v1/chat/completions", json=payload) as resp:
            if resp.status != 200:
                text = await resp.text()
                return Result(rec["session"], rec["turn"], False, resp.headers.get("X-KVRoute-Replica"), 0, 0, 0,
                              None, None, f"HTTP {resp.status}: {text[:200]}")
            replica = resp.headers.get("X-KVRoute-Replica")
            prompt_tokens = int(resp.headers.get("X-KVRoute-Prompt-Tokens", 0))
            matched = int(resp.headers.get("X-KVRoute-Matched-Tokens", 0))
            buf = b""
            async for raw in resp.content.iter_any():
                buf += raw
                while b"\n\n" in buf:
                    event, buf = buf.split(b"\n\n", 1)
                    line = event.strip()
                    if not line.startswith(b"data:"):
                        continue
                    data = line[5:].strip()
                    if data == b"[DONE]":
                        continue
                    msg = json.loads(data)
                    if msg.get("usage"):
                        usage_tokens = msg["usage"].get("completion_tokens")
                    for ch in msg.get("choices", []):
                        if (ch.get("delta") or {}).get("content"):
                            if ttft is None:
                                ttft = time.perf_counter() - t0
                            chunks += 1
        e2e = time.perf_counter() - t0
        return Result(rec["session"], rec["turn"], True, replica, prompt_tokens, matched,
                      usage_tokens if usage_tokens is not None else chunks, ttft, e2e)
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        return Result(rec["session"], rec["turn"], False, None, 0, 0, 0, None, None, repr(e))


async def _scrape(session: aiohttp.ClientSession, urls: list[str]) -> dict[str, tuple[float, float]]:
    out = {}
    for url in urls:
        try:
            async with session.get(url + "/metrics", timeout=aiohttp.ClientTimeout(total=5)) as resp:
                m = parse_vllm_metrics(await resp.text())
            out[url] = (m.prefix_queries or 0.0, m.prefix_hits or 0.0)
        except Exception:  # noqa: BLE001
            pass
    return out


async def replay(router: str, records: list[dict], model: str = "mock-model", speedup: float = 1.0,
                 timeout_s: float = 900.0) -> tuple[list[Result], dict]:
    router = router.rstrip("/")
    sessions: dict[str, list[dict]] = {}
    for r in records:
        sessions.setdefault(r["session"], []).append(r)
    for turns in sessions.values():
        turns.sort(key=lambda r: r["turn"])

    results: list[Result] = []
    conn = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(connector=conn, timeout=aiohttp.ClientTimeout(total=timeout_s)) as http:
        async with http.get(router + "/router/stats") as resp:
            stats = await resp.json()
        replica_urls = [r["url"] for r in stats["replicas"]]
        before = await _scrape(http, replica_urls)

        t_start = time.perf_counter()

        async def run_session(turns: list[dict]) -> None:
            await asyncio.sleep(max(0.0, turns[0]["start"] / speedup - (time.perf_counter() - t_start)))
            for i, rec in enumerate(turns):
                if i > 0:
                    await asyncio.sleep(rec["think_time"] / speedup)
                results.append(await _send(http, router, rec, model))

        await asyncio.gather(*(run_session(t) for t in sessions.values()))
        duration = time.perf_counter() - t_start
        after = await _scrape(http, replica_urls)

    dq = sum(after[u][0] - before.get(u, (0, 0))[0] for u in after)
    dh = sum(after[u][1] - before.get(u, (0, 0))[1] for u in after)
    return results, summarize(results, duration, replica_hit_rate=(dh / dq) if dq else None, policy=stats["policy"])


def summarize(results: list[Result], duration: float, replica_hit_rate: float | None = None,
              policy: str | None = None) -> dict:
    ok = [r for r in results if r.ok]
    ttft = [r.ttft_s for r in ok if r.ttft_s is not None]
    e2e = [r.e2e_s for r in ok if r.e2e_s is not None]
    tpot = [(r.e2e_s - r.ttft_s) / (r.output_tokens - 1) for r in ok
            if r.ttft_s is not None and r.output_tokens > 1]
    prompt = sum(r.prompt_tokens for r in ok)
    matched = sum(r.matched_tokens for r in ok)
    per_replica = Counter(r.replica for r in ok)
    loads = list(per_replica.values())
    return {
        "policy": policy,
        "requests": len(results),
        "errors": len(results) - len(ok),
        "duration_s": duration,
        "req_per_s": len(ok) / duration if duration else 0.0,
        "output_tok_per_s": sum(r.output_tokens for r in ok) / duration if duration else 0.0,
        "ttft_p50_s": percentile(ttft, 50),
        "ttft_p90_s": percentile(ttft, 90),
        "ttft_p99_s": percentile(ttft, 99),
        "ttft_mean_s": statistics.fmean(ttft) if ttft else None,
        "e2e_p50_s": percentile(e2e, 50),
        "e2e_p99_s": percentile(e2e, 99),
        "tpot_p50_s": percentile(tpot, 50),
        "router_est_hit_rate": matched / prompt if prompt else None,
        "replica_hit_rate": replica_hit_rate,
        "requests_per_replica": dict(sorted(per_replica.items())),
        "load_imbalance": (max(loads) / (sum(loads) / len(loads))) if loads else None,
    }


def write_results(results: list[Result], summary: dict, out_dir: str | Path, tag: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / f"{tag}_requests.jsonl").open("w") as f:
        for r in results:
            f.write(json.dumps(asdict(r)) + "\n")
    (out / f"{tag}_summary.json").write_text(json.dumps(summary, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--router", default="http://localhost:8000")
    ap.add_argument("--trace", required=True)
    ap.add_argument("--model", default="mock-model", help="model name sent in requests")
    ap.add_argument("--speedup", type=float, default=1.0, help="divide arrival and think times by this")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    results, summary = asyncio.run(replay(args.router, load(args.trace), args.model, args.speedup))
    tag = args.tag or f"{Path(args.trace).stem}_{summary['policy']}"
    write_results(results, summary, args.out_dir, tag)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
