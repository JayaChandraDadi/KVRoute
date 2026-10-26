"""End-to-end: router in front of simulated replicas, all in one event loop."""

import json

import aiohttp
import pytest
from aiohttp.test_utils import TestServer

from kvroute.mock_replica import MockEngine, MockServer
from kvroute.policies import make_policy
from kvroute.replay import replay
from kvroute.router import Router
from kvroute.tokenizer import WordTokenizer
from kvroute.traces import generate_chat


async def start_stack(n=3, policy="lmetric", time_scale=0.05, poll_interval=0.2):
    replicas = []
    for _ in range(n):
        srv = TestServer(MockServer(MockEngine(num_gpu_blocks=2048, time_scale=time_scale)).build_app())
        await srv.start_server()
        replicas.append(srv)
    router = Router([str(s.make_url("")) for s in replicas], make_policy(policy), WordTokenizer(),
                    poll_interval=poll_interval)
    rsrv = TestServer(router.build_app())
    await rsrv.start_server()
    return router, rsrv, replicas


async def stop_stack(rsrv, replicas):
    await rsrv.close()
    for s in replicas:
        await s.close()


def chat(system: str, user: str, max_tokens: int = 4, stream: bool = False):
    return {"model": "mock-model", "max_tokens": max_tokens, "stream": stream,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}


SYSTEM_A = " ".join(f"alpha{i}" for i in range(300))
SYSTEM_B = " ".join(f"beta{i}" for i in range(300))


@pytest.mark.asyncio
async def test_shared_prefix_follows_the_cache():
    router, rsrv, replicas = await start_stack()
    try:
        async with aiohttp.ClientSession() as http:
            url = str(rsrv.make_url("/v1/chat/completions"))
            async with http.post(url, json=chat(SYSTEM_A, "first question")) as r:
                assert r.status == 200
                first = r.headers["X-KVRoute-Replica"]
                assert int(r.headers["X-KVRoute-Matched-Tokens"]) == 0
                body = await r.json()
                assert body["usage"]["completion_tokens"] == 4
            async with http.post(url, json=chat(SYSTEM_A, "second question")) as r:
                assert r.headers["X-KVRoute-Replica"] == first
                assert int(r.headers["X-KVRoute-Matched-Tokens"]) >= 288  # 18 full blocks of the system prompt
                body = await r.json()
                assert body["usage"]["prompt_tokens_details"]["cached_tokens"] >= 288  # the replica really hit
            async with http.post(url, json=chat(SYSTEM_B, "unrelated")) as r:
                assert int(r.headers["X-KVRoute-Matched-Tokens"]) == 0
    finally:
        await stop_stack(rsrv, replicas)


@pytest.mark.asyncio
async def test_streaming_passthrough_and_metrics():
    router, rsrv, replicas = await start_stack(n=2)
    try:
        async with aiohttp.ClientSession() as http:
            payload = chat(SYSTEM_A, "stream please", max_tokens=5, stream=True)
            payload["stream_options"] = {"include_usage": True}
            async with http.post(str(rsrv.make_url("/v1/chat/completions")), json=payload) as r:
                assert r.headers["Content-Type"].startswith("text/event-stream")
                raw = (await r.read()).decode()
            events = [line[6:] for line in raw.split("\n\n") if line.startswith("data: ")]
            assert events[-1] == "[DONE]"
            deltas = [json.loads(e) for e in events[:-1]]
            assert sum(1 for d in deltas if d["choices"]) == 5
            assert deltas[-1]["usage"]["completion_tokens"] == 5

            async with http.get(str(rsrv.make_url("/metrics"))) as r:
                text = await r.text()
            assert 'kvroute_requests_total{policy="lmetric",replica=' in text
            assert "kvroute_ttft_seconds_bucket" in text

            async with http.get(str(rsrv.make_url("/router/stats"))) as r:
                stats = await r.json()
            assert stats["policy"] == "lmetric"
            assert sum(rep["routed"] for rep in stats["replicas"]) == 1
    finally:
        await stop_stack(rsrv, replicas)


@pytest.mark.asyncio
async def test_failover_when_replica_is_down():
    router, rsrv, replicas = await start_stack(n=2, policy="round_robin", poll_interval=0)
    await replicas[0].close()  # replica0 is gone
    try:
        async with aiohttp.ClientSession() as http:
            for _ in range(3):
                async with http.post(str(rsrv.make_url("/v1/chat/completions")), json=chat(SYSTEM_A, "q")) as r:
                    assert r.status == 200
                    assert r.headers["X-KVRoute-Replica"] == "replica1"
    finally:
        await stop_stack(rsrv, replicas[1:])


@pytest.mark.asyncio
async def test_policy_switch_and_reset():
    router, rsrv, replicas = await start_stack(n=2)
    try:
        async with aiohttp.ClientSession() as http:
            async with http.post(str(rsrv.make_url("/router/policy")), json={"policy": "least_load"}) as r:
                assert (await r.json())["policy"] == "least_load"
            async with http.post(str(rsrv.make_url("/router/policy")), json={"policy": "nope"}) as r:
                assert r.status == 400
            async with http.post(str(rsrv.make_url("/v1/chat/completions")), json=chat(SYSTEM_A, "q")) as r:
                assert r.status == 200
            assert sum(len(rep.prefix) for rep in router.replicas) > 0
            async with http.post(str(rsrv.make_url("/router/reset?replicas=1"))) as r:
                out = await r.json()
            assert out["replicas"] == {"replica0": 200, "replica1": 200}
            assert sum(len(rep.prefix) for rep in router.replicas) == 0
    finally:
        await stop_stack(rsrv, replicas)


@pytest.mark.asyncio
async def test_replay_reports_hits_for_cache_aware_policy():
    records = generate_chat(sessions=12, rate=50.0, turns=(3, 4), think_time=0.05, max_tokens=(4, 8), seed=3)
    router, rsrv, replicas = await start_stack(n=3, policy="lmetric", time_scale=0.02)
    try:
        results, summary = await replay(str(rsrv.make_url("")), records, speedup=1.0)
        assert summary["errors"] == 0 and summary["requests"] == len(records)
        assert summary["ttft_p50_s"] is not None and summary["output_tok_per_s"] > 0
        # Later turns resend earlier history, so a cache-aware router gets hits
        # on the replica side, and its own estimate agrees roughly.
        assert summary["replica_hit_rate"] > 0.3
        assert abs(summary["router_est_hit_rate"] - summary["replica_hit_rate"]) < 0.15
    finally:
        await stop_stack(rsrv, replicas)
