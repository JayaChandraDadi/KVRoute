"""KVRoute: cache-aware request router in front of vLLM replicas.

Speaks the OpenAI API (/v1/completions, /v1/chat/completions, streaming or
not). For every request it tokenizes the prompt, hashes its full KV blocks,
asks the policy which replica to use given each replica's estimated cached
prefix and in-flight load, and streams the replica's response back.

    python -m kvroute.router --replicas http://gpu0:8000,http://gpu1:8000 \
        --policy lmetric --tokenizer hf:Qwen/Qwen2.5-0.5B-Instruct
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from dataclasses import dataclass, field

import aiohttp
from aiohttp import web
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from .metrics import RouterMetrics, VllmMetrics, parse_vllm_metrics
from .policies import Candidate, Policy, make_policy
from .prefix_index import PrefixIndex, block_hashes
from .tokenizer import Tokenizer, load_tokenizer

log = logging.getLogger("kvroute")

ROUTER_KEY = web.AppKey("router", object)
DEFAULT_CAPACITY_BLOCKS = 16_384
FAILURES_BEFORE_DOWN = 3


@dataclass
class ReplicaState:
    index: int
    name: str
    url: str
    prefix: PrefixIndex
    in_flight: int = 0
    healthy: bool = True
    failures: int = 0
    routed: int = 0
    prompt_tokens: int = 0
    matched_tokens: int = 0
    scraped: VllmMetrics = field(default_factory=VllmMetrics)

    def stats(self) -> dict:
        s = self.scraped
        hit_rate = (s.prefix_hits / s.prefix_queries) if s.prefix_queries else None
        return {
            "name": self.name,
            "url": self.url,
            "healthy": self.healthy,
            "in_flight": self.in_flight,
            "routed": self.routed,
            "prompt_tokens": self.prompt_tokens,
            "matched_tokens": self.matched_tokens,
            "router_hit_rate": self.matched_tokens / self.prompt_tokens if self.prompt_tokens else None,
            "index_blocks": len(self.prefix),
            "index_capacity_blocks": self.prefix.capacity_blocks,
            "replica_running": s.running,
            "replica_waiting": s.waiting,
            "replica_kv_cache_usage": s.kv_usage,
            "replica_prefix_hit_rate": hit_rate,
        }


class Router:
    def __init__(
        self,
        replica_urls: list[str],
        policy: Policy,
        tokenizer: Tokenizer,
        block_size: int = 16,
        capacity_blocks: int | None = None,
        poll_interval: float = 1.0,
        request_timeout: float = 600.0,
        metrics: RouterMetrics | None = None,
    ) -> None:
        if not replica_urls:
            raise ValueError("at least one replica URL is required")
        self.policy = policy
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.fixed_capacity = capacity_blocks
        self.poll_interval = poll_interval
        self.request_timeout = request_timeout
        self.metrics = metrics or RouterMetrics()
        self.replicas = [
            ReplicaState(i, f"replica{i}", url.rstrip("/"), PrefixIndex(capacity_blocks or DEFAULT_CAPACITY_BLOCKS))
            for i, url in enumerate(replica_urls)
        ]
        self._session: aiohttp.ClientSession | None = None
        self._poller: asyncio.Task | None = None

    # ------------------------------------------------------------------ app
    def build_app(self) -> web.Application:
        app = web.Application(client_max_size=64 * 1024 * 1024)
        app[ROUTER_KEY] = self
        app.router.add_post("/v1/completions", self._completions)
        app.router.add_post("/v1/chat/completions", self._chat)
        app.router.add_get("/v1/models", self._models)
        app.router.add_get("/metrics", self._metrics)
        app.router.add_get("/health", self._health)
        app.router.add_get("/router/stats", self._stats)
        app.router.add_get("/router/policy", self._get_policy)
        app.router.add_post("/router/policy", self._set_policy)
        app.router.add_post("/router/reset", self._reset)
        app.on_startup.append(self._on_startup)
        app.on_cleanup.append(self._on_cleanup)
        return app

    async def _on_startup(self, app: web.Application) -> None:
        timeout = aiohttp.ClientTimeout(total=self.request_timeout, sock_connect=5)
        self._session = aiohttp.ClientSession(timeout=timeout, connector=aiohttp.TCPConnector(limit=0))
        if self.poll_interval > 0:
            self._poller = asyncio.create_task(self._poll_loop())

    async def _on_cleanup(self, app: web.Application) -> None:
        if self._poller:
            self._poller.cancel()
            try:
                await self._poller
            except asyncio.CancelledError:
                pass
        if self._session:
            await self._session.close()

    # -------------------------------------------------------------- routing
    def _encode(self, kind: str, payload: dict) -> list[int]:
        if kind == "chat":
            return self.tokenizer.encode_chat(payload.get("messages") or [])
        prompt = payload.get("prompt", "")
        if isinstance(prompt, list) and prompt and isinstance(prompt[0], (str, list)):
            prompt = prompt[0]  # batched prompts: route on the first one
        return self.tokenizer.encode_prompt(prompt)

    def _candidates(self, hashes: list[int], exclude: set[int]) -> list[Candidate]:
        healthy = [r for r in self.replicas if r.healthy and r.index not in exclude]
        if not healthy:  # all marked down: try anything not yet tried
            healthy = [r for r in self.replicas if r.index not in exclude]
        return [
            Candidate(r.index, r.name, r.in_flight, r.prefix.match(hashes) * self.block_size) for r in healthy
        ]

    async def _completions(self, request: web.Request) -> web.StreamResponse:
        return await self._proxy(request, "/v1/completions", "completion")

    async def _chat(self, request: web.Request) -> web.StreamResponse:
        return await self._proxy(request, "/v1/chat/completions", "chat")

    async def _proxy(self, request: web.Request, path: str, kind: str) -> web.StreamResponse:
        body = await request.read()
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            return web.json_response({"error": "request body must be JSON"}, status=400)

        t_start = time.perf_counter()
        tokens = self._encode(kind, payload)
        hashes = block_hashes(tokens, self.block_size)
        tried: set[int] = set()
        headers = {"Content-Type": "application/json"}
        if "Authorization" in request.headers:
            headers["Authorization"] = request.headers["Authorization"]

        while True:
            candidates = self._candidates(hashes, tried)
            if not candidates:
                return web.json_response({"error": "no replica available"}, status=503)
            decision = self.policy.choose(len(tokens), candidates)
            replica = self.replicas[decision.index]
            matched = next(c.matched_tokens for c in candidates if c.index == decision.index)
            self.metrics.decision_seconds.observe(time.perf_counter() - t_start)

            # Optimistic: later requests should see this prefix as cached here.
            replica.prefix.insert(hashes)
            replica.in_flight += 1
            self.metrics.in_flight.labels(replica.name).set(replica.in_flight)
            prepared = False
            try:
                async with self._session.post(replica.url + path, data=body, headers=headers) as upstream:
                    replica.failures = 0
                    replica.routed += 1
                    replica.prompt_tokens += len(tokens)
                    replica.matched_tokens += matched
                    self.metrics.requests.labels(replica.name, self.policy.name).inc()
                    self.metrics.prompt_tokens.labels(replica.name).inc(len(tokens))
                    self.metrics.matched_tokens.labels(replica.name).inc(matched)
                    self.metrics.index_blocks.labels(replica.name).set(len(replica.prefix))

                    resp = web.StreamResponse(
                        status=upstream.status,
                        headers={
                            "Content-Type": upstream.headers.get("Content-Type", "application/json"),
                            "X-KVRoute-Replica": replica.name,
                            "X-KVRoute-Prompt-Tokens": str(len(tokens)),
                            "X-KVRoute-Matched-Tokens": str(matched),
                        },
                    )
                    await resp.prepare(request)
                    prepared = True
                    first = True
                    async for chunk in upstream.content.iter_any():
                        if first:
                            self.metrics.ttft_seconds.labels(replica.name).observe(time.perf_counter() - t_start)
                            first = False
                        await resp.write(chunk)
                    await resp.write_eof()
                    self.metrics.e2e_seconds.labels(replica.name).observe(time.perf_counter() - t_start)
                    return resp
            except (aiohttp.ClientConnectionError, asyncio.TimeoutError) as e:
                self.metrics.upstream_errors.labels(replica.name).inc()
                if prepared:  # already streaming to the client: cannot retry
                    log.warning("upstream %s failed mid-stream: %s", replica.name, e)
                    raise
                replica.failures += 1
                if replica.failures >= FAILURES_BEFORE_DOWN:
                    replica.healthy = False
                log.warning("upstream %s unreachable (%s); retrying elsewhere", replica.name, e)
                tried.add(replica.index)
            finally:
                replica.in_flight -= 1
                self.metrics.in_flight.labels(replica.name).set(replica.in_flight)

    # ---------------------------------------------------------- monitoring
    async def _poll_loop(self) -> None:
        while True:
            await asyncio.gather(*(self._poll_one(r) for r in self.replicas))
            await asyncio.sleep(self.poll_interval)

    async def _poll_one(self, r: ReplicaState) -> None:
        try:
            async with self._session.get(r.url + "/metrics", timeout=aiohttp.ClientTimeout(total=2)) as resp:
                text = await resp.text()
            scraped = parse_vllm_metrics(text)
        except Exception:  # noqa: BLE001 - any failure counts as unreachable
            r.failures += 1
            if r.failures >= FAILURES_BEFORE_DOWN:
                r.healthy = False
            self.metrics.replica_up.labels(r.name).set(0)
            return
        r.failures = 0
        r.healthy = True
        r.scraped = scraped
        if self.fixed_capacity is None and scraped.num_gpu_blocks:
            r.prefix.resize(scraped.num_gpu_blocks)
        m = self.metrics
        m.replica_up.labels(r.name).set(1)
        if scraped.running is not None:
            m.replica_running.labels(r.name).set(scraped.running)
        if scraped.waiting is not None:
            m.replica_waiting.labels(r.name).set(scraped.waiting)
        if scraped.kv_usage is not None:
            m.replica_kv_usage.labels(r.name).set(scraped.kv_usage)
        if scraped.prefix_queries:
            m.replica_hit_rate.labels(r.name).set(scraped.prefix_hits / scraped.prefix_queries)

    # ------------------------------------------------------------ admin
    async def _models(self, request: web.Request) -> web.Response:
        for r in self.replicas:
            if not r.healthy:
                continue
            try:
                async with self._session.get(r.url + "/v1/models") as resp:
                    return web.Response(body=await resp.read(), status=resp.status, content_type="application/json")
            except aiohttp.ClientError:
                continue
        return web.json_response({"error": "no replica available"}, status=503)

    async def _metrics(self, request: web.Request) -> web.Response:
        return web.Response(body=generate_latest(self.metrics.registry), headers={"Content-Type": CONTENT_TYPE_LATEST})

    async def _health(self, request: web.Request) -> web.Response:
        up = sum(r.healthy for r in self.replicas)
        return web.json_response({"healthy_replicas": up, "replicas": len(self.replicas)}, status=200 if up else 503)

    async def _stats(self, request: web.Request) -> web.Response:
        return web.json_response(
            {"policy": self.policy.name, "block_size": self.block_size, "replicas": [r.stats() for r in self.replicas]}
        )

    async def _get_policy(self, request: web.Request) -> web.Response:
        return web.json_response({"policy": self.policy.name, "params": vars(self.policy)})

    async def _set_policy(self, request: web.Request) -> web.Response:
        body = await request.json()
        try:
            self.policy = make_policy(body["policy"], **body.get("params", {}))
        except (KeyError, TypeError, ValueError) as e:
            return web.json_response({"error": str(e)}, status=400)
        return web.json_response({"policy": self.policy.name, "params": vars(self.policy)})

    async def _reset(self, request: web.Request) -> web.Response:
        """Clear router state between benchmark runs; with ?replicas=1 also
        clear every replica's prefix cache (vLLM needs VLLM_SERVER_DEV_MODE=1)."""
        for r in self.replicas:
            r.prefix.clear()
            r.routed = r.prompt_tokens = r.matched_tokens = 0
        results = {}
        if request.query.get("replicas") in ("1", "true"):
            for r in self.replicas:
                try:
                    async with self._session.post(r.url + "/reset_prefix_cache") as resp:
                        results[r.name] = resp.status
                except aiohttp.ClientError as e:
                    results[r.name] = str(e)
        return web.json_response({"reset": True, "replicas": results})


def main() -> None:
    ap = argparse.ArgumentParser(description="Cache-aware router for vLLM replicas")
    ap.add_argument("--replicas", required=True, help="comma-separated replica base URLs")
    ap.add_argument("--policy", default="lmetric", choices=["round_robin", "least_load", "prefix", "linear", "lmetric"])
    ap.add_argument("--tokenizer", default="word", help="'word' (mock replicas) or 'hf:<model>' (vLLM)")
    ap.add_argument("--block-size", type=int, default=16, help="must match the replicas' KV block size")
    ap.add_argument("--capacity-blocks", type=int, default=None,
                    help="prefix-index size per replica (default: replica's num_gpu_blocks from /metrics)")
    ap.add_argument("--poll-interval", type=float, default=1.0)
    ap.add_argument("--max-imbalance", type=int, default=8, help="prefix policy: allowed load gap")
    ap.add_argument("--hotspot-factor", type=float, default=None, help="lmetric: enable hotspot guard")
    ap.add_argument("--linear-alpha", type=float, default=1.0)
    ap.add_argument("--linear-beta", type=float, default=256.0)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    params = {
        "prefix": {"max_imbalance": args.max_imbalance},
        "lmetric": {"hotspot_factor": args.hotspot_factor},
        "linear": {"alpha": args.linear_alpha, "beta": args.linear_beta},
    }.get(args.policy, {})
    router = Router(
        [u.strip() for u in args.replicas.split(",") if u.strip()],
        make_policy(args.policy, **params),
        load_tokenizer(args.tokenizer),
        block_size=args.block_size,
        capacity_blocks=args.capacity_blocks,
        poll_interval=args.poll_interval,
    )
    web.run_app(router.build_app(), host=args.host, port=args.port, access_log=None)


if __name__ == "__main__":
    main()
