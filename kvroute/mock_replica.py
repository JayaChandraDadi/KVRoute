"""A vLLM stand-in for running the whole system on a laptop or in CI.

It serves the same OpenAI endpoints and the same /metrics names as vLLM and
simulates the part that matters for routing: a continuous-batching engine
with an LRU prefix cache of KV blocks. Step time follows a simple cost
model -- a fixed overhead, plus a per-token cost for prefill, plus a
per-sequence cost for decode -- so a prefix-cache hit really does shorten
time to first token, and a busy replica really is slower.

It produces no meaningful text and its timings are a model, not a GPU
measurement. Use it to develop and test the router and the benchmark
harness; use real vLLM replicas (deploy/docker-compose.yml) for numbers.

    python -m kvroute.mock_replica --port 8001
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
import uuid
from collections import deque
from dataclasses import dataclass, field

from aiohttp import web

from .prefix_index import PrefixIndex, block_hashes
from .tokenizer import WordTokenizer

ENGINE_KEY = web.AppKey("engine", object)


@dataclass
class CostModel:
    step_overhead_ms: float = 3.0  # scheduler + launch overhead per engine step
    prefill_ms_per_token: float = 0.04  # ~25k prefill tok/s
    decode_ms_per_seq: float = 0.12  # memory-bound decode cost per running sequence

    def step_ms(self, prefill_tokens: int, decode_seqs: int) -> float:
        return self.step_overhead_ms + prefill_tokens * self.prefill_ms_per_token + decode_seqs * self.decode_ms_per_seq


@dataclass
class Seq:
    rid: str
    tokens: list[int]
    max_tokens: int
    out: asyncio.Queue
    hashes: list[int] = field(default_factory=list)
    cached: int = 0
    prefilled: int = 0
    generated: int = 0
    scheduled: bool = False

    @property
    def prefill_done(self) -> bool:
        return self.prefilled >= len(self.tokens)


class MockEngine:
    def __init__(
        self,
        block_size: int = 16,
        num_gpu_blocks: int = 4096,
        max_num_seqs: int = 64,
        max_batched_tokens: int = 2048,
        cost: CostModel | None = None,
        time_scale: float = 1.0,
    ) -> None:
        self.block_size = block_size
        self.num_gpu_blocks = num_gpu_blocks
        self.max_num_seqs = max_num_seqs
        self.max_batched_tokens = max_batched_tokens
        self.cost = cost or CostModel()
        self.time_scale = time_scale
        self.cache = PrefixIndex(num_gpu_blocks)
        self.waiting: deque[Seq] = deque()
        self.running: list[Seq] = []
        self.prefix_queries = 0
        self.prefix_hits = 0
        self.prompt_tokens = 0
        self.generation_tokens = 0
        self._wake = asyncio.Event()
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    def submit(self, tokens: list[int], max_tokens: int) -> Seq:
        seq = Seq(uuid.uuid4().hex[:12], tokens or [0], max(1, max_tokens), asyncio.Queue())
        self.waiting.append(seq)
        self._wake.set()
        return seq

    def kv_usage(self) -> float:
        blocks = sum((len(s.tokens) + s.generated + self.block_size - 1) // self.block_size for s in self.running)
        return min(1.0, blocks / self.num_gpu_blocks)

    def reset_cache(self) -> None:
        self.cache.clear()

    def _admit(self) -> None:
        while self.waiting and len(self.running) < self.max_num_seqs:
            seq = self.waiting.popleft()
            seq.hashes = block_hashes(seq.tokens, self.block_size)
            hit_blocks = self.cache.match(seq.hashes)
            # Like vLLM, the last prompt token is always recomputed.
            seq.cached = min(hit_blocks * self.block_size, len(seq.tokens) - 1)
            seq.prefilled = seq.cached
            seq.scheduled = True
            self.prefix_queries += len(seq.tokens)
            self.prefix_hits += seq.cached
            self.prompt_tokens += len(seq.tokens)
            self.running.append(seq)

    async def _loop(self) -> None:
        while True:
            if not self.running and not self.waiting:
                self._wake.clear()
                await self._wake.wait()
            self._admit()

            # One engine step: every decoding sequence gets one token, the
            # remaining token budget goes to prefill chunks (FCFS).
            decoding = [s for s in self.running if s.prefill_done]
            budget = self.max_batched_tokens - len(decoding)
            prefill_plan: list[tuple[Seq, int]] = []
            for s in self.running:
                if s.prefill_done or budget <= 0:
                    continue
                chunk = min(budget, len(s.tokens) - s.prefilled)
                prefill_plan.append((s, chunk))
                budget -= chunk
            step_ms = self.cost.step_ms(sum(c for _, c in prefill_plan), len(decoding))
            await asyncio.sleep(step_ms / 1000.0 * self.time_scale)

            finished: list[Seq] = []
            for s in decoding:
                s.generated += 1
                self.generation_tokens += 1
                s.out.put_nowait(f"tok{s.generated} ")
                if s.generated >= s.max_tokens:
                    finished.append(s)
            for s, chunk in prefill_plan:
                s.prefilled += chunk
                if s.prefill_done:
                    self.cache.insert(s.hashes)  # computed blocks become reusable
                    s.generated += 1  # the step that finishes prefill samples the first token
                    self.generation_tokens += 1
                    s.out.put_nowait(f"tok{s.generated} ")
                    if s.generated >= s.max_tokens:
                        finished.append(s)
            for s in finished:
                self.running.remove(s)
                s.out.put_nowait(None)

    def metrics_text(self, model: str) -> str:
        lbl = f'model_name="{model}",engine="0"'
        return "\n".join(
            [
                "# TYPE vllm:num_requests_running gauge",
                f"vllm:num_requests_running{{{lbl}}} {len(self.running)}",
                "# TYPE vllm:num_requests_waiting gauge",
                f"vllm:num_requests_waiting{{{lbl}}} {len(self.waiting)}",
                "# TYPE vllm:kv_cache_usage_perc gauge",
                f"vllm:kv_cache_usage_perc{{{lbl}}} {self.kv_usage():.6f}",
                "# TYPE vllm:prefix_cache_queries counter",
                f"vllm:prefix_cache_queries_total{{{lbl}}} {self.prefix_queries}",
                "# TYPE vllm:prefix_cache_hits counter",
                f"vllm:prefix_cache_hits_total{{{lbl}}} {self.prefix_hits}",
                "# TYPE vllm:prompt_tokens counter",
                f"vllm:prompt_tokens_total{{{lbl}}} {self.prompt_tokens}",
                "# TYPE vllm:generation_tokens counter",
                f"vllm:generation_tokens_total{{{lbl}}} {self.generation_tokens}",
                "# TYPE vllm:cache_config_info gauge",
                f'vllm:cache_config_info{{block_size="{self.block_size}",num_gpu_blocks="{self.num_gpu_blocks}",'
                f'enable_prefix_caching="True",engine="0"}} 1.0',
                "",
            ]
        )


class MockServer:
    def __init__(self, engine: MockEngine, model: str = "mock-model") -> None:
        self.engine = engine
        self.model = model
        self.tokenizer = WordTokenizer()

    def build_app(self) -> web.Application:
        app = web.Application(client_max_size=64 * 1024 * 1024)
        app[ENGINE_KEY] = self.engine
        app.router.add_post("/v1/completions", self._completions)
        app.router.add_post("/v1/chat/completions", self._chat)
        app.router.add_get("/v1/models", self._models)
        app.router.add_get("/metrics", self._metrics)
        app.router.add_get("/health", self._health)
        app.router.add_post("/reset_prefix_cache", self._reset)
        app.on_startup.append(lambda app: self.engine.start() or _noop())
        app.on_cleanup.append(lambda app: self.engine.stop())
        return app

    async def _completions(self, request: web.Request) -> web.StreamResponse:
        body = await request.json()
        prompt = body.get("prompt", "")
        if isinstance(prompt, list) and prompt and isinstance(prompt[0], (str, list)):
            prompt = prompt[0]
        return await self._generate(request, body, self.tokenizer.encode_prompt(prompt), chat=False)

    async def _chat(self, request: web.Request) -> web.StreamResponse:
        body = await request.json()
        return await self._generate(request, body, self.tokenizer.encode_chat(body.get("messages") or []), chat=True)

    async def _generate(self, request, body, tokens, chat: bool) -> web.StreamResponse:
        max_tokens = int(body.get("max_tokens") or body.get("max_completion_tokens") or 16)
        seq = self.engine.submit(tokens, max_tokens)
        created = int(time.time())
        obj = "chat.completion" if chat else "text_completion"
        usage = lambda: {  # noqa: E731
            "prompt_tokens": len(tokens),
            "completion_tokens": seq.generated,
            "total_tokens": len(tokens) + seq.generated,
            "prompt_tokens_details": {"cached_tokens": seq.cached},
        }

        if not body.get("stream"):
            pieces = []
            while (piece := await seq.out.get()) is not None:
                pieces.append(piece)
            text = "".join(pieces)
            choice = {"index": 0, "finish_reason": "length"}
            choice.update({"message": {"role": "assistant", "content": text}} if chat else {"text": text})
            return web.json_response(
                {"id": seq.rid, "object": obj, "created": created, "model": self.model, "choices": [choice],
                 "usage": usage()}
            )

        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        chunk_obj = "chat.completion.chunk" if chat else "text_completion"
        while (piece := await seq.out.get()) is not None:
            delta = {"delta": {"content": piece}} if chat else {"text": piece}
            event = {"id": seq.rid, "object": chunk_obj, "created": created, "model": self.model,
                     "choices": [{"index": 0, **delta, "finish_reason": None}]}
            await resp.write(f"data: {json.dumps(event)}\n\n".encode())
        if (body.get("stream_options") or {}).get("include_usage"):
            event = {"id": seq.rid, "object": chunk_obj, "created": created, "model": self.model, "choices": [],
                     "usage": usage()}
            await resp.write(f"data: {json.dumps(event)}\n\n".encode())
        await resp.write(b"data: [DONE]\n\n")
        await resp.write_eof()
        return resp

    async def _health(self, request: web.Request) -> web.Response:
        return web.Response(text="ok")

    async def _models(self, request: web.Request) -> web.Response:
        return web.json_response({"object": "list", "data": [{"id": self.model, "object": "model"}]})

    async def _metrics(self, request: web.Request) -> web.Response:
        return web.Response(text=self.engine.metrics_text(self.model), content_type="text/plain")

    async def _reset(self, request: web.Request) -> web.Response:
        self.engine.reset_cache()
        return web.json_response({"reset": True})


async def _noop() -> None:
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Simulated vLLM replica (OpenAI API + vLLM metrics)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--model", default="mock-model")
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--num-gpu-blocks", type=int, default=4096)
    ap.add_argument("--max-num-seqs", type=int, default=64)
    ap.add_argument("--max-batched-tokens", type=int, default=2048)
    ap.add_argument("--prefill-ms-per-token", type=float, default=CostModel.prefill_ms_per_token)
    ap.add_argument("--decode-ms-per-seq", type=float, default=CostModel.decode_ms_per_seq)
    ap.add_argument("--step-overhead-ms", type=float, default=CostModel.step_overhead_ms)
    ap.add_argument("--time-scale", type=float, default=1.0, help="<1 runs faster than the modelled GPU")
    args = ap.parse_args()
    engine = MockEngine(
        block_size=args.block_size,
        num_gpu_blocks=args.num_gpu_blocks,
        max_num_seqs=args.max_num_seqs,
        max_batched_tokens=args.max_batched_tokens,
        cost=CostModel(args.step_overhead_ms, args.prefill_ms_per_token, args.decode_ms_per_seq),
        time_scale=args.time_scale,
    )
    web.run_app(MockServer(engine, args.model).build_app(), host=args.host, port=args.port, access_log=None,
                print=None)


if __name__ == "__main__":
    main()
