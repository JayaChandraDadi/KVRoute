"""Prometheus metrics exported by the router, and a parser for vLLM's /metrics."""

from __future__ import annotations

from dataclasses import dataclass

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram
from prometheus_client.parser import text_string_to_metric_families

_LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120)


class RouterMetrics:
    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        self.registry = registry or CollectorRegistry()
        r = self.registry
        lbl = ["replica"]
        # Router-side accounting
        self.requests = Counter("kvroute_requests", "Requests routed", lbl + ["policy"], registry=r)
        self.prompt_tokens = Counter("kvroute_prompt_tokens", "Prompt tokens routed", lbl, registry=r)
        self.matched_tokens = Counter(
            "kvroute_matched_tokens", "Prompt tokens the router expected to hit the replica's prefix cache", lbl,
            registry=r,
        )
        self.upstream_errors = Counter("kvroute_upstream_errors", "Failed upstream calls", lbl, registry=r)
        self.in_flight = Gauge("kvroute_in_flight", "Requests routed and not finished", lbl, registry=r)
        self.index_blocks = Gauge("kvroute_index_blocks", "Blocks in the router's prefix index", lbl, registry=r)
        self.decision_seconds = Histogram(
            "kvroute_decision_seconds", "Time to tokenize, hash and pick a replica",
            buckets=(1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2), registry=r,
        )
        self.ttft_seconds = Histogram(
            "kvroute_ttft_seconds", "Router-observed time to first response byte", lbl, buckets=_LATENCY_BUCKETS,
            registry=r,
        )
        self.e2e_seconds = Histogram(
            "kvroute_e2e_seconds", "Router-observed request latency", lbl, buckets=_LATENCY_BUCKETS, registry=r
        )
        # Scraped from each replica's /metrics
        self.replica_up = Gauge("kvroute_replica_up", "1 if the replica's /metrics was reachable", lbl, registry=r)
        self.replica_running = Gauge("kvroute_replica_running", "vllm:num_requests_running", lbl, registry=r)
        self.replica_waiting = Gauge("kvroute_replica_waiting", "vllm:num_requests_waiting", lbl, registry=r)
        self.replica_kv_usage = Gauge("kvroute_replica_kv_cache_usage", "vllm:kv_cache_usage_perc", lbl, registry=r)
        self.replica_hit_rate = Gauge(
            "kvroute_replica_prefix_hit_rate", "vllm prefix_cache_hits / prefix_cache_queries (lifetime)", lbl,
            registry=r,
        )


@dataclass
class VllmMetrics:
    running: float | None = None
    waiting: float | None = None
    kv_usage: float | None = None
    prefix_queries: float | None = None
    prefix_hits: float | None = None
    num_gpu_blocks: int | None = None
    block_size: int | None = None


def parse_vllm_metrics(text: str) -> VllmMetrics:
    """Sum vLLM's per-engine series into one value per metric."""
    out = VllmMetrics()

    def add(field: str, value: float) -> None:
        cur = getattr(out, field)
        setattr(out, field, value if cur is None else cur + value)

    for family in text_string_to_metric_families(text):
        for s in family.samples:
            name = s.name
            if name == "vllm:num_requests_running":
                add("running", s.value)
            elif name == "vllm:num_requests_waiting":
                add("waiting", s.value)
            elif name == "vllm:kv_cache_usage_perc":
                out.kv_usage = max(out.kv_usage or 0.0, s.value)
            elif name == "vllm:prefix_cache_queries_total":
                add("prefix_queries", s.value)
            elif name == "vllm:prefix_cache_hits_total":
                add("prefix_hits", s.value)
            elif name.startswith("vllm:cache_config_info"):
                blocks = s.labels.get("num_gpu_blocks")
                if blocks not in (None, "", "None"):
                    out.num_gpu_blocks = int(blocks)
                bs = s.labels.get("block_size")
                if bs not in (None, "", "None"):
                    out.block_size = int(bs)
    return out
