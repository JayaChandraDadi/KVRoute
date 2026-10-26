"""Routing policies.

Every policy sees, for each healthy replica, the router's estimate of how
many prompt tokens are already cached there and how many requests it is
currently serving, and returns the replica to send the request to.

* ``round_robin``  -- ignores state.
* ``least_load``   -- fewest in-flight requests (what a plain load balancer does).
* ``prefix``       -- longest cached prefix, falling back to least load when a
                      replica is more than ``max_imbalance`` requests busier than
                      the least-loaded one (the usual cache-affinity heuristic).
* ``linear``       -- alpha * new_prefill_tokens + beta * batch_size, the
                      weighted-sum form many production routers use; needs its
                      weights tuned per workload.
* ``lmetric``      -- new_prefill_tokens x batch_size (LMetric, OSDI '26). The
                      product needs no weights: a replica wins either by saving
                      prefill or by being less loaded, in proportion.

LMetric details the paper leaves open, decided here:
* new prefill tokens are at least 1 (vLLM always recomputes the last prompt
  token, even on a full hit);
* batch size counts the request being routed (in_flight + 1), so idle
  replicas still compare by prefill cost instead of all scoring zero;
* ties go to the less-loaded replica, then the lower replica index;
* an optional hotspot guard removes replicas whose load exceeds
  ``hotspot_factor`` x the mean (+1), the fallback the paper describes for
  heavily skewed prefix popularity.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Candidate:
    index: int  # position in the replica list (stable tie-break)
    name: str
    in_flight: int  # requests currently routed there and not finished
    matched_tokens: int  # prompt tokens the router believes are cached there


@dataclass
class Decision:
    index: int
    new_prefill_tokens: int
    score: float


def _new_prefill(prompt_tokens: int, matched: int) -> int:
    return max(1, prompt_tokens - matched)


class Policy:
    name = "base"

    def choose(self, prompt_tokens: int, candidates: list[Candidate]) -> Decision:  # pragma: no cover
        raise NotImplementedError


class RoundRobin(Policy):
    name = "round_robin"

    def __init__(self) -> None:
        self._next = 0

    def choose(self, prompt_tokens, candidates):
        c = candidates[self._next % len(candidates)]
        self._next += 1
        return Decision(c.index, _new_prefill(prompt_tokens, c.matched_tokens), 0.0)


class LeastLoad(Policy):
    name = "least_load"

    def choose(self, prompt_tokens, candidates):
        c = min(candidates, key=lambda c: (c.in_flight, c.index))
        return Decision(c.index, _new_prefill(prompt_tokens, c.matched_tokens), float(c.in_flight))


class PrefixAffinity(Policy):
    name = "prefix"

    def __init__(self, max_imbalance: int = 8) -> None:
        self.max_imbalance = max_imbalance

    def choose(self, prompt_tokens, candidates):
        min_load = min(c.in_flight for c in candidates)
        eligible = [c for c in candidates if c.in_flight - min_load <= self.max_imbalance]
        c = min(eligible, key=lambda c: (-c.matched_tokens, c.in_flight, c.index))
        return Decision(c.index, _new_prefill(prompt_tokens, c.matched_tokens), float(-c.matched_tokens))


class Linear(Policy):
    name = "linear"

    def __init__(self, alpha: float = 1.0, beta: float = 256.0) -> None:
        # Defaults: one queued request is worth 256 prefill tokens.
        self.alpha = alpha
        self.beta = beta

    def choose(self, prompt_tokens, candidates):
        def score(c):
            return self.alpha * _new_prefill(prompt_tokens, c.matched_tokens) + self.beta * (c.in_flight + 1)

        c = min(candidates, key=lambda c: (score(c), c.in_flight, c.index))
        return Decision(c.index, _new_prefill(prompt_tokens, c.matched_tokens), score(c))


class LMetric(Policy):
    name = "lmetric"

    def __init__(self, hotspot_factor: float | None = None) -> None:
        self.hotspot_factor = hotspot_factor

    def choose(self, prompt_tokens, candidates):
        pool = candidates
        if self.hotspot_factor is not None and len(candidates) > 1:
            mean = sum(c.in_flight for c in candidates) / len(candidates)
            limit = self.hotspot_factor * mean + 1
            cooled = [c for c in candidates if c.in_flight <= limit]
            pool = cooled or candidates

        def score(c):
            return _new_prefill(prompt_tokens, c.matched_tokens) * (c.in_flight + 1)

        c = min(pool, key=lambda c: (score(c), c.in_flight, c.index))
        return Decision(c.index, _new_prefill(prompt_tokens, c.matched_tokens), float(score(c)))


POLICIES = {
    "round_robin": RoundRobin,
    "least_load": LeastLoad,
    "prefix": PrefixAffinity,
    "linear": Linear,
    "lmetric": LMetric,
}


def make_policy(name: str, **kwargs) -> Policy:
    if name not in POLICIES:
        raise ValueError(f"unknown policy {name!r}; choose from {sorted(POLICIES)}")
    return POLICIES[name](**kwargs)
