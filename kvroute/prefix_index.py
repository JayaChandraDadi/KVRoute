"""Router-side model of each replica's prefix cache.

vLLM caches KV in fixed-size blocks and identifies a block by a hash chained
over all tokens up to and including it, so block i only matches if blocks
0..i-1 also match. The router reproduces that: it hashes each full block of
a request's tokens with the same chaining and keeps, per replica, an LRU set
of block hashes it has sent there. The number of leading hashes a replica
already holds is the router's estimate of that replica's cache hit.

The estimate can drift from the replica's real cache (the replica may evict
blocks the router still remembers), so the capacity is set from the
replica's own KV-cache size and the replica's reported hit rate is scraped
alongside for comparison.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence

_ROOT = 0x9E3779B97F4A7C15  # arbitrary non-zero seed for the first block


def block_hashes(tokens: Sequence[int], block_size: int) -> list[int]:
    """Chained hashes of every *full* block (a partial last block is never
    cached by vLLM, so it is ignored here too)."""
    hashes: list[int] = []
    prev = _ROOT
    for start in range(0, len(tokens) - block_size + 1, block_size):
        prev = hash((prev, tuple(tokens[start : start + block_size])))
        hashes.append(prev)
    return hashes


class PrefixIndex:
    """LRU set of block hashes believed to be cached on one replica."""

    def __init__(self, capacity_blocks: int) -> None:
        if capacity_blocks <= 0:
            raise ValueError("capacity_blocks must be positive")
        self.capacity_blocks = capacity_blocks
        self._blocks: OrderedDict[int, None] = OrderedDict()

    def __len__(self) -> int:
        return len(self._blocks)

    def match(self, hashes: Sequence[int]) -> int:
        """Number of leading blocks of `hashes` present (stops at first miss)."""
        n = 0
        for h in hashes:
            if h not in self._blocks:
                break
            n += 1
        return n

    def insert(self, hashes: Sequence[int]) -> None:
        """Record that the replica now holds these blocks (most recently used),
        evicting the least recently used blocks beyond capacity."""
        for h in hashes:
            if h in self._blocks:
                self._blocks.move_to_end(h)
            else:
                self._blocks[h] = None
        while len(self._blocks) > self.capacity_blocks:
            self._blocks.popitem(last=False)

    def resize(self, capacity_blocks: int) -> None:
        self.capacity_blocks = max(1, capacity_blocks)
        while len(self._blocks) > self.capacity_blocks:
            self._blocks.popitem(last=False)

    def clear(self) -> None:
        self._blocks.clear()
