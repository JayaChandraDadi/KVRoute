"""KVRoute: cache-aware request routing for vLLM replicas."""

from .policies import POLICIES, make_policy
from .prefix_index import PrefixIndex, block_hashes
from .router import Router

__all__ = ["POLICIES", "PrefixIndex", "Router", "block_hashes", "make_policy"]
__version__ = "0.1.0"
