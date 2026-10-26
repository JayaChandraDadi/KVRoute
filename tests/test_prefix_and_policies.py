import pytest

from kvroute.metrics import parse_vllm_metrics
from kvroute.policies import Candidate, make_policy
from kvroute.prefix_index import PrefixIndex, block_hashes
from kvroute.replay import percentile
from kvroute.tokenizer import WordTokenizer


# --------------------------------------------------------------- hashing
def test_block_hashes_ignore_partial_block():
    assert len(block_hashes(list(range(33)), 16)) == 2
    assert block_hashes(list(range(15)), 16) == []


def test_block_hashes_are_prefix_chained():
    a = block_hashes(list(range(64)), 16)
    b = block_hashes(list(range(48)) + [999] * 16, 16)
    assert a[:3] == b[:3] and a[3] != b[3]
    # Same block content at a different position must hash differently.
    c = block_hashes([7] * 32, 16)
    assert c[0] != c[1]


def test_index_match_stops_at_first_miss():
    idx = PrefixIndex(100)
    hs = block_hashes(list(range(64)), 16)
    idx.insert(hs[:2])
    assert idx.match(hs) == 2
    idx.insert([hs[3]])  # a non-contiguous block does not extend the match
    assert idx.match(hs) == 2


def test_index_lru_eviction_and_touch():
    idx = PrefixIndex(3)
    idx.insert([1, 2, 3])
    idx.insert([1])  # touch 1 -> 2 is now least recently used
    idx.insert([4])
    assert idx.match([2]) == 0 and idx.match([1]) == 1 and len(idx) == 3
    idx.resize(1)
    assert len(idx) == 1 and idx.match([4]) == 1


# --------------------------------------------------------------- policies
def cands(*specs):
    return [Candidate(i, f"r{i}", load, matched) for i, (load, matched) in enumerate(specs)]


def test_round_robin_cycles():
    p = make_policy("round_robin")
    assert [p.choose(100, cands((0, 0), (0, 0), (0, 0))).index for _ in range(4)] == [0, 1, 2, 0]


def test_least_load_breaks_ties_by_index():
    p = make_policy("least_load")
    assert p.choose(100, cands((3, 0), (1, 0), (1, 0))).index == 1


def test_prefix_affinity_respects_imbalance_cap():
    p = make_policy("prefix", max_imbalance=4)
    assert p.choose(1000, cands((2, 900), (0, 0))).index == 0  # within cap: follow the cache
    assert p.choose(1000, cands((9, 900), (0, 0))).index == 1  # too busy: fall back


def test_lmetric_prefers_cache_when_load_equal():
    p = make_policy("lmetric")
    d = p.choose(1000, cands((2, 0), (2, 800)))
    assert d.index == 1 and d.new_prefill_tokens == 200 and d.score == 200 * 3


def test_lmetric_trades_cache_against_load():
    p = make_policy("lmetric")
    # r1 saves 800 prefill tokens but has 9 requests: 200*10=2000 > 1000*1
    assert p.choose(1000, cands((0, 0), (9, 800))).index == 0
    # with only 3 requests it wins: 200*4=800 < 1000
    assert p.choose(1000, cands((0, 0), (3, 800))).index == 1


def test_lmetric_full_hit_still_costs_one_token():
    d = make_policy("lmetric").choose(64, cands((0, 64)))
    assert d.new_prefill_tokens == 1


def test_lmetric_hotspot_guard_excludes_overloaded_replica():
    p = make_policy("lmetric", hotspot_factor=1.5)
    # mean load 10, limit 16: replica 0 (25 in flight) is excluded despite its hit
    assert p.choose(1000, cands((25, 1000), (3, 0), (2, 0))).index == 2


def test_linear_policy_weights():
    p = make_policy("linear", alpha=1.0, beta=100.0)
    # r0: 1000 + 100*1 = 1100 ; r1: 200 + 100*5 = 700
    assert p.choose(1000, cands((0, 0), (4, 800))).index == 1


def test_unknown_policy():
    with pytest.raises(ValueError):
        make_policy("random")


# --------------------------------------------------------------- misc
def test_word_tokenizer_chat_prefix_property():
    tok = WordTokenizer()
    sys = {"role": "system", "content": "you are helpful " * 20}
    t1 = tok.encode_chat([sys, {"role": "user", "content": "hello there"}])
    t2 = tok.encode_chat([sys, {"role": "user", "content": "hello there"},
                          {"role": "assistant", "content": "hi"}, {"role": "user", "content": "more"}])
    assert t2[: len(t1) - 1] == t1[:-1]  # everything but the generation marker is a shared prefix


def test_parse_vllm_metrics_sums_engines():
    text = """# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{engine="0",model_name="m"} 3
vllm:num_requests_running{engine="1",model_name="m"} 2
# TYPE vllm:prefix_cache_queries counter
vllm:prefix_cache_queries_total{engine="0",model_name="m"} 1000
# TYPE vllm:prefix_cache_hits counter
vllm:prefix_cache_hits_total{engine="0",model_name="m"} 250
# TYPE vllm:kv_cache_usage_perc gauge
vllm:kv_cache_usage_perc{engine="0",model_name="m"} 0.42
# TYPE vllm:cache_config_info gauge
vllm:cache_config_info{block_size="16",num_gpu_blocks="12345",engine="0"} 1.0
"""
    m = parse_vllm_metrics(text)
    assert m.running == 5 and m.prefix_queries == 1000 and m.prefix_hits == 250
    assert m.kv_usage == pytest.approx(0.42) and m.num_gpu_blocks == 12345 and m.block_size == 16


def test_percentile_interpolates():
    assert percentile([1, 2, 3, 4], 50) == pytest.approx(2.5)
    assert percentile([5], 99) == 5
    assert percentile([], 50) is None
