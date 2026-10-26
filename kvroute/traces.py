"""Synthetic workload traces with realistic prefix sharing.

Two shapes that make cache-aware routing matter:

* ``chat``  -- multi-turn conversations. Each session starts with one of a
  few shared system prompts, and every turn resends the whole history plus a
  new user message, so turn n+1 shares turn n's full prompt as its prefix.
* ``agent`` -- tool-using agents. Each agent type has a long shared prefix
  (instructions + tool schemas); every step appends the previous action and a
  tool observation, with short outputs and short think times.

Sessions are closed-loop: a session's next turn is sent only after the
previous one finishes, plus its think time, the way a user or agent behaves.
Session start times follow a Poisson process.

Trace format (JSONL), one record per request:
    {"session": "chat-3", "turn": 0, "start": 1.82, "think_time": 0.0,
     "messages": [...], "max_tokens": 128}
``start`` is used for turn 0 only; later turns use ``think_time``.

    python -m kvroute.traces --kind chat --sessions 200 --rate 4 -o traces/chat.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

_SYLLABLES = ["ka", "lo", "mi", "ten", "ra", "vu", "sel", "do", "pin", "qua", "zor", "eli", "nat", "bru", "fen",
              "gai", "hox", "ju", "ser", "tal", "ux", "wen", "yo", "cri", "dal", "mor", "pex", "sti", "vor", "lin"]


class _Words:
    def __init__(self, rng: random.Random, vocab: int = 3000) -> None:
        self.rng = rng
        self.vocab = [
            "".join(rng.choice(_SYLLABLES) for _ in range(rng.randint(1, 3))) for _ in range(vocab)
        ]

    def text(self, n: int) -> str:
        return " ".join(self.rng.choice(self.vocab) for _ in range(n))


def generate_chat(
    sessions: int = 200,
    rate: float = 4.0,
    system_prompts: int = 4,
    system_words: int = 400,
    turns: tuple[int, int] = (3, 8),
    user_words: tuple[int, int] = (20, 120),
    max_tokens: tuple[int, int] = (64, 256),
    think_time: float = 3.0,
    seed: int = 0,
) -> list[dict]:
    rng = random.Random(seed)
    words = _Words(rng)
    systems = [words.text(system_words) for _ in range(system_prompts)]
    records = []
    t = 0.0
    for s in range(sessions):
        t += rng.expovariate(rate)
        history = [{"role": "system", "content": rng.choice(systems)}]
        for turn in range(rng.randint(*turns)):
            history.append({"role": "user", "content": words.text(rng.randint(*user_words))})
            out = rng.randint(*max_tokens)
            records.append(
                {
                    "session": f"chat-{s}",
                    "turn": turn,
                    "start": round(t, 4),
                    "think_time": 0.0 if turn == 0 else round(rng.expovariate(1.0 / think_time), 4),
                    "messages": list(history),
                    "max_tokens": out,
                }
            )
            # The replayed model's reply is synthetic; the next turn carries a
            # stand-in of the same length so prompt lengths grow realistically.
            history.append({"role": "assistant", "content": words.text(out)})
    return records


def generate_agent(
    sessions: int = 120,
    rate: float = 3.0,
    agent_types: int = 3,
    prefix_words: int = 1500,
    steps: tuple[int, int] = (4, 12),
    observation_words: tuple[int, int] = (40, 250),
    max_tokens: tuple[int, int] = (32, 96),
    think_time: float = 0.8,
    seed: int = 1,
) -> list[dict]:
    rng = random.Random(seed)
    words = _Words(rng)
    prefixes = [words.text(prefix_words) for _ in range(agent_types)]
    records = []
    t = 0.0
    for s in range(sessions):
        t += rng.expovariate(rate)
        history = [
            {"role": "system", "content": rng.choice(prefixes)},
            {"role": "user", "content": "task: " + words.text(rng.randint(15, 60))},
        ]
        for step in range(rng.randint(*steps)):
            out = rng.randint(*max_tokens)
            records.append(
                {
                    "session": f"agent-{s}",
                    "turn": step,
                    "start": round(t, 4),
                    "think_time": 0.0 if step == 0 else round(rng.expovariate(1.0 / think_time), 4),
                    "messages": list(history),
                    "max_tokens": out,
                }
            )
            history.append({"role": "assistant", "content": "call " + words.text(out)})
            history.append({"role": "user", "content": "observation: " + words.text(rng.randint(*observation_words))})
    return records


def generate(kind: str, seed: int = 0, scale: float = 1.0, rate_scale: float = 1.0) -> list[dict]:
    """``chat``, ``agent`` or ``mixed``; ``scale`` multiplies session counts,
    ``rate_scale`` multiplies arrival rates."""
    if kind == "chat":
        return generate_chat(sessions=max(1, int(200 * scale)), rate=4.0 * rate_scale, seed=seed)
    if kind == "agent":
        return generate_agent(sessions=max(1, int(120 * scale)), rate=3.0 * rate_scale, seed=seed + 1)
    if kind == "mixed":
        recs = generate(kind="chat", seed=seed, scale=scale, rate_scale=rate_scale)
        recs += generate(kind="agent", seed=seed, scale=scale, rate_scale=rate_scale)
        return recs
    raise ValueError(f"unknown trace kind {kind!r}")


def save(records: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def load(path: str | Path) -> list[dict]:
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate a synthetic chat / agent trace")
    ap.add_argument("--kind", choices=["chat", "agent", "mixed"], default="chat")
    ap.add_argument("--scale", type=float, default=1.0, help="multiply the number of sessions")
    ap.add_argument("--rate-scale", type=float, default=1.0, help="multiply session arrival rate")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("-o", "--output", required=True)
    args = ap.parse_args()
    records = generate(args.kind, args.seed, args.scale, args.rate_scale)
    save(records, args.output)
    sessions = len({r["session"] for r in records})
    print(f"wrote {len(records)} requests from {sessions} sessions to {args.output}")


if __name__ == "__main__":
    main()
