"""Benchmark workloads: seeded, reproducible request schedules (see docs/benchmarking.md).

A schedule is a list of requests, each with the second at which it is sent and its exact JSON
body. Arrivals follow a Poisson process (exponential gaps at the profile's rate). Everything is
derived from (profile, seed) with Python's own seeded generator, so the same seed gives the same
schedule on every machine, and a shorter run is a prefix of a longer one.

The "prefix" profiles model prompts that start with one of a few shared system prompts, the case
prefix caching is for: each prompt is a shared prefix followed by a suffix of its own.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

DATASET_VERSION = "prompts-v1"
# Digest of the dataset's parsed content: editing prompts-v1 changes it (and fails a test). Make a
# prompts-v2 instead, so results measured on v1 stay comparable with each other.
DATASET_SHA256 = "5c6e9220c3b01251bba234f2a991e0e316e63bf7fb109a78f3dbf4a11e5178f9"
_DATASET_PATH = Path(__file__).parent / "data" / f"{DATASET_VERSION}.jsonl"

# Synthetic prompts use ordinary token ids from this range: below Llama 3's special tokens
# (128000 and up) and away from the byte-level tokens at the start of the vocabulary.
TOKEN_ID_RANGE = (1000, 100_000)


@dataclass(frozen=True)
class Profile:
    name: str
    description: str
    rate_rps: float           # mean arrival rate (Poisson), requests per second
    num_requests: int
    kind: str                 # "chat": dataset questions; "tokens": synthetic prompts of exact lengths;
                              # "prefix": a shared synthetic prefix + a synthetic suffix of its own
    prompt_tokens: tuple[tuple[int, float], ...] = ()       # "tokens": (length, weight); "prefix": the suffix's
    output_tokens: tuple[tuple[int, float], ...] = ((64, 1.0),)  # (length, weight); forced with ignore_eos
    # "prefix": prefix_count shared prefixes of prefix_tokens tokens each. A request picks prefix k
    # with weight 1 / (k + 1) ** prefix_zipf: 0 picks them uniformly, 1 makes a few of them popular.
    prefix_count: int = 0
    prefix_tokens: int = 0
    prefix_zipf: float = 0.0


PROFILES: dict[str, Profile] = {
    "short": Profile(
        name="short",
        description="short chat questions from prompts-v1, 64 output tokens each, 4 requests/s",
        rate_rps=4.0, num_requests=120, kind="chat", output_tokens=((64, 1.0),)),
    "mixed": Profile(
        name="mixed",
        description="synthetic prompts of 32/128/480 tokens with 32/128/384 output tokens, 2 requests/s",
        rate_rps=2.0, num_requests=120, kind="tokens",
        prompt_tokens=((32, 0.5), (128, 0.3), (480, 0.2)),
        output_tokens=((32, 0.4), (128, 0.4), (384, 0.2))),
    # Prefix caching (Day 43). Prefix lengths are whole 16-token pages, so a hit reuses all of them.
    "shared_prefix": Profile(
        name="shared_prefix",
        description="4 shared 384-token prefixes (uniform) + a 64-token suffix each, 64 output tokens, 4 requests/s",
        rate_rps=4.0, num_requests=120, kind="prefix", prompt_tokens=((64, 1.0),), output_tokens=((64, 1.0),),
        prefix_count=4, prefix_tokens=384),
    "memory_pressure": Profile(
        name="memory_pressure",
        description="16 shared 448-token prefixes (Zipf, s = 1) + a 32-token suffix each, 128 / 256 output "
                    "tokens, 1.5 requests/s",
        rate_rps=1.5, num_requests=120, kind="prefix", prompt_tokens=((32, 1.0),),
        output_tokens=((128, 0.5), (256, 0.5)), prefix_count=16, prefix_tokens=448, prefix_zipf=1.0),
}


@dataclass(frozen=True)
class PlannedRequest:
    index: int
    at_s: float               # when to send it, in seconds after the run starts
    endpoint: str             # "/v1/chat/completions" or "/v1/completions"
    prompt_id: str            # dataset id ("q07") or "tokens-<length>"
    max_tokens: int
    body: dict[str, Any]      # the exact JSON body sent


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def load_dataset() -> list[dict[str, str]]:
    """The prompt dataset, checked against its pinned digest."""
    entries = [json.loads(line) for line in _DATASET_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    digest = hashlib.sha256(_canonical(entries)).hexdigest()
    if digest != DATASET_SHA256:
        raise ValueError(f"{_DATASET_PATH.name} changed (digest {digest}); add a new dataset version instead")
    return entries


def _weighted(rng: random.Random, choices: tuple[tuple[int, float], ...]) -> int:
    values = [value for value, _ in choices]
    weights = [weight for _, weight in choices]
    return rng.choices(values, weights=weights)[0]


def shared_prefixes(profile: Profile, seed: int) -> list[list[int]]:
    """A "prefix" profile's shared prefixes for this seed (from a stream of their own: the arrival and
    content streams are the same as without them). Other profiles have none."""
    if profile.kind != "prefix":
        return []
    if profile.prefix_count < 1 or profile.prefix_tokens < 1:
        raise ValueError(f"profile {profile.name!r} needs prefix_count and prefix_tokens >= 1")
    rng = random.Random(f"neuroserve:{profile.name}:{seed}:prefixes")
    return [[rng.randrange(*TOKEN_ID_RANGE) for _ in range(profile.prefix_tokens)]
            for _ in range(profile.prefix_count)]


def build_schedule(profile: Profile, seed: int, model: str, *, num_requests: int | None = None,
                   rate_rps: float | None = None) -> list[PlannedRequest]:
    """The requests of one run, in send order. Same arguments, same schedule."""
    n = profile.num_requests if num_requests is None else num_requests
    rate = profile.rate_rps if rate_rps is None else rate_rps
    if n < 1 or rate <= 0:
        raise ValueError("a schedule needs at least one request and a positive rate")
    # Separate streams for arrival times and content, seeded from strings (hashed with SHA-512 by
    # Python, so stable across platforms): changing one never shifts the other.
    arrivals = random.Random(f"neuroserve:{profile.name}:{seed}:arrivals")
    content = random.Random(f"neuroserve:{profile.name}:{seed}:content")
    dataset = load_dataset() if profile.kind == "chat" else []
    prefixes, prefix_weights = shared_prefixes(profile, seed), []
    if profile.kind == "prefix":
        prefix_weights = [1.0 / (k + 1) ** profile.prefix_zipf for k in range(profile.prefix_count)]

    schedule = []
    t = 0.0
    for index in range(n):
        t += arrivals.expovariate(rate)  # Poisson process: exponential gaps with mean 1/rate
        max_tokens = _weighted(content, profile.output_tokens)
        common = {"model": model, "max_tokens": max_tokens, "stream": True,
                  "stream_options": {"include_usage": True},
                  "neuroserve": {"ignore_eos": True}}  # exactly max_tokens tokens: lengths are controlled
        if profile.kind == "chat":
            entry = content.choice(dataset)
            endpoint, prompt_id = "/v1/chat/completions", entry["id"]
            body = {**common, "messages": [{"role": "user", "content": entry["prompt"]}]}
        elif profile.kind == "tokens":
            length = _weighted(content, profile.prompt_tokens)
            ids = [content.randrange(*TOKEN_ID_RANGE) for _ in range(length)]
            endpoint, prompt_id = "/v1/completions", f"tokens-{length}"
            body = {**common, "prompt": ids}
        elif profile.kind == "prefix":
            k = content.choices(range(profile.prefix_count), weights=prefix_weights)[0]
            length = _weighted(content, profile.prompt_tokens)
            ids = prefixes[k] + [content.randrange(*TOKEN_ID_RANGE) for _ in range(length)]
            endpoint, prompt_id = "/v1/completions", f"prefix-{k}+{length}"
            body = {**common, "prompt": ids}
        else:
            raise ValueError(f"unknown profile kind {profile.kind!r}")
        schedule.append(PlannedRequest(index, round(t, 6), endpoint, prompt_id, max_tokens, body))
    return schedule


def schedule_digest(schedule: list[PlannedRequest]) -> str:
    """A fingerprint of the whole schedule: equal digests mean identical arrivals and bodies."""
    return hashlib.sha256(_canonical([asdict(p) for p in schedule])).hexdigest()
