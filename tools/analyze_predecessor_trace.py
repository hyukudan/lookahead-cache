#!/usr/bin/env python3
"""Replay predecessor-cache traces without inferring latency from hit rates.

Native JSONL uses one event="draft" record per completed draft, with context_id,
vocabulary_epoch, vocabulary_size, predecessors, cache_hits, capacity and mode.
An epoch changes whenever the cache is recreated or invalidated. Native traces
must begin before the first draft of every recorded context. The first observed
epoch starts cold; reported hits are checked against that exact direct-map replay.

Legacy window records contain p, pos, t and d[5]. They omit vocabulary identity,
so --format legacy requires --assume-fixed-vocab and produces simulations only.
The legacy predecessor chain is t,d[0],d[1],d[2],d[3], regardless of how many
drafts the target subsequently verifies or accepts.
"""

import argparse
import json
from pathlib import Path
import sys

from evidence import digest, load_payload


UINT32_MAX = 2**32 - 1
MAX_CAPACITY = 65536


def unsigned(value, name, maximum=2**64 - 1, minimum=0):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in {minimum}..{maximum}")
    return value


def slot_for(predecessor, capacity):
    if capacity == 0:
        return None
    value = predecessor
    value ^= value >> 16
    value = (value * 0x7FEB352D) & UINT32_MAX
    value ^= value >> 15
    value = (value * 0x846CA68B) & UINT32_MAX
    value ^= value >> 16
    return value % capacity


class Cache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.slots = {}
        self.hits = 0
        self.misses = 0
        self.replacements = 0

    def clear(self):
        self.slots.clear()

    def step(self, predecessor):
        slot = slot_for(predecessor, self.capacity)
        hit = slot is not None and self.slots.get(slot) == predecessor
        if hit:
            self.hits += 1
        else:
            self.misses += 1
            if slot is not None:
                if slot in self.slots:
                    self.replacements += 1
                self.slots[slot] = predecessor
        return hit


def tokens(value, name, count=None):
    if not isinstance(value, list) or not value or len(value) > 1024:
        raise ValueError(f"{name} must be a nonempty token list of at most 1024 entries")
    if count is not None and len(value) != count:
        raise ValueError(f"{name} must have {count} entries")
    return [unsigned(token, name, UINT32_MAX - 1) for token in value]


def native_record(record):
    if record.get("event") != "draft":
        raise ValueError("native input requires event=draft records")
    context = unsigned(record.get("context_id"), "context_id")
    epoch = unsigned(record.get("vocabulary_epoch"), "vocabulary_epoch")
    vocabulary = unsigned(record.get("vocabulary_size"), "vocabulary_size", UINT32_MAX, 1)
    capacity = unsigned(record.get("capacity"), "capacity", MAX_CAPACITY)
    predecessors = tokens(record.get("predecessors"), "predecessors")
    mode = record.get("mode")
    if mode not in ("cached", "uncached", "baseline", "fallback"):
        raise ValueError("mode must be cached, uncached, baseline or fallback")
    active = mode in ("cached", "uncached")
    if active and ((capacity == 0) != (mode == "uncached")):
        raise ValueError("cached mode requires positive capacity; uncached mode requires zero")
    reported = record.get("cache_hits")
    if active:
        if not isinstance(reported, list) or len(reported) != len(predecessors):
            raise ValueError("cache_hits must contain one boolean or 0/1 per predecessor")
        if any(type(hit) not in (int, bool) or hit not in (0, 1) for hit in reported):
            raise ValueError("cache_hits entries must be booleans or 0/1")
        reported = [bool(hit) for hit in reported]
    elif reported is not None:
        if (not isinstance(reported, list) or len(reported) != len(predecessors) or
                any(type(hit) not in (int, bool) or hit != 0 for hit in reported)):
            raise ValueError("baseline/fallback drafts can only report one false/0 per predecessor")
    return context, epoch, vocabulary, capacity, predecessors, reported, active


def analyze(records, capacities, input_format="native", assume_fixed_vocab=False):
    if input_format not in ("native", "legacy"):
        raise ValueError("unsupported input format")
    if input_format == "legacy" and not assume_fixed_vocab:
        raise ValueError("legacy traces omit vocabulary identity; explicitly pass --assume-fixed-vocab")
    if input_format == "native" and assume_fixed_vocab:
        raise ValueError("--assume-fixed-vocab applies only to legacy traces")
    if not capacities or len(capacities) > 32 or len(set(capacities)) != len(capacities):
        raise ValueError("choose 1..32 distinct capacities")
    for capacity in capacities:
        unsigned(capacity, "capacity", MAX_CAPACITY)
    contexts = {}
    completed = set()
    drafts = steps = skipped = transitions = observed_hits = observed_misses = 0
    for line, record in enumerate(records, 1):
        try:
            if not isinstance(record, dict):
                raise ValueError("each trace record must be an object")
            if input_format == "legacy":
                context = unsigned(record.get("p"), "p")
                if "done" in record:
                    if type(record["done"]) is not int or record["done"] != 1:
                        raise ValueError("legacy completion marker must be done=1")
                    if context in completed:
                        raise ValueError("repeated completion; appended runs need separate input files")
                    completed.add(context)
                    continue
                if context in completed:
                    raise ValueError("prompt reused after completion; analyze separate runs independently")
                position = unsigned(record.get("pos"), "pos")
                predecessors = [unsigned(record.get("t"), "t", UINT32_MAX - 1),
                                *tokens(record.get("d"), "d", 5)[:4]]
                epoch, vocabulary, capacity, reported, active = 0, None, None, None, True
            else:
                context, epoch, vocabulary, capacity, predecessors, reported, active = native_record(record)
                position = None
            state = contexts.get(context)
            if state is None:
                if len(contexts) >= 4096:
                    raise ValueError("at most 4096 contexts per input file")
                state = {"epoch": epoch, "vocabulary": vocabulary, "capacity": capacity,
                         "peak_vocabulary": vocabulary, "last_position": None,
                         "simulated": {c: Cache(c) for c in capacities},
                         "observed": Cache(capacity) if input_format == "native" else None}
                contexts[context] = state
            elif epoch != state["epoch"]:
                if epoch < state["epoch"]:
                    raise ValueError("vocabulary_epoch must increase, never return to an earlier cache")
                for cache in state["simulated"].values():
                    cache.clear()
                state.update(epoch=epoch, vocabulary=vocabulary, capacity=capacity,
                             observed=Cache(capacity))
                transitions += 1
            elif (vocabulary, capacity) != (state["vocabulary"], state["capacity"]):
                raise ValueError("vocabulary size/capacity changed without a new epoch")
            if vocabulary is not None:
                state["peak_vocabulary"] = max(state["peak_vocabulary"], vocabulary)
            if position is not None:
                previous = state["last_position"]
                if previous is not None and position <= previous:
                    raise ValueError("legacy positions must increase within each prompt")
                state["last_position"] = position
            drafts += 1
            if not active:
                skipped += 1
                continue
            for index, predecessor in enumerate(predecessors):
                for cache in state["simulated"].values():
                    cache.step(predecessor)
                if reported is not None:
                    expected = state["observed"].step(predecessor)
                    if expected != reported[index]:
                        raise ValueError(f"reported hit differs from cold-start replay at draft step {index}; "
                                         "the trace may omit a draft, reset, epoch change or initial warm state")
                    observed_hits += int(expected)
                    observed_misses += int(not expected)
                steps += 1
        except (TypeError, ValueError) as error:
            raise ValueError(f"record {line}: {error}") from error
    if not drafts or not steps:
        raise ValueError("no cache-eligible draft steps to analyze")
    scenarios = []
    for capacity in capacities:
        caches = [state["simulated"][capacity] for state in contexts.values()]
        hits, misses = sum(c.hits for c in caches), sum(c.misses for c in caches)
        scenarios.append({
            "capacity": capacity, "simulated_hits": hits, "simulated_misses": misses,
            "simulated_hit_rate": hits / (hits + misses),
            "occupied_slot_replacements": sum(c.replacements for c in caches),
            "payload_bytes_if_all_contexts_retained_at_peak_sizes": None if input_format == "legacy" else
                sum(state["peak_vocabulary"] for state in contexts.values()) * capacity * 4,
        })
    return {
        "schema_version": 1, "status": "analyzed", "input_format": input_format,
        "hardware_execution_verified": False, "latency_or_speedup_estimated": False,
        "contexts": len(contexts), "drafts": drafts, "cache_eligible_steps": steps,
        "skipped_baseline_or_fallback_drafts": skipped, "epoch_transitions": transitions,
        "assumptions": ["Every context begins with an empty cache.",
                        "Epoch changes cover every reset, mapping, weights and numerical-identity change."]
                       if input_format == "native" else
                       ["The complete ordered vocabulary and model remain fixed within each prompt.",
                        "No unrecorded cache invalidation occurs; every prompt begins with an empty cache.",
                        "Legacy ctx is a context length, not a vocabulary or cache identity."],
        "reported_cache_activity": None if input_format == "legacy" else {
            "hits": observed_hits, "misses": observed_misses,
            "hit_rate": observed_hits / (observed_hits + observed_misses),
            "consistent_with_direct_map_replay": True,
            "independently_attested": False,
        },
        "capacity_simulations": scenarios,
    }


def read_records(path):
    with path.open("rb") as handle:
        line = 0
        while True:
            raw = handle.readline(1024 * 1024 + 1)
            if not raw:
                return
            line += 1
            if len(raw) > 1024 * 1024:
                raise ValueError(f"line {line}: trace record exceeds 1 MiB")
            if not raw.strip():
                continue
            try:
                yield load_payload(raw.decode("utf-8"))
            except (UnicodeDecodeError, ValueError) as error:
                raise ValueError(f"line {line}: {error}") from error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--format", choices=("native", "legacy"), default="native")
    parser.add_argument("--assume-fixed-vocab", action="store_true")
    parser.add_argument("--capacities", default="0,4,8,16,32,64")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; choose a new analysis file")
    capacities = [int(value) for value in args.capacities.split(",")]
    before = digest(args.input)
    result = analyze(read_records(args.input), capacities, args.format, args.assume_fixed_vocab)
    if digest(args.input) != before:
        raise ValueError("trace changed while it was analyzed")
    result["input_sha256"] = before
    result["analyzer_sha256"] = digest(Path(__file__))
    result["validation_helpers_sha256"] = digest(Path(__file__).with_name("evidence.py"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"output": str(args.output), "drafts": result["drafts"],
                      "cache_eligible_steps": result["cache_eligible_steps"],
                      "hardware_execution_verified": False}))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError) as error:
        print(f"predecessor trace: {error}", file=sys.stderr)
        sys.exit(1)
