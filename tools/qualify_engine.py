#!/usr/bin/env python3
"""Qualify the opt-in HIP engine adapter on one real device, then measure it.

Runs the same local engine executable with its original head, the uncached
adapter, and a cached adapter. This is the batch application's fixed-window,
greedy, single-node route. It never contacts a server or downloads model assets.
"""

import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import statistics
import subprocess
import sys
import time

from evidence import digest, fields, finite, integer, load_payload, require


PREFIX = "STRIX_LOOKAHEAD_RUN "
MODEL_TEST_PREFIX = "STRIX_LOOKAHEAD_MODEL_TEST "
CAPACITY_ENV = "STRIX_LOOKAHEAD_CAPACITY"
RECEIPT_ENV = "STRIX_LOOKAHEAD_RECEIPT"
TRACE_ENV = "STRIX_LOOKAHEAD_TRACE"
VOCABULARY_LIMIT = 129280
DEFAULT_PAYLOAD_BUDGET = 64 * 1024 * 1024
MAX_CAPACITY = DEFAULT_PAYLOAD_BUDGET // (VOCABULARY_LIMIT * 4)
ORDERS = (
    ("original", "uncached", "cached"),
    ("uncached", "cached", "original"),
    ("cached", "original", "uncached"),
    ("cached", "uncached", "original"),
    ("original", "cached", "uncached"),
    ("uncached", "original", "cached"),
)


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def tokens(value, name):
    require(isinstance(value, list) and bool(value), f"{name} must be a nonempty token list")
    require(all(type(token) is int and 0 <= token < VOCABULARY_LIMIT for token in value),
            f"{name} contains an invalid token ID")
    return value


def read_prompt(path):
    text = path.read_text(encoding="utf-8").strip()
    require(bool(re.fullmatch(r"[0-9]+(?:\s*,\s*[0-9]+)*", text)),
            f"prompt must be a comma-separated decimal token list: {path}")
    return tokens([int(value.strip()) for value in text.split(",")], str(path))


def file_stat(path):
    value = path.stat()
    require(path.is_file(), f"not a regular file: {path}")
    return {"size_bytes": value.st_size, "mtime_ns": value.st_mtime_ns,
            "ctime_ns": value.st_ctime_ns, "device": value.st_dev, "inode": value.st_ino}


def inspect_asset(path, expected=None):
    before = file_stat(path)
    require(before["size_bytes"] > 0, f"empty input asset: {path}")
    fingerprint = digest(path)
    require(file_stat(path) == before, f"input changed while hashing: {path}")
    if expected is not None:
        require(fingerprint == expected.lower(), f"SHA256 mismatch: {path}")
    return {"path": str(path), "sha256": fingerprint, "stat": before}


def unchanged(assets, executable, executable_sha256):
    for asset in assets:
        require(file_stat(Path(asset["path"])) == asset["stat"],
                f"input asset metadata changed: {asset['path']}")
    require(digest(executable) == executable_sha256, "engine executable changed during the campaign")


def write_record(directory, record):
    temporary = directory / "record.json.tmp"
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, allow_nan=False)
        handle.write("\n")
    temporary.replace(directory / "record.json")


def validate_hex_list(value, count, name):
    require(isinstance(value, list) and len(value) == count,
            f"{name} must contain one digest per request")
    require(all(isinstance(item, str) and re.fullmatch(r"[0-9a-f]{16}", item) for item in value),
            f"{name} contains an invalid FNV64 digest")


def validate_receipt(receipt, config, prompts, variant, qualification):
    fields(receipt, {
        "schema_version": 1, "backend": "hip", "transport": "single",
        "world": 1, "mode": "spec", "sampler": "greedy",
        "layers": config["layers"], "max_positions": config["max_positions"],
        "vocabulary": config["vocabulary"], "window": f"fixed:{config['window_rows']}",
        "generate": config["generate"], "sync_start": True,
        "prompt_rows": config["prompt_rows"],
    })
    concurrency = integer(receipt, "concurrency", 1)
    require(concurrency in config["concurrency"], "unexpected receipt concurrency")
    expected_prompts = [prompts[i % len(prompts)] for i in range(concurrency)]
    actual_prompts = receipt.get("prompts")
    require(isinstance(actual_prompts, list) and len(actual_prompts) == concurrency,
            "receipt must contain one prompt token list per request")
    for values in actual_prompts:
        tokens(values, "receipt prompt")
    require(actual_prompts == expected_prompts, "receipt prompt IDs differ from the supplied input")
    emitted = receipt.get("tokens")
    require(isinstance(emitted, list) and len(emitted) == concurrency,
            "receipt must contain one output token list per request")
    for index, values in enumerate(emitted):
        tokens(values, f"request {index}")
        require(len(values) == config["generate"], "output token count differs from requested generation")
    # --sync-start completes all prompts before measuring decode-only steps.
    require(integer(receipt, "decode_tokens", 1) == concurrency * (config["generate"] - 1),
            "decode token count does not match the synchronized workload")
    for key in ("decode_seconds", "wall_decode_seconds", "request_seconds"):
        require(finite(receipt.get(key), key) > 0, f"{key} must be positive")
    require(receipt["request_seconds"] + 1e-9 >= receipt["wall_decode_seconds"],
            "request time is shorter than its decode interval")
    gate = receipt.get("gate")
    require(isinstance(gate, dict), "missing engine correctness gate")
    fields(gate, {"enabled": qualification, "check": qualification, "gather": qualification,
                  "passed": True if qualification else None,
                  "differing_rows": 0, "differing_states": 0})
    compared = integer(gate, "compared_rows")
    require(compared > 0 if qualification else compared == 0,
            "unexpected number of compared rows in the engine gate")
    validate_hex_list(receipt.get("kept_rows_fnv64"), concurrency, "kept_rows_fnv64")
    if qualification:
        validate_hex_list(receipt.get("state_fnv64"), concurrency, "state_fnv64")
    else:
        require(receipt.get("state_fnv64") is None, "timing must not claim a final-state comparison")
    adapter = receipt.get("adapter")
    require(isinstance(adapter, dict), "missing adapter receipt")
    requested = variant != "original"
    fields(adapter, {"compiled": True, "requested": requested, "active": requested,
                     "fallback_windows": 0, "errors": 0})
    capacity = integer(adapter, "capacity")
    expected_capacity = config["capacity"] if variant == "cached" else 0
    require(capacity == expected_capacity, "effective adapter capacity differs from the requested variant")
    steps = integer(adapter, "steps")
    hits, misses = integer(adapter, "hits"), integer(adapter, "misses")
    require(hits + misses == steps, "adapter hit and miss counts do not add up to steps")
    require(steps > 0 if requested else steps == 0, "requested adapter did not run, or original path used it")
    if variant != "cached":
        require(hits == 0, "original/uncached variant reported cache hits")
    require(integer(adapter, "vocabulary_size", 1 if requested else 0) <= VOCABULARY_LIMIT,
            "invalid adapter vocabulary size")
    integer(adapter, "map_rebuilds")
    payload = integer(adapter, "payload_bytes")
    require(payload > 0 if variant == "cached" else payload == 0,
            "unexpected resident cache payload for the requested variant")
    return concurrency


def read_receipts(path, config, prompts, variant, qualification):
    receipts = {}
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.startswith(PREFIX):
                continue
            receipt = load_payload(line[len(PREFIX):])
            concurrency = validate_receipt(receipt, config, prompts, variant, qualification)
            require(concurrency not in receipts, "duplicate receipt for a concurrency")
            receipts[concurrency] = receipt
    require(set(receipts) == set(config["concurrency"]), "missing engine receipts; use the adapter-enabled patched application")
    return receipts


def read_model_test_receipt(path):
    receipts = []
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith(MODEL_TEST_PREFIX):
                receipt = load_payload(line[len(MODEL_TEST_PREFIX):])
                fields(receipt, {"schema_version": 1, "backend": "hip",
                                 "adapter_compiled": True, "real_weights": True,
                                 "draft_outputs_bitwise_equal": True, "failures": 0})
                receipts.append(receipt)
    require(len(receipts) == 1,
            "native model test must emit exactly one successful draft-output equivalence receipt; missing/skipped checks cannot qualify the adapter")
    return receipts[0]


def compare_receipts(current, reference, qualification):
    for concurrency in reference:
        candidate, expected = current[concurrency], reference[concurrency]
        require(candidate["tokens"] == expected["tokens"],
                f"generated token IDs differ at concurrency {concurrency}")
        if qualification:
            require(candidate["kept_rows_fnv64"] == expected["kept_rows_fnv64"],
                    f"kept-row digests differ at concurrency {concurrency}")
            require(candidate["state_fnv64"] == expected["state_fnv64"],
                    f"final-state digests differ at concurrency {concurrency}")


def validate_trace(path, receipts, variant, capacity):
    from analyze_predecessor_trace import analyze, native_record, read_records

    if variant == "original":
        drafts = 0
        for record in read_records(path):
            native_record(record)
            require(record.get("mode") == "baseline", "original path trace contains an adapter step")
            drafts += 1
        require(drafts > 0, "original qualification trace is empty")
        return {"status": "baseline_only", "drafts": drafts, "cache_eligible_steps": 0}
    result = analyze(read_records(path), [0, capacity], input_format="native")
    observed = result["reported_cache_activity"]
    for key in ("hits", "misses"):
        require(observed[key] == sum(receipt["adapter"][key] for receipt in receipts.values()),
                f"trace {key} do not match the application receipts")
    return result


def stop_process(process):
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
        process.wait(timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
        except OSError:
            pass
        process.wait()


def execute(command, environment, directory, label, timeout):
    stdout_path, stderr_path = directory / f"{label}.stdout.log", directory / f"{label}.stderr.log"
    result = {"command": command, "started_at_utc": now(), "timed_out": False,
              "stdout": stdout_path.name, "stderr": stderr_path.name}
    started = time.monotonic_ns()
    with stdout_path.open("x", encoding="utf-8") as stdout, stderr_path.open("x", encoding="utf-8") as stderr:
        try:
            process = subprocess.Popen(command, env=environment, stdout=stdout, stderr=stderr,
                                       start_new_session=os.name == "posix")
            try:
                result["exit_code"] = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                result["timed_out"] = True
                stop_process(process)
                result["exit_code"] = 124
            except KeyboardInterrupt:
                stop_process(process)
                result["exit_code"] = 130
                result["interrupted"] = True
        except OSError as error:
            result["exit_code"] = 127
            stderr.write(str(error) + "\n")
    result["process_wall_seconds_including_load"] = (time.monotonic_ns() - started) / 1e9
    result["finished_at_utc"] = now()
    result["stdout_sha256"] = digest(stdout_path)
    result["stderr_sha256"] = digest(stderr_path)
    return result


def base_command(executable, model, draft, prompt_paths, config):
    command = [str(executable), str(model), "--draft", str(draft), "--transport", "single",
               "--mode", "spec", "--prompts", ",".join(map(str, prompt_paths)),
               "--n", ",".join(map(str, config["concurrency"])),
               "--window", f"fixed:{config['window_rows']}", "--generate", str(config["generate"]),
               "--layers", str(config["layers"]), "--max-positions", str(config["max_positions"]),
               "--prompt-rows", str(config["prompt_rows"]), "--vocab", config["vocabulary"],
               "--sync-start", "--draft-ranks", "1"]
    if config["engram_direct"]:
        command.append("--engram-direct")
    if config["tp_attention"]:
        command.append("--tp-attention")
    return command


def summarize(runs, config):
    summaries = []
    for concurrency in config["concurrency"]:
        modes = {}
        for variant in ORDERS[0]:
            samples = [next(row for row in run["receipts"] if row["concurrency"] == concurrency)
                       for run in runs if run["phase"] == "measurement" and run["variant"] == variant]
            values = [sample["wall_decode_seconds"] for sample in samples]
            require(len(values) == config["repeats"], "incomplete timing sample set")
            median = statistics.median(values)
            finite(median, "median decode time")
            throughput = concurrency * (config["generate"] - 1) / median
            finite(throughput, "median decode throughput")
            modes[variant] = {
                "wall_decode_seconds": values, "median_wall_decode_seconds": median,
                "median_decode_tokens_per_second": throughput,
                "adapter_steps": [sample["adapter"]["steps"] for sample in samples],
                "adapter_hits": [sample["adapter"]["hits"] for sample in samples],
                "adapter_misses": [sample["adapter"]["misses"] for sample in samples],
                "adapter_map_rebuilds": [sample["adapter"]["map_rebuilds"] for sample in samples],
                "adapter_payload_bytes": [sample["adapter"]["payload_bytes"] for sample in samples],
            }
        original = modes["original"]["median_wall_decode_seconds"]
        uncached = modes["uncached"]["median_wall_decode_seconds"]
        cached = modes["cached"]["median_wall_decode_seconds"]
        original_ratio, uncached_ratio = original / cached, uncached / cached
        finite(original_ratio, "original/cached time ratio")
        finite(uncached_ratio, "uncached/cached time ratio")
        summaries.append({"concurrency": concurrency, "modes": modes,
                          "original_over_cached_time_ratio": original_ratio,
                          "uncached_over_cached_time_ratio": uncached_ratio})
    return summaries


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", required=True, type=Path)
    parser.add_argument("--native-test", required=True, type=Path,
                        help="Adapter-enabled native drafter test executable; runs with the same target/draft GGUF assets before A/B")
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--draft", required=True, type=Path)
    parser.add_argument("--prompt-file", required=True, action="append", type=Path,
                        help="Comma-separated token IDs; repeat to supply distinct requests")
    parser.add_argument("--output", required=True, type=Path, help="New evidence directory (must not exist)")
    parser.add_argument("--asset", action="append", type=Path, default=[],
                        help="Additional immutable local input, such as a weight overlay, to fingerprint")
    parser.add_argument("--model-sha256")
    parser.add_argument("--draft-sha256")
    parser.add_argument("--concurrency", default="1,2,4")
    parser.add_argument("--layers", type=int, default=40)
    parser.add_argument("--generate", type=int, default=128)
    parser.add_argument("--max-positions", type=int, default=4096)
    parser.add_argument("--prompt-rows", type=int, default=8)
    parser.add_argument("--window-rows", type=int, default=6)
    parser.add_argument("--vocab", default="full", help="full or a numeric core size; file-based vocabularies are excluded")
    parser.add_argument("--capacity", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup-rounds", type=int, default=0)
    parser.add_argument("--timeout-seconds", type=float, default=1800)
    parser.add_argument("--engram-direct", action="store_true")
    parser.add_argument("--tp-attention", action="store_true")
    parser.add_argument("--trace", action="store_true",
                        help="Record and replay-check predecessor traces during qualification only, never timing")
    args = parser.parse_args()
    try:
        require(bool(re.fullmatch(r"[1-7](?:,[1-7])*", args.concurrency)),
                "concurrency must be comma-separated values 1..7; eight requests fill all eight verifier rows and leave no draft budget")
        concurrency = [int(value) for value in args.concurrency.split(",")]
        require(len(set(concurrency)) == len(concurrency), "duplicate concurrency values")
        require(4 <= args.layers <= 40,
                "layers must be 4..40 for target hidden capture; fewer than 40 is a truncated diagnostic, not the full checkpoint")
        require(3 <= args.generate <= 1048576,
                "generate must be 3..1048576 so at least two decode tokens remain after prefill and the adapter can run")
        require(1 <= args.max_positions <= 2**32 - 1, "invalid max-positions")
        require(1 <= args.prompt_rows <= 8 and 2 <= args.window_rows <= 6,
                "prompt-rows must be 1..8 and window-rows must be 2..6")
        require(1 <= args.capacity <= MAX_CAPACITY,
                f"capacity must be 1..{MAX_CAPACITY} to fit the adapter's 64 MiB per-context full-vocabulary payload budget")
        require(1 <= args.repeats <= 31 and 0 <= args.warmup_rounds <= 8,
                "repeats must be 1..31; warmup-rounds must be 0..8")
        require(math.isfinite(args.timeout_seconds) and args.timeout_seconds > 0, "timeout must be positive and finite")
        require(args.vocab == "full" or (re.fullmatch(r"[0-9]+", args.vocab) and 1 <= int(args.vocab) <= VOCABULARY_LIMIT),
                "vocab must be full or an integer 1..129280")
        for value in (args.model_sha256, args.draft_sha256):
            require(value is None or re.fullmatch(r"[0-9a-fA-F]{64}", value), "expected SHA256 must have 64 hexadecimal characters")
    except ValueError as error:
        parser.error(str(error))
    config = {"concurrency": concurrency, "layers": args.layers, "generate": args.generate,
              "max_positions": args.max_positions, "prompt_rows": args.prompt_rows,
              "window_rows": args.window_rows, "vocabulary": args.vocab, "capacity": args.capacity,
              "repeats": args.repeats, "warmup_rounds": args.warmup_rounds,
              "timeout_seconds": args.timeout_seconds, "engram_direct": args.engram_direct,
              "tp_attention": args.tp_attention, "qualification_trace": args.trace}
    return args, config


def main():
    args, config = arguments()
    directory = args.output.absolute()
    directory.mkdir(parents=True, exist_ok=False)
    record = {"schema_version": 1, "kind": "single_node_hip_engine_qualification",
              "model_scope": "full_40_layers" if config["layers"] == 40 else "truncated_diagnostic",
              "started_at_utc": now(), "status": "running", "configuration": config,
              "runs": [], "native_model_test": None, "validation_errors": [], "summary": None,
              "interpretation": "Real local engine runs are required. Fixed-window greedy single-node results do not establish sampled decoding, CUDA, two-node performance, or superiority for other workloads. Decode ratios exclude model loading; process wall times include it.",
              "build_source_binding": "not_attested", "input_hash_policy": "Full asset SHA256 once; file metadata checked before and after each run. Executable SHA256 checked each time.",
              "order": "three rotating orders followed by their reversed orders",
              "digest_limit": "Token IDs are compared exactly. FNV64 row and state checks are digests, not a collision-free bitwise proof.",
              "exit_code": None}
    write_record(directory, record)
    exit_code = 1
    try:
        executable = args.executable.resolve(strict=True)
        require(executable.is_file(), "engine executable is not a file")
        executable_hash = digest(executable)
        native_test = args.native_test.resolve(strict=True)
        require(native_test.is_file(), "native test executable is not a file")
        native_hash = digest(native_test)
        model, draft = args.model.resolve(strict=True), args.draft.resolve(strict=True)
        prompt_paths = [path.resolve(strict=True) for path in args.prompt_file]
        require(all("," not in str(path) for path in prompt_paths), "prompt paths cannot contain commas")
        prompt_values = [read_prompt(path) for path in prompt_paths]
        require(max(map(len, prompt_values)) + config["generate"] + config["window_rows"] <= config["max_positions"],
                "max-positions must cover prompt, generated tokens, and a verification window")
        print("Fingerprinting local model assets once before the campaign...", file=sys.stderr, flush=True)
        # These packs may select many tensors from multiple directories or a
        # mix manifest. A list of arbitrary --asset files does not establish
        # complete coverage of those inputs, so this initial campaign excludes
        # them explicitly instead of silently treating the GGUF as sufficient.
        for name in ("STRIX_DS41_TRELLIS", "STRIX_DS41_TRELLIS_MIX"):
            require(not os.environ.get(name),
                    f"{name} is outside this campaign's input inventory; use an environment without external pack/mix selections")
        assets = [inspect_asset(model, args.model_sha256), inspect_asset(draft, args.draft_sha256)]
        assets.extend(inspect_asset(path) for path in prompt_paths)
        assets.extend(inspect_asset(path.resolve(strict=True)) for path in args.asset)
        overlay = os.environ.get("STRIX_DS41_OVERLAY")
        overlay_path = Path(overlay).resolve(strict=True) if overlay else None
        if overlay_path is not None and str(overlay_path) not in {asset["path"] for asset in assets}:
            assets.append(inspect_asset(overlay_path))
        root = Path(__file__).resolve().parent.parent
        record["inputs"] = assets
        record["executable"] = {"path": str(executable), "sha256": executable_hash}
        record["native_test_executable"] = {"path": str(native_test), "sha256": native_hash}
        record["source_sha256"] = {name: digest(root / name) for name in ("tools/qualify_engine.py", "tools/evidence.py")}
        if args.trace:
            record["source_sha256"]["tools/analyze_predecessor_trace.py"] = digest(root / "tools/analyze_predecessor_trace.py")
        environment = os.environ.copy()
        environment.pop(CAPACITY_ENV, None)
        environment.pop(TRACE_ENV, None)
        if overlay_path is not None:
            # Pin symlink resolution as well as contents: every child opens the
            # exact overlay file that was fingerprinted above.
            environment["STRIX_DS41_OVERLAY"] = str(overlay_path)
        environment[RECEIPT_ENV] = "1"
        environment["LC_ALL"] = "C"
        reported_environment = {name: value for name, value in environment.items()
                                if name.startswith(("STRIX_", "HIP_", "ROCR_", "HSA_", "CUDA_", "OMP_", "MKL_", "OPENBLAS_"))
                                or name in ("LD_LIBRARY_PATH", "LC_ALL")}
        for name in reported_environment:
            if re.search(r"(?:^|_)(?:TOKEN|KEY|SECRET|PASSWORD|PASSWD|CREDENTIAL|AUTH|COOKIE)(?:_|$)", name):
                reported_environment[name] = "<redacted>"
        record["environment"] = {"reported": reported_environment,
                                 "base_environment_sha256": hashlib.sha256(json.dumps(environment, sort_keys=True).encode()).hexdigest(),
                                 "runtime_variant_switch": CAPACITY_ENV,
                                 "trace_policy": "optional separate files during qualification; disabled during warmup and measurement",
                                 "additional_input_notice": "STRIX_DS41_OVERLAY is resolved and fingerprinted automatically. External TRELLIS/MIX selections are excluded because their complete inventory is not established. Use --asset for any other explicitly supplied local input."}
        # Target verification can correct a changed proposal. Prove proposal,
        # Markov and confidence equivalence first using the native real-weight
        # fixture, independently of the end-to-end token/state gate below.
        native_environment = environment.copy()
        for name in (CAPACITY_ENV, TRACE_ENV, "STRIX_DS41_DSPARK_BENCH", "STRIX_DS41_DSPARK_DUMP",
                     "STRIX_DS41_DSPARK_VOCAB_DIR"):
            native_environment.pop(name, None)
        native_environment["STRIX_DS41_Q2_GGUF"] = str(model)
        native_environment["STRIX_DS41_DSPARK_GGUF"] = str(draft)
        unchanged(assets, executable, executable_hash)
        require(digest(native_test) == native_hash, "native test executable changed before qualification")
        print("native-model-equivalence: running real-weight drafter tests", file=sys.stderr, flush=True)
        native_run = {"execution": execute([str(native_test)], native_environment, directory,
                                           "native-model-equivalence", config["timeout_seconds"]),
                      "model": str(model), "draft": str(draft), "receipt": None,
                      "executable_sha256_before": native_hash,
                      "executable_sha256_after": None,
                      "optional_vocabulary_pack_tests": "disabled; campaign uses only full/numeric vocabularies"}
        record["native_model_test"] = native_run
        write_record(directory, record)
        native_run["executable_sha256_after"] = digest(native_test)
        unchanged(assets, executable, executable_hash)
        require(native_run["executable_sha256_after"] == native_hash,
                "native test executable changed during qualification")
        native_code = native_run["execution"]["exit_code"]
        if native_code == 77:
            record["status"] = "unavailable"
            exit_code = 77
            raise RuntimeError("native model test reported unavailable (exit 77); no A/B runs started")
        if native_code == 130:
            raise KeyboardInterrupt
        require(native_code == 0, f"native model test failed with exit code {native_code}; no A/B runs started")
        native_run["receipt"] = read_model_test_receipt(directory / native_run["execution"]["stdout"])
        write_record(directory, record)
        command = base_command(executable, model, draft, prompt_paths, config)
        reference = None
        timing_reference = None
        schedule = [("qualification", 0, ORDERS[0])]
        schedule.extend(("warmup", index, ORDERS[index % len(ORDERS)]) for index in range(config["warmup_rounds"]))
        schedule.extend(("measurement", index, ORDERS[index % len(ORDERS)]) for index in range(config["repeats"]))
        for phase, repetition, order in schedule:
            for variant in order:
                unchanged(assets, executable, executable_hash)
                child_environment = environment.copy()
                if variant != "original":
                    child_environment[CAPACITY_ENV] = str(config["capacity"] if variant == "cached" else 0)
                qualified = phase == "qualification"
                run_command = [*command, *(["--check", "--gather"] if qualified else ["--spec-only"])]
                label = f"{phase}-{repetition:02d}-{variant}"
                trace_path = directory / f"{label}.trace.jsonl"
                if args.trace and qualified:
                    child_environment[TRACE_ENV] = str(trace_path)
                print(f"{label}: running local engine", file=sys.stderr, flush=True)
                run = {"phase": phase, "repetition": repetition, "variant": variant,
                       "capacity_environment": child_environment.get(CAPACITY_ENV), "receipts": [],
                       "execution": execute(run_command, child_environment, directory, label, config["timeout_seconds"])}
                record["runs"].append(run)
                if trace_path.is_file():
                    run["trace"] = {"path": trace_path.name, "sha256": digest(trace_path),
                                    "size_bytes": trace_path.stat().st_size}
                write_record(directory, record)
                unchanged(assets, executable, executable_hash)
                code = run["execution"]["exit_code"]
                if code == 77:
                    record["status"] = "unavailable"
                    exit_code = 77
                    raise RuntimeError("engine reported unavailable (exit 77); no performance result")
                if code == 130:
                    raise KeyboardInterrupt
                require(code == 0, f"{label} failed with exit code {code}; inspect its saved logs")
                receipts = read_receipts(directory / run["execution"]["stdout"], config, prompt_values, variant, qualified)
                run["receipts"] = list(receipts.values())
                if args.trace and qualified:
                    require(trace_path.is_file(), "requested qualification trace was not produced")
                    run["trace"]["analysis"] = validate_trace(trace_path, receipts, variant, config["capacity"])
                if reference is None:
                    require(qualified and variant == "original", "original qualification must run first")
                    reference = receipts
                else:
                    compare_receipts(receipts, reference, qualified)
                if not qualified:
                    if timing_reference is None:
                        timing_reference = receipts
                    else:
                        for concurrency in receipts:
                            require(receipts[concurrency]["kept_rows_fnv64"] == timing_reference[concurrency]["kept_rows_fnv64"],
                                    f"timing kept-row digests differ at concurrency {concurrency}")
                write_record(directory, record)
        unchanged(assets, executable, executable_hash)
        record["source_sha256_after"] = {name: digest(root / name) for name in record["source_sha256"]}
        require(record["source_sha256_after"] == record["source_sha256"], "collector sources changed during the campaign")
        record["summary"] = summarize(record["runs"], config)
        record["status"] = "passed"
        record["exact_output_tokens_equal"] = True
        record["qualification_row_and_state_digests_equal"] = True
        exit_code = 0
    except KeyboardInterrupt:
        record["status"] = "interrupted"
        record["validation_errors"].append("campaign interrupted; partial records and logs retained")
        exit_code = 130
    except (OSError, ValueError, RuntimeError) as error:
        if record["status"] != "unavailable":
            record["status"] = "failed"
        record["validation_errors"].append(str(error))
    record["exit_code"] = exit_code
    record["finished_at_utc"] = now()
    write_record(directory, record)
    print(json.dumps({"status": record["status"], "output": str(directory / "record.json"),
                      "exit_code": exit_code, "validation_errors": record["validation_errors"]}, allow_nan=False))
    return exit_code


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError) as error:
        print(f"engine qualification: {error}", file=sys.stderr)
        sys.exit(1)
