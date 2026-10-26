"""Validation and provenance shared by the local evidence collectors.

Source snapshots describe the working tree. They are deliberately not presented
as proof that a binary was built from that tree; a binary digest identifies the
exact executable that was run, without making that stronger claim.
"""

import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys


def digest(path):
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def snapshot(paths):
    result = {}
    for name, path in paths.items():
        try:
            result[name] = digest(path)
        except OSError as error:
            result[name] = {"error": str(error)}
    return result


def provenance(source_paths, executables):
    return {
        "source_sha256": snapshot(source_paths),
        "executable_sha256": snapshot(executables),
        "build_source_binding": "not_attested",
        "interpretation": "Source digests describe the working tree at run time; executable digests identify the binaries run. No source-to-binary build attestation is claimed.",
    }


def finish_provenance(record, source_paths, executables):
    sources_after = snapshot(source_paths)
    executables_after = snapshot(executables)
    record["source_sha256_after"] = sources_after
    record["executable_sha256_after"] = executables_after
    errors = []
    for label, before, after in (
            ("source", record["source_sha256"], sources_after),
            ("executable", record["executable_sha256"], executables_after)):
        for name in before:
            if not isinstance(before[name], str) or not isinstance(after[name], str):
                errors.append(f"cannot fingerprint {label}: {name}")
            elif before[name] != after[name]:
                errors.append(f"{label} changed during the run: {name}")
    record["unchanged_during_run"] = not errors
    return errors


def invoke(command):
    try:
        process = subprocess.run(command, text=True, capture_output=True, check=False,
                                 encoding="utf-8", errors="replace")
        code, stdout, stderr = process.returncode, process.stdout, process.stderr
    except OSError as error:
        code, stdout, stderr = 127, "", str(error)
    return {"command": [Path(command[0]).name, *command[1:]],
            "exit_code": code, "stdout": stdout, "stderr": stderr}


def load_payload(stdout):
    def invalid_constant(value):
        raise ValueError(f"non-finite JSON constant: {value}")
    def finite_float(value):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"non-finite JSON number: {value}")
        return number
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicated JSON key: {key}")
            result[key] = value
        return result
    value = json.loads(stdout, parse_constant=invalid_constant,
                       parse_float=finite_float, object_pairs_hook=unique_object)
    if not isinstance(value, dict):
        raise ValueError("benchmark payload must be a JSON object")
    return value


def require(condition, message):
    if not condition:
        raise ValueError(message)


def fields(payload, expected):
    for key, value in expected.items():
        actual = payload.get(key)
        require(type(actual) is type(value) and actual == value,
                f"unexpected or missing benchmark field: {key}")


def integer(payload, key, minimum=0):
    value = payload.get(key)
    require(type(value) is int and minimum <= value <= 2**64 - 1,
            f"{key} must be an unsigned 64-bit integer >= {minimum}")
    return value


def finite(value, name):
    require(type(value) in (int, float) and 0 <= value <= sys.float_info.max and math.isfinite(value),
            f"{name} must be finite and nonnegative")
    return value


def approximately(actual, expected, name):
    finite(actual, name)
    require(math.isclose(actual, expected, rel_tol=1e-7, abs_tol=1e-9),
            f"inconsistent benchmark field: {name}")


def validate_cpu(payload):
    fields(payload, {
        "benchmark": "cpu-reference-pair-component",
        "baseline": "same_cpu_reference_capacity_zero",
        "optimized_engine_comparison": False, "synthetic_inputs": True,
        "single_thread": True, "device_measurement": False,
        "end_to_end_inference": False, "all_outputs_equivalent": True,
        "rank": 256,
    })
    for key in ("vocabulary_size", "embedding_rows", "capacity_entries"):
        integer(payload, key, 1)
    count = integer(payload, "steps_per_repeat", 1) * integer(payload, "repeats", 1)
    modes = payload.get("modes")
    require(isinstance(modes, list) and len(modes) == 4, "CPU benchmark must contain four modes")
    names = []
    for mode in modes:
        require(isinstance(mode, dict), "invalid CPU benchmark mode")
        require(isinstance(mode.get("mode"), str), "CPU mode name must be a string")
        names.append(mode.get("mode"))
        fields(mode, {"all_outputs_equivalent": True})
        require(integer(mode, "hits") + integer(mode, "misses") == count,
                "CPU mode hit/miss counts do not match measured steps")
        baseline = integer(mode, "baseline_total_ns")
        cached = integer(mode, "cached_total_ns")
        for key in ("baseline_p50_step_ns", "baseline_p95_step_ns",
                    "cached_p50_step_ns", "cached_p95_step_ns"):
            integer(mode, key)
        if cached:
            approximately(mode.get("component_speed_ratio"), baseline / cached, "component_speed_ratio")
        else:
            require(mode.get("component_speed_ratio") is None, "zero-time CPU ratio must be null")
    require(set(names) == {"warm", "cold", "mixed", "collision"}, "unexpected or duplicated CPU modes")


def validate_device(payload, backend):
    fields(payload, {
        "status": "ok", "measurement": "pair_cache_component_only",
        "synthetic_inputs": True, "fixed_base_logits": True,
        "end_to_end_inference": False, "baseline": "same_two_kernel_capacity_zero",
        "backend": backend.upper(), "rank": 256,
    })
    for key in ("vocab", "embedding_rows", "capacity"):
        integer(payload, key, 1)
    steps = integer(payload, "steps", 1)
    repeats = integer(payload, "repeats", 1)
    require(type(payload.get("graph")) is bool, "graph must be a boolean")
    cases = payload.get("cases")
    require(isinstance(cases, list) and 3 <= len(cases) <= 4, "device benchmark must contain three or four cases")
    names = []
    for case in cases:
        require(isinstance(case, dict), "invalid device benchmark case")
        require(isinstance(case.get("pattern"), str), "device case name must be a string")
        names.append(case.get("pattern"))
        fields(case, {"equivalent": True})
        hits = integer(case, "hits")
        require(hits <= steps * repeats, "device hits exceed measured steps")
        approximately(case.get("hit_rate"), hits / (steps * repeats), "hit_rate")
        means = {}
        for key in ("uncached_ms", "cached_ms"):
            values = case.get(key)
            require(isinstance(values, list) and len(values) == repeats,
                    f"{key} sample count does not match repeats")
            for value in values:
                finite(value, key)
            means[key] = sum(values) / repeats
            approximately(case.get(f"{key}_mean"), means[key], f"{key}_mean")
        if means["cached_ms"]:
            approximately(case.get("component_speedup"), means["uncached_ms"] / means["cached_ms"], "component_speedup")
        else:
            require(case.get("component_speedup") is None, "zero-time device ratio must be null")
        prefill = integer(case, "warmup_prefill_steps") > 0
        fields(case, {"warm_cache_prefilled": prefill, "initial_hotset_fill_excluded": prefill})
    require(len(set(names)) == len(names) and {"cold", "warm", "mixed"} <= set(names)
            and set(names) <= {"cold", "warm", "mixed", "collision"}, "unexpected or duplicated device cases")
