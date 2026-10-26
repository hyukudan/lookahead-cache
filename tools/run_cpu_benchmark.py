#!/usr/bin/env python3
"""Run the synthetic CPU component benchmark and retain its provenance."""

import argparse
import datetime
import json
import os
from pathlib import Path
import platform
import sys

from evidence import finish_provenance, invoke, load_payload, provenance, validate_cpu


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--pin", action="store_true", help="Use the first permitted CPU on Linux")
    parser.add_argument("benchmark_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    executable = args.executable.resolve(strict=True)
    output = args.output.absolute()
    if output.exists():
        parser.error("output already exists; choose a new evidence file")
    extra = args.benchmark_args
    if extra[:1] == ["--"]:
        extra = extra[1:]
    command = [str(executable), *extra]
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    if args.pin and not affinity:
        parser.error("CPU affinity is not available on this platform")
    pinned_cpu = affinity[0] if args.pin else None
    if pinned_cpu is not None:
        os.sched_setaffinity(0, {pinned_cpu})
    root = Path(__file__).resolve().parent.parent
    source_paths = ["src/pair.cpp", "src/cache.cpp", "include/strix_lookahead/pair.h",
                    "include/strix_lookahead/cache.h", "examples/pair_benchmark.cpp",
                    "CMakeLists.txt", "Makefile", "tools/run_cpu_benchmark.py", "tools/evidence.py"]
    source_paths = {name: root / name for name in source_paths}
    executables = {executable.name: executable}
    observed = provenance(source_paths, executables)
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    execution = invoke(command)
    code = execution["exit_code"]
    result = None
    errors = []
    if code == 0:
        try:
            result = load_payload(execution["stdout"])
            validate_cpu(result)
        except ValueError as error:
            errors.append(str(error))
            result = None
            code = 1
    errors.extend(finish_provenance(observed, source_paths, executables))
    if errors:
        code = 1
    cpu_model = platform.processor() or "unreported"
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
    flags_file = executable.parent / "CMakeFiles/strix_lookahead.dir/flags.make"
    compile_flags = None
    if flags_file.is_file():
        compile_flags = next((line.split("=", 1)[1].strip()
                              for line in flags_file.read_text().splitlines()
                              if line.startswith("CXX_FLAGS =")), None)
    evidence = {
        "schema_version": 2,
        "started_at_utc": started,
        "finished_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "hardware": {"cpu_model": cpu_model, "machine": platform.machine(),
                     "system": platform.system(), "pinned_cpu": pinned_cpu},
        "command": [executable.name, *extra],
        "library_compile_flags": compile_flags,
        "library_compile_flags_origin": "Adjacent CMake flags.make when available; not a binary attestation.",
        "provenance": observed,
        "status": "passed" if code == 0 else "failed",
        "collector_exit_code": code if 0 <= code <= 255 else 1,
        "execution": execution,
        "validation_errors": errors,
        "interpretation": "Synthetic CPU operator comparison against the same uncached reference; no LLM or GPU performance claim.",
        "result": result,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation avoids silently replacing earlier measurements.
    with output.open("x", encoding="utf-8") as handle:
        json.dump(evidence, handle, indent=2, allow_nan=False)
        handle.write("\n")
    summary = {"output": str(output), "cpu_model": cpu_model,
               "status": evidence["status"], "exit_code": code}
    if code == 0:
        summary.update(all_outputs_equivalent=True, modes=result["modes"])
    print(json.dumps(summary, indent=2))
    return code if 0 < code <= 255 else 1 if code else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError) as error:
        print(f"benchmark evidence: {error}", file=sys.stderr)
        sys.exit(1)
