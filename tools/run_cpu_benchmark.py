#!/usr/bin/env python3
"""Run the synthetic CPU component benchmark and retain its provenance."""

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys


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
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    process = subprocess.run(command, text=True, capture_output=True, check=False)
    if process.returncode:
        sys.stderr.write(process.stderr)
        raise RuntimeError(f"benchmark exited with status {process.returncode}")
    result = json.loads(process.stdout)
    if (result.get("all_outputs_equivalent") is not True
            or result.get("end_to_end_inference") is not False
            or result.get("device_measurement") is not False):
        raise RuntimeError("benchmark did not return the expected CPU evidence contract")
    root = Path(__file__).resolve().parent.parent
    source_paths = ["src/pair.cpp", "include/strix_lookahead/pair.h",
                    "include/strix_lookahead/cache.h", "examples/pair_benchmark.cpp"]
    source_hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                     for name in source_paths}
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
        "schema_version": 1,
        "started_at_utc": started,
        "finished_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "hardware": {"cpu_model": cpu_model, "machine": platform.machine(),
                     "system": platform.system(), "pinned_cpu": pinned_cpu},
        "command": [executable.name, *extra],
        "library_compile_flags": compile_flags,
        "source_sha256": source_hashes,
        "interpretation": "Synthetic CPU operator comparison against the same uncached reference; no LLM or GPU performance claim.",
        "result": result,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation avoids silently replacing earlier measurements.
    with output.open("x", encoding="utf-8") as handle:
        json.dump(evidence, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"output": str(output), "cpu_model": cpu_model,
                      "all_outputs_equivalent": True, "modes": result["modes"]}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError) as error:
        print(f"benchmark evidence: {error}", file=sys.stderr)
        sys.exit(1)
