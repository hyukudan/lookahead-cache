#!/usr/bin/env python3
"""Run native correctness before timing and retain an explicit qualification record."""

import argparse
import datetime
import json
from pathlib import Path
import sys

from evidence import finish_provenance, invoke, load_payload, provenance, validate_device


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True, choices=["hip", "cuda"])
    parser.add_argument("--build-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("benchmark_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; choose a new evidence file")
    root = Path(__file__).resolve().parent.parent
    build = args.build_dir.resolve(strict=True)
    test = build / f"sl_{args.backend}_test"
    benchmark = build / f"sl_{args.backend}_benchmark"
    if not test.is_file() or not benchmark.is_file():
        parser.error("build the native qualification test and benchmark first")
    extra = args.benchmark_args
    if extra[:1] == ["--"]:
        extra = extra[1:]
    suffix = "hip.hip" if args.backend == "hip" else "cuda.cu"
    sources = ["include/strix_lookahead/cache.h", "include/strix_lookahead/pair.h",
               "include/strix_lookahead/device.h", "src/pair.cpp",
               "src/cache.cpp",
               f"src/device_{suffix}", "src/device_kernels.hpp",
               "src/device_runtime.hpp", "support/device_runtime.hpp",
               "tests/device_test.cpp", "examples/device_benchmark.cpp",
               "CMakeLists.txt", "tools/qualify_device.py", "tools/evidence.py"]
    source_paths = {name: root / name for name in sources}
    executables = {test.name: test, benchmark.name: benchmark}
    record = {
        "schema_version": 2, "backend": args.backend,
        "started_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "provenance": provenance(source_paths, executables),
        "end_to_end_inference": False,
        "correctness": invoke([str(test)]),
        "benchmark": None,
        "validation_errors": [],
    }
    code = record["correctness"]["exit_code"]
    if code == 0:
        record["benchmark"] = invoke([str(benchmark), *extra])
        code = record["benchmark"]["exit_code"]
        if code == 0:
            try:
                payload = load_payload(record["benchmark"]["stdout"])
                validate_device(payload, args.backend)
                record["benchmark"]["result"] = payload
            except ValueError as error:
                record["validation_errors"].append(str(error))
                code = 1
    record["validation_errors"].extend(finish_provenance(record["provenance"], source_paths, executables))
    if record["validation_errors"]:
        code = 1
    record["qualification_exit_code"] = code if 0 <= code <= 255 else 1
    record["status"] = "passed" if code == 0 else "unavailable" if code == 77 else "failed"
    record["finished_at_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"status": record["status"], "output": str(args.output),
                      "backend": args.backend, "exit_code": code}))
    # Preserve 77 instead of letting a skipped test appear as qualification.
    return code if 0 <= code <= 255 else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError) as error:
        print(f"device qualification: {error}", file=sys.stderr)
        sys.exit(1)
