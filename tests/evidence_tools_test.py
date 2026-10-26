#!/usr/bin/env python3
"""Reject false qualification, retain failed runs, and track exact executables."""

import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parent.parent


def cpu_payload():
    return {
        "benchmark": "cpu-reference-pair-component",
        "baseline": "same_cpu_reference_capacity_zero",
        "optimized_engine_comparison": False, "synthetic_inputs": True,
        "single_thread": True, "device_measurement": False,
        "end_to_end_inference": False, "all_outputs_equivalent": True,
        "rank": 256, "vocabulary_size": 3, "embedding_rows": 4,
        "capacity_entries": 2, "steps_per_repeat": 2, "repeats": 1,
        "modes": [{"mode": name, "all_outputs_equivalent": True,
                   "hits": 0, "misses": 2, "baseline_total_ns": 20,
                   "cached_total_ns": 10, "baseline_p50_step_ns": 10,
                   "baseline_p95_step_ns": 10, "cached_p50_step_ns": 5,
                   "cached_p95_step_ns": 5, "component_speed_ratio": 2}
                  for name in ("warm", "cold", "mixed", "collision")],
    }


def device_payload():
    return {
        "status": "ok", "measurement": "pair_cache_component_only",
        "synthetic_inputs": True, "fixed_base_logits": True,
        "end_to_end_inference": False, "baseline": "same_two_kernel_capacity_zero",
        "backend": "HIP", "rank": 256, "vocab": 3, "embedding_rows": 4,
        "capacity": 2, "steps": 2, "repeats": 1, "graph": True,
        "cases": [{"pattern": name, "equivalent": True, "hits": 0,
                   "hit_rate": 0, "uncached_ms": [2], "cached_ms": [1],
                   "uncached_ms_mean": 2, "cached_ms_mean": 1,
                   "component_speedup": 2, "warmup_prefill_steps": 0,
                   "warm_cache_prefilled": False, "initial_hotset_fill_excluded": False}
                  for name in ("cold", "warm", "mixed")],
    }


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.output_index = 0

    def tearDown(self):
        self.temporary.cleanup()

    def script(self, name, output="", code=0, extra=""):
        path = self.directory / name
        path.write_text(f"#!{sys.executable}\nimport sys\n{extra}\n"
                        f"sys.stdout.write({output!r})\nsys.exit({code})\n", encoding="utf-8")
        path.chmod(0o755)
        return path

    def run_collector(self, device=False, output=None):
        self.output_index += 1
        output = output or self.directory / f"evidence-{self.output_index}.json"
        if device:
            command = [sys.executable, str(ROOT / "tools/qualify_device.py"),
                       "--backend", "hip", "--build-dir", str(self.directory)]
        else:
            command = [sys.executable, str(ROOT / "tools/run_cpu_benchmark.py"),
                       "--executable", str(self.directory / "sl_pair_benchmark")]
        process = subprocess.run([*command, "--output", str(output)], text=True,
                                 capture_output=True, check=False, cwd=self.directory)
        return process, json.loads(output.read_text()) if output.is_file() else None, output

    def test_valid_payloads_and_binary_fingerprints(self):
        self.script("sl_pair_benchmark", json.dumps(cpu_payload()))
        self.script("sl_hip_test", "native correctness passed\n")
        self.script("sl_hip_benchmark", json.dumps(device_payload()))
        for device in (False, True):
            process, record, _ = self.run_collector(device)
            self.assertEqual(process.returncode, 0, process.stderr)
            self.assertEqual(record["status"], "passed")
            self.assertEqual(record["schema_version"], 2)
            observed = record["provenance"]
            self.assertEqual(observed["build_source_binding"], "not_attested")
            self.assertTrue(observed["unchanged_during_run"])
            self.assertEqual(observed["executable_sha256"], observed["executable_sha256_after"])
            self.assertTrue(all(len(value) == 64 for value in observed["executable_sha256"].values()))

    def test_device_rejects_invalid_contracts(self):
        self.script("sl_hip_test")
        valid_json = json.dumps(device_payload())
        invalid = ["", "{", "{}", "[]", "NaN",
                   valid_json[:-1] + ', "extra": 1e999}',
                   valid_json[:-1] + ', "backend": "CUDA"}']
        for change in (lambda p: p.update(backend="CUDA"),
                       lambda p: p.update(end_to_end_inference=True),
                       lambda p: p["cases"][0].update(equivalent=False),
                       lambda p: p["cases"][0].update(cached_ms=[float("nan")]),
                       lambda p: p["cases"][0].update(uncached_ms=[-1]),
                       lambda p: p["cases"][0].update(pattern=[]),
                       lambda p: p["cases"][0].update(component_speedup=10)):
            payload = copy.deepcopy(device_payload())
            change(payload)
            invalid.append(json.dumps(payload))
        for stdout in invalid:
            with self.subTest(stdout=stdout):
                self.script("sl_hip_benchmark", stdout)
                process, record, _ = self.run_collector(True)
                self.assertEqual(process.returncode, 1, process.stderr)
                self.assertEqual(record["status"], "failed")
                self.assertTrue(record["validation_errors"])
                self.assertEqual(record["benchmark"]["stdout"], stdout)

    def test_cpu_rejects_invalid_contracts(self):
        valid_json = json.dumps(cpu_payload())
        invalid = ["", "{", "{}", "[]", valid_json[:-1] + ', "extra": 1e999}',
                   valid_json[:-1] + ', "all_outputs_equivalent": false}']
        for change in (lambda p: p.update(baseline="something_else"),
                       lambda p: p.update(all_outputs_equivalent=False),
                       lambda p: p["modes"][0].update(misses=9),
                       lambda p: p["modes"][0].update(mode=[]),
                       lambda p: p["modes"][0].update(component_speed_ratio=99)):
            payload = copy.deepcopy(cpu_payload())
            change(payload)
            invalid.append(json.dumps(payload))
        for stdout in invalid:
            with self.subTest(stdout=stdout):
                self.script("sl_pair_benchmark", stdout)
                process, record, _ = self.run_collector()
                self.assertEqual(process.returncode, 1, process.stderr)
                self.assertEqual(record["status"], "failed")
                self.assertIsNone(record["result"])
                self.assertEqual(record["execution"]["stdout"], stdout)

    def test_nonzero_runs_are_preserved(self):
        self.script("sl_pair_benchmark", "CPU failed\n", 5)
        self.script("sl_hip_test", "native correctness passed\n")
        self.script("sl_hip_benchmark", "device failed\n", 6)
        for device, expected in ((False, 5), (True, 6)):
            process, record, _ = self.run_collector(device)
            self.assertEqual(process.returncode, expected)
            self.assertEqual(record["status"], "failed")

    def test_unavailable_correctness_never_runs_benchmark(self):
        self.script("sl_hip_test", "no GPU\n", 77)
        marker = self.directory / "benchmark-was-run"
        self.script("sl_hip_benchmark", json.dumps(device_payload()),
                    extra=f"open({str(marker)!r}, 'w').close()")
        process, record, _ = self.run_collector(True)
        self.assertEqual(process.returncode, 77)
        self.assertEqual(record["status"], "unavailable")
        self.assertIsNone(record["benchmark"])
        self.assertFalse(marker.exists())

    def test_binary_changes_fail_evidence(self):
        self.script("sl_pair_benchmark", json.dumps(cpu_payload()),
                    extra="with open(__file__, 'a') as handle: handle.write('# changed\\n')")
        process, record, _ = self.run_collector()
        self.assertEqual(process.returncode, 1)
        self.assertEqual(record["status"], "failed")
        self.assertFalse(record["provenance"]["unchanged_during_run"])
        self.assertTrue(any("executable changed" in value for value in record["validation_errors"]))

    def test_existing_record_is_not_overwritten(self):
        self.script("sl_pair_benchmark", json.dumps(cpu_payload()))
        output = self.directory / "existing.json"
        output.write_text('{"keep":true}\n')
        process, record, _ = self.run_collector(output=output)
        self.assertNotEqual(process.returncode, 0)
        self.assertEqual(record, {"keep": True})


if __name__ == "__main__":
    unittest.main()
