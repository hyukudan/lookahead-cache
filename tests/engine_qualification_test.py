#!/usr/bin/env python3
"""Runner contract tests using explicitly synthetic executables, never a GPU.

The mock receipts below exercise campaign ordering and rejection logic. They
are not engine qualification or performance evidence.
"""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest


ROOT = Path(__file__).resolve().parent.parent
MOCK_NATIVE = r'''
import json, os, pathlib, sys
scenario = os.environ.get("SL_ENGINE_TEST_SCENARIO", "")
for key in ("STRIX_LOOKAHEAD_CAPACITY", "STRIX_LOOKAHEAD_TRACE", "STRIX_DS41_DSPARK_BENCH",
            "STRIX_DS41_DSPARK_DUMP", "STRIX_DS41_DSPARK_VOCAB_DIR"):
    if key in os.environ:
        print("native fixture unexpectedly received " + key, file=sys.stderr)
        sys.exit(9)
for key in ("STRIX_DS41_Q2_GGUF", "STRIX_DS41_DSPARK_GGUF"):
    if not pathlib.Path(os.environ[key]).is_file():
        sys.exit(8)
if scenario == "native_failure":
    print("synthetic native check failure", file=sys.stderr)
    sys.exit(2)
if scenario == "native_unavailable":
    sys.exit(77)
if scenario == "native_skipped":
    print("real-weight checks skipped in this explicitly synthetic executable")
    sys.exit(0)
receipt = {"schema_version": 1, "backend": "hip", "adapter_compiled": True,
           "real_weights": True, "fixture": "real_weights_synthetic_hidden",
           "draft_outputs_bitwise_equal": True, "failures": 0}
if scenario == "native_mismatch":
    receipt.update(draft_outputs_bitwise_equal=False, failures=1)
if scenario == "native_wrong_fixture":
    receipt["fixture"] = "synthetic_weights_synthetic_hidden"
if scenario == "native_missing_fixture":
    del receipt["fixture"]
print("STRIX_LOOKAHEAD_MODEL_TEST " + json.dumps(receipt))
if scenario == "native_duplicate":
    print("STRIX_LOOKAHEAD_MODEL_TEST " + json.dumps(receipt))
'''

MOCK_ENGINE = r'''
import json, os, pathlib, sys, time
args = sys.argv[1:]
def option(name):
    return args[args.index(name) + 1]
scenario = os.environ.get("SL_ENGINE_TEST_SCENARIO", "")
if scenario == "unavailable":
    print("mock device unavailable", file=sys.stderr)
    sys.exit(77)
if scenario == "nonzero":
    print("mock native failure", file=sys.stderr)
    sys.exit(2)
if scenario == "timeout":
    time.sleep(2)
if scenario == "no_receipt":
    print("ordinary mock output without the required receipt protocol")
    sys.exit(0)
if scenario == "malformed":
    print('STRIX_LOOKAHEAD_RUN {')
    sys.exit(0)
requested = "STRIX_LOOKAHEAD_CAPACITY" in os.environ
capacity = int(os.environ.get("STRIX_LOOKAHEAD_CAPACITY", "0"))
variant = "cached" if capacity else "uncached" if requested else "original"
qualification = "--check" in args
prompts = [[int(item.strip()) for item in pathlib.Path(path).read_text().strip().split(",")]
           for path in option("--prompts").split(",")]
concurrencies = [int(item) for item in option("--n").split(",")]
generate = int(option("--generate"))
trace = None
if "STRIX_LOOKAHEAD_TRACE" in os.environ:
    trace = open(os.environ["STRIX_LOOKAHEAD_TRACE"], "x")
if not qualification and trace:
    sys.exit(9)
for n in concurrencies:
    # Five repeated predecessors: one miss then four hits for positive capacity.
    # As in the eight-row verifier, saturation or one remaining decode token
    # leaves no room for a speculative window. The mock must not hide that.
    exercised = requested and n < 8 and generate >= 3
    hits = 4 if capacity and exercised else 0
    steps = 5 if exercised else 0
    receipt = {
        "schema_version": 1, "backend": "hip", "transport": "single", "world": 1,
        "mode": "spec", "sampler": "greedy", "layers": int(option("--layers")),
        "max_positions": int(option("--max-positions")), "vocabulary": option("--vocab"),
        "window": option("--window"), "generate": generate, "concurrency": n,
        "sync_start": "--sync-start" in args, "prompt_rows": int(option("--prompt-rows")),
        "prompts": [prompts[i % len(prompts)] for i in range(n)],
        "tokens": [[20 + i + j for j in range(generate)] for i in range(n)],
        "decode_tokens": n * (generate - 1), "decode_seconds": 1.0,
        "wall_decode_seconds": 1.5 if variant != "cached" else 1.0,
        "request_seconds": 2.0,
        "gate": {"enabled": qualification, "check": qualification, "gather": qualification,
                 "passed": True if qualification else None, "compared_rows": 3 * n if qualification else 0,
                 "differing_rows": 0, "differing_states": 0},
        "kept_rows_fnv64": ["0000000000000001" if qualification else "0000000000000002"] * n,
        "state_fnv64": ["0000000000000003"] * n if qualification else None,
        "adapter": {"compiled": True, "requested": requested, "active": requested,
                    "capacity": capacity, "vocabulary_size": 129280, "steps": steps,
                    "hits": hits, "misses": steps - hits, "map_rebuilds": 1 if requested else 0,
                    "fallback_windows": 0, "errors": 0, "payload_bytes": capacity * 129280 * 4 * n},
    }
    if scenario == "tokens" and variant == "cached":
        receipt["tokens"][0][0] += 1
    if scenario == "digest" and variant == "cached":
        receipt["state_fnv64"][0] = "0000000000000004"
    if scenario == "ignored" and requested:
        receipt["adapter"].update(requested=False, active=False)
    if scenario == "fallback" and requested:
        receipt["adapter"]["fallback_windows"] = 1
    if scenario == "zero_work" and requested:
        receipt["adapter"].update(steps=0, hits=0, misses=0)
    if scenario == "wrong_capacity" and variant == "cached":
        receipt["adapter"]["capacity"] += 1
    if scenario == "gate":
        receipt["gate"].update(passed=False, differing_rows=1)
    if scenario == "decode_count":
        receipt["decode_tokens"] += 1
    if scenario == "false_time":
        receipt["wall_decode_seconds"] = 0
    if scenario == "tiny_time" and not qualification:
        receipt["wall_decode_seconds"] = 1e-320
    if scenario == "boolean_prompt":
        receipt["prompts"][0][0] = True
    if scenario == "no_state":
        receipt["state_fnv64"] = None
    if scenario == "missing" and n == concurrencies[-1]:
        continue
    if trace:
        record = {"event": "draft", "context_id": n, "vocabulary_epoch": 1,
                  "vocabulary_size": 129280, "predecessors": [1] * 5,
                  "capacity": capacity, "mode": "baseline" if not requested else variant,
                  "cache_hits": [0, 1, 1, 1, 1] if capacity else [0] * 5}
        if scenario == "trace_hit" and capacity:
            record["cache_hits"][0] = 1
        trace.write(json.dumps(record) + "\n")
    encoded = json.dumps(receipt)
    if scenario == "nonfinite":
        encoded = encoded[:-1] + ', "extra": 1e999}'
    print("STRIX_LOOKAHEAD_RUN " + encoded)
if trace:
    trace.close()
if scenario == "mutate_model":
    with open(args[0], "ab") as handle:
        handle.write(b"changed")
if scenario == "mutate_overlay":
    with open(os.environ["STRIX_DS41_OVERLAY"], "ab") as handle:
        handle.write(b"changed")
print("mock diagnostics retained", file=sys.stderr)
'''


@unittest.skipUnless(os.name == "posix", "mock executable uses a POSIX shebang; no hardware is exercised")
class EngineQualificationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.engine = self.directory / "mock-engine"
        self.engine.write_text(f"#!{sys.executable}\n" + textwrap.dedent(MOCK_ENGINE))
        self.engine.chmod(0o755)
        self.native = self.directory / "mock-native-test"
        self.native.write_text(f"#!{sys.executable}\n" + textwrap.dedent(MOCK_NATIVE))
        self.native.chmod(0o755)
        self.model = self.directory / "synthetic-model-fixture"
        self.draft = self.directory / "synthetic-draft-fixture"
        self.prompt = self.directory / "prompt.ids"
        self.model.write_bytes(b"mock model bytes; not a trained model")
        self.draft.write_bytes(b"mock draft bytes; not a trained model")
        self.prompt.write_text("1,2\n")
        self.count = 0

    def tearDown(self):
        self.temporary.cleanup()

    def run_campaign(self, scenario="", extra=(), environment_overrides=None):
        self.count += 1
        output = self.directory / f"campaign-{self.count}"
        command = [sys.executable, str(ROOT / "tools/qualify_engine.py"),
                   "--executable", str(self.engine), "--native-test", str(self.native), "--model", str(self.model),
                   "--draft", str(self.draft), "--prompt-file", str(self.prompt),
                   "--output", str(output), "--concurrency", "1,2", "--generate", "8",
                   "--max-positions", "64", "--repeats", "1", *extra]
        environment = os.environ.copy()
        for name in ("STRIX_DS41_OVERLAY", "STRIX_DS41_TRELLIS", "STRIX_DS41_TRELLIS_MIX"):
            environment.pop(name, None)
        environment["SL_ENGINE_TEST_SCENARIO"] = scenario
        # Ambient opt-ins must not contaminate the original or timing arms.
        environment["STRIX_LOOKAHEAD_CAPACITY"] = "99"
        environment["STRIX_LOOKAHEAD_TRACE"] = str(self.directory / "ambient-trace-must-not-exist")
        environment["STRIX_DS41_DSPARK_BENCH"] = "1"
        environment["STRIX_DS41_DSPARK_DUMP"] = str(self.directory / "ambient-dump-must-not-exist")
        environment["STRIX_DS41_DSPARK_VOCAB_DIR"] = str(self.directory / "ambient-vocabulary-must-not-be-read")
        for name, value in (environment_overrides or {}).items():
            environment[name] = str(value)
        process = subprocess.run(command, env=environment, text=True, capture_output=True, check=False)
        record_path = output / "record.json"
        record = json.loads(record_path.read_text()) if record_path.is_file() else None
        return process, record, output

    def test_complete_campaign_rotates_orders_and_preserves_inputs(self):
        process, record, output = self.run_campaign(extra=("--repeats", "3"))
        self.assertEqual(process.returncode, 0, process.stderr + process.stdout)
        self.assertEqual(record["status"], "passed")
        self.assertEqual(len(record["runs"]), 12)
        self.assertTrue(record["exact_output_tokens_equal"])
        self.assertEqual(record["model_scope"], "full_40_layers")
        self.assertEqual(record["build_source_binding"], "not_attested")
        native = record["native_model_test"]
        self.assertTrue(native["receipt"]["draft_outputs_bitwise_equal"])
        self.assertEqual(native["receipt"]["fixture"], "real_weights_synthetic_hidden")
        self.assertEqual(native["model"], str(self.model.resolve()))
        self.assertEqual(native["draft"], str(self.draft.resolve()))
        self.assertEqual(native["executable_sha256_before"], native["executable_sha256_after"])
        self.assertEqual(len(native["executable_sha256_before"]), 64)
        self.assertTrue((output / native["execution"]["stdout"]).is_file())
        self.assertFalse((self.directory / "ambient-dump-must-not-exist").exists())
        measured = [run["variant"] for run in record["runs"] if run["phase"] == "measurement"]
        self.assertEqual(measured, ["original", "uncached", "cached", "uncached", "cached", "original",
                                    "cached", "original", "uncached"])
        self.assertFalse((self.directory / "ambient-trace-must-not-exist").exists())
        self.assertEqual(record["summary"][0]["original_over_cached_time_ratio"], 1.5)
        for run in record["runs"]:
            execution = run["execution"]
            self.assertTrue((output / execution["stdout"]).is_file())
            self.assertIn("mock diagnostics retained", (output / execution["stderr"]).read_text())
            self.assertEqual(len(execution["stdout_sha256"]), 64)
            if run["variant"] == "original":
                self.assertIsNone(run["capacity_environment"])
        self.assertEqual(len(record["inputs"]), 3)

    def test_rejects_false_or_incomplete_success(self):
        for scenario in ("no_receipt", "malformed", "tokens", "digest", "ignored", "fallback",
                         "zero_work", "wrong_capacity", "gate", "decode_count", "false_time",
                         "no_state", "missing", "nonfinite", "tiny_time", "boolean_prompt"):
            with self.subTest(scenario=scenario):
                process, record, _ = self.run_campaign(scenario)
                self.assertEqual(process.returncode, 1, process.stderr + process.stdout)
                self.assertEqual(record["status"], "failed")
                self.assertTrue(record["validation_errors"])
                self.assertIsNone(record["summary"])
                self.assertTrue(record["runs"])

    def test_unavailable_and_failures_keep_raw_logs(self):
        for scenario, status, code in (("unavailable", "unavailable", 77), ("nonzero", "failed", 1)):
            with self.subTest(scenario=scenario):
                process, record, output = self.run_campaign(scenario)
                self.assertEqual(process.returncode, code)
                self.assertEqual(record["status"], status)
                self.assertEqual(len(record["runs"]), 1)
                self.assertTrue((output / record["runs"][0]["execution"]["stderr"]).read_text())

    def test_native_gate_must_run_and_pass_before_any_engine_arm(self):
        for scenario in ("native_failure", "native_skipped", "native_mismatch", "native_duplicate", "native_unavailable"):
            with self.subTest(scenario=scenario):
                process, record, output = self.run_campaign(scenario)
                unavailable = scenario == "native_unavailable"
                self.assertEqual(process.returncode, 77 if unavailable else 1)
                self.assertEqual(record["status"], "unavailable" if unavailable else "failed")
                self.assertEqual(record["runs"], [])
                self.assertIsNone(record["summary"])
                self.assertTrue(record["validation_errors"])
                self.assertTrue((output / record["native_model_test"]["execution"]["stdout"]).is_file())

    def test_native_fixture_scope_is_required(self):
        for scenario in ("native_wrong_fixture", "native_missing_fixture"):
            with self.subTest(scenario=scenario):
                process, record, output = self.run_campaign(scenario)
                self.assertEqual(process.returncode, 1, process.stderr + process.stdout)
                self.assertEqual(record["status"], "failed")
                self.assertEqual(record["runs"], [])
                self.assertIsNone(record["summary"])
                self.assertTrue(any("fixture" in error for error in record["validation_errors"]))
                self.assertTrue((output / record["native_model_test"]["execution"]["stdout"]).is_file())

    def test_timeout_is_failure_with_partial_record(self):
        process, record, _ = self.run_campaign("timeout", extra=("--timeout-seconds", "0.05"))
        self.assertEqual(process.returncode, 1)
        self.assertEqual(record["status"], "failed")
        # Under load, the same short deadline may expire in the mandatory
        # native gate before the engine arm starts. Either must retain its log.
        execution = (record["runs"][0] if record["runs"] else record["native_model_test"])["execution"]
        self.assertTrue(execution["timed_out"])
        self.assertEqual(execution["exit_code"], 124)

    def test_asset_mutation_and_hash_mismatch_fail(self):
        process, record, _ = self.run_campaign("mutate_model")
        self.assertEqual(process.returncode, 1)
        self.assertTrue(any("metadata changed" in error for error in record["validation_errors"]))
        process, record, _ = self.run_campaign(extra=("--model-sha256", "0" * 64))
        self.assertEqual(process.returncode, 1)
        self.assertEqual(record["runs"], [])
        self.assertTrue(any("SHA256 mismatch" in error for error in record["validation_errors"]))

    def test_workloads_that_cannot_exercise_the_adapter_are_rejected(self):
        for extra, message in ((("--concurrency", "8"), "leave no draft budget"),
                               (("--concurrency", "1,2,4,8"), "leave no draft budget"),
                               (("--generate", "2"), "adapter can run"),
                               (("--layers", "3"), "target hidden capture"),
                               (("--capacity", "130"), "64 MiB")):
            with self.subTest(extra=extra):
                process, record, _ = self.run_campaign(extra=extra)
                self.assertNotEqual(process.returncode, 0)
                self.assertIsNone(record)
                self.assertIn(message, process.stderr)

    def test_truncated_model_is_explicitly_diagnostic(self):
        process, record, _ = self.run_campaign(extra=("--layers", "4"))
        self.assertEqual(process.returncode, 0, process.stderr + process.stdout)
        self.assertEqual(record["model_scope"], "truncated_diagnostic")
        self.assertEqual(record["configuration"]["layers"], 4)

    def test_environment_overlay_is_pinned_and_uninventoried_packs_are_rejected(self):
        overlay = self.directory / "overlay-fixture"
        overlay.write_bytes(b"synthetic overlay asset")
        alias = self.directory / "overlay-link"
        alias.symlink_to(overlay)
        process, record, _ = self.run_campaign(environment_overrides={"STRIX_DS41_OVERLAY": alias})
        self.assertEqual(process.returncode, 0, process.stderr + process.stdout)
        self.assertEqual(record["environment"]["reported"]["STRIX_DS41_OVERLAY"], str(overlay.resolve()))
        matched = [asset for asset in record["inputs"] if asset["path"] == str(overlay.resolve())]
        self.assertEqual(len(matched), 1)
        self.assertEqual(len(matched[0]["sha256"]), 64)
        process, record, _ = self.run_campaign("mutate_overlay", environment_overrides={"STRIX_DS41_OVERLAY": alias})
        self.assertEqual(process.returncode, 1)
        self.assertEqual(record["status"], "failed")
        self.assertTrue(any("metadata changed" in error and str(overlay.resolve()) in error
                            for error in record["validation_errors"]))
        for name in ("STRIX_DS41_TRELLIS", "STRIX_DS41_TRELLIS_MIX"):
            with self.subTest(name=name):
                process, record, _ = self.run_campaign(environment_overrides={name: self.directory})
                self.assertEqual(process.returncode, 1)
                self.assertEqual(record["runs"], [])
                self.assertTrue(any(name in error and "input inventory" in error for error in record["validation_errors"]))

    def test_trace_is_qualification_only_and_replayed(self):
        process, record, output = self.run_campaign(extra=("--trace",))
        self.assertEqual(process.returncode, 0, process.stderr + process.stdout)
        for run in record["runs"]:
            if run["phase"] == "qualification":
                self.assertTrue((output / run["trace"]["path"]).is_file())
                if run["variant"] == "cached":
                    self.assertEqual(run["trace"]["analysis"]["reported_cache_activity"]["hits"], 8)
            else:
                self.assertNotIn("trace", run)
        process, record, _ = self.run_campaign("trace_hit", extra=("--trace",))
        self.assertEqual(process.returncode, 1)
        self.assertEqual(record["status"], "failed")
        self.assertTrue(any("reported hit differs" in error for error in record["validation_errors"]))

    def test_existing_directory_is_preserved(self):
        output = self.directory / "existing"
        output.mkdir()
        marker = output / "keep"
        marker.write_text("keep")
        process, record, _ = self.run_campaign(extra=("--output", str(output)))
        self.assertEqual(process.returncode, 1)
        self.assertIsNone(record)
        self.assertEqual(marker.read_text(), "keep")


if __name__ == "__main__":
    unittest.main()
