#!/usr/bin/env python3
"""Keep trace replay exact across contexts, invalidation and missing identity."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
SPEC = importlib.util.spec_from_file_location("trace_analyzer", ROOT / "tools/analyze_predecessor_trace.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def draft(context=1, epoch=4, vocabulary=1024, predecessors=None, hits=None, mode="cached", capacity=16):
    return {"event": "draft", "context_id": context, "vocabulary_epoch": epoch,
            "vocabulary_size": vocabulary, "predecessors": predecessors or [0, 0, 1, 1, 0],
            "cache_hits": hits if hits is not None else [0, 1, 0, 1, 0],
            "capacity": capacity, "mode": mode}


class TraceTests(unittest.TestCase):
    def test_hash_vectors_and_zero_capacity(self):
        keys = (0, 1, 2, 3, 127, 65535, 129279, 4294967294)
        self.assertEqual([MODULE.slot_for(k, 16) for k in keys], [0, 0, 1, 13, 15, 10, 13, 10])
        self.assertEqual([MODULE.slot_for(k, 31) for k in keys], [0, 27, 24, 20, 3, 3, 2, 24])
        self.assertIsNone(MODULE.slot_for(0, 0))

    def test_context_isolation_and_epoch_invalidation(self):
        records = [draft(),
                   draft(context=2, predecessors=[0] * 5, hits=[0, 1, 1, 1, 1]),
                   draft(predecessors=[0, 2, 2, 0, 1], hits=[1, 0, 1, 1, 0]),
                   draft(epoch=5, vocabulary=1030)]
        result = MODULE.analyze(records, [0, 16, 31])
        self.assertEqual(result["epoch_transitions"], 1)
        self.assertEqual(result["contexts"], 2)
        self.assertEqual(result["cache_eligible_steps"], 20)
        self.assertEqual(result["reported_cache_activity"]["hits"], 11)
        self.assertEqual(result["reported_cache_activity"]["misses"], 9)
        baseline, actual, alternative = result["capacity_simulations"]
        self.assertEqual((baseline["simulated_hits"], baseline["simulated_misses"]), (0, 20))
        self.assertEqual((actual["simulated_hits"], actual["simulated_misses"]), (11, 9))
        self.assertGreater(alternative["simulated_hits"], actual["simulated_hits"])
        self.assertEqual(actual["payload_bytes_if_all_contexts_retained_at_peak_sizes"], (1030 + 1024) * 16 * 4)
        self.assertFalse(result["hardware_execution_verified"])
        self.assertFalse(result["latency_or_speedup_estimated"])

    def test_fallback_does_not_fill_cache_and_uncached_is_explicit(self):
        result = MODULE.analyze([draft(mode="fallback", hits=[0] * 5), draft()], [16])
        self.assertEqual(result["skipped_baseline_or_fallback_drafts"], 1)
        self.assertEqual(result["reported_cache_activity"]["hits"], 2)
        uncached = MODULE.analyze([draft(mode="uncached", capacity=0, hits=[0] * 5)], [0, 16])
        self.assertEqual(uncached["reported_cache_activity"]["hits"], 0)
        self.assertEqual(uncached["capacity_simulations"][1]["simulated_hits"], 2)

    def test_reported_hits_and_changed_identity_fail_closed(self):
        invalid = [
            [draft(hits=[1, 1, 0, 1, 0])],
            [draft(), draft(vocabulary=1025)],
            [draft(), draft(capacity=32)],
            [draft(), draft(epoch=3)],
            [draft(mode="uncached", capacity=16)],
            [draft(mode="cached", capacity=0)],
            [draft(hits=[0, 1.0, 0, 1, 0])],
            [draft(mode="fallback", hits=[1] * 5)],
            [draft(mode="fallback", hits=[0]), draft()],
            [draft(context=True)],
            [draft(predecessors=[0, 0, 4294967295, 1, 0])],
        ]
        for records in invalid:
            with self.subTest(records=records), self.assertRaises(ValueError):
                MODULE.analyze(records, [16])

    def test_legacy_requires_opt_in_and_uses_all_proposals(self):
        # Only one proposal was verified, but all five Markov steps executed.
        record = {"p": 0, "pos": 0, "ctx": 4096, "t": 0, "d": [0, 1, 1, 0, 999], "R": 2, "a": 0}
        with self.assertRaisesRegex(ValueError, "assume-fixed-vocab"):
            MODULE.analyze([record], [16], "legacy")
        result = MODULE.analyze([record], [16], "legacy", True)
        self.assertEqual(result["cache_eligible_steps"], 5)
        self.assertEqual(result["capacity_simulations"][0]["simulated_hits"], 2)
        self.assertIsNone(result["reported_cache_activity"])
        self.assertIsNone(result["capacity_simulations"][0]["payload_bytes_if_all_contexts_retained_at_peak_sizes"])
        # Same ctx length cannot alias distinct requests.
        other = dict(record, p=1)
        isolated = MODULE.analyze([record, other], [16], "legacy", True)
        self.assertEqual(isolated["capacity_simulations"][0]["simulated_hits"], 4)
        with self.assertRaisesRegex(ValueError, "positions must increase"):
            MODULE.analyze([record, record], [16], "legacy", True)
        with self.assertRaisesRegex(ValueError, "reused after completion"):
            MODULE.analyze([record, {"p": 0, "done": 1}, dict(record, pos=1)], [16], "legacy", True)

    def test_cli_strict_json_provenance_and_exclusive_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "trace.jsonl", root / "report.json"
            command = [sys.executable, str(ROOT / "tools/analyze_predecessor_trace.py"),
                       "--input", str(source), "--output", str(output)]
            source.write_text(json.dumps(draft()) + "\n", encoding="utf-8")
            process = subprocess.run(command, text=True, capture_output=True, check=False)
            self.assertEqual(process.returncode, 0, process.stderr)
            report = json.loads(output.read_text())
            self.assertEqual(len(report["input_sha256"]), 64)
            self.assertEqual(len(report["analyzer_sha256"]), 64)
            saved = output.read_bytes()
            process = subprocess.run(command, text=True, capture_output=True, check=False)
            self.assertNotEqual(process.returncode, 0)
            self.assertEqual(output.read_bytes(), saved)
            output.unlink()
            valid = json.dumps(draft())
            for malformed in (valid[:-1] + ', "extra": 1e999}',
                              valid[:-1] + ', "extra": NaN}',
                              valid[:-1] + ', "capacity": 16}', "[]", "{", ""):
                with self.subTest(malformed=malformed):
                    source.write_text(malformed + "\n", encoding="utf-8")
                    process = subprocess.run(command, text=True, capture_output=True, check=False)
                    self.assertNotEqual(process.returncode, 0, process.stdout)
                    self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
