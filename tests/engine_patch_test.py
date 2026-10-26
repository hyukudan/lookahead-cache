#!/usr/bin/env python3
"""Exercise real git patching without a GPU, model or remote repository."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import apply_engine_patch as adapter


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


class PatchTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.checkout = self.root / "engine"
        self.package = self.root / "integration"
        self.checkout.mkdir()
        self.package.mkdir()
        self.command("init", "-q")
        (self.checkout / "source.cpp").write_text("original\n")
        self.command("add", "source.cpp")
        self.command("-c", "user.name=Adapter Test", "-c", "user.email=adapter-test@example.invalid",
                     "commit", "-qm", "fixture")
        self.head = self.command("rev-parse", "HEAD").strip()
        (self.checkout / "source.cpp").write_text("reviewed\n")
        patch = self.command("diff", "--", "source.cpp")
        (self.checkout / "source.cpp").write_text("original\n")
        (self.package / "adapter.patch").write_text(patch)
        self.manifest = {"schema_version": 1, "engine_base_commit": self.head,
                         "patch_file": "adapter.patch", "patch_sha256": digest(patch),
                         "files": [{"path": "source.cpp", "before_sha256": digest("original\n"),
                                    "after_sha256": digest("reviewed\n")} ]}
        self.save_manifest()

    def tearDown(self):
        self.temporary.cleanup()

    def command(self, *args):
        result = subprocess.run(["git", "-C", str(self.checkout), *args], capture_output=True,
                                text=True, check=True)
        return result.stdout

    def save_manifest(self):
        (self.package / "manifest.json").write_text(json.dumps(self.manifest))

    def run_adapter(self, action="check"):
        return adapter.operate(self.checkout, action, self.package)

    def test_check_apply_reverse_preserve_unrelated_files(self):
        other = self.checkout / "unrelated.txt"
        other.write_text("user work\n")
        self.assertEqual(self.run_adapter()["status"], "checked")
        self.assertEqual((self.checkout / "source.cpp").read_text(), "original\n")
        self.assertEqual(self.run_adapter("apply")["status"], "applied")
        self.assertEqual((self.checkout / "source.cpp").read_text(), "reviewed\n")
        self.assertEqual(self.run_adapter("reverse")["status"], "reversed")
        self.assertEqual((self.checkout / "source.cpp").read_text(), "original\n")
        self.assertEqual(other.read_text(), "user work\n")
        self.assertEqual(self.command("rev-parse", "HEAD").strip(), self.head)

    def test_wrong_commit_or_modified_target_is_rejected(self):
        self.manifest["engine_base_commit"] = "0" * 40
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, "revision differs"):
            self.run_adapter("apply")
        self.manifest["engine_base_commit"] = self.head
        self.save_manifest()
        (self.checkout / "source.cpp").write_text("user change\n")
        with self.assertRaisesRegex(ValueError, "required before"):
            self.run_adapter("apply")
        self.assertEqual((self.checkout / "source.cpp").read_text(), "user change\n")

    def test_changed_patch_and_staged_changes_are_rejected(self):
        patch = self.package / "adapter.patch"
        original = patch.read_text()
        patch.write_text(original + "changed\n")
        with self.assertRaisesRegex(ValueError, "patch differs"):
            self.run_adapter("apply")
        patch.write_text(original)
        (self.checkout / "source.cpp").write_text("staged change\n")
        self.command("add", "source.cpp")
        (self.checkout / "source.cpp").write_text("original\n")
        with self.assertRaisesRegex(ValueError, "staged"):
            self.run_adapter("apply")

    def test_reapplication_or_modified_reverse_is_rejected(self):
        self.run_adapter("apply")
        with self.assertRaisesRegex(ValueError, "required before"):
            self.run_adapter("apply")
        (self.checkout / "source.cpp").write_text("reviewed plus user change\n")
        with self.assertRaisesRegex(ValueError, "required after"):
            self.run_adapter("reverse")

    def test_unlisted_patch_target_is_rejected_before_writing(self):
        patch = self.package / "adapter.patch"
        text = patch.read_text() + ("diff --git a/unlisted.txt b/unlisted.txt\n"
                                   "new file mode 100644\n--- /dev/null\n+++ b/unlisted.txt\n"
                                   "@@ -0,0 +1 @@\n+unexpected\n")
        patch.write_text(text)
        self.manifest["patch_sha256"] = digest(text)
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, "paths differ"):
            self.run_adapter("apply")
        self.assertFalse((self.checkout / "unlisted.txt").exists())
        self.assertEqual((self.checkout / "source.cpp").read_text(), "original\n")

    def test_symlinks_and_path_traversal_are_rejected(self):
        source = self.checkout / "source.cpp"
        source.unlink()
        outside = self.root / "outside.cpp"
        outside.write_text("original\n")
        source.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "symlinks"):
            self.run_adapter("apply")
        self.manifest["files"][0]["path"] = "../outside.cpp"
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, "unsafe"):
            self.run_adapter("apply")
        self.assertEqual(outside.read_text(), "original\n")


if __name__ == "__main__":
    unittest.main()
