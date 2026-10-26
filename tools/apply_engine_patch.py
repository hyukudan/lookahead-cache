#!/usr/bin/env python3
"""Check, apply or reverse the private adapter against its exact engine revision.

This changes only the local checkout. It never commits, pushes, downloads a model
or changes a repository remote. Unrelated working-tree files are left alone.
"""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys

from evidence import load_payload


PACKAGE = Path(__file__).resolve().parent.parent / "integrations/engine_lab"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(checkout, *arguments):
    result = subprocess.run(["git", "-C", str(checkout), *arguments], text=True,
                            capture_output=True, check=False)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip()
                           or f"git command failed with code {result.returncode}")
    return result.stdout.strip()


def hexadecimal(value, length):
    return isinstance(value, str) and len(value) == length and all(c in "0123456789abcdef" for c in value)


def safe_file(root, name):
    if not isinstance(name, str):
        raise ValueError("manifest paths must be strings")
    path = PurePosixPath(name)
    if (path.is_absolute() or not path.parts or str(path) != name
            or any(part in (".", "..", ".git") for part in path.parts)):
        raise ValueError(f"unsafe manifest path: {name}")
    current = root
    for part in path.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"symlinks are not accepted in adapter paths: {name}")
    if not current.is_file():
        raise ValueError(f"expected existing regular file: {name}")
    if not current.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"path escapes its checkout: {name}")
    return current


def load_manifest(package):
    manifest = load_payload((package / "manifest.json").read_text(encoding="utf-8"))
    if type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1:
        raise ValueError("unsupported integration manifest")
    if not hexadecimal(manifest.get("engine_base_commit"), 40):
        raise ValueError("manifest must bind an exact engine commit")
    if not hexadecimal(manifest.get("patch_sha256"), 64):
        raise ValueError("manifest must bind the patch contents")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("manifest has no adapter files")
    seen = set()
    for entry in files:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise ValueError("invalid manifest entry")
        if entry["path"] in seen:
            raise ValueError("duplicate adapter file")
        seen.add(entry["path"])
        if any(not hexadecimal(entry.get(key), 64) for key in ("before_sha256", "after_sha256")):
            raise ValueError("adapter file must have before/after fingerprints")
    patch = safe_file(package, manifest.get("patch_file"))
    if sha256(patch) != manifest["patch_sha256"]:
        raise ValueError("adapter patch differs from the manifest")
    return manifest, patch


def operate(checkout, action="check", package=PACKAGE):
    if action not in ("check", "apply", "reverse"):
        raise ValueError("unknown patch action")
    checkout = checkout.resolve(strict=True)
    manifest, patch = load_manifest(package.resolve(strict=True))
    root = Path(git(checkout, "rev-parse", "--show-toplevel")).resolve(strict=True)
    if root != checkout:
        raise ValueError("--checkout must name the engine repository root")
    head = git(checkout, "rev-parse", "HEAD")
    if head != manifest["engine_base_commit"]:
        raise ValueError(f"engine revision differs from the supported commit {manifest['engine_base_commit']}")
    paths = [entry["path"] for entry in manifest["files"]]
    for name in paths:
        safe_file(checkout, name)
    changed = git(checkout, "apply", "--numstat", "-z", "--", str(patch))
    patch_paths = []
    for row in changed.split("\0"):
        if not row:
            continue
        columns = row.split("\t", 2)
        if len(columns) != 3 or not columns[0].isdigit() or not columns[1].isdigit():
            raise ValueError("adapter must contain ordinary text file changes")
        patch_paths.append(columns[2])
    if len(patch_paths) != len(paths) or set(patch_paths) != set(paths):
        raise ValueError("patch paths differ from the integration manifest")
    if git(checkout, "diff", "--cached", "--name-only", "--", *paths):
        raise ValueError("adapter paths have staged changes; use a clean isolated checkout")
    before_key = "after_sha256" if action == "reverse" else "before_sha256"
    after_key = "before_sha256" if action == "reverse" else "after_sha256"
    for entry in manifest["files"]:
        path = safe_file(checkout, entry["path"])
        if sha256(path) != entry[before_key]:
            raise ValueError(f"{entry['path']} differs from the required {before_key.removesuffix('_sha256')} contents")
    options = ["--reverse"] if action == "reverse" else []
    # git apply checks all hunks before writing. Do not use --3way, --reject or
    # --unsafe-paths: a partially applicable adapter must never be accepted.
    git(checkout, "apply", "--check", "--whitespace=error-all", *options, "--", str(patch))
    if action != "check":
        git(checkout, "apply", "--whitespace=error-all", *options, "--", str(patch))
        for entry in manifest["files"]:
            if sha256(safe_file(checkout, entry["path"])) != entry[after_key]:
                raise RuntimeError(f"post-apply fingerprint mismatch: {entry['path']}; inspect the checkout")
    return {"status": "checked" if action == "check" else "reversed" if action == "reverse" else "applied",
            "engine_commit": head, "patch_sha256": manifest["patch_sha256"],
            "files": paths, "native_execution": False, "remote_modified": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", required=True, type=Path)
    parser.add_argument("--action", choices=("check", "apply", "reverse"), default="check")
    args = parser.parse_args()
    print(json.dumps(operate(args.checkout, args.action), indent=2))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError) as error:
        print(f"private engine adapter: {error}", file=sys.stderr)
        sys.exit(1)
