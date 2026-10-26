# Private HIP engine adapter

This package connects the resident HIP proposal head to StrixLookahead. It is a
local, opt-in integration for the exact engine revision recorded in
[`manifest.json`](manifest.json):
[`683e69e91eb2203addf016e2cfdead8217e65fa2`](https://github.com/hyukudan/strix-halo-engine-lab/tree/683e69e91eb2203addf016e2cfdead8217e65fa2).

The engine repository is public. This patch and the private library are maintained
in this private repository. Applying the patch does not create a commit, move a
branch, change a remote, or publish any code.

## Apply or reverse

Use an isolated engine checkout at the pinned commit. From the StrixLookahead
repository:

```sh
python3 tools/apply_engine_patch.py --checkout /path/to/engine --action check
python3 tools/apply_engine_patch.py --checkout /path/to/engine --action apply
```

The installer checks the engine commit, patch SHA256, exact file list and the
before/after SHA256 of every changed file. It rejects staged edits on those files,
symlink paths and a partially applicable patch. Unrelated local files are preserved.

To restore the exact original contents, provided the patched files have not been
edited further:

```sh
python3 tools/apply_engine_patch.py --checkout /path/to/engine --action reverse
```

An engine update requires reviewing and regenerating this package against the new
revision; the installer deliberately does not guess how to merge another version.

## Build and qualification

The engine option `STRIX_ENABLE_LOOKAHEAD=ON` requires an installed
`StrixLookahead::hip` target, library version 0.2.1 or newer. Version 0.3.0 supplies
this integration and its campaign tools. The option defaults to off. Runtime
`STRIX_LOOKAHEAD_CAPACITY` chooses the original head when unset, the uncached
library when `0`, or the cache when positive.

See the [complete build and qualification procedure](../../docs/engine-integration.md)
for the installed library, engine build, native model regressions and three-arm
campaign. The integration is HIP-only. The existing CUDA component is separate.

This revision was compiled for `gfx1151`, and the patch was applied and reversed
against an isolated checkout. Execution on a physical GPU with the real model
assets remains a separate qualification gate. Compilation is not numerical or
performance validation on Strix Halo.

## Provenance and license

The patch contains modifications and context from the engine at the pinned
revision. Its Apache License 2.0 is included unchanged as
[`ENGINE_LICENSE`](ENGINE_LICENSE); that license covers the engine-derived patch.
Original engine identifiers, comments and required attribution are retained.
Modified files carry integration notices. This license copy does not relicense the
separate private StrixLookahead implementation.

The manifest fingerprints both the original and resulting engine files. No model
weights, prompts, credentials or external benchmark artifacts are included.
