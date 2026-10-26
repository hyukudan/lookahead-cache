# CPU component measurements

These are project-owned synthetic measurements of the standalone CPU rank-256
pair operator against the same implementation with cache capacity zero. They
contain no model assets, external-engine results or physical GPU measurements.

See [the results and methodology](../../docs/results.md) before interpreting
ratios. The [campaign summary](2026-10-26-summary.json) gives the aggregation
rules, hardware/compiler settings, source and executable hashes, and SHA256
values for every collector record below.

| Candidates | Cache entries | Independent process records |
|---:|---:|---|
| 32,768 | 16 | [Run 1](2026-10-26-v32768-c16-run1.json), [run 2](2026-10-26-v32768-c16-run2.json), [run 3](2026-10-26-v32768-c16-run3.json) |
| 32,768 | 128 | [Run 1](2026-10-26-v32768-c128-run1.json), [run 2](2026-10-26-v32768-c128-run2.json), [run 3](2026-10-26-v32768-c128-run3.json) |
| 129,280 | 16 | [Run 1](2026-10-26-v129280-c16-run1.json), [run 2](2026-10-26-v129280-c16-run2.json), [run 3](2026-10-26-v129280-c16-run3.json) |

Each record is unmodified collector JSON. It retains the executable's stdout
and stderr, its parsed timing/count aggregates, validation status, and before/
after source and executable fingerprints. All nine measurements passed.
Individual per-call duration samples are not emitted by the benchmark; do not
treat the aggregate p50/p95 values as an available sample trace.

No absolute local filesystem paths, hostname, model files or executables are included.
The preflight and collector console logs remain in the ignored research build
directory; the nine measured records and their numerical results are preserved
here in full. No outlier or slower run was removed.
