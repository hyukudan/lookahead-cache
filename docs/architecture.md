# Architecture

## One contract, hardware-specific implementations

HIP is the first device implementation target. CUDA should reuse cache identity
and lifetime semantics with its own allocation, streams and events. Host pointers
must never be reinterpreted as portable device buffers.

| Layer | Responsibility |
| --- | --- |
| Reference core | Entry identity, leases, eviction, publication and statistics |
| Planned device backend | Device storage, lookup and completion ordering |
| Model adapter | Vocabulary mapping, predecessor IDs and original bias calculation |
| Inference engine | Logits, selection, target verification and state commit/rollback |

The current cache preallocates payload and metadata. Lookup, reservation and release
must not allocate. The payload budget excludes metadata and allocator overhead;
it is not a total process-memory limit.

## Identity and ownership

The caller supplies three opaque 32-byte identities: weights, numerical profile,
and ordered vocabulary mapping. The library cannot verify them against model files.
An entry key is a global predecessor token ID; vector positions follow the fixed
ordered vocabulary. A numerical identity must cover precision and reduction order.

Readers pin committed entries. Writers reserve unpublished slots. Publication
requires completion of the entire fill. Active readers and writers prevent unsafe
reuse. Each acquired lease has an individual ticket; copying it does not create
another lease.

Strix Halo's shared memory does not remove producer/consumer ordering or memory
bandwidth contention. A future HIP backend must keep lookup and useful cached data
on the device execution path. Copying the host reference vector for every hit is
not the intended design.
