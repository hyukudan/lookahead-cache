# Integration

## Reference cache lifecycle

1. Call `sl_cache_config_init`, set vocabulary size, entry capacity and read-lease
   capacity, then supply identities and an optional payload budget.
2. Create the cache and check the returned status.
3. Acquire by global predecessor ID. `SL_OK` returns a pinned read lease;
   `SL_MISS` means no committed entry exists. A fill in progress returns `SL_BUSY`.
4. On a miss, reserve a slot. Compute the complete vector with the engine's
   existing arithmetic and write every element before committing. Abort on failure.
5. Acquire the committed entry, consume it, then release the read lease.
6. Destroy only after all readers and writers have finished.

`SL_PRESENT` means a committed entry exists. `SL_BUSY` can also indicate exhausted
slots or reader tickets. Use the uncached path when cache capacity is unavailable.

A reservation can evict an unpinned victim immediately. Aborting that reservation
does not restore the victim. Fill only the returned writable range and treat lease
metadata as immutable. Copies represent the same ticket, so complete a lease once.
Do not reuse a pointer after release, commit, abort or destruction.
An acquire/reserve output object must not already contain a live lease. These
operations overwrite it on success and clear it on failure. Release, commit and
abort clear their supplied lease only on success; a failure preserves that object.

## Vocabulary and exactness

Cache only vectors fully determined by the predecessor token and the fixed cache
identity. Context-dependent logits or corrections cannot share this key without
additional identity and invalidation rules; the reference cache does not supply
those rules.

A reduced vocabulary is an ordered list, not just a length. This version requires
a complete vector for that list. To change its mapping, release all leases and reset
with the new identity. Create a new cache when dimensions change. Reset invalidates
entries while preserving cumulative counters.

The engine owns the dot product, logit addition, confidence, tie-breaking,
verification and rollback. Check cache byte preservation separately from
end-to-end token equivalence.

## Planned HIP/CUDA integration

GPU publication must follow completion of the producer kernel. Reader leases must
remain active until consuming kernels complete, not merely until their launches
return. Device completion and failure must be represented explicitly in the backend
contract. The current host API does not provide asynchronous device readiness.

Keep caches local to each rank. An adapter should avoid transporting complete bias
vectors between nodes on every hit. Benchmark the existing distributed execution
mode before changing where proposal selection occurs.
