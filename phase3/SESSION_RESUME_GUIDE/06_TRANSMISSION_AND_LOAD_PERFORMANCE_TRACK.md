# Transmission and Load-Performance Track

This file makes the weight-transfer problem an explicit Phase 3 workstream.

## 1. Why this track exists

The project was designed to support a fast shared-memory data path, but recent live runs still show very slow model load.

That is not a cosmetic issue.

If Phase 3 only reaches "eventually loads, then crashes later," it is still not a usable customer path.

## 2. Current live evidence

Recent bounded / long runs show:

- guest transport falling back to `BAR1`
- messages like `mmap shmem ... failed` and `Cannot resolve GPA for shmem`
- long stretches of `poll call_id=0x0032`
- host `HtoD progress` moving slowly over long wall-clock windows

This means the intended fast path is not currently proven to be active.

## 3. Working hypothesis

There are at least two separate performance problems:

1. **Fast path not active**
   - shared-memory registration is failing
   - the guest uses `BAR1` fallback instead
2. **Architecture still serialized**
   - guest transport uses a blocking RPC model
   - `cuda_transport_call()` is mutex-serialized
   - mediator handles CUDA calls synchronously
   - executor holds a global mutex during each CUDA call

So even after `shmem` is fixed, load time may still remain too slow until serialization is reduced.

## 4. Required questions

Any assistant continuing Phase 3 should answer these in order:

1. Why exactly does `shmem` registration fail on the VM?
2. What privileges or service settings are missing (`mlock`, `pagemap`, sandbox, capabilities, limits)?
3. After `shmem` is active, what serialization remains in:
   - guest shim / transport
   - vgpu-stub
   - mediator
   - host executor
4. Which improvement gives the largest load-time reduction first?

## 5. Minimum evidence to collect

For any serious model-load run, collect:

- VM line proving `shmem` or `BAR1`
- host line proving `HtoD progress`
- rough wall-clock pace of progress
- whether the path is still one blocking transfer / response at a time

Do not describe the transmission path as "fast shared memory" unless the run proves it.

## 6. Near-term action priority

Priority order for this track:

1. make `shmem` registration succeed reliably
2. measure whether that materially improves load time
3. if still too slow, remove serialization bottlenecks in the call path

## 7. Relationship to the error-tracing track

This performance track does **not** replace model-load error tracing.

Both must continue in parallel:

- Track A: remove the current load / MMQ / runner failure
- Track B: make model load fast enough to be a real product path
