# Next Action Plan: Dual Track

This is the recommended next plan if resuming from a cold session.

The plan now has **two simultaneous tracks**:

1. fix the active model-load failure
2. investigate and reduce the transmission / weight-load delay

## 1. Why this plan is first

Do not begin with another blind long run.

Reason:

- the quick checks already show the basic mediated path is working
- current evidence points to a later failure in GGML CUDA MMQ / graph reserve
- recent runs also show that the guest is not using the intended shared-memory fast path
- the best-effort first move is to fix the MMQ / graph blocker while, in parallel, proving why the load path is slow

Practical goal:

- get one successful Phase 1 response even if it uses a less optimized CUDA path
- do not lose sight of the load-performance problem while pursuing the first Phase 1 proof

## 2. Track A: model-load blocker

Target: remove or reduce the MMQ / graph-reserve path as the active blocker.

Most promising immediate angles:

1. confirm whether the deployed `libggml-cuda.so` was built with `GGML_CUDA_GRAPHS=OFF`
2. if not, rebuild/deploy with graphs off using the existing Hopper build path
3. if a runtime switch exists and is easy to apply, force a non-MMQ / non-graph path for one bounded retest

Why this is justified:

- `build_libggml_cuda_hopper_docker.sh` explicitly says it sets `GGML_CUDA_GRAPHS=OFF` to avoid E5
- the latest long-run log still showed `mmq_x_best=0`
- `ggml_backend_cuda_graph_reserve` appears in the failing stack

## 3. Track B: transmission / load-performance

Target: explain why weight transfer is slow and move the system toward the intended fast path.

Most important current facts:

1. recent long runs show `mmap shmem ... failed` and `using BAR1`
2. `cuda_transport_call()` serializes calls under a global mutex
3. mediator and executor currently process CUDA calls in a blocking, effectively single-flight way

Immediate questions to answer:

1. why does shared-memory registration fail on the VM (`mlock`, GPA resolution, privileges, sandbox, or `pagemap` access)?
2. after shared memory is active, what serialization remains in guest transport, vgpu-stub, mediator, and executor?
3. which improvement is highest value first:
   - make `shmem` reliable
   - reduce per-copy round-trips
   - batch or stream HtoD

Output expected from this track:

- one short documented explanation of the current slow path
- one ordered list of fixes needed to make model load materially faster

## 4. Ordered steps

### Step 0: quick baseline before touching anything

Re-run the short checks:

- VM:
  - `systemctl is-active ollama`
  - `journalctl -u ollama -b --no-pager | grep 'inference compute' | tail -3`
- Host:
  - `grep 'module-load' /tmp/mediator.log | tail -20`
- Preflight:
  - `bash phase3/run_preflight_gemm_ex_vm.sh`

Expected:

- `active`
- `library=CUDA`
- `compute=9.0`
- no new E1/E4 surprise
- preflight still passes
- transport path is known (`shmem` or `BAR1`) for the current run

### Step 1: verify the intended graph-related build policy

Check these files first:

- `build_libggml_cuda_hopper_docker.sh`
- `BUILD_LIBGGML_CUDA_HOPPER.md`
- `BUILD_AND_DEPLOY_LIBGGML_CUDA_PHASE3.md`

Important known fact:

- `build_libggml_cuda_hopper_docker.sh` contains `-DGGML_CUDA_GRAPHS=OFF`

Question to answer before rebuilding:

- is the currently deployed `libggml-cuda.so` definitely from that build path, or only known to contain `sm_90`?

If the answer is "unknown", treat that as insufficient and rebuild/deploy a known-good graphs-off artifact.

### Step 2: rebuild and deploy the graphs-off Hopper library

Use the existing documented path instead of inventing a new one.

Primary intent:

- keep `sm_90`
- make the deployed library definitely correspond to the anti-E5 build settings

After deploy, verify:

- file timestamp changed as expected
- `strings /usr/local/lib/ollama/cuda_v12/libggml-cuda.so | grep sm_90`
- Ollama still discovers GPU mode correctly after restart

### Step 3: do one bounded full-load retest

Only after Step 2.

This is the first longer experiment that is worth the time.

Use a bounded run, not an unbounded wait:

- prefer the existing bounded trace / monitoring scripts over ad hoc curls
- keep host and VM log correlation
- stop if the same MMQ signature appears again

What success looks like:

- no `mmq_x_best=0`
- no `mmq.cuh:3884`
- no abort in `ggml_backend_cuda_graph_reserve`
- ideally a real HTTP 200 response

### Step 4: in parallel, verify the transmission path

Before or during the bounded retest, collect:

1. VM evidence for `shmem registered` vs `using BAR1`
2. host evidence for `HtoD progress`
3. whether progress cadence implies one blocking copy at a time

If the run is still `BAR1`, fix that before claiming the transport design is performing as intended.

### Step 5: if the same failure remains

Then the next branch is:

1. capture a better crash artifact
2. verify whether MMQ itself, not just graphs, must be bypassed
3. consider forcing a path that relies on `cublasGemmEx` instead of the failing quantized MMQ path

At that point, the question is no longer "is the pipeline alive?" but "which exact GGML CUDA execution path must be disabled or replaced for Phase 1 proof?"

## 5. What not to do first

Do not start with:

- another blind 2h-4h run before the graph/MMQ build question is answered
- another broad reread of the whole `phase3` tree
- another E1 deep dive unless current logs re-show `401312` / `INVALID_IMAGE`
- treating the transmission problem as "later"
- claiming shared memory is active without live proof

## 6. Decision tree

If preflight fails:

- stop and re-check the mediated CUDA/cuBLAS path first

If preflight passes and full load still dies with MMQ:

- stay focused on E5
- but continue the transport/performance track in parallel

If the run is still using `BAR1`:

- prioritize making `shmem` registration succeed

If `shmem` becomes active but load is still too slow:

- focus next on serialization and per-call round-trip reduction

If full load no longer dies with MMQ but dies elsewhere:

- update the active blocker classification
- then branch to the new error with the same checkpoint discipline

## 7. The single best next move

If resuming under time pressure, do this first:

- rebuild/deploy the Hopper `libggml-cuda.so` from the existing graphs-off path
- run one bounded full-load retest
- at the same time, record whether the run is on `shmem` or `BAR1`

That is currently the highest-value combined next experiment.
