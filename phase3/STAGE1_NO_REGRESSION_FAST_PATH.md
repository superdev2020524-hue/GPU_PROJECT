# Stage 1 No-Regression Fast Path

This document exists for one purpose: finish the Stage 1 milestone quickly without reopening already-solved branches.

## Stage 1 finish line

Stage 1 is complete only when all three are proven on the same live path:

1. Deterministic correctness: the deterministic prompt bundle returns correct, parseable answers.
2. Speed: cold and warm runs meet the agreed targets.
3. Residency: `keep_alive` keeps the model resident and explicit unload clears it.

## Problems that have repeatedly wasted time

1. Artifact drift: source changed, but the live loaded binary or `.so` was not the rebuilt one.
2. Split deployment state: host stub, mediator, guest shim, Ollama binary, and Ollama shared libraries were not all on the intended version at the same time.
3. Restart-order mistakes: mediator restart alone, or service restart alone, was treated as enough when a full VM reboot or QEMU rebuild was actually required.
4. Mixed-layer edits: changing host, guest, and Ollama behavior in the same cycle made causality unclear.
5. Dirty runtime evidence: old logs, old `vgpu_current_call.txt`, old stage files, or old runner state contaminated new conclusions.
6. False promotion of candidates: a new earlier failure was treated like the root problem before proving whether it was only a temporary regression.
7. Wrong live artifact assumptions: top-level Ollama rebuilds did not update the loaded CUDA backend shared objects.
8. Path confusion: a previously solved transport/bootstrap branch reappeared and masked the deeper live blocker.
9. Acceptance too early: the milestone gate was rerun before the active live blocker was actually cleared.
10. Missing baseline control: there was no single mandatory preflight proving the exact host and guest artifact set before each serious repro.

## Non-negotiable rules

1. Keep exactly one active error.
2. Treat every new earlier failure as a candidate until the current active error is closed or disproved.
3. Change only one layer per cycle:
   - host QEMU stub
   - mediator
   - guest shim
   - Ollama binary/shared libraries
   - service/runtime configuration
4. Do not interpret a runtime result until the live artifact path is proven.
5. Use one clean bounded repro payload for comparison until the active error changes.
6. If a run fails earlier than the last proven checkpoint, classify it as a regression and repair the baseline first.
7. Do not widen the search area while an earlier regression is still active.
8. Do not rerun the full Stage 1 gate while the active runtime blocker is still earlier than the milestone path.

## Mandatory baseline proof before any serious repro

Before each bounded repro, record all of the following:

1. Host QEMU/stub state:
   - installed `qemu` RPM version
   - current domid/root socket path
   - mediator running on a truncated `/tmp/mediator.log`
2. Guest shim state:
   - deployed guest shim path
   - rebuilt library install paths
   - if needed, hash of the deployed guest shim artifact
3. Ollama live artifact state:
   - `LD_LIBRARY_PATH`
   - `OLLAMA_LIBRARY_PATH`
   - loaded runner `/proc/<pid>/maps` sample when applicable
4. Clean trace state:
   - clear `vgpu_current_call.txt`
   - clear `vgpu_host_response_verify.log`
   - clear `vgpu_status_poll.log`
   - clear runner stage logs
   - truncate `ollama` stderr log

If any of these are missing, the repro is not baseline-safe.

## Fixed checkpoint ladder

Every repro must be classified against this exact ladder:

1. `0x0001` `cuInit`
2. `0x00f0` GPU info
3. `0x0030` first alloc
4. first `0x003c` HtoD wave
5. `before_LoadModelFromFile`
6. `after_LoadModelFromFile`
7. `before_NewContextWithModel`
8. `before_backend_graph_reserve i=1`
9. `after_backend_graph_reserve i=1`
10. `before_backend_graph_reserve i=2`
11. `after_backend_graph_reserve i=2`
12. response returned
13. `/api/ps` residency published

The active error is always "the earliest checkpoint that no longer advances on the current baseline."

## Fast execution loop

1. Freeze the baseline.
2. Run one bounded repro with the standard trace set.
3. Compare the result to the last proven checkpoint.
4. If the branch regressed earlier, repair only that regression.
5. If the branch stayed on the same checkpoint, instrument only that checkpoint.
6. Apply one change in one layer.
7. Redeploy and prove the live artifact path.
8. Re-run the same bounded repro.
9. Promote the next blocker only if the current checkpoint is clearly passed.

## Current forced direction

Do not reopen solved endpoint/bootstrap debates unless the ladder falls earlier than `0x0001` or `0x00f0`.

The current live direction is:

1. Stay on `P1-E` as the active error.
2. Treat `P1-Q` as the current active-branch gate.
3. Focus on the second CUDA split reserve path.
4. Focus specifically on guest-shim bulk transfer during `cuLibraryLoadData`.
5. Focus specifically on `write_bar1_data_words()` / `do_single_cuda_call()` and the matching handoff path in the guest shim.

## When to rerun the Stage 1 gate

Run `phase1_milestone_gate.py` only after all of the following are true on the current branch:

1. `after_backend_graph_reserve i=2` is observed.
2. One bounded deterministic request returns `HTTP 200`.
3. `/api/ps` shows the model resident after `keep_alive`.

Until then, the gate suite is not the driver; the active runtime checkpoint is.

## What counts as a successful close

Stage 1 can be declared complete only when:

1. Cold deterministic request succeeds correctly.
2. Warm deterministic request succeeds correctly.
3. Accuracy bundle passes.
4. Latency bundle passes.
5. Residency bundle passes.
6. The result is reproduced from a clean baseline, not a one-off lucky run.
