# Refresh and GPU detection — investigation (Mar 18)

*Investigation requested: Is the current problem the same "GPU not detected after refresh" that was fixed before, or different?*

---

## 1. What was fixed before (and what you remember)

- **Symptom:** During model load, the scheduler calls `GPUDevices()` again to refresh free VRAM. That path used **wrong dir order** in `bootstrapDevices(ctx, []string{ml.LibOllamaPath, dir}, devFilter)` (parent first) → refresh bootstrap failed to see CUDA → **"unable to refresh free memory, using old values"** and/or scheduler got empty/wrong GPU list.
- **Fix:** Patch **refresh path** in `discover/runner.go` (around line 340) to use `[]string{dir, ml.LibOllamaPath}` (GPU lib dir first), same as initial bootstrap. Applied via `transfer_ollama_go_patches.py` / `patch_discover_runner_go()`.
- **"Destroy and restore":** When GPU mode was broken (e.g. wrong CUBLAS shim, wrong paths), the procedure in **RESTORE_GPU_LOGIC_CHECKLIST.md** and **GPU_MODE_DO_NOT_BREAK.md** restores: vgpu.conf (paths, no LD_PRELOAD), remove CUBLAS/CUBLASLt from `/opt/vgpu/lib`, patched ollama.bin (device.go, server.go, discover/runner.go), verify with journalctl. So "restore" = re-apply those pieces; it is the same set of fixes, not a different bug.

---

## 2. Current VM state (verified)

| Check | Result |
|-------|--------|
| Ollama running | `active` |
| **GPU mode** | **Yes.** Journal: `inference compute ... library=CUDA ... "NVIDIA H100 80GB HBM3"`, `total="80.0 GiB"`, `total_vram="80.0 GiB"`. No "filtering device which didn't fully initialize", no library=cpu. |
| Initial discovery | Correct (CUDA, 80 GiB). |
| **Refresh message** | **Still present:** `msg="unable to refresh free memory, using old values"` (e.g. when a generate runner starts). |
| **discover/runner.go (VM)** | **Refresh patch is in source:** line 340 area has `bootstrapDevices(ctx, []string{dir, ml.LibOllamaPath}, devFilter)` (dir first). Initial bootstrap line 109: `dirs = []string{dir, ml.LibOllamaPath}`. |
| ml/device.go (VM) | `NeedsInitValidation()` returns `d.Library == "ROCm"` only (CUDA skipped). |
| Binary in use | Built Mar 17 13:00 from `/home/test-4/ollama/ollama.bin`; source on disk has the refresh patch. |

---

## 3. Is this the same problem or different?

- **Same symptom:** We still see **"unable to refresh free memory, using old values"** when the generate runner starts.
- **Same fix is already in source:** The **refresh dir order** (`[]string{dir, ml.LibOllamaPath}`) is present in the VM’s `discover/runner.go` at the refresh path. So the original “wrong order → CUDA not found at refresh” fix **is** applied in the current source.
- **So either:**
  1. **Binary not built from this source:** The running binary might have been built before the refresh patch was applied (or from a different tree). Then we’d still get the warning even though source is correct. **Action:** Rebuild `ollama.bin` on the VM from the current patched source, reinstall, restart, and trigger a generate; see if "unable to refresh free memory" disappears.
  2. **Refresh fails for a different reason:** Even with correct dir order, the refresh bootstrap can fail (e.g. refresh runner timeout, `cuMemGetInfo` failure in the refresh context, or refresh runner not getting `/opt/vgpu/lib` so it doesn’t see the GPU). Then the warning would persist. **Action:** If rebuild doesn’t remove the warning, treat it as “refresh still failing for another reason” and dig into refresh-runner env, timeouts, and refresh-path CUDA calls (see ERROR_TRACKING_STATUS.md §5).

---

## 4. Relation to “no alloc/HtoD” and past mistakes

- **No alloc/HtoD:** In recent runs, the generate runner never reaches `cuMemAlloc` (0x0030) or `cuMemcpyHtoD_v2` (0x0032); we only see the same 6 init/context RPCs. Runner **does** get `LD_LIBRARY_PATH=/opt/vgpu/lib:...` (verified via server log). So the blocker is either: (a) scheduler skips GPU for this load because refresh failed (empty/wrong list), so runner uses CPU and never does alloc/HtoD, or (b) something else before the first alloc (e.g. different code path, or blocking elsewhere).
- **Past mistake (not checking GPU mode):** We have previously proceeded without confirming that Ollama was actually in GPU mode. **Now:** We confirmed **Ollama is in GPU mode** (library=CUDA, 80 GiB in journal). So the current investigation is from a correct baseline: GPU mode is on; the open issue is refresh warning + load path not reaching alloc/HtoD.

---

## 5. Rebuild and retest (Mar 18) — done

- **Done:** Rebuilt `ollama.bin` on VM from current source (discover/runner.go has refresh patch at line 340), installed to `/usr/local/bin/ollama.bin.real`, restarted ollama. Triggered generate (tinyllama, 70s timeout).
- **Result:** **"unable to refresh free memory, using old values" still appears** in journal when the generate runner starts. Inference compute shows library=CUDA, 80 GiB; runner gets LD_LIBRARY_PATH with /opt/vgpu/lib first. No SUBMIT 0x0030/0x0032 in verify log.
- **Conclusion:** Refresh is failing for a **reason other than** dir order (patch is in the binary). Next: investigate refresh-runner env, timeout, or cuMemGetInfo in refresh path (ERROR_TRACKING_STATUS.md §5).

## 6. Recommended next steps (if continuing)

1. ~~Rebuild from current VM source and re-test refresh~~ **Done; warning persists.**

2. **If refresh warning persists**  
   Treat as “refresh failing despite correct dir order”: add/inspect logging around the refresh path (e.g. does the refresh bootstrap runner get `/opt/vgpu/lib`? does it return devices? does `cuMemGetInfo` run and succeed?). See ERROR_TRACKING_STATUS.md §5 (OLLAMA_DEBUG=1, load path, scheduler using “old values”).

3. **Confirm load path and alloc/HtoD**  
   Run with **OLLAMA_DEBUG=1** and capture whether the load path uses GPU or CPU and where it stops; check for SUBMIT 0x0030/0x0032 in verify log after any fix.

---

## 7. Short answers to your questions

- **Are we currently unable to detect the GPU after refreshing?**  
  **Initial** detection is fine (GPU mode, CUDA, 80 GiB). The **refresh** step (during model load) still logs "unable to refresh free memory, using old values", so **refresh** is still failing.

- **Is this the same problem you remember?**  
  The **symptom** is the same (refresh warning). The **fix we applied** (dir order in refresh path) **is** in the VM source. So either the running binary doesn’t have that fix (rebuild needed), or the same symptom now has a **different cause** (refresh runner env, timeout, or CUDA call in refresh path).

- **Destroy/restore:** That refers to **RESTORE_GPU_LOGIC_CHECKLIST.md**: re-apply vgpu.conf, remove CUBLAS shims, patched binary, etc. It’s the same set of fixes; not a different bug. After “restore,” GPU mode and discovery work; refresh may still warn if the binary wasn’t rebuilt with the refresh patch or if refresh fails for another reason.

- **Proceeding:** We **did** confirm Ollama is in GPU mode before drawing conclusions. Next is: rebuild from current source and re-check refresh; then, if needed, investigate why refresh still fails and why the load path doesn’t reach alloc/HtoD.
