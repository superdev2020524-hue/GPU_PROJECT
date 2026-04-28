# Fix "filtering device which didn't fully initialize" — Skip CUDA init validation

*Mar 16, 2026*

## Cause

Ollama’s second-pass “init validation” starts a runner with `CUDA_VISIBLE_DEVICES=<our-vGPU>` and `GGML_CUDA_INIT=1`. That runner exits or crashes before answering the `/info` request, so `bootstrapDevices()` returns 0 devices and the GPU is filtered out with **“filtering device which didn't fully initialize”**.

## Fix

Skip the second-pass validation for CUDA so the device list from the first pass is kept:

- In **`ml/device.go`**, change `NeedsInitValidation()` so it returns `true` only for ROCm, not for CUDA:
  - From: `return d.Library == "ROCm" || d.Library == "CUDA"`
  - To: `return d.Library == "ROCm"`

Then rebuild the Ollama binary and install it on the VM.

## Status on TEST-4

- **Source on VM is already patched.**  
  The patched `device.go`, `server.go`, and `discover/runner.go` were transferred to `/home/test-4/ollama/` via `transfer_ollama_go_patches.py`.  
  Check: `grep -A3 'func (d DeviceInfo) NeedsInitValidation' /home/test-4/ollama/ml/device.go` → should show `return d.Library == "ROCm"` only.

- **Build on VM:** Use `/usr/local/go/bin/go` (Go 1.26.1) if present; default `go` may be 1.18. Always check with `go version` and `/usr/local/go/bin/go version` before concluding the VM cannot build.

## Which method is more effective: build on VM or locally?

**From the phase3 docs** (PHASE3_GPU_AND_TRANSPORT_STATUS.md, VM_TEST3_GPU_MODE_STATUS.md, RESTORE_GPU_LOGIC_CHECKLIST.md):

- **Build on VM** was used when the VM had a suitable Go (e.g. TEST-3 had **Go 1.26** at `/usr/local/go/bin/go`). Then `transfer_ollama_go_patches.py` does everything: transfer patched source → build on VM → install. One command, no copying the tree.
- **Build locally** is the documented alternative when the VM does *not* have the right Go: apply patches locally (`apply_ollama_vgpu_patches.py`), run `go build -o ollama.bin .` on a machine with **Go 1.23+**, then `scp ollama.bin` to the VM and install (Option B in PHASE3_GPU_AND_TRANSPORT_STATUS.md).

**For TEST-4:** Check for Go 1.23+ on the VM first: run `/usr/local/go/bin/go version` (often 1.26.x). If present, build on the VM with `cd /home/test-4/ollama && /usr/local/go/bin/go build -o ollama.bin .`. Do not assume the VM cannot build without checking.

**Summary:** Prefer **local build** when the VM lacks the right Go version; prefer **VM build** when the VM already has Go 1.23+ (simpler, one script). The same pattern is used for guest shims: phase3 docs recommend building heavy artifacts (e.g. libvgpu-cuda.so.1) on a machine with more resources and copying the result to the VM to avoid OOM (see TRANSFER_LIBVGPU_CUDA_SCRIPT_INVESTIGATION.md, STATUS_FOR_NEW_VM_COMPLETE.md).

---

## What you need to do

To get discovery working (no “didn’t fully initialize” and GPU kept):

1. **Build a patched `ollama.bin`** on a machine that has **Go 1.23 or newer**:
   - Either use the **VM tree** (already patched): copy `/home/test-4/ollama` to the build host, then:
     ```bash
     cd /path/to/ollama && go build -o ollama.bin .
     ```
   - Or use the **local phase3 tree**: run `python3 apply_ollama_vgpu_patches.py` in `phase3` (patches `phase3/ollama-src`), then build in that tree:
     ```bash
     cd phase3/ollama-src && go build -o ollama.bin .
     ```

2. **Install on the VM** (e.g. TEST-4):
   ```bash
   scp ollama.bin test-4@10.25.33.12:/tmp/
   ssh test-4@10.25.33.12 "echo Calvin@123 | sudo -S systemctl stop ollama && sudo cp /tmp/ollama.bin /usr/local/bin/ollama.bin && sudo systemctl start ollama"
   ```

3. **Restart Ollama** so discovery runs again with the new binary. You should then see the GPU kept (no “filtering device which didn’t fully initialize”) and `library=CUDA`, `total_vram` non-zero.

## Optional: build on the VM

If you install **Go 1.23+** on the VM (e.g. from https://go.dev/dl/ or a newer package), you can build there:

```bash
cd /home/test-4/ollama && go build -o ollama.bin .
sudo systemctl stop ollama && sudo cp ollama.bin /usr/local/bin/ollama.bin && sudo systemctl start ollama
```

## References

- **ACTUAL_OLLAMA_ERROR_CAPTURED.md** — How the “actual Oyu” was captured and what the log shows.
- **VM_TEST3_GPU_MODE_STATUS.md** — Same fix used on TEST-3; “Skip CUDA init validation” section.
- **patches/skip_cuda_init_validation_for_vgpu.patch** — Patch file for `ml/device.go`.
