# GPU Mode Fix for Snap Ollama (test-3)

## Problem
- Ollama (snap) reports `library=cpu` and `id=cpu` because the backend scanner only looks for `libggml-*.so` in the **top-level** of the lib dir (`/snap/ollama/105/lib/ollama/`).
- In the snap, `libggml-cuda.so` lives only in `cuda_v12/`, so the scanner never finds it.
- Snap filesystem is read-only, so we cannot add a symlink there.

## Approach: Bind-mount writable copy
1. **Copy** the snap's `lib/ollama` into a writable dir: `/var/snap/ollama/common/ollama_lib_writable/`.
2. **Add** in that copy:
   - Top-level symlink: `libggml-cuda.so` → `cuda_v12/libggml-cuda.so`
   - Replace `cuda_v12/libcuda.so.1` with our vGPU shim (from `/opt/vgpu/lib/libcuda.so.1`).
3. **Bind-mount** that copy over the snap path so the scanner sees the top-level `libggml-cuda.so`:
   ```bash
   sudo mount --bind /var/snap/ollama/common/ollama_lib_writable /snap/ollama/105/lib/ollama
   ```
4. **Service override** (no LD_PRELOAD to avoid SEGV in main process):
   - `LD_LIBRARY_PATH=/var/snap/ollama/common/lib:/snap/ollama/105/lib/ollama:/snap/ollama/105/lib/ollama/cuda_v12`
   - `OLLAMA_LIBRARY_PATH=...` and `OLLAMA_LLM_LIBRARY=cuda_v12`

## Shim constructor change
- The vGPU shim’s load-time constructors (resolve_libc_file_funcs_early and resolve_libc_file_funcs_at_load) are **disabled** (no-op) so that when the **main** ollama process loads the shim via LD_PRELOAD it does not SEGV.
- With the bind-mount approach we do **not** use LD_PRELOAD for the main process; the **runner** loads our shim when it loads `libggml-cuda.so` (which depends on `libcuda.so.1` from the mounted dir).

## One-time setup (run on VM)
```bash
# 1) Stop Ollama and unmount if already mounted
sudo systemctl stop snap.ollama.listener.service
sudo umount /snap/ollama/105/lib/ollama 2>/dev/null || true

# 2) Full copy (can take 1–2 min from squashfs)
sudo rm -rf /var/snap/ollama/common/ollama_lib_writable
sudo mkdir -p /var/snap/ollama/common/ollama_lib_writable
sudo cp -r /snap/ollama/105/lib/ollama/. /var/snap/ollama/common/ollama_lib_writable/

# 3) Add top-level libggml-cuda and replace libcuda with shim
sudo ln -sf cuda_v12/libggml-cuda.so /var/snap/ollama/common/ollama_lib_writable/libggml-cuda.so
sudo cp /opt/vgpu/lib/libcuda.so.1 /var/snap/ollama/common/ollama_lib_writable/cuda_v12/libcuda.so.1

# 4) Bind mount and start
sudo mount --bind /var/snap/ollama/common/ollama_lib_writable /snap/ollama/105/lib/ollama
sudo systemctl start snap.ollama.listener.service
```

## Verify
```bash
# Should show id=gpu or library=cuda and total_vram > 0
sudo journalctl -u snap.ollama.listener -n 30 --no-pager | grep -E "inference compute|library=|total_vram"
```

## Status on test-3 (snap)
- Bind mount and top-level `libggml-cuda.so` are in place; `/snap/ollama/105/lib/ollama/libggml-cuda.so` exists.
- Discovery still reports `id=cpu` and **never** calls our shim (`/tmp/cuda_get_count_called.txt` is never written).
- So the snap’s discovery path does **not** load `libggml-cuda.so` from the mounted dir (likely different path or logic in snap context).
- **Recommendation:** For GPU mode with the vGPU shim, use a **non-snap** Ollama install (e.g. official install script or .deb) so you can add the top-level symlink under `/usr/local/lib/ollama/` as in `ROOT_CAUSE_FIXED.md`; that configuration is known to work.

## Persistence
The bind mount is lost on reboot. To make it persistent, add to `/etc/rc.local` or a systemd service:
- After snap.ollama is available, run the same `mount --bind` as above.
