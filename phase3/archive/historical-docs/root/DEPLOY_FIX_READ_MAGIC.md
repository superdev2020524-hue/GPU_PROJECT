# Deploy "failed to read magic" fix to VM

## What was fixed (locally)

In `guest-shim/libvgpu_cuda.c`:

1. **Real libc FILE* helpers**  
   `ensure_real_libc_resolved()` now resolves `fopen`, `fgets`, and `fread` from `libc.so.6` and stores them in `g_real_fopen_global`, `g_real_fgets_global`, `g_real_fread_global`.

2. **Excluded paths (e.g. model blobs)**  
   For paths excluded from interception (e.g. `/.ollama/models/`, `/models/blobs/`), `fopen` uses `g_real_fopen_global` so the GGUF loader gets a real `FILE*`.

3. **fgets/fread on untracked streams**  
   When the stream was not opened by us (e.g. model file opened with real `fopen`), we now call libc’s real `fgets`/`fread` instead of `read()` syscalls, so `FILE*` buffering stays consistent and "failed to read magic" is avoided.

## Deploy to VM

The updated file is large; use one of these methods.

### Option A: scp with sshpass (recommended)

```bash
cd /path/to/gpu/phase3
sshpass -p 'Calvin@123' scp -o StrictHostKeyChecking=no guest-shim/libvgpu_cuda.c test-11@10.25.33.111:/home/test-11/phase3/guest-shim/libvgpu_cuda.c
```

Then on the VM:

```bash
cd ~/phase3 && make guest
sudo cp guest-shim/libvgpu-cuda.so.1 /opt/vgpu/lib/libcuda.so.1
sudo systemctl restart ollama
```

### Option B: Chunked transfer (not recommended)

The script `reliable_file_copy.py` can corrupt large files (shell escaping of base64 chunks). Prefer Option A (scp). If you must use chunked transfer, verify the file on the VM after copy (e.g. `head -20 ~/phase3/guest-shim/libvgpu_cuda.c` should start with `/*` and `* libvgpu_cuda.c`).

### Verify

On the VM, run a short inference (e.g. `ollama run llama3.2:1b 'Hi'` or a single `/api/generate` call).  
If the fix is applied, the model should load without "gguf_init_from_file_impl: failed to read magic" and inference should complete (or at least get past model load).
