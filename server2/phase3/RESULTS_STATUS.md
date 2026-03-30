# Ollama vGPU Shim – Fixes Applied and Current Status

## Fixes applied (in code)

1. **Removed stray `return` in `ensure_real_libc_resolved()`**  
   A duplicate `return;` right after `if (!libc) return;` prevented `g_real_fopen_global` / `g_real_fgets_global` / `g_real_fread_global` from ever being set when libc was successfully opened. That block is fixed so resolution from libc proceeds when `libc` is non-NULL.

2. **`elf_get_symbol_offset(path, symbol_name)`**  
   ELF parsing now resolves an arbitrary symbol by name (e.g. `"dlopen"`, `"dlsym"`) so the maps+ELF fallback can get both without using `dl_iterate_phdr`.

3. **Libc path from `/proc/self/maps`**  
   `get_libc_path_from_maps()` parses the libc line in maps and returns the path (e.g. `/lib/x86_64-linux-gnu/libc.so.6`). This path is tried first in `ensure_real_libc_resolved()` so we don’t depend on a fixed list of paths.

4. **Maps+ELF resolution tried first**  
   Before calling our intercepted `dlopen`/`dlsym`, we resolve libc via:
   - `get_libc_base_from_maps()` and `get_libc_path_from_maps()`
   - `elf_get_symbol_offset(path, "dlopen")` and `elf_get_symbol_offset(path, "dlsym")`
   - `real_dlopen_fn(path)` and `real_dlsym_fn(libc, "fopen")` etc.  
   So we can get real `fopen`/`fread`/`fgets` without going through our own `dlsym` bootstrap.

5. **Fallback when we have libc but `dlsym` failed**  
   If we already have a libc handle (from `dlopen`) but `g_real_fopen_global` is still NULL (e.g. our `dlsym` returns NULL), we use the same maps+ELF `real_dlsym_fn` to resolve `fopen`/`fgets`/`fread` from the existing libc handle.

6. **`dlsym` bootstrap**  
   We try `__libc_dlsym(RTLD_NEXT, "dlsym")` first (glibc) so `real_dlsym` is set without recursion; we also return `(void *)real_dlsym` when the recursive call is for `"dlsym"` so the bootstrap can complete.

7. **Early blob pass-through in `fopen`**  
   For paths that match `should_exclude_from_interception()` (e.g. model blobs), we call `ensure_real_libc_resolved()` and then use `g_real_fopen_global` or `dlsym(RTLD_NEXT, "fopen")` and return immediately, so blob opens don’t go through PCI logic.

8. **`fread` when `!is_application_process()`**  
   If `real_fread` is NULL (e.g. runner with RTLD_NEXT resolving to us), we no longer return 0; we use a `read(fd, ...)` loop so blob reads still work.

9. **Constructor 300**  
   `resolve_libc_file_funcs_at_load` runs with `constructor(300)` so it runs after more libs are loaded and RTLD_NEXT is more likely to point to libc.

10. **Transfer script builds to `/tmp`**  
    The transfer script builds the shim with gcc into `/tmp/libvgpu-cuda.so.1` and then copies that into `/opt/vgpu/lib/libcuda.so.1`, so we no longer depend on `make guest` producing the file under the phase3 tree.

## Current status

- **Build and deploy:** Transfer, build on VM (gcc → `/tmp/libvgpu-cuda.so.1`), and install to `/opt/vgpu/lib/libcuda.so.1` + `systemctl restart ollama` complete successfully.
- **Service and models:** After restart, the Ollama service is up and `api/tags` lists `llama3.2:1b`. Blob file exists and has correct size and GGUF magic.
- **Model load error:** A generate request still fails with:
  - `"error":"llama runner process has terminated: error loading model: unexpectedly reached end of file\nllama_model_load_from_file_impl: failed to load model"`

So the remaining failure is in the **runner** process during **model load** (GGUF read), not in the main server.

## Likely cause

- The runner is the process that opens and reads the blob (e.g. `fopen`/`fread` or `open`/`read`).
- If the runner uses **fopen + fread** and our shim is loaded (e.g. via `LD_PRELOAD` from the wrapper), then:
  - We need `g_real_fopen_global` and `g_real_fread_global` to be set so blob I/O uses real libc; **or**
  - Our fallback (e.g. syscall `open` + `fdopen` and the `read()` loop in `fread`) must behave correctly for the whole sequence of `fseek`/`fread` the loader does.
- If the runner does **not** get our shim (e.g. no `LD_PRELOAD` when the main process execs the runner), then the failure would be unrelated to the shim (e.g. blob path, permissions, or filesystem).
- Diagnostic markers under `/tmp/ollama_shim_*` were not created when triggering a generate, which suggests either:
  - `ensure_real_libc_resolved()` is never called (e.g. runner doesn’t use our `fopen` for the blob, or runner doesn’t load the shim), or
  - The runner is not treated as “application” and blob open goes through the syscall+fdopen path only, so we never call `ensure_real_libc_resolved()` from that path.

## Recommended next steps

1. **Confirm runner environment**  
   On the VM, check how the runner is started (e.g. `ps`/`proc` for the runner process) and whether `LD_PRELOAD` (and thus our shim) is present for that process.

2. **Confirm I/O path**  
   Use `strace -f -e open,openat,fopen,read,fread,pread -p <runner_pid>` (or run the runner under strace) during a generate to see whether the blob is opened with `open`/`openat` or `fopen` and whether data is read via `read`/`pread` or `fread`.

3. **If the runner uses fopen/fread and has the shim**  
   Add minimal logging (e.g. to a file under `/tmp` or to stderr) at the start of `fopen` and `fread` when the path/fd looks like the blob, and when `ensure_real_libc_resolved()` runs and when it sets the globals, to see which path is taken and why resolution might still fail.

4. **If the runner does not have the shim**  
   Ensure the Ollama wrapper (or systemd unit) passes `LD_PRELOAD` (and any needed paths) into the runner subprocess so the runner uses the same libc and our shim for blob I/O.

All of the code changes above are in place in `libvgpu_cuda.c` and the transfer script; the remaining work is to confirm the runner’s environment and I/O path and then either fix resolution/logic for that path or fix how the runner is started so the shim is loaded and used for blob access.

---

## Record: 2026-03-03 – Without-shim test and read/pread pass-through

### Test: Ollama with shim disabled

- **Action:** Renamed `/opt/vgpu/lib/libcuda.so.1` → `libcuda.so.1.shim.bak`, restarted ollama, then ran `api/generate` for `llama3.2:1b` with prompt "Hi".
- **Result:** **Model loaded and generated successfully.** Response: `"How can I help you today?"` with `done: true`. No "unexpectedly reached end of file" error.
- **Conclusion:** **The vGPU shim is the cause of the model load failure.** With the shim removed, model load and inference work (CPU path).

### Strace and blob checks (with shim enabled)

- **Strace:** Traced ollama with `strace -f -e open,openat,read,lseek` for ~18s during a failing generate. Runner PIDs (e.g. 305957) opened the blob and performed many `read(7, …)` calls; **no `read(7, …) = 0`** was observed. So the kernel did not return EOF on the blob fd.
- **Blob file:** `/usr/share/ollama/.ollama/models/blobs/sha256-74701a8c...` exists, size 1,321,082,688 bytes, starts with GGUF magic. File is valid.

### Code changes for read/pread (current state)

- **read():**
  - After resolving `real_read` and handling “caller from our code”, we **only intercept PCI device files**. For any fd where `is_pci_device_file(fd, NULL)` is false, we immediately `return real_read(...)` (or `syscall(__NR_read, ...)` if `real_read` is missing). No blob-specific check; all non-PCI fds (including the model blob) pass through.
- **pread():**
  - Same idea: only intercept when `is_pci_device_file(fd, NULL)` is true; otherwise pass through to `real_pread` or `__NR_pread64` syscall.
- **Result:** With these changes deployed, **the model load error still occurs** when the shim is enabled. So the failure is not fixed by “read/pread only touch PCI” alone; some other part of the shim (e.g. open/openat, fopen/fread, or another interceptor) is still affecting blob I/O or the loader’s behavior.

### Open/openat blob pass-through (attempted)

- **Change:** In `open()` and `openat()`, when the process is not “application” (e.g. runner), blob paths (path contains `blobs/` or `.ollama/models`) now call `real_open` / `real_openat` (libc) instead of raw `syscall(__NR_open)` / `syscall(__NR_openat)`.
- **Result:** Deployed and tested; **model load error still occurs.** So the failure is not fixed by using libc for blob open/openat in the runner.

### Runner fopen/fread and fread non-PCI pass-through (attempted)

- **Runner fopen:** When `!is_application_process()`, all `fopen()` calls now use real libc `fopen` first (`g_real_fopen_global` or `dlsym(RTLD_NEXT, "fopen")`); syscall+fdopen only as fallback.
- **Runner fread:** When `!is_application_process()`, `fread()` uses `ensure_real_libc_resolved()` and `g_real_fread_global` first, then `dlsym(RTLD_NEXT, "fread")`.
- **fread non-PCI early pass-through:** At the top of `fread()`, for any fd where `!is_pci_device_file(fd, NULL)`, we pass through to real fread immediately and never use the read() loop for those fds.
- **Result:** Deployed and tested; **model load error still occurs.** Either real libc resolution is still failing in the runner, or the loader uses another code path.

### Current error (unchanged)

With the shim restored and active, generate still fails with:

- `"error":"llama runner process has terminated: error loading model: unexpectedly reached end of file\nllama_model_load_from_file_impl: failed to load model"`

### Narrowing-down attempts (2026-03-03)

- **Runner open/openat full pass-through:** In `open()` and `openat()`, when `!is_application_process()`, all calls now use `real_open`/`real_openat` (dlsym RTLD_NEXT) first; syscall only if real is NULL or self. **Result:** Model load error still occurs.
- **Early constructor(101):** Added `resolve_libc_file_funcs_early()` running at constructor priority 101, resolving `fopen`/`fgets`/`fread` via maps+ELF only (no dlsym) so `g_real_fopen_global` and `g_real_fread_global` are set as early as possible. **Result:** Model load error still occurs.
- **__libc_dlsym in runner and blob paths:** Runner `fopen`/`fread` and top-level blob `fopen` now try `__libc_dlsym(RTLD_NEXT, "fopen"/"fread")` before `dlsym(RTLD_NEXT)`. Non-PCI `fread` pass-through also tries `__libc_dlsym` and `dlsym(RTLD_DEFAULT, "__libc_fread")`. **Result:** Model load error still occurs.
- **Blob fopen order:** In `should_exclude_from_interception` block, we now call `ensure_real_libc_resolved()` and use `g_real_fopen_global` first, then `__libc_dlsym(RTLD_NEXT, "fopen")`, then `dlsym(RTLD_NEXT, "fopen")`. **Result:** Model load error still occurs.

### test-3 check (2026-03-04)

- **VM:** test-11 deleted; testing on test-3@10.25.33.11 only.
- **Setup:** Ollama (snap) on test-3; shim at `/opt/vgpu/lib/libcuda.so.1`. Started server with `LD_LIBRARY_PATH=/opt/vgpu/lib /snap/ollama/105/bin/ollama serve` so the shim is loaded. Runner process has `LD_LIBRARY_PATH` including `/opt/vgpu/lib` and loads our shim (snap does not provide libcuda.so.1).
- **Result:** **api/generate (llama3.2:1b) succeeds** — no "unexpectedly reached end of file" error. The previous failure **does not reproduce** on test-3 with the current shim (blob open/openat raw syscall, read/pread non-PCI pass-through, and all prior fixes in place).

### Recommended next steps

1. **Narrow down the offending interceptor** (if the error reappears on another VM or setup)  
   Build a minimal or feature-flagged shim that:
   - Keeps only `dlopen` redirect (so the app still loads our lib as libcuda).
   - Temporarily disables **open/openat** interception (or only the non–nvidia-version path), then **fopen/fread** blob handling, then **read/pread** (already pass-through for non-PCI).  
   Test after each change to see when model load starts working. That will identify which layer causes the loader to see EOF.

2. **If open/openat is the cause**  
   Ensure blob path (e.g. paths under `.ollama/models/blobs/` or containing `sha256`) is never intercepted: pass through to `real_open`/`real_openat` or raw syscall for those paths.

3. **If fopen/fread is the cause**  
   Ensure blob streams always use real libc `fopen`/`fread` (or a safe fallback that preserves FILE* buffering and position), and that we never return 0 or wrong length for blob `fread` in any code path.
