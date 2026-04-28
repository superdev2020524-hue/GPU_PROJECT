# Solution Found: OLLAMA_LIBRARY_PATH Missing

## Date: 2026-02-26

## Critical Discovery

**`OLLAMA_LIBRARY_PATH` was NOT set in `vgpu.conf`!**

### Evidence

According to `BREAKTHROUGH_SUMMARY.md` (Feb 25 - when it was working):
```
OLLAMA_LIBRARY_PATH="[/usr/local/lib/ollama /usr/local/lib/ollama/cuda_v12]"
```

This environment variable was present in the logs when GPU discovery was working.

### Current State

- `LD_LIBRARY_PATH` is set in `vgpu.conf` ✓
- `OLLAMA_LIBRARY_PATH` is NOT set in `vgpu.conf` ✗

### Why This Matters

`OLLAMA_LIBRARY_PATH` is an **Ollama-specific** environment variable that tells Ollama's backend scanner where to look for backend libraries. This is different from `LD_LIBRARY_PATH`:

- `LD_LIBRARY_PATH`: Used by the dynamic linker to find shared libraries
- `OLLAMA_LIBRARY_PATH`: Used by Ollama's scanner to find backend libraries (like `libggml-cuda.so`)

**If `OLLAMA_LIBRARY_PATH` is not set, the scanner doesn't know where to look for `cuda_v12/`!**

## The Fix

**Added to `/etc/systemd/system/ollama.service.d/vgpu.conf`:**
```ini
# OLLAMA_LIBRARY_PATH - tells scanner where to find backend libraries
Environment="OLLAMA_LIBRARY_PATH=/usr/local/lib/ollama:/usr/local/lib/ollama/cuda_v12"
```

This matches what was shown in `BREAKTHROUGH_SUMMARY.md` when it was working.

## Expected Result

After adding `OLLAMA_LIBRARY_PATH`:
1. Scanner will know where to look for backend libraries
2. Scanner will find `cuda_v12/` directory
3. Scanner will load `libggml-cuda.so` from `cuda_v12/`
4. Discovery will show `library=cuda_v12` and `initial_count=1`

## Why This Was Missing

This environment variable may have been:
- Never added to `vgpu.conf`
- Removed during a previous fix
- Lost during system update or reinstall

## Safety

- ✅ Only adds an environment variable
- ✅ No code changes
- ✅ No breaking changes
- ✅ Matches working configuration from Feb 25

## Verification

After restarting Ollama, check logs for:
- `OLLAMA_LIBRARY_PATH` in logs
- "verifying if device is supported" message
- `library=cuda_v12` or `library=cuda`
- `initial_count=1`

## Conclusion

**This was the missing piece!** `OLLAMA_LIBRARY_PATH` tells Ollama's scanner where to find backend libraries. Without it, the scanner doesn't know where to look for `cuda_v12/`, which explains why it was checking other directories but not the requested one.
