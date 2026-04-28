# Final Solution Applied - OLLAMA_LIBRARY_PATH

## Date: 2026-02-26

## Root Cause Identified

**`OLLAMA_LIBRARY_PATH` was missing from `vgpu.conf`!**

### Evidence

According to `BREAKTHROUGH_SUMMARY.md` (Feb 25 - when it was working):
```
OLLAMA_LIBRARY_PATH="[/usr/local/lib/ollama /usr/local/lib/ollama/cuda_v12]"
```

This environment variable was present in the logs when GPU discovery was working.

### Why This Matters

`OLLAMA_LIBRARY_PATH` is an **Ollama-specific** environment variable that tells Ollama's backend scanner where to look for backend libraries. This is different from `LD_LIBRARY_PATH`:

- `LD_LIBRARY_PATH`: Used by dynamic linker to find shared libraries
- `OLLAMA_LIBRARY_PATH`: Used by Ollama's scanner to find backend libraries (like `libggml-cuda.so`)

**Without `OLLAMA_LIBRARY_PATH`, the scanner doesn't know where to look for `cuda_v12/`!**

This explains why:
- Scanner checks `cuda_v13` and `vulkan` (other directories it finds)
- Scanner does NOT check `cuda_v12/` (doesn't know where to look)
- Library is not loaded (scanner can't find it)

## Fix Applied

### Step 1: Added OLLAMA_LIBRARY_PATH

**Added to `/etc/systemd/system/ollama.service.d/vgpu.conf`:**
```ini
# OLLAMA_LIBRARY_PATH - tells scanner where to find backend libraries
Environment="OLLAMA_LIBRARY_PATH=/usr/local/lib/ollama:/usr/local/lib/ollama/cuda_v12"
```

### Step 2: Removed Duplicate Entry

A duplicate entry was accidentally created (one with quotes, one without), causing Ollama to crash. The unquoted duplicate has been removed.

**Removed:**
```ini
Environment=OLLAMA_LIBRARY_PATH=/usr/local/lib/ollama:/usr/local/lib/ollama/cuda_v12
```

**Kept (correct format):**
```ini
Environment="OLLAMA_LIBRARY_PATH=/usr/local/lib/ollama:/usr/local/lib/ollama/cuda_v12"
```

## Current Status

- ✅ `OLLAMA_LIBRARY_PATH` is set (with quotes, correct format)
- ✅ Duplicate entry removed
- ⏳ Need to verify Ollama starts correctly
- ⏳ Need to verify discovery works

## Verification Steps

1. **Check OLLAMA_LIBRARY_PATH is set correctly:**
   ```bash
   sudo grep OLLAMA_LIBRARY_PATH /etc/systemd/system/ollama.service.d/vgpu.conf
   ```
   Should show only ONE line with quotes.

2. **Verify systemd syntax:**
   ```bash
   sudo systemd-analyze verify ollama.service
   ```
   Should show no errors.

3. **Restart Ollama:**
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl restart ollama
   ```

4. **Check if Ollama starts:**
   ```bash
   systemctl is-active ollama
   ```
   Should show "active" (not crashing).

5. **Wait for discovery (12 seconds) and check results:**
   ```bash
   sleep 12
   journalctl -u ollama --since "15 seconds ago" | grep -E "verifying|library=|initial_count|OLLAMA_LIBRARY_PATH"
   ```

## Expected Results

After the fix:

1. ✅ Ollama starts without crashing
2. ✅ `OLLAMA_LIBRARY_PATH` appears in logs
3. ✅ "verifying if device is supported" message appears
4. ✅ `library=cuda_v12` or `library=cuda` in discovery logs
5. ✅ `initial_count=1` (GPU detected)
6. ✅ GPU mode active

## Why This Solution Works

1. **Matches working configuration:** This is exactly what was in `BREAKTHROUGH_SUMMARY.md` when it was working on Feb 25
2. **Tells scanner where to look:** `OLLAMA_LIBRARY_PATH` is the mechanism Ollama uses to find backend libraries
3. **Safe change:** Only adds an environment variable, no code changes
4. **No breaking changes:** All other working parts preserved

## Files Modified

- `/etc/systemd/system/ollama.service.d/vgpu.conf` - Added `OLLAMA_LIBRARY_PATH` (removed duplicate)

## Safety

- ✅ Only environment variable changes
- ✅ No code modifications
- ✅ Matches working configuration from Feb 25
- ✅ All other working parts preserved

## Conclusion

**`OLLAMA_LIBRARY_PATH` was the missing piece!** This environment variable tells Ollama's scanner where to find backend libraries. Without it, the scanner doesn't know where to look for `cuda_v12/`, which explains why it was checking other directories but not the requested one.

**The fix has been applied.** After restarting Ollama and verifying it starts correctly (no crashes), the scanner should be able to find and load `libggml-cuda.so` from `cuda_v12/`.

## Next Steps

1. Verify Ollama starts correctly (no SEGV crashes)
2. Verify `OLLAMA_LIBRARY_PATH` appears in logs
3. Verify discovery shows `library=cuda_v12` and `initial_count=1`
4. Verify GPU mode is active
