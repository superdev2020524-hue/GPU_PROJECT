# Critical Fix: Duplicate OLLAMA_LIBRARY_PATH Causing Crash

## Date: 2026-02-26

## Critical Issue

**Ollama is crashing with SEGV after adding `OLLAMA_LIBRARY_PATH`!**

### Root Cause

When `OLLAMA_LIBRARY_PATH` was added to `vgpu.conf`, a **duplicate entry** was created:
1. `Environment="OLLAMA_LIBRARY_PATH=/usr/local/lib/ollama:/usr/local/lib/ollama/cuda_v12"` (correct, with quotes)
2. `Environment=OLLAMA_LIBRARY_PATH=/usr/local/lib/ollama:/usr/local/lib/ollama/cuda_v12` (incorrect, without quotes)

The **unquoted duplicate** is causing a syntax error in systemd configuration, which makes Ollama crash.

### The Fix

**Remove the unquoted duplicate entry:**

```bash
sudo sed -i '/^Environment=OLLAMA_LIBRARY_PATH=/d' /etc/systemd/system/ollama.service.d/vgpu.conf
```

This removes the unquoted line, keeping only the correctly quoted version.

### Verification

After removing the duplicate:

1. **Verify only one entry remains:**
   ```bash
   sudo grep OLLAMA_LIBRARY_PATH /etc/systemd/system/ollama.service.d/vgpu.conf
   ```
   Should show only one line with quotes.

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
   Should show "active".

5. **Check discovery results:**
   ```bash
   sleep 12
   journalctl -u ollama --since "15 seconds ago" | grep -E "verifying|library=|initial_count|OLLAMA_LIBRARY_PATH"
   ```

## Expected Result

After removing the duplicate:

1. ✅ Ollama starts without crashing
2. ✅ `OLLAMA_LIBRARY_PATH` is set correctly (one entry with quotes)
3. ✅ Scanner can find `cuda_v12/` directory
4. ✅ Library loads and GPU is detected

## Why This Happened

The duplicate was likely created when:
- Multiple attempts were made to add `OLLAMA_LIBRARY_PATH`
- Different methods were used (some with quotes, some without)
- The unquoted version was added accidentally

## Correct Format

The correct format in `vgpu.conf` is:
```ini
Environment="OLLAMA_LIBRARY_PATH=/usr/local/lib/ollama:/usr/local/lib/ollama/cuda_v12"
```

**Important:** The quotes are required for systemd environment variables with colons in the value.

## Status

- ✅ Duplicate entry has been removed
- ⏳ Need to verify Ollama starts correctly
- ⏳ Need to verify discovery works

## Next Steps

1. Verify Ollama starts after removing duplicate
2. Check if discovery works with `OLLAMA_LIBRARY_PATH` set correctly
3. Verify GPU is detected
