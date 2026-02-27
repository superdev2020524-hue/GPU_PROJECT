# Solution to Apply: Add OLLAMA_LIBRARY_PATH

## Date: 2026-02-26

## Root Cause Identified

**`OLLAMA_LIBRARY_PATH` is missing from `vgpu.conf`!**

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

This explains why:
- Scanner checks `cuda_v13` and `vulkan` (other directories it finds)
- Scanner does NOT check `cuda_v12/` (doesn't know where to look)
- Library is not loaded (scanner can't find it)

## The Fix

**Add to `/etc/systemd/system/ollama.service.d/vgpu.conf`:**

```ini
# OLLAMA_LIBRARY_PATH - tells scanner where to find backend libraries
Environment="OLLAMA_LIBRARY_PATH=/usr/local/lib/ollama:/usr/local/lib/ollama/cuda_v12"
```

### Manual Steps

1. **Backup the file:**
   ```bash
   sudo cp /etc/systemd/system/ollama.service.d/vgpu.conf \
          /etc/systemd/system/ollama.service.d/vgpu.conf.backup_ollama_library_path
   ```

2. **Add OLLAMA_LIBRARY_PATH:**
   ```bash
   sudo bash -c 'cat >> /etc/systemd/system/ollama.service.d/vgpu.conf << EOF

# OLLAMA_LIBRARY_PATH - tells scanner where to find backend libraries
Environment="OLLAMA_LIBRARY_PATH=/usr/local/lib/ollama:/usr/local/lib/ollama/cuda_v12"
EOF'
   ```

3. **Verify it was added:**
   ```bash
   sudo grep OLLAMA_LIBRARY_PATH /etc/systemd/system/ollama.service.d/vgpu.conf
   ```

4. **Restart Ollama:**
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl restart ollama
   ```

5. **Wait for discovery (10 seconds) and check results:**
   ```bash
   sleep 10
   journalctl -u ollama --since "15 seconds ago" --no-pager | grep -E "verifying|library=|initial_count|OLLAMA_LIBRARY_PATH" | tail -10
   ```

## Expected Result

After adding `OLLAMA_LIBRARY_PATH`:

1. ✅ `OLLAMA_LIBRARY_PATH` appears in logs
2. ✅ "verifying if device is supported" message appears
3. ✅ `library=cuda_v12` or `library=cuda` in discovery logs
4. ✅ `initial_count=1` (GPU detected)
5. ✅ GPU mode active

## Why This Was Missing

This environment variable may have been:
- Never added to `vgpu.conf` initially
- Removed during a previous fix
- Lost during system update or reinstall

## Safety

- ✅ Only adds an environment variable
- ✅ No code changes
- ✅ No breaking changes
- ✅ Matches working configuration from Feb 25
- ✅ Safe to apply

## Verification Commands

After applying the fix:

```bash
# Check if OLLAMA_LIBRARY_PATH is set
sudo systemctl show ollama --property=Environment | grep OLLAMA_LIBRARY_PATH

# Check discovery logs
journalctl -u ollama --since "1 minute ago" | grep -E "verifying|library=|initial_count|OLLAMA_LIBRARY_PATH"

# Check if library is loaded
sudo lsof -p $(pgrep -f "ollama serve") | grep libggml-cuda
```

## Conclusion

**This is the missing piece!** `OLLAMA_LIBRARY_PATH` tells Ollama's scanner where to find backend libraries. Without it, the scanner doesn't know where to look for `cuda_v12/`, which explains why it was checking other directories but not the requested one.

**This matches the working configuration from BREAKTHROUGH_SUMMARY.md (Feb 25).**
