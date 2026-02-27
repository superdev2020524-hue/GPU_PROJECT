# OLLAMA_LIBRARY_PATH Fix Applied

## Date: 2026-02-26

## Status

**`OLLAMA_LIBRARY_PATH` has been added to `vgpu.conf`**

### Current State

The file `/etc/systemd/system/ollama.service.d/vgpu.conf` now contains:
```ini
# OLLAMA_LIBRARY_PATH
Environment="OLLAMA_LIBRARY_PATH=/usr/local/lib/ollama:/usr/local/lib/ollama/cuda_v12"
Environment=OLLAMA_LIBRARY_PATH=/usr/local/lib/ollama:/usr/local/lib/ollama/cuda_v12
```

**Note:** There appears to be a duplicate entry (one with quotes, one without). The one with quotes is the correct format.

### Next Steps

1. **Remove duplicate entry** (if needed):
   ```bash
   sudo sed -i '/^Environment=OLLAMA_LIBRARY_PATH=/d' /etc/systemd/system/ollama.service.d/vgpu.conf
   ```
   This removes the unquoted duplicate, keeping only the quoted version.

2. **Restart Ollama:**
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl restart ollama
   ```

3. **Verify it's working:**
   ```bash
   sleep 12
   journalctl -u ollama --since "15 seconds ago" | grep -E "verifying|library=|initial_count|OLLAMA_LIBRARY_PATH"
   ```

## Expected Results

After restarting Ollama with `OLLAMA_LIBRARY_PATH` set:

1. ✅ `OLLAMA_LIBRARY_PATH` appears in logs
2. ✅ "verifying if device is supported" message appears
3. ✅ `library=cuda_v12` or `library=cuda` in discovery logs
4. ✅ `initial_count=1` (GPU detected)
5. ✅ GPU mode active

## Why This Should Work

According to `BREAKTHROUGH_SUMMARY.md` (Feb 25 - when it was working):
- `OLLAMA_LIBRARY_PATH` was present in logs
- Library was loaded and verified
- GPU discovery was working

**This matches the working configuration!**

## If Still Not Working

If `OLLAMA_LIBRARY_PATH` is set but library still not loading:

1. Check for duplicate entries and remove them
2. Verify format is correct: `Environment="OLLAMA_LIBRARY_PATH=/usr/local/lib/ollama:/usr/local/lib/ollama/cuda_v12"`
3. Ensure Ollama was restarted after adding it
4. Check logs for any errors
5. Verify library is accessible from the paths specified

## Conclusion

**`OLLAMA_LIBRARY_PATH` has been added.** This was the missing piece that tells Ollama's scanner where to find backend libraries. After restarting Ollama, the scanner should be able to find and load `libggml-cuda.so` from `cuda_v12/`.
