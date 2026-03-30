# Next Steps - Error Capture System

## Current Status

✅ **Infrastructure Complete:**
- Enhanced write() interceptor implemented in `libvgpu_cuda.c`
- Error capture scripts deployed
- Systemd configuration updated
- Analysis tools ready

⚠️ **Issue:**
- Error logs not being created (`/tmp/ollama_errors*.log` don't exist)
- This indicates write() interceptor is not intercepting

## Root Cause Analysis

The write() interceptor may not be working because:

1. **Library not rebuilt** - Enhanced code exists but library wasn't recompiled
2. **Symbol conflict** - Another library exports write() and takes precedence
3. **Library loading order** - libvgpu-cuda.so not loaded early enough
4. **Function not exported** - write() not properly exported from library

## Verification Steps

### Step 1: Verify Library Was Rebuilt

```bash
ssh test-10@10.25.33.110
cd ~/phase3/guest-shim

# Check library timestamp
ls -lh /usr/lib64/libvgpu-cuda.so

# Check if write() is in the library
nm -D /usr/lib64/libvgpu-cuda.so | grep " write$"
```

**Expected**: Should show `T write` (exported function)

### Step 2: Verify Library is Loaded

```bash
# Check if library is loaded by Ollama
sudo lsof -p $(pgrep -f "ollama serve" | head -1) | grep libvgpu-cuda

# Check LD_PRELOAD
sudo cat /proc/$(pgrep -f "ollama serve" | head -1)/environ | tr '\0' '\n' | grep LD_PRELOAD
```

**Expected**: Should show libvgpu-cuda.so in process and LD_PRELOAD set

### Step 3: Test Write Interceptor Manually

```bash
# Create a test program that writes to stderr
cat > /tmp/test_write.c << 'EOF'
#include <unistd.h>
int main() {
    const char *msg = "TEST ERROR MESSAGE\n";
    write(2, msg, 19);
    return 0;
}
EOF

gcc -o /tmp/test_write /tmp/test_write.c

# Run with LD_PRELOAD
LD_PRELOAD=/usr/lib64/libvgpu-cuda.so /tmp/test_write

# Check if log was created
cat /tmp/ollama_errors_full.log
```

**Expected**: Should see test message in log file

## Solution: Force Rebuild and Verify

### Complete Rebuild Process

```bash
ssh test-10@10.25.33.110
cd ~/phase3/guest-shim

# 1. Verify source has enhanced code
grep -c "ollama_errors_full.log" libvgpu_cuda.c

# 2. Clean old libraries
sudo rm -f /usr/lib64/libvgpu-*.so

# 3. Rebuild
sudo ./install.sh

# 4. Verify write() is exported
nm -D /usr/lib64/libvgpu-cuda.so | grep " write$"

# 5. Reload systemd
sudo systemctl daemon-reload

# 6. Restart Ollama
sudo systemctl restart ollama

# 7. Monitor for logs
tail -f /tmp/ollama_errors_full.log
```

## Alternative: Direct Error Capture

If write() interceptor continues to not work, we can capture errors directly:

### Method 1: Strace Capture

```bash
sudo strace -p $(pgrep -f "ollama serve" | head -1) -s 2000 -e trace=write 2>&1 | grep -E "(error|failed|ggml|cuda)" > /tmp/strace_errors.log
```

### Method 2: Journalctl Filtering

```bash
journalctl -u ollama -f | grep -iE "(error|failed|timeout|discover|ggml|cuda)" > /tmp/journalctl_errors.log
```

### Method 3: Systemd Stderr

The systemd stderr redirection should already be capturing to `/tmp/ollama_stderr.log`:

```bash
tail -f /tmp/ollama_stderr.log
```

## Expected Behavior Once Working

When the write() interceptor is active:

1. **Log files created immediately** when Ollama starts
2. **Format**: `[timestamp.nanoseconds] PID=pid SIZE=bytes: message`
3. **Full messages**: No truncation, complete error text
4. **Real-time capture**: Errors appear as they're written

## Success Criteria

✅ Error logs exist: `/tmp/ollama_errors_full.log` and `/tmp/ollama_errors_filtered.log`
✅ Logs contain error messages from discovery
✅ Full error messages (not truncated)
✅ Can analyze errors with `analyze_errors.sh`

## Quick Test Command

```bash
# One-liner to test if interceptor works
ssh test-10@10.25.33.110 'LD_PRELOAD=/usr/lib64/libvgpu-cuda.so sh -c "echo test >&2" && cat /tmp/ollama_errors_full.log 2>/dev/null | tail -1'
```

If this shows the test message, the interceptor is working.

## Next Action

**Immediate next step**: Verify the library was actually rebuilt with the enhanced code and that write() is exported. Then restart Ollama and monitor for error logs.

If logs still don't appear, use alternative capture methods (strace, journalctl) to get the error messages we need.
