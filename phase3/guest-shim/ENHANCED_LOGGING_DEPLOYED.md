# Enhanced Logging Deployed

## What Was Done

Enhanced logging has been added to device query functions to track what's happening in runner subprocesses:

1. **`cuDeviceGetCount()`**:
   - Now logs PID when called
   - Logs success with PID and return value (count=1)

2. **`cuDeviceGetAttribute()`**:
   - Now logs PID when called
   - Logs success with PID, attribute ID, and return value

## Purpose

This enhanced logging will help us determine:
- If device query functions are being called in runner subprocesses
- What PIDs are calling these functions (main process vs runner subprocesses)
- What values are being returned
- If compute capability attributes (75/76) are being queried

## How to Check

After Ollama restarts, check the logs:

```bash
# Check for cuDeviceGetCount() calls
sudo journalctl -u ollama --since "2 minutes ago" --no-pager | grep -i "cuDeviceGetCount"

# Check for cuDeviceGetAttribute() calls
sudo journalctl -u ollama --since "2 minutes ago" --no-pager | grep -i "cuDeviceGetAttribute"

# Check for compute capability queries (attributes 75 and 76)
sudo journalctl -u ollama --since "2 minutes ago" --no-pager | grep -E "(attrib=75|attrib=76)"

# Check final GPU detection status
sudo journalctl -u ollama --since "2 minutes ago" --no-pager | grep -E "(initial_count|library=|compute=)"
```

## Expected Results

If device query functions ARE being called:
- You'll see logs with PIDs showing the functions are called
- Check if the PIDs match runner subprocesses
- Check what values are being returned
- If `initial_count=0` persists, the values might be wrong

If device query functions are NOT being called:
- No logs will appear
- This confirms runner subprocesses aren't calling these functions
- Need to investigate why `ggml_backend_cuda_init` isn't calling them

## Current Status

- ✅ Enhanced logging deployed
- ✅ Ollama restarted
- ⚠️ Need to check logs to see if functions are being called
- ⚠️ Still showing `initial_count=0` (need to verify if functions are called)

## Next Steps

1. **Check the logs** using the commands above
2. **If functions ARE being called**:
   - Verify the return values are correct (count=1, CC=9.0)
   - Check if the PIDs match runner subprocesses
   - If values are correct but `initial_count=0`, investigate why Ollama isn't using them

3. **If functions are NOT being called**:
   - Investigate why `ggml_backend_cuda_init` isn't calling them
   - Check if there's an error check failing before device queries
   - Verify runner subprocesses have our shims loaded
