# Constructor Fix Applied - Root Cause Identified and Fixed

## Date: 2026-02-26

## Root Cause Identified

### The Problem
The constructor in `libvgpu_cuda.c` was checking `LD_PRELOAD` FIRST, and if it was set, it would skip checking for OLLAMA environment variables. 

**Issue**: The runner subprocess inherits `LD_PRELOAD` from the main process, so it would take the first branch (detecting via LD_PRELOAD) and never check for OLLAMA env vars. This meant the constructor never detected the runner process as an Ollama process.

### The Fix
Modified the constructor to check for OLLAMA environment variables FIRST, before checking LD_PRELOAD. This ensures that runner processes are detected correctly even if they inherit LD_PRELOAD from the main process.

## Code Changes

**File**: `phase3/guest-shim/libvgpu_cuda.c`

**Change**: Reordered the detection logic to check OLLAMA env vars first:

```c
/* CRITICAL FIX: Check for OLLAMA environment variables FIRST
 * Runner processes may inherit LD_PRELOAD from main process,
 * so we need to check OLLAMA vars before LD_PRELOAD to detect runners correctly */
const char *ollama_lib = getenv("OLLAMA_LLM_LIBRARY");
const char *ollama_path = getenv("OLLAMA_LIBRARY_PATH");
if (ollama_lib || ollama_path) {
    /* OLLAMA environment variables present - this is likely Ollama/runner process */
    is_app = 1;
    const char *ollama_msg = "[libvgpu-cuda] constructor: Ollama process detected (via OLLAMA env vars), initializing\n";
    syscall(__NR_write, 2, ollama_msg, strlen(ollama_msg));
} else {
    /* Check for LD_PRELOAD - main process has this */
    const char *ld_preload = getenv("LD_PRELOAD");
    if (ld_preload && strstr(ld_preload, "libvgpu")) {
        /* We have LD_PRELOAD with our shims - likely an application process */
        is_app = 1;
        const char *app_msg = "[libvgpu-cuda] constructor: Application process detected (via LD_PRELOAD)\n";
        syscall(__NR_write, 2, app_msg, strlen(app_msg));
    }
    /* ... rest of logic ... */
}
```

## Deployment Status

- ✅ Fix applied to source code
- ✅ File transferred to VM
- ✅ Library rebuilt
- ⚠️ Testing pending (needs restart and discovery run)

## Expected Results

After the fix:
1. Constructor will check OLLAMA env vars FIRST
2. Runner process will be detected even if it has LD_PRELOAD
3. Constructor logs should show: "Ollama process detected (via OLLAMA env vars)"
4. Shim will initialize in runner process
5. Discovery should show `initial_count=1` and `library=cuda`

## Next Steps

1. Restart Ollama (if not already restarted)
2. Trigger discovery by running a model
3. Check discovery logs for `initial_count=1` and `library=cuda`
4. Verify constructor logs show "Ollama process detected (via OLLAMA env vars)"

## Summary

**Root Cause**: Constructor checked LD_PRELOAD before OLLAMA env vars, causing runner processes to be missed.

**Fix**: Reordered detection logic to check OLLAMA env vars first.

**Status**: Fix deployed, needs testing to verify GPU detection works.
