# Constructor Priority Fix Status

## Date: 2026-02-26

## Current Status

### ✅ Completed

1. **Runtime API shim constructor priority** - Updated to 101 ✓
2. **NVML shim constructor priority** - Updated to 101 ✓
3. **Both shims rebuilt and deployed** ✓

### ❌ In Progress

**Driver API shim constructor priority** - File corrupted during fix attempt
- File: `libvgpu_cuda.c`
- Issue: Syntax errors after sed replacement
- Status: Needs restoration from working version

## What Happened

Attempted to update constructor priority using `sed`:
```bash
sed -i "s/__attribute__((constructor))$/__attribute__((constructor(101)))/" libvgpu_cuda.c
```

This caused:
- Duplicate constructor lines
- Missing function declaration
- Syntax errors at lines 2012 and 2188

## Solution

1. **Restore `libvgpu_cuda.c` from working version** (local or git)
2. **Apply constructor priority fix manually** using proper editor or more precise sed
3. **Rebuild and test**

## Constructor Priority Explanation

GCC constructor priorities:
- **Priority 101** = runs early (before default 65535)
- **Default 65535** = runs late
- **Lower number = runs earlier**

By using priority 101, constructors should run BEFORE Ollama's discovery, ensuring device count = 1 is available when discovery checks.

## Expected Impact

Once all three shims have constructor priority 101:
- Constructors will run early (before discovery)
- Device count will be set to 1 before discovery runs
- Discovery should see `initial_count=1`
- `libggml-cuda.so` should be loaded
- GPU mode should be active

## Next Steps

1. Restore `libvgpu_cuda.c` from working version
2. Apply constructor priority fix carefully (line-by-line or using editor)
3. Rebuild and deploy
4. Verify constructor runs before discovery
5. Verify GPU mode is active
