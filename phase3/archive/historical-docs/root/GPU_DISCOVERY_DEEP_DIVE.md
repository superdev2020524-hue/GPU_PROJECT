# GPU Discovery Deep Dive - Analysis and Solution

## Root Cause Identified

After deep investigation, the root cause of GPU mode not activating is:

### The Problem

1. **Libraries are NOT loading into the process** - Despite correct symlinks, `libcuda.so.1` and `libnvidia-ml.so.1` are not appearing in process memory maps
2. **cuInit() must be called early** - According to `COMPLETE_SOLUTION.txt`, `cuInit()` must be called BEFORE `cuDeviceGetPCIBusId()` for Ollama's GPU matching to work
3. **Constructor is empty for safety** - We made the constructor completely empty to prevent VM crashes, but this means `cuInit()` is never called early
4. **Lazy initialization may be too late** - `ensure_init()` calls `cuInit()` lazily, but if Ollama calls `cuDeviceGetPCIBusId()` before other CUDA functions, the matching may fail

### Current Flow

1. Ollama starts and logs "discovering available GPUs..."
2. Ollama tries to load `libcuda.so.1` and `libnvidia-ml.so.1` via `dlopen()`
3. **PROBLEM**: Libraries are not loading (not in process memory maps)
4. If libraries don't load, `cuInit()` is never called
5. If `cuInit()` is never called, GPU discovery fails
6. Result: `library=cpu`

### Why Libraries Aren't Loading

Possible reasons:
1. **Symlinks point to wrong library** - Some symlinks still point to real NVIDIA library instead of shim
2. **Library resolution fails** - `dlopen()` can't find the library despite symlinks
3. **Go runtime bypasses symlinks** - Go may use direct syscalls that bypass normal library resolution
4. **Library not in search path** - Despite `LD_LIBRARY_PATH`, library isn't found

## Evidence from Investigation

### What We Found

1. **Process Memory Maps**: NO libraries found (`libcuda`, `libvgpu`, `libnvidia-ml` not in `/proc/PID/maps`)
2. **Symlinks**: Most are correct, but some NVML symlinks may still point to real library
3. **Environment**: `LD_LIBRARY_PATH` is set correctly in systemd
4. **Service**: Running without `force_load_shim` wrapper ✓
5. **GPU Mode**: Still showing `library=cpu` ⚠

### Key Insight from Codebase

From `COMPLETE_SOLUTION.txt`:
- **Ollama requires `cuInit()` to be called BEFORE `cuDeviceGetPCIBusId()`**
- **The matching requires cuInit() to be called BEFORE cuDeviceGetPCIBusId()**
- **If cuInit() isn't called early enough, the matching fails**
- **Solution**: Call `cuInit(0)` in library constructor

But we can't do this safely because:
- Constructor loads into ALL processes (systemd, sshd, etc.)
- Early initialization is unsafe (libc/pthreads not ready)
- Can cause VM crashes

## Solution Options

### Option 1: Safe Early Initialization in ensure_init()

Modify `ensure_init()` to be more aggressive about calling `cuInit()` early, but only for application processes:

```c
static int ensure_init(void)
{
    /* For application processes, try to initialize immediately */
    if (is_safe_to_check_process() && is_application_process()) {
        if (!g_initialized) {
            /* Call cuInit() immediately for application processes */
            CUresult rc = cuInit(0);
            if (rc == CUDA_SUCCESS) {
                return CUDA_SUCCESS;
            }
        }
    }
    /* ... rest of function ... */
}
```

**Pros**: Safe, only initializes for application processes
**Cons**: Still relies on `is_application_process()` which may fail

### Option 2: Force Library Loading via LD_PRELOAD

Use `LD_PRELOAD` in systemd to force libraries to load:

```ini
[Service]
Environment="LD_PRELOAD=/usr/lib64/libvgpu-cuda.so:/usr/lib64/libvgpu-nvml.so"
```

**Pros**: Forces libraries to load
**Cons**: Go runtime may clear `LD_PRELOAD`, and we're avoiding this method

### Option 3: Fix Symlinks and Ensure Libraries Load

1. Verify ALL symlinks point to shim (not real library)
2. Ensure libraries are in `ldconfig` cache
3. Test that `dlopen("libcuda.so.1")` actually works
4. Check if Go runtime is bypassing normal library resolution

**Pros**: Addresses root cause (libraries not loading)
**Cons**: May not work if Go uses direct syscalls

### Option 4: Use force_load_shim Wrapper (Rejected)

We removed this because it was causing issues, but it might be necessary if symlinks don't work.

## Recommended Solution

**Combination of Option 1 and Option 3**:

1. **Fix all symlinks** - Ensure every `libcuda.so.1` and `libnvidia-ml.so.1` symlink points to our shim
2. **Make ensure_init() more aggressive** - Call `cuInit()` immediately when safe for application processes
3. **Add LD_PRELOAD as backup** - Even though Go may clear it, it might help for initial loading
4. **Verify library loading** - Check process memory maps after restart to confirm libraries load

## Next Steps

1. Fix remaining NVML symlinks that point to real library
2. Modify `ensure_init()` to call `cuInit()` more aggressively
3. Add `LD_PRELOAD` to systemd as backup mechanism
4. Restart Ollama and verify libraries load
5. Check GPU mode after inference

## Testing Checklist

- [ ] All symlinks point to shim (CUDA and NVML)
- [ ] Libraries appear in process memory maps
- [ ] `cuInit()` is called early (check logs)
- [ ] GPU mode shows `library=cuda` (not `library=cpu`)
- [ ] No VM crashes or system instability
