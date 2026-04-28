# Actual Investigation Findings - No Assumptions

## Verified Facts

### 1. Main Process Has NO CUDA/NVML Libraries
- **PID**: 51795 (`/usr/local/bin/ollama serve`)
- **Libraries in memory**: Standard libraries only (libc, libm, libstdc++, libdl, librt)
- **CUDA/NVML libraries**: **NONE FOUND**
- **Conclusion**: Libraries are NOT loaded at startup in the main process

### 2. Process Count
- Only ONE ollama process exists (main `ollama serve`)
- No runner subprocesses found during idle state
- Runner subprocesses may be created on-demand during inference

### 3. Bundled Libraries Status
- `/usr/local/lib/ollama/cuda_v12/libcuda.so.1` → points to our shim ✓
- `/usr/local/lib/ollama/cuda_v13/libcuda.so.1` → points to our shim ✓
- Symlinks are correct

### 4. System Library Symlinks
- All system library symlinks point to our shims ✓
- `/usr/local/lib/libcuda.so.1` → our shim ✓
- `/usr/lib64/libcuda.so.1` → our shim ✓
- `/usr/lib/x86_64-linux-gnu/libcuda.so.1` → our shim ✓

### 5. Shim Files Exist
- `/usr/lib64/libvgpu-cuda.so` exists
- `/usr/lib64/libvgpu-nvml.so` exists
- Both have correct SONAMEs

## What We DON'T Know Yet

### 1. When Do Libraries Load?
- **Unknown**: When does Ollama actually call `dlopen()`?
- **Unknown**: Does it happen during discovery or only when CUDA backend is needed?
- **Unknown**: Does it happen in main process or runner subprocesses?

### 2. Why Aren't Libraries Loading?
- **Possible reasons**:
  1. Ollama never calls `dlopen()` because discovery fails early
  2. `dlopen()` is called but fails silently
  3. Libraries load in runner subprocesses (not checked yet)
  4. Go runtime prevents library loading
  5. Some condition prevents library loading

### 3. Is Discovery Actually Running?
- **Unknown**: Does "discovering available GPUs..." log mean discovery is running?
- **Unknown**: What happens during discovery?
- **Unknown**: Does discovery succeed or fail?

### 4. Are Runner Subprocesses Created?
- **Unknown**: When are runner subprocesses created?
- **Unknown**: Do they have libraries loaded?
- **Not checked**: Need to check during active inference

## What Needs to Be Checked

### Immediate Checks Needed

1. **Check runner subprocesses during inference**
   - Trigger inference
   - Check if runner subprocesses are created
   - Check if they have CUDA/NVML libraries loaded

2. **Trace actual dlopen() calls**
   - Use `strace` or `ltrace` to see if `dlopen()` is called
   - See what paths are used
   - See if calls succeed or fail

3. **Check discovery process in detail**
   - What exactly happens during "discovering available GPUs..."?
   - Does NVML discovery succeed?
   - Does it proceed to CUDA loading?

4. **Check for errors**
   - Are there any errors during discovery?
   - Are libraries failing to load silently?
   - Check stderr for any messages

5. **Verify NVML shim works**
   - Test NVML shim manually
   - Verify `nvmlInit_v2()` works
   - Check if Ollama can find NVML functions

## Honest Assessment

### What I Know For Sure
1. Main process has NO CUDA/NVML libraries loaded
2. All symlinks are correct
3. Shim files exist and have correct SONAMEs
4. Bundled libraries point to our shims

### What I Assumed (But Don't Know)
1. ❌ I assumed libraries should load at startup - **WRONG**
2. ❌ I assumed discovery happens in main process - **UNKNOWN**
3. ❌ I assumed `dlopen()` is called - **NOT VERIFIED**
4. ❌ I assumed runner subprocesses exist - **NOT FOUND**

### What I Need to Actually Check
1. ✅ When do libraries actually load? (Need to trace)
2. ✅ Do runner subprocesses have libraries? (Need to check during inference)
3. ✅ Is `dlopen()` called? (Need to trace)
4. ✅ Does discovery succeed? (Need to check logs in detail)
5. ✅ Are there any errors? (Need to check stderr)

## Next Steps - Proper Investigation

1. **During active inference**:
   - Check all processes (main + runners)
   - Check libraries in each process
   - See when libraries load

2. **Trace library loading**:
   - Use `strace -e trace=openat,open` to see file opens
   - Use `ltrace` to see `dlopen()` calls
   - See what actually happens

3. **Check discovery logs in detail**:
   - Get full discovery log output
   - See if NVML discovery succeeds
   - See if it proceeds to CUDA

4. **Test shims manually**:
   - Verify NVML shim works
   - Verify CUDA shim works
   - See if they can be loaded

5. **Check for errors**:
   - Look for any error messages
   - Check if libraries fail to load
   - See what prevents loading

## Conclusion

I made assumptions about when and how libraries load. The actual fact is:
- **Main process has NO CUDA/NVML libraries**
- **Libraries are NOT loaded at startup**

I need to investigate:
- **WHEN** libraries actually load (if at all)
- **WHERE** they load (main process or runners)
- **WHY** they're not loading (if they should be)

This requires actual tracing and monitoring, not assumptions.
