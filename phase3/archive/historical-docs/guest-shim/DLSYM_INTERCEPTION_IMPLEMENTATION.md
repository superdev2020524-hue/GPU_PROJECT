# dlsym Interception Implementation

## Implementation Summary

### Route 2.1: Safe dlsym() Interception
**Status:** ✅ Completed

**File:** `phase3/guest-shim/libvgpu_cuda.c` (lines 297-467)

**Implementation Details:**
- Added `dlsym()` interception function to catch CUDA function lookups from `libggml-cuda.so`
- Uses safe bootstrap mechanism to avoid infinite recursion
- Logs all CUDA function lookups (functions starting with "cu" or "cuda")
- Redirects CUDA function lookups to our shims when possible using `RTLD_DEFAULT`

**Bootstrap Strategy:**
- Uses `initialized` flag to prevent re-entry into bootstrap
- Uses `bootstrap_guard` to prevent infinite recursion
- Attempts to get real `dlsym` via `dlsym(RTLD_NEXT, "dlsym")` with recursion protection
- Falls back gracefully if bootstrap fails

**Logging:**
- Logs all CUDA function lookups with handle and symbol name
- Logs when functions are redirected to our shims
- Logs when functions are found via real_dlsym

### Route 2.2: Enhanced dlopen() Interception
**Status:** ✅ Completed

**File:** `phase3/guest-shim/libvgpu_cuda.c` (lines 258-272)

**Enhancement:**
- Added logging for `libggml-cuda.so` loads
- Logs when `libggml-cuda.so` is loaded via `dlopen()`
- Notes that our `dlsym()` interceptor will catch its function lookups

### Route 2.3: Runtime API Shim dlsym Logging
**Status:** ⏭️ Skipped (Optional)

**Reason:** The Runtime API shim (`libvgpu_cudart.c`) doesn't need separate dlsym interception since:
- The Driver API shim already intercepts dlsym for all CUDA functions
- Runtime API functions are called directly, not via dlsym
- Adding dlsym interception here could cause conflicts

### Route 2.4: Build and Test
**Status:** ✅ Ready for Testing

**Build Requirements:**
- `-ldl` flag: ✅ Already included in `install.sh` (line 208)
- `-D_GNU_SOURCE`: ✅ Already defined in build command (line 202)
- Version script: ✅ Not needed for Driver API shim

**Next Steps:**
1. Rebuild shims using `install.sh`
2. Restart Ollama service
3. Check logs for dlsym interception messages
4. Verify CUDA function lookups are being caught and redirected

## Expected Behavior

### When libggml-cuda.so loads:
1. `dlopen("libggml-cuda.so")` is called
2. Our interceptor logs: `"[libvgpu-cuda] dlopen(...) - libggml-cuda.so loading"`
3. `libggml-cuda.so` initializes and may call `dlsym()` to resolve CUDA functions

### When libggml-cuda.so calls dlsym():
1. Our `dlsym()` interceptor is called
2. Logs: `"[libvgpu-cuda] dlsym(handle=..., "cuDeviceGetAttribute") called"`
3. Tries to resolve from our shim using `RTLD_DEFAULT`
4. If found, logs: `"[libvgpu-cuda] dlsym() REDIRECTED "cuDeviceGetAttribute" to shim"`
5. Returns our shim function, which returns compute capability 9.0

### Success Criteria:
- ✅ dlsym interception logs show CUDA function lookups
- ✅ Function redirection works (our shims are called)
- ✅ `compute=9.0` appears in Ollama logs (instead of `compute=0.0`)
- ✅ Device is not filtered as "didn't fully initialize"
- ✅ Ollama uses GPU mode instead of CPU mode

## Risk Mitigation

### Bootstrap Issues:
- ✅ Uses `initialized` flag to prevent re-entry
- ✅ Uses `bootstrap_guard` to prevent infinite recursion
- ✅ Falls back gracefully if bootstrap fails

### Recursion Issues:
- ✅ Static flags prevent re-entry
- ✅ Caches `real_dlsym` pointer
- ✅ Avoids calling our own `dlsym()` during bootstrap (with guard)

### Performance:
- ✅ Only intercepts CUDA-related functions (fast string comparison)
- ✅ Uses `strncmp` for fast prefix matching
- ✅ Minimal overhead for non-CUDA function lookups

### Compatibility:
- ⚠️ Needs testing with different glibc versions
- ⚠️ Needs testing with Go binaries (CGo)
- ⚠️ Needs verification it doesn't break system libraries

## Files Modified

1. **`phase3/guest-shim/libvgpu_cuda.c`**
   - Added `dlsym()` interception function (lines 297-467)
   - Enhanced `dlopen()` logging for `libggml-cuda.so` (lines 258-272)

2. **`phase3/guest-shim/OLLAMA_COMPUTE_CAPABILITY_SOURCE.md`** (new)
   - Documents Route 1 research findings

3. **`phase3/guest-shim/DLSYM_INTERCEPTION_IMPLEMENTATION.md`** (this file)
   - Documents Route 2 implementation details
