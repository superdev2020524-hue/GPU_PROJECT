# Build Errors Found During New VM Testing

## Date: 2026-02-27

## Test Environment
- VM: test-11@10.25.33.111
- Test: Building GOAL register from scratch on new VM

## Errors Found

### 1. Compilation Errors in `libvgpu_cuda.c`

**Error Messages:**
```
/tmp/GOAL/BUILD/../SOURCE/libvgpu_cuda.c:1882:5: error: expected identifier or '(' before 'if'
/tmp/GOAL/BUILD/../SOURCE/libvgpu_cuda.c:1903:5: error: expected identifier or '(' before 'if'
/tmp/GOAL/BUILD/../SOURCE/libvgpu_cuda.c:1915:5: error: expected identifier or '(' before 'if'
/tmp/GOAL/BUILD/../SOURCE/libvgpu_cuda.c:1949:5: error: expected identifier or '(' before 'if'
/tmp/GOAL/BUILD/../SOURCE/libvgpu_cuda.c:2011:5: error: expected identifier or '(' before 'return'
/tmp/GOAL/BUILD/../SOURCE/libvgpu_cuda.c:2012:1: error: expected identifier or '(' before '}' token
```

**Root Cause:**
- Duplicate code block in `fgets()` function
- Lines 1834-1879 contain duplicate `skip_flag` handling code that's already handled at lines 1730-1764
- The comment at line 1834 says "DISABLED FOR TESTING (now restored):" but the code is NOT commented out
- This creates unreachable code and breaks function structure

**Location:**
- `SOURCE/libvgpu_cuda.c` lines 1834-1879

**Fix:**
- Remove or properly comment out the duplicate code block (lines 1834-1879)
- The correct version is in `phase3/guest-shim/libvgpu_cuda.c`

### 2. Compilation Error in `cuda_transport.c`

**Error Message:**
```
/tmp/GOAL/BUILD/../SOURCE/cuda_transport.c:988:13: error: static declaration of 'libvgpu_set_skip_interception' follows non-static declaration
```

**Root Cause:**
- Function `libvgpu_set_skip_interception()` is declared as `static` but may be referenced elsewhere
- The function uses runtime resolution via `dlsym()` which is correct, but the `static` keyword conflicts

**Location:**
- `SOURCE/cuda_transport.c` line 988

**Fix:**
- Verify the function declaration matches the working version in `phase3/guest-shim/cuda_transport.c`
- The correct version uses proper static declaration with runtime resolution

## Impact

- **libvgpu-cuda.so**: Failed to build (compilation errors)
- **libvgpu-nvml.so**: Failed to build (depends on cuda_transport.c)
- **libvgpu-cudart.so**: Built successfully (doesn't depend on problematic code)

## Solution

### Immediate Fix

The corrected source files are in `phase3/guest-shim/`:
- `phase3/guest-shim/libvgpu_cuda.c` - Correct version
- `phase3/guest-shim/cuda_transport.c` - Correct version

**Action Required:**
1. Copy corrected files to GOAL register:
   ```bash
   cp phase3/guest-shim/libvgpu_cuda.c phase3/GOAL/SOURCE/libvgpu_cuda.c
   cp phase3/guest-shim/cuda_transport.c phase3/GOAL/SOURCE/cuda_transport.c
   ```

2. Rebuild and test:
   ```bash
   cd phase3/GOAL/BUILD
   sudo bash install.sh
   ```

### Verification

After copying corrected files, verify:
- All 3 libraries build successfully
- No compilation errors
- Libraries install to `/usr/lib64/`
- Symlinks created correctly

## Status

- ✅ **Issue Identified**: Compilation errors found
- ✅ **Root Cause**: Duplicate code in `libvgpu_cuda.c`, static declaration issue in `cuda_transport.c`
- ✅ **Solution**: Copy corrected files from `phase3/guest-shim/`
- ⏳ **Action Required**: Update GOAL register with corrected source files

## Notes

- The GOAL register source files were copied from an earlier version that had these issues
- The working versions are in `phase3/guest-shim/`
- This demonstrates the importance of testing on a clean VM before finalizing the GOAL register
