# Confirmed: Functions Are NOT Being Called

## ✅ Final Verification

**Functions are NOT being called, despite all infrastructure working:**

1. **Libraries ARE loaded** ✓
   - 10 references to libvgpu libraries in process memory
   - Both CUDA and NVML shims present

2. **Symbols ARE resolvable** ✓
   - `dlsym()` can find `cuInit` at 0x73262803b5a0
   - `dlsym()` can find `nvmlInit_v2` at 0x73262802b720

3. **Discovery IS happening** ✓
   - Log shows: "discovering available GPUs..."
   - strace shows: opens `/usr/local/lib/ollama/libcuda.so.1` (our symlink)

4. **Functions are exported** ✓
   - All required symbols present in libraries

## ❌ The Problem

**Functions are NEVER called!**

### Evidence

- **No log messages**: Despite logging in:
  - `cuInit()` - no messages
  - `nvmlInit_v2()` - no messages  
  - `ensure_init()` - no messages
  - `cuDeviceGetCount()` - no messages

- **Discovery happens but doesn't call functions**
  - "discovering available GPUs..." appears
  - But no function calls follow
  - GPU mode stays CPU

## 🔍 Root Cause

**Ollama's discovery process:**
1. Opens our library (confirmed via strace) ✓
2. Loads it into memory (confirmed via process maps) ✓
3. Can resolve symbols (confirmed via dlsym test) ✓
4. **But NEVER calls the initialization functions** ✗

## 💡 Why This Happens

### Possible Reasons

1. **Discovery uses a different mechanism**
   - Maybe checks library existence but doesn't call functions
   - Or uses a wrapper that fails before calling functions
   - Or uses a different API path entirely

2. **Discovery fails early**
   - Maybe checks prerequisites first (like `/proc/driver/nvidia/version`)
   - If prerequisite checks fail, never reaches function calls
   - Or uses a different discovery method that doesn't use our functions

3. **NVML discovery fails first**
   - Ollama might use NVML for initial discovery
   - If NVML discovery fails, never loads CUDA
   - Even though NVML shim is loaded, it's not called

4. **Wrapper function issues**
   - Ollama has `ggml_nvml_init` wrapper
   - Wrapper might check library loading but not call functions
   - Or wrapper fails before reaching our functions

## 🎯 The Solution

Since discovery doesn't call our functions, we have two options:

### Option 1: Force Early Initialization (Risky)

Add initialization to constructor that:
- Only runs for application processes
- Only runs when safe
- Initializes CUDA/NVML early

**Risk**: Could cause VM crashes if not done carefully

### Option 2: Make Discovery Call Functions (Preferred)

Understand why discovery doesn't call functions and fix it:
- Check if there are prerequisite checks that fail
- Verify if wrapper functions are the issue
- Ensure discovery actually invokes initialization

**Risk**: Lower, but requires understanding Ollama's discovery mechanism

## 📋 Next Steps

1. **Understand discovery mechanism**
   - How does Ollama actually do discovery?
   - What are the prerequisite checks?
   - Why doesn't it call functions?

2. **Consider early initialization**
   - If discovery won't call functions, we need early init
   - But must be extremely careful to avoid VM crashes
   - Test thoroughly before deploying

3. **Check NVML discovery**
   - Maybe NVML discovery fails first
   - If so, fix NVML discovery to allow CUDA to load

## 🎯 Key Insight

**Everything works except function invocation:**
- Libraries load ✓
- Symbols resolve ✓
- Discovery happens ✓
- But functions are never called ✗

This is why GPU mode stays CPU - discovery loads our libraries but never invokes the initialization functions that would activate GPU mode.
