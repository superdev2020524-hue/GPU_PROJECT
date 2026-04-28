# Critical Finding: Functions Are NOT Being Called

## ✅ Confirmed

1. **Libraries ARE loaded**
   - 10 references to libvgpu libraries in runner process
   - Both CUDA and NVML shims in memory
   - LD_PRELOAD is working

2. **Discovery is happening**
   - Log shows: "discovering available GPUs..."
   - strace shows: opens `/usr/local/lib/ollama/libcuda.so.1` (our symlink)

3. **Functions are exported**
   - `cuInit` exists in library
   - `nvmlInit_v2` exists in library
   - All required symbols present

## ❌ The Problem

**Functions are NOT being called!**

### Evidence

- **No log messages**: Despite logging in `cuInit()` and `nvmlInit_v2()`, no messages appear
- **GPU mode is CPU**: Discovery happens but reports CPU
- **Libraries loaded but unused**: Libraries are in memory but functions aren't invoked

### What This Means

**Ollama's discovery process:**
1. Opens our library (confirmed via strace)
2. Loads it into memory (confirmed via process maps)
3. But **never calls** `cuInit()` or `nvmlInit_v2()`

## 🔍 Why Functions Aren't Called

### Possible Reasons

1. **Discovery uses a different mechanism**
   - Maybe checks library existence but doesn't call functions
   - Or uses a wrapper that fails before calling our functions
   - Or uses a different API path

2. **Discovery fails early**
   - Maybe checks something else first (like `/proc/driver/nvidia/version`)
   - If that check fails, never reaches function calls
   - Or uses a different discovery method

3. **Library loaded but symbols not resolved**
   - Maybe `dlsym()` fails to find our functions
   - Or functions are found but not called
   - Or there's a symbol versioning issue

4. **NVML discovery fails first**
   - Ollama might use NVML for initial discovery
   - If NVML discovery fails, never loads CUDA
   - Even though NVML shim is loaded, it's not called

## 💡 Next Steps

1. **Verify symbol resolution**
   - Check if `dlsym()` can find our functions
   - Test if functions are callable when loaded
   - Verify symbol versioning

2. **Check discovery mechanism**
   - Understand how Ollama actually does discovery
   - Check if it uses wrapper functions
   - Verify if there are prerequisite checks

3. **Consider early initialization**
   - If discovery doesn't call functions, we might need early init
   - But must be extremely careful to avoid VM crashes
   - Test thoroughly before deploying

4. **Check NVML discovery**
   - strace didn't show NVML opens
   - Maybe NVML discovery happens differently
   - Or NVML discovery fails silently

## 🎯 Key Insight

**The infrastructure works perfectly:**
- Libraries load ✓
- Symlinks work ✓
- Discovery finds our library ✓
- Functions are exported ✓

**But discovery never calls our functions.**

This is why GPU mode stays CPU - discovery doesn't actually invoke the functions that would initialize CUDA/NVML.
