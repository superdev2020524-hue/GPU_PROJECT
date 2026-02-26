# Next Step: Discovery Timing Issue

## Date: 2026-02-26

## ✅ Constructor Fix Complete

The Runtime API shim constructor fix is **complete and working**:
- LD_PRELOAD order fixed (Driver API before Runtime API)
- Constructor calls cuInit() directly as external function
- Device count functions return count=1
- All constructor functions working correctly

## ⚠️ Remaining Issue

**`libggml-cuda.so` is NOT being loaded**

Even though:
- Constructor works ✓
- Device count functions return count=1 ✓
- NVML device count function implemented (returns count=1) ✓

Discovery still shows `initial_count=0` and doesn't load `libggml-cuda.so`.

## 🔍 Root Cause Hypothesis

Ollama's discovery might be:
1. **Checking device count BEFORE our constructors run**
   - Discovery might happen very early in the process
   - Our constructors might run too late
   - Device count might be checked before it's set to 1

2. **Using a different method that doesn't call our functions**
   - Discovery might use `dlsym()` to find functions directly
   - Might check library symbols instead of calling functions
   - Might use a different API we're not intercepting

3. **Checking something else we're not intercepting**
   - Might check PCI devices directly
   - Might check library loading capability
   - Might use a different discovery mechanism

## 📋 Next Steps

1. **Verify discovery timing**
   - Check when discovery runs vs when constructors run
   - Ensure device count is available BEFORE discovery runs
   - Consider making device count available even earlier

2. **Check discovery mechanism**
   - Verify if discovery uses `dlsym()` to find functions
   - Check if discovery calls our functions or uses different method
   - Understand exactly how Ollama determines `initial_count`

3. **Ensure early availability**
   - Make device count available at library load time
   - Consider using constructor priorities
   - Ensure functions return correct values immediately

4. **Alternative approach**
   - Intercept discovery at a different level
   - Modify how Ollama checks for GPUs
   - Ensure discovery sees device count = 1

## Key Insight

**The constructor fix is working, but discovery might be checking device count before constructors run or using a different method.**

We need to ensure device count = 1 is available BEFORE discovery runs, or intercept discovery at a different level.
