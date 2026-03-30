# Root Cause Confirmed

## ✅ Final Verification

**All infrastructure works:**
- ✅ Libraries ARE loaded in runner (10 references in process memory)
- ✅ Symbols ARE resolvable (dlsym() test passed)
- ✅ LD_PRELOAD is set in runner
- ✅ Early interception works for all prerequisite checks
- ✅ Discovery happens ("discovering available GPUs...")

**But functions are NEVER called:**
- ❌ No `cuInit()` messages (despite fprintf() at START of function)
- ❌ No `nvmlInit_v2()` messages (despite fprintf() at START of function)
- ❌ No `ensure_init()` messages
- ❌ GPU mode stays CPU

## 🔍 Root Cause

**Ollama's discovery mechanism:**
1. Loads libraries via `dlopen()` ✓
2. Can resolve symbols via `dlsym()` ✓
3. **But NEVER calls initialization functions** ✗

This is why GPU mode stays CPU - discovery loads our libraries but never invokes the initialization functions that would activate GPU mode.

## 💡 Why This Happens

Ollama's discovery likely:
- Checks if library is valid (can be loaded)
- Checks if symbols exist (can be resolved)
- But doesn't actually call functions until later
- Or uses a wrapper that fails before calling functions
- Or discovery happens in a way that doesn't require function calls

## 🎯 Solution

Since discovery won't call functions, we need **early initialization in constructor**.

**But must be extremely careful:**
- Only for application processes
- Only when safe to check process type
- Minimal initialization (just set flags)
- Actual heavy init happens on first function call

Since we're using `LD_PRELOAD` (not `/etc/ld.so.preload`), constructor should be safer because:
- Only loads into processes that have LD_PRELOAD set
- Not loaded into ALL system processes
- Can check process type before initializing

## 📋 Next Steps

1. **Add safe early initialization to constructor**
   - Check if it's safe to check process type
   - Check if it's an application process
   - If both pass, call `cuInit()` and `nvmlInit_v2()` early
   - But only minimal initialization, heavy init on first call

2. **Test thoroughly**
   - Ensure no VM crashes
   - Verify GPU mode activates
   - Check that functions are called
