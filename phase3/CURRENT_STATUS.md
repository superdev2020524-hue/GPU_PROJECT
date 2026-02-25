# Current Status: Libraries Loading But Not Called

## ✅ Major Breakthrough

**Both NVML and CUDA shims are now loading via LD_PRELOAD!**

### Evidence
- **Main process**: `libvgpu-nvml.so` and `libvgpu-cuda.so` in memory maps
- **Runner process**: Both libraries loaded (10 references total: 5 CUDA + 5 NVML)
- **LD_PRELOAD**: Set to `/usr/lib64/libvgpu-nvml.so:/usr/lib64/libvgpu-cuda.so`

### What Changed
Added `LD_PRELOAD` to systemd configuration:
```
Environment="LD_PRELOAD=/usr/lib64/libvgpu-nvml.so:/usr/lib64/libvgpu-cuda.so"
```

Even though the config previously said not to use LD_PRELOAD (because Go runtime clears it), it actually works for the main process and runner subprocesses inherit it.

## ⚠️ Current Problem

**Libraries are loaded but not being called:**
- GPU mode is still CPU
- No shim messages in logs
- Discovery doesn't seem to be calling our functions

## 🔍 Analysis

### Why Libraries Load But Aren't Called

1. **Constructors are empty** (for safety)
   - Our constructors are completely empty to prevent VM crashes
   - Initialization only happens when functions are called (lazy initialization)

2. **Discovery might not call functions**
   - Discovery might check for library existence but not call functions
   - Or discovery might fail before reaching our code
   - Or discovery uses a different mechanism

3. **dlopen() interception has safety delays**
   - Our dlopen() interception bypasses interception for first 20 calls
   - This might prevent discovery from using our shims

## 💡 Possible Solutions

### Option 1: Ensure Early Initialization
- Add safe initialization to constructors
- But this risks VM crashes (why we made them empty)

### Option 2: Verify Discovery Calls
- Use strace to see what discovery actually does
- Check if it calls NVML functions
- Check if it uses dlopen() or direct calls

### Option 3: Check Error Conditions
- Discovery might be failing silently
- Check if there are error conditions preventing discovery
- Verify if discovery logs show any errors

### Option 4: Force Function Calls
- If discovery checks library existence but doesn't call functions,
- We might need to ensure functions are called during library load
- But this risks early initialization issues

## 📋 Next Steps

1. **Verify discovery behavior**
   - Use strace to see what discovery actually does
   - Check if it calls NVML functions
   - Check if it uses dlopen() or direct calls

2. **Check if functions are being called**
   - Add more logging to see if functions are called
   - Check if messages are suppressed
   - Verify if stderr is being captured

3. **Consider safe initialization**
   - If discovery doesn't call functions, we might need early init
   - But must be extremely careful to avoid VM crashes
   - Test thoroughly before deploying

## 🎯 Key Insight

**Libraries ARE loading, which is huge progress!**

The fact that both libraries are in process memory means:
- LD_PRELOAD works
- Libraries are accessible
- The infrastructure is in place

Now we just need to ensure discovery actually calls our functions.
