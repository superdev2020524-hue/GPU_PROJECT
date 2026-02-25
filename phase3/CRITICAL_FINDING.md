# Critical Finding: Libraries Are NOT Loading

## Verified Fact

**Main process has ZERO CUDA/NVML libraries loaded.**

This is the root cause. Despite all our fixes:
- ✅ Correct SONAMEs
- ✅ Correct symlinks (all paths)
- ✅ Bundled libraries point to shims
- ✅ Shim files exist

**Libraries are still NOT loading into the process.**

## What This Means

1. **Ollama is NOT calling `dlopen()` for CUDA/NVML libraries**
   - OR
2. **`dlopen()` is being called but failing silently**
   - OR
3. **Libraries load in a different process (runner subprocesses) that we haven't checked yet**
   - OR
4. **Ollama uses a different mechanism to load libraries**

## Next Critical Steps

1. **Verify if runner subprocesses have libraries**
   - Need to check during active inference
   - May require different approach (script on VM)

2. **Trace actual `dlopen()` calls**
   - Use `ltrace` or `strace` to see if called
   - See what paths are used
   - See if calls succeed

3. **Check if NVML discovery succeeds**
   - If NVML fails, CUDA may never load
   - Need to verify NVML shim is working

4. **Check for errors**
   - Libraries may be failing to load silently
   - Need to check stderr and logs

## The Real Question

**Why isn't Ollama loading CUDA/NVML libraries at all?**

This is the fundamental question that needs to be answered before we can fix the issue.
