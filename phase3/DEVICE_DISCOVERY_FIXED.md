# Device Discovery Fixed!

## ✅ Major Progress

**Device discovery is now working:**
- ✅ Class is read correctly (`class=0x030200` instead of `0x002331`)
- ✅ Device discovery succeeds in serve process
- ✅ `cuInit()` finds device: "device found at 0000:00:05.0"

## 🔧 What Was Fixed

**Bug:** `fgets()` interception was using `strstr()` which matched `/devices/` when checking for `/device`, causing wrong values to be returned.

**Fix:** Changed to exact path matching using `strcmp()` on the end of the path.

## ❌ Remaining Issue

**GPU mode is still CPU:**
- Device discovery works in serve process
- But GPU mode doesn't activate
- Need to check if runner subprocess also needs device discovery

## 🎯 Next Steps

1. **Check if runner subprocess calls cuInit()**
   - Runner might also need to discover device
   - Or discovery might happen in runner, not serve

2. **Verify discovery happens in the right process**
   - Maybe discovery needs to happen in runner?
   - Or both processes need it?

3. **Check if Ollama needs additional information**
   - Maybe cuInit() isn't enough?
   - Or maybe there are other checks after cuInit()?

## 💡 Key Insight

**Device discovery is fixed, but GPU mode still doesn't activate.**

This suggests that either:
- Discovery needs to happen in runner subprocess too
- Or Ollama needs additional information beyond just cuInit() succeeding
