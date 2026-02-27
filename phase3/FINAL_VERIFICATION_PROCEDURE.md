# Final Verification Procedure

## Date: 2026-02-27

## Purpose
Verify that all fixes are working and the H100 GPU is detected and used by Ollama.

---

## Step 1: Fresh Restart and Bootstrap Discovery Verification

### 1.1 Restart Ollama Service
```bash
sudo systemctl restart ollama
sleep 8
```

### 1.2 Trigger Bootstrap Discovery
```bash
timeout 5 ollama list 2>&1
```

### 1.3 Check Bootstrap Discovery Logs
```bash
cat /tmp/ollama_stderr.log 2>&1 | strings | grep -E 'bootstrap discovery|initial_count|evaluating.*devices' | tail -10
```

**Expected Output:**
- `bootstrap discovery took` with duration
- `initial_count=1` (NOT 0)
- No errors about device filtering

### 1.4 Check GGML CHECK Logs
```bash
cat /tmp/ollama_stderr.log 2>&1 | strings | grep -E 'GGML CHECK|VERIFY.*Direct|CC_major' | tail -10
```

**Expected Output:**
- `[GGML CHECK] major=9 minor=0 multiProcessorCount=132 totalGlobalMem=85899345920 warpSize=32`
- `[libvgpu-cudart] VERIFY: Direct memory at 0x148/0x14C: major=9 minor=0`
- `[libvgpu-cudart] cudaGetDeviceProperties_v2() returning: name=..., CC_major=9 CC_minor=0`

---

## Step 2: Device Detection During Model Execution

### 2.1 Check Device Detection Logs
```bash
cat /tmp/ollama_stderr.log 2>&1 | strings | grep -E 'found.*CUDA devices|ggml_cuda_init' | tail -5
```

**Expected Output:**
- `ggml_cuda_init: found 1 CUDA devices:`
- Device properties logged correctly

### 2.2 Verify CUDA API Calls
```bash
cat /tmp/ollama_stderr.log 2>&1 | strings | grep -E 'cudaGetDeviceCount.*SUCCESS|cuDeviceGetCount.*SUCCESS' | tail -5
```

**Expected Output:**
- `[libvgpu-cudart] cudaGetDeviceCount() SUCCESS: returning count=1`
- `[libvgpu-cuda] cuDeviceGetCount() SUCCESS: returning count=1`

---

## Step 3: Model Execution Verification

### 3.1 Run a Small Model
```bash
ollama run llama3.2:1b "Hello, how are you?"
```

**Expected Behavior:**
- Model loads successfully
- Response generated
- No errors about GPU unavailable

### 3.2 Check Model Execution Logs
```bash
cat /tmp/ollama_stderr.log 2>&1 | strings | tail -100 | grep -E 'cudaMalloc|cudaMemcpy|found.*CUDA|Device 0' | tail -10
```

**Expected Output:**
- CUDA memory operations logged
- Device 0 referenced
- No CPU fallback messages

### 3.3 Verify GPU Usage (Optional)
```bash
# If nvidia-smi or similar tools are available
# Check that GPU is being used during model execution
```

---

## Step 4: Debug Mode Verification (Optional)

### 4.1 Enable Debug Mode
```bash
export OLLAMA_DEBUG=1
sudo systemctl restart ollama
sleep 8
```

### 4.2 Run Commands with Debug
```bash
ollama list
ollama run llama3.2:1b "Test"
```

### 4.3 Check Debug Logs
```bash
cat /tmp/ollama_stderr.log 2>&1 | strings | grep -E 'DEBUG|bootstrap|initial_count|library=' | tail -20
```

---

## Step 5: Troubleshooting (If Issues Persist)

### 5.1 Clear Cache (If initial_count Still 0)
```bash
rm -rf ~/.ollama/cache/*
sudo systemctl restart ollama
sleep 8
```

### 5.2 Verify Library Loading
```bash
ldd /usr/lib64/libvgpu-cudart.so | head -10
strings /usr/lib64/libvgpu-cudart.so | grep -E 'GGML CHECK|computeCapability' | head -5
```

### 5.3 Check File Permissions
```bash
ls -l /usr/lib64/libvgpu-*.so
ls -l ~/.ollama/models/blobs/ 2>/dev/null | head -5
```

### 5.4 Verify Structure Layout
```bash
# Check that computeCapabilityMajor/Minor are in the library
nm -D /usr/lib64/libvgpu-cudart.so | grep -E 'computeCapability|cudaGetDeviceProperties'
```

---

## Success Criteria

### ✅ Bootstrap Discovery
- `initial_count=1` in logs
- GGML CHECK logs show correct values (major=9, minor=0, SM=132, mem=80GB)
- No device filtering errors

### ✅ Model Execution
- `ggml_cuda_init: found 1 CUDA devices:` in logs
- Model runs successfully
- CUDA memory operations logged
- No CPU fallback

### ✅ All APIs Working
- `cuDeviceGetCount()` returns 1
- `cudaGetDeviceCount()` returns 1
- `nvmlInit()` succeeds
- `cudaGetDeviceProperties_v2()` returns correct values

---

## Quick Verification Command

Run this single command to check all key indicators:

```bash
sudo systemctl restart ollama && sleep 8 && timeout 5 ollama list 2>&1 > /dev/null && sleep 3 && echo "=== BOOTSTRAP DISCOVERY ===" && cat /tmp/ollama_stderr.log 2>&1 | strings | grep -E 'bootstrap discovery|initial_count' | tail -5 && echo "=== GGML CHECK ===" && cat /tmp/ollama_stderr.log 2>&1 | strings | grep -E 'GGML CHECK|VERIFY.*Direct' | tail -5 && echo "=== DEVICE DETECTION ===" && cat /tmp/ollama_stderr.log 2>&1 | strings | grep -E 'found.*CUDA devices|ggml_cuda_init' | tail -5
```

---

## Expected Final Status

Once all verifications pass:
- ✅ Bootstrap discovery: `initial_count=1`
- ✅ Device properties: Compute capability 9.0, 132 SMs, 80GB memory
- ✅ Model execution: GPU detected and used
- ✅ All CUDA/NVML APIs: Working correctly

**H100 GPU integration is fully verified and working!**
