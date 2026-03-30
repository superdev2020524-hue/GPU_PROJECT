# SHIM Extension Plan: Making Ollama Discovery Work

## Critical Discovery

From VM investigation, we found that **Ollama scans PCI devices directly** by reading:
- `/sys/bus/pci/devices/*/vendor`
- `/sys/bus/pci/devices/*/device`  
- `/sys/bus/pci/devices/*/class`

Our vGPU device exists at `0000:00:05.0` with:
- Vendor: `0x10de` (NVIDIA) ✓
- Device: `0x2331` (H100 PCIe) ✓
- Class: `0x030200` (3D controller) ✓

**But Ollama still reports `library=cpu`**, which means:
1. Ollama finds the PCI device but can't match it with NVML devices
2. Or Ollama's discovery happens in a subprocess that doesn't inherit `LD_PRELOAD`
3. Or Ollama uses a different discovery path that bypasses NVML

## Research Findings

From web search, key insights:
1. **Ollama stops searching after first library load failure** - doesn't try alternative paths
2. **Ollama uses `dlsym()` to load NVML functions** - if symbols fail, discovery fails
3. **Ollama matches PCI devices with NVML devices by PCI bus ID** - format must match exactly
4. **Discovery may happen in subprocess** - `LD_PRELOAD` might not be inherited

## Extension Strategy

### Phase 1: Intercept PCI Device File Reads

**Problem**: Ollama reads PCI device files directly. We need to ensure it sees correct values.

**Solution**: Intercept `read()`, `pread()`, `readv()`, and `preadv()` calls to PCI device files.

**Implementation**:
- Add `read()` interception for `/sys/bus/pci/devices/*/vendor`, `device`, `class`
- Return correct values: `0x10de`, `0x2331`, `0x030200`
- Log all PCI device file reads for debugging

### Phase 2: Intercept dlsym Calls

**Problem**: Ollama uses `dlsym()` to load NVML functions. If symbols aren't found, discovery fails.

**Solution**: Intercept `dlsym()` calls and log what symbols Ollama is looking for.

**Implementation**:
- Add `dlsym()` interception with logging
- Ensure all required NVML symbols are exported
- Provide fallback for missing symbols

### Phase 3: Ensure Discovery in Main Process

**Problem**: Discovery might happen in a subprocess that doesn't inherit `LD_PRELOAD`.

**Solution**: Ensure shim is loaded in all processes, including subprocesses.

**Implementation**:
- Add constructor to both shims that logs process info
- Check if discovery happens in subprocess
- If needed, inject shim via `LD_PRELOAD` in subprocess environment

### Phase 4: Intercept Directory Scanning

**Problem**: Ollama scans `/sys/bus/pci/devices/` directory. We need to ensure our device is visible.

**Solution**: Intercept `readdir()`, `readdir64()`, and `opendir()` calls.

**Implementation**:
- Intercept directory operations on `/sys/bus/pci/devices/`
- Ensure our device (`0000:00:05.0`) appears in directory listings
- Log all directory scans for debugging

### Phase 5: Add Comprehensive Logging

**Problem**: We need to see exactly what Ollama is doing during discovery.

**Solution**: Add detailed logging to all interception points.

**Implementation**:
- Log all PCI device file reads
- Log all `dlsym()` calls
- Log all directory operations
- Log all NVML function calls
- Create a discovery trace log

## Implementation Priority

1. **Phase 1: PCI Device File Read Interception** (CRITICAL)
   - This is the most likely cause - Ollama reads PCI files directly
   - Need to intercept `read()` calls to ensure correct values

2. **Phase 2: dlsym Interception** (HIGH)
   - Need to see what symbols Ollama is looking for
   - May reveal missing functions

3. **Phase 3: Subprocess Detection** (MEDIUM)
   - Check if discovery happens in subprocess
   - May need to inject shim in subprocess environment

4. **Phase 4: Directory Scanning** (LOW)
   - Less likely to be the issue, but worth checking
   - Ensure device is visible in directory listings

5. **Phase 5: Comprehensive Logging** (ONGOING)
   - Add throughout implementation
   - Critical for debugging

## Expected Outcome

After implementation:
1. Ollama scans PCI devices and finds our vGPU
2. Ollama loads NVML library (our shim)
3. Ollama calls NVML functions (all work correctly)
4. Ollama matches PCI device with NVML device by PCI bus ID
5. Ollama reports `library=cuda_v12` instead of `library=cpu`

## Testing Strategy

1. Deploy extended shim
2. Restart Ollama
3. Check logs for:
   - PCI device file reads
   - `dlsym()` calls
   - NVML function calls
   - Discovery process
4. Verify Ollama reports `library=cuda_v12`
