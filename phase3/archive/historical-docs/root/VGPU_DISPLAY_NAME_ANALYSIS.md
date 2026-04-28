# VGPU Display Name Analysis: "HEXACORE vH100 CAP"

This document analyzes how to make CUDA and lspci show **"HEXACORE vH100 CAP"** instead of the current "NVIDIA H100 80GB HBM3" in Phase 3.

---

## Summary

**A simple search-and-replace + rebuild is NOT sufficient.** You must:

1. **CUDA/NVML** – Change the display name in several places and preserve it when host info overwrites it.
2. **lspci** – Update or provide a custom PCI ID database; not part of the project build.
3. **GPU specs** – If "HEXACORE vH100 CAP" implies different specs (e.g. fewer SMs, less VRAM), update `gpu_properties.h` accordingly.

---

## Where the Display Name Appears

### 1. CUDA Device Name

**Sources:**

| Location | Purpose |
|----------|---------|
| `guest-shim/gpu_properties.h` | `GPU_DEFAULT_NAME` – base default string |
| `guest-shim/libvgpu_cudart.c` | `cudaGetDeviceProperties_v2()` – sets `prop->name` |
| `guest-shim/libvgpu_cuda.c` | `init_gpu_defaults()` → `g_gpu_info.name` (used by `cuDeviceGetName`, `cuDeviceGetProperties`) |
| `guest-shim/libvgpu_nvml.c` | `nvmlDeviceGetName()` – uses `GPU_DEFAULT_NAME` or host info |

**Important:** After the guest connects to the host, `fetch_gpu_info()` in `libvgpu_cuda.c` calls `CUDA_CALL_GET_GPU_INFO`. The host returns real GPU info from `cuDeviceGetName()`, which **overwrites** `g_gpu_info.name`. So changing only `GPU_DEFAULT_NAME` would be overridden once host info is fetched.

### 2. lspci Output

**How lspci works:**

- Reads vendor/device IDs from `/sys/bus/pci/devices/<bdf>/vendor` and `device`.
- Looks up the human-readable name in the **pci.ids** database (e.g. `/usr/share/hwdata/pci.ids` or `/usr/share/misc/pci.ids`).

**Current IDs:** vendor=`0x10de` (NVIDIA), device=`0x2331` (H100 PCIe).

**Implication:** lspci’s name is controlled by the pci.ids database on the guest system, not by project code.

---

## Required Changes

### A. CUDA / NVML Display Name

**Step 1 – `gpu_properties.h` (both copies):**

- `phase3/guest-shim/gpu_properties.h`
- `phase3/GOAL/SOURCE/gpu_properties.h` (if you use GOAL)

```c
#define GPU_DEFAULT_NAME            "HEXACORE vH100 CAP"
```

**Step 2 – Preserve name when host info is applied** in `phase3/guest-shim/libvgpu_cuda.c`:

In `fetch_gpu_info()`, after applying live info from the host, keep using your configured name:

```c
    if (rc == 0 && recv_len >= sizeof(live_info)) {
        g_gpu_info = live_info;
        /* Override name with our configured vGPU display name */
        strncpy(g_gpu_info.name, GPU_DEFAULT_NAME, sizeof(g_gpu_info.name) - 1);
        g_gpu_info.name[sizeof(g_gpu_info.name) - 1] = '\0';
        sanitize_gpu_info(&g_gpu_info);
        ...
```

**Step 3 – NVML** (`libvgpu_nvml.c`):

NVML already uses `GPU_DEFAULT_NAME` for defaults. If NVML fetches from host (CUDA_CALL_GET_GPU_INFO path), the host info overwrites the name. If you want NVML to always show "HEXACORE vH100 CAP", add a similar override after applying host info there.

### B. lspci Display Name

**Option 1 – Custom pci.ids on the guest:**

1. Copy the system pci.ids:
   ```bash
   cp /usr/share/hwdata/pci.ids /usr/share/hwdata/pci.ids.bak
   # or: cp /usr/share/misc/pci.ids /usr/share/misc/pci.ids.bak
   ```

2. Edit the NVIDIA section and change device `2331`:
   ```
   10de  NVIDIA Corporation
     2331  HEXACORE vH100 CAP
   ```
   (or add/change the matching subsystem entry)

3. Or use a custom file and point lspci at it:
   ```bash
   lspci -i /path/to/custom/pci.ids
   ```

**Option 2 – install script:**

Add to your guest `install.sh` (or similar) a step that patches or replaces the relevant part of pci.ids.

### C. GPU Specs (If Different)

"HEXACORE" may denote a vGPU profile (e.g. 6-way partition). If so, specs differ from a full H100:

- **VRAM** – e.g. ~13 GB if 1/6 of 80 GB
- **SMs** – fewer SMs if partitioned (e.g. 22 instead of 132)
- **Compute capability** – usually unchanged (9.0 for H100)

If you want to match that profile:

1. Look up the official specs for "HEXACORE vH100 CAP" (NVIDIA docs, licensing).
2. Update `gpu_properties.h` accordingly (`GPU_DEFAULT_TOTAL_MEM`, `GPU_DEFAULT_SM_COUNT`, etc.).

If you keep full H100 behavior, leave the current specs.

---

## Files to Modify

| File | Change |
|------|--------|
| `phase3/guest-shim/gpu_properties.h` | `GPU_DEFAULT_NAME` → `"HEXACORE vH100 CAP"` |
| `phase3/GOAL/SOURCE/gpu_properties.h` | Same (if used) |
| `phase3/guest-shim/libvgpu_cuda.c` | In `fetch_gpu_info()`, override `g_gpu_info.name` with `GPU_DEFAULT_NAME` after applying live info |
| Guest VM pci.ids | Patch device `10de:2331` → "HEXACORE vH100 CAP" (or use custom file with `lspci -i`) |

---

## Direct Answers

**Q: How can I analyze all files in Phase 3 so VGPU properties appear as HEXACORE vH100 CAP?**

A: Focus on the locations listed above. Grep for `GPU_DEFAULT_NAME`, `g_gpu_info.name`, `cudaGetDeviceProperties`, `cuDeviceGetName`, `nvmlDeviceGetName`, and `fetch_gpu_info`. The analysis above covers these.

**Q: Would it suffice to search for GPU performance data worldwide, replace the H100 with it, and simply rebuild?**

A: **No.** You need to:

1. Change the **display name** in the listed sources and add the override in `fetch_gpu_info()`.
2. Handle **lspci** via pci.ids (outside the build).
3. Optionally update **specs** in `gpu_properties.h` if you want to match the HEXACORE profile.

---

## Rebuild Steps

1. Edit `gpu_properties.h` and `libvgpu_cuda.c`.
2. Rebuild the guest shim libraries:
   ```bash
   cd phase3/guest-shim
   gcc -shared -fPIC -o libvgpu-cuda.so libvgpu_cuda.c cuda_transport.c -I. -I../include ...
   # (use your actual build command)
   ```
3. Deploy the updated libraries to the guest VM.
4. Update pci.ids on the guest for lspci.
5. Restart Ollama (or your workload) and verify with `nvidia-smi`, `ollama` logs, and `lspci`.
