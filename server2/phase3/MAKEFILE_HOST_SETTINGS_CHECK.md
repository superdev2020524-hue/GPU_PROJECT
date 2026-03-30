# Makefile Host Settings Verification

## Current Makefile Settings

### CUDA Configuration
```makefile
CUDA_PATH  ?= /usr/local/cuda
CUDA_LIB   = $(CUDA_PATH)/lib64
CUDA_INC   = $(CUDA_PATH)/include
NVCC       ?= nvcc
NVCC_FLAGS = -O2 -arch=sm_90 -x cu      # sm_90 = H100
```

**Status:** ✅ **Likely Correct**
- Default path `/usr/local/cuda` is standard for CUDA installations
- Can be overridden with `CUDA_PATH=/custom/path make host`
- `sm_90` is correct for H100 GPU

**Verification Needed:**
```bash
# On host, check if CUDA is at default location:
ls -l /usr/local/cuda
which nvcc

# If CUDA is elsewhere, override:
CUDA_PATH=/opt/cuda make host
```

### QEMU Build Configuration
```makefile
RPM_BUILD      ?= $(HOME)/vgpu-build/rpmbuild
QEMU_BIN       = /usr/lib64/xen/bin/qemu-system-i386  # (implied from guide)
```

**Status:** ✅ **Correct**
- From your logs: `QEMU_BIN="/usr/lib64/xen/bin/qemu-system-i386"` matches
- QEMU has vgpu-cuda device: Confirmed working
- RPM build directory is user-configurable (good)

### Mediator Binary
```makefile
MEDIATOR_BIN  = mediator_phase3
```

**Status:** ✅ **Correct**
- From your logs: `./mediator_phase3` is running
- Binary name matches

### Build Output Location
```makefile
# Builds in current directory (phase3/)
MEDIATOR_BIN  = mediator_phase3  # (no path prefix)
```

**Status:** ✅ **Correct**
- From your logs: Mediator runs from `/root/phase3/mediator_phase3`
- Builds in current directory, which is correct

### Installation Paths
```makefile
PREFIX     ?= /usr/local
ETC_PREFIX ?= /etc/vgpu
```

**Status:** ⚠️ **May Need Verification**
- Default installs to `/usr/local/bin/mediator_phase3`
- But your mediator is running from `/root/phase3/mediator_phase3`
- This suggests you're running it directly, not from installed location

**Current Usage:**
- Mediator is run from build directory: `/root/phase3/mediator_phase3`
- This is fine for development/testing
- For production, you might want to use `make install`

## Potential Issues

### 1. CUDA Path May Vary
**Issue:** CUDA might not be at `/usr/local/cuda` on all systems

**Solution:** Makefile already supports override:
```bash
CUDA_PATH=/opt/cuda make host
```

**Recommendation:** Add check in Makefile to detect CUDA location automatically

### 2. Missing libvgpu-cudart.so Build
**Issue:** The Makefile builds `libvgpu-cuda.so` but we also need `libvgpu-cudart.so`

**Current Makefile:**
```makefile
guest: $(SHIM_CUDA_LIB) $(SHIM_NVML_LIB)
```

**Missing:** `libvgpu-cudart.so` build target

**Status:** ❌ **Needs Update**
- We've been building `libvgpu-cudart.so` manually
- Should be added to Makefile

### 3. QEMU Binary Path Hardcoded in Guide
**Issue:** QEMU path is mentioned in guide but not in Makefile

**Status:** ✅ **OK** - This is just for verification, not build

### 4. Socket Path Configuration
**Issue:** Socket paths are defined in source code, not Makefile

**Status:** ✅ **OK** - This is correct, socket paths are runtime configuration

## Recommendations

### 1. Add CUDA Auto-Detection
Add to Makefile:
```makefile
# Auto-detect CUDA if not set
ifeq ($(CUDA_PATH),/usr/local/cuda)
  ifneq ($(wildcard /usr/local/cuda/include/cuda.h),)
    # CUDA found at default location
  else
    # Try common alternatives
    ifneq ($(wildcard /opt/cuda/include/cuda.h),)
      CUDA_PATH := /opt/cuda
    else
      $(warning CUDA not found at /usr/local/cuda, set CUDA_PATH manually)
    endif
  endif
endif
```

### 2. Add libvgpu-cudart.so Build Target
```makefile
SHIM_CUDART_SRC = $(SHIM_DIR)/libvgpu_cudart.c
SHIM_CUDART_LIB = $(SHIM_DIR)/libvgpu-cudart.so

guest: $(SHIM_CUDA_LIB) $(SHIM_NVML_LIB) $(SHIM_CUDART_LIB)

$(SHIM_CUDART_LIB): $(SHIM_CUDART_SRC) $(SHIM_TRANSPORT_SRC) \
                    $(SHIM_DIR)/cuda_transport.h $(SHIM_DIR)/gpu_properties.h \
                    $(INC_DIR)/cuda_protocol.h
	$(CC) -shared -fPIC $(CFLAGS) $(SHIM_INCLUDES) \
		-o $@ $(SHIM_CUDART_SRC) -ldl
	@echo "[SHIM] $@"
```

### 3. Add libvgpu-cublas.so Build Target
We also need CUBLAS shim:
```makefile
SHIM_CUBLAS_SRC = $(SHIM_DIR)/libvgpu_cublas.c
SHIM_CUBLAS_LIB = $(SHIM_DIR)/libvgpu-cublas.so.12

$(SHIM_CUBLAS_LIB): $(SHIM_CUBLAS_SRC) $(SHIM_DIR)/gpu_properties.h
	$(CC) -shared -fPIC $(CFLAGS) $(SHIM_INCLUDES) \
		-Wl,--version-script=$(SHIM_DIR)/cublas_version.lds \
		-o $@ $(SHIM_CUBLAS_SRC) -ldl
	@echo "[SHIM] $@"
```

### 4. Add Verification Target
```makefile
verify-host:
	@echo "Verifying host build environment..."
	@echo -n "  CUDA_PATH: "; \
	if [ -d "$(CUDA_PATH)" ]; then \
		echo "$(CUDA_PATH) ✓"; \
	else \
		echo "$(CUDA_PATH) ✗ (not found)"; \
		echo "  Set CUDA_PATH=/path/to/cuda"; \
	fi
	@echo -n "  nvcc: "; \
	if command -v $(NVCC) >/dev/null 2>&1; then \
		echo "$(shell which $(NVCC)) ✓"; \
	else \
		echo "✗ (not found)"; \
	fi
	@echo -n "  QEMU: "; \
	if [ -x "/usr/lib64/xen/bin/qemu-system-i386" ]; then \
		echo "/usr/lib64/xen/bin/qemu-system-i386 ✓"; \
	else \
		echo "✗ (not found)"; \
	fi
	@echo -n "  SQLite: "; \
	if pkg-config --exists sqlite3 2>/dev/null; then \
		echo "✓"; \
	else \
		echo "✗ (install sqlite-devel)"; \
	fi
```

## Summary

| Setting | Makefile Value | Current Host | Status |
|---------|---------------|--------------|--------|
| CUDA_PATH | `/usr/local/cuda` | Unknown | ⚠️ Verify |
| QEMU_BIN | (not in Makefile) | `/usr/lib64/xen/bin/qemu-system-i386` | ✅ Matches |
| Mediator binary | `mediator_phase3` | `mediator_phase3` | ✅ Matches |
| RPM_BUILD | `$(HOME)/vgpu-build/rpmbuild` | Unknown | ⚠️ Verify |
| Build location | Current dir | `/root/phase3/` | ✅ Matches |
| Missing targets | - | `libvgpu-cudart.so` | ❌ Needs addition |

## Action Items

1. ✅ **Verify CUDA path** on host matches Makefile default
2. ❌ **Add libvgpu-cudart.so** build target to Makefile
3. ❌ **Add libvgpu-cublas.so** build target to Makefile  
4. ✅ **Add verify-host target** to check environment
5. ⚠️ **Document CUDA_PATH override** in beginner guide
