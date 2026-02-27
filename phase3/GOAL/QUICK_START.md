# Quick Start Guide

## For New VM Deployment

### Step 1: Copy GOAL Register

Copy the entire `phase3/GOAL/` directory to your new VM.

### Step 2: Build Shim Libraries

```bash
cd phase3/GOAL/BUILD
sudo bash install.sh
```

This will:
- Compile all shim libraries from `../SOURCE/`
- Install to `/usr/lib64/`
- Create system symlinks
- Register with `ldconfig`

### Step 3: Verify Installation

```bash
cd phase3/GOAL/TEST_SCRIPTS

# Test C/C++ CUDA detection
./test_cuda_detection.sh

# Test Python CUDA detection
python3 test_python_cuda.py
```

Both should show:
- ✓ CUDA initialized successfully
- ✓ GPU detected (device count = 1)

### Step 4: Use in Your Applications

**No special configuration needed!** Just use CUDA/NVML as normal:

```python
# Python example
import ctypes
cuda = ctypes.CDLL("libcuda.so.1")  # Automatically uses vGPU shim
# ... use CUDA functions ...
```

```c
// C example
#include <dlfcn.h>
void *handle = dlopen("libcuda.so.1", RTLD_LAZY);  // Automatically uses vGPU shim
// ... use CUDA functions ...
```

## What's Included

- ✅ All source files (.c, .h)
- ✅ All include files
- ✅ Build script (install.sh)
- ✅ Test scripts
- ✅ Complete documentation

## Directory Structure

```
GOAL/
├── SOURCE/        # All C source files and headers
├── INCLUDE/       # Protocol headers
├── BUILD/         # Build script (install.sh)
├── TEST_SCRIPTS/  # Test programs
└── DOCUMENTATION/ # Technical docs
```

## That's It!

The GOAL register is completely self-contained. Everything needed to build and deploy vGPU shims is included.

---

**See `INSTALLATION.md` for detailed instructions.**
**See `VERIFICATION.md` for troubleshooting.**
