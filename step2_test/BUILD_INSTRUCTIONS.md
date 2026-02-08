# BUILD INSTRUCTIONS
**Date:** 2026-02-08

---

## IMPORTANT: Different Components Build on Different Machines

### Dom0 (Host) - Requires CUDA
- **MEDIATOR daemon** - Needs CUDA libraries
- **CUDA test executable** - For testing CUDA functionality

### VM (Guest) - NO CUDA Needed
- **VM client** - Only needs standard C libraries
- Reads MMIO, writes to NFS

---

## BUILDING ON DOM0 (Host)

### Prerequisites:
```bash
# Install CUDA toolkit (if not already installed)
# Install build tools
yum install gcc make  # or apt-get on Debian/Ubuntu
```

### Build MEDIATOR:
```bash
cd /home/david/Downloads/gpu/step2_test
make dom0
```

This creates:
- `build-dom0/mediator_async` - MEDIATOR daemon
- `build-dom0/cuda_vector_add` - CUDA test executable

### Test CUDA (optional):
```bash
make test-cuda
```

### Install MEDIATOR (optional):
```bash
sudo make install-dom0
```

---

## BUILDING ON VM (Guest)

### Prerequisites:
```bash
# Install build tools (no CUDA needed!)
apt-get install build-essential  # Debian/Ubuntu
# or
yum install gcc make            # RHEL/CentOS
```

### Build VM Client:
```bash
cd /path/to/step2_test
make vm
```

This creates:
- `build-vm/vm_client_vector` - VM client executable

### Install VM Client (optional):
```bash
sudo make install-vm
```

---

## QUICK START

### Step 1: Build on Dom0
```bash
# On Dom0
cd /home/david/Downloads/gpu/step2_test
make dom0
```

### Step 2: Build on VM
```bash
# On VM (copy source files or mount shared directory)
cd /path/to/step2_test
make vm
```

### Step 3: Run MEDIATOR on Dom0
```bash
# On Dom0
sudo ./build-dom0/mediator_async
```

### Step 4: Run VM Client on VM
```bash
# On VM
sudo ./build-vm/vm_client_vector 100 200
```

---

## FILE TRANSFER OPTIONS

### Option 1: NFS Share
Mount the source directory on VM via NFS, then build on VM.

### Option 2: Copy Files
Copy `vm_client_vector.c` to VM and build there.

### Option 3: Build on Host, Copy Binary
Build VM client on host (without CUDA flags), then copy binary to VM.

---

## TROUBLESHOOTING

### "nvcc: command not found" on VM
- **Solution:** This is expected! VM client doesn't need CUDA.
- Build VM client with: `make vm` (no CUDA required)

### "CUDA library not found" on Dom0
- **Solution:** Install CUDA toolkit on Dom0
- Check: `which nvcc` and `ls /usr/local/cuda/lib64/libcudart.so*`

### "Cannot find cuda_vector_add.h" on VM
- **Solution:** VM client doesn't need CUDA header.
- Only MEDIATOR needs CUDA headers (builds on Dom0)

---

## SUMMARY

| Component | Build Location | CUDA Required | Output |
|-----------|---------------|---------------|--------|
| MEDIATOR | Dom0 | ✅ Yes | `build-dom0/mediator_async` |
| VM Client | VM | ❌ No | `build-vm/vm_client_vector` |

**Remember:** MEDIATOR on Dom0, VM Client on VM!
