# Building the guest shim (libvgpu-cuda.so.1) on your local PC

Building on your PC avoids the heavy gcc run on the VM and prevents the soft lockup that happened on TEST-4. You do **not** need CUDA or any GPU libraries on the PC to build the **guest** shim.

---

## What you need on the PC

1. **GCC** (C compiler with C11 support)  
   - Ubuntu/Debian: `sudo apt-get install build-essential`  
   - That gives you gcc and standard headers.

2. **The phase3 tree**  
   - The same repo/directory you use for the transfer scripts (e.g. `.../gpu/phase3`).  
   - The build only uses sources and headers from this tree; no system CUDA or extra libraries.

**You do not need:**  
- CUDA toolkit / nvidia drivers  
- Any GPU  
- Any special “relevant libraries” beyond what comes with build-essential (glibc, pthread, libdl are standard).

The guest shim is a **proxy** loaded via `LD_PRELOAD`; it only links with **-ldl** and **-lpthread**. It does not link against libcuda.

---

## Build command (run from phase3 directory)

```bash
cd /path/to/phase3

gcc -shared -fPIC -O2 -Wall -Wextra -std=c11 -D_GNU_SOURCE \
    -Iinclude -Iguest-shim \
    -o libvgpu-cuda.so.1 \
    guest-shim/libvgpu_cuda.c guest-shim/cuda_transport.c \
    -ldl -lpthread
```

This produces **libvgpu-cuda.so.1** in the current directory.

---

## Possible issues when running the .so on the VM

### 1. Glibc version (main one to watch)

The .so is built against your PC’s **glibc**. If your PC has a **newer** glibc than the VM, the VM may fail to load the library with something like:

- `version 'GLIBC_2.34' not found`  
- or `libc.so.6: version 'GLIBC_XX' not found`

**Ways to avoid that:**

- **Build on a system with the same or older glibc as the VM.**  
  - If the VM is Ubuntu 22.04, build on Ubuntu 22.04 (or older).  
  - If you build on Ubuntu 24.04, the VM (22.04) might not have the required glibc version.

- **Or build in a container/chroot that matches the VM:**  
  - e.g. Docker with the same distro/version as the VM, then run the same gcc command inside the container and copy the resulting `libvgpu-cuda.so.1` to the VM.

**Check glibc on VM (after SSH):**  
`ldd --version`  
**Check glibc on PC:**  
`ldd --version`

If the PC’s glibc is **newer** than the VM’s, prefer building in a 22.04 (or VM-matching) environment so the .so doesn’t depend on symbols the VM doesn’t have.

### 2. Architecture

VM and PC must use the same CPU architecture (e.g. both **x86_64**). If your PC is x86_64 and the VM is x86_64, you’re fine. For ARM VMs you’d need a cross-compiler or an ARM build environment.

### 3. No other “relevant libraries” required

The shim does not link to CUDA, so you don’t install “CUDA libraries” on the PC for this build. Standard build-essential is enough.

---

## After building on the PC

1. Copy **libvgpu-cuda.so.1** to the VM (e.g. scp).
2. On the VM:
   ```bash
   sudo cp /path/to/libvgpu-cuda.so.1 /opt/vgpu/lib/libvgpu-cuda.so.1
   sudo systemctl restart ollama
   ```

You can skip running `transfer_libvgpu_cuda.py` (which would rebuild on the VM). Use the transfer script only when you need to push **source** changes and don’t want to build on the PC.

---

## Summary

| Question | Answer |
|----------|--------|
| Do I need CUDA on my PC to build the guest shim? | **No.** |
| Do I need any extra “relevant libraries”? | **No.** Just a normal C toolchain (e.g. `build-essential`). |
| What can go wrong? | **Glibc:** if the PC’s glibc is newer than the VM’s, the VM may fail to load the .so. Build on the same or older distro as the VM (or in a matching container). |
| Same for libvgpu-cublas? | Same idea: only standard C + phase3 headers, no CUDA. Build with the same gcc command pattern but for the CUBLAS shim sources. |
