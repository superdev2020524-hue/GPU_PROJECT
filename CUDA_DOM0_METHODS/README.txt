================================================================================
                    CUDA DOM0 METHODS - README
                    Guide to using these method files
================================================================================

FOLDER STRUCTURE:
-----------------
CUDA_DOM0_METHODS/
├── 00_MASTER_PLAN.txt          - Overview of all methods
├── PROGRESS_TRACKER.txt         - Track which methods are completed
├── README.txt                   - This file
│
├── 15_NAMESPACE_ISOLATION.txt  - Method 15: Namespace isolation
├── 16_KERNEL_MODULE_CPUID_INTERCEPT.txt - Method 16: Kernel module
├── 16_cpuid_intercept_module.c - Kernel module source code
├── 17_CUDA_PACKAGE_MANAGER.txt - Method 17: Package manager install
├── 22_FUSE_FILESYSTEM.txt      - Method 22: FUSE filesystem
└── 22_cpuinfo_fuse.c           - FUSE source code

HOW TO USE:
-----------
1. Start with 00_MASTER_PLAN.txt to understand all methods
2. Check PROGRESS_TRACKER.txt to see what's been tried
3. Follow the step-by-step guide in each method file
4. Update PROGRESS_TRACKER.txt when completing methods

CURRENT STATUS:
--------------
- nvidia-smi: WORKING
- CUDA runtime: FAILING (error 801)
- Root cause: CUDA uses CPUID to detect virtualization

NEXT STEPS:
-----------
Start with Method 16 (Kernel Module) as it's the most viable remaining option.

BACKUP LOCATIONS:
-----------------
- libcuda.so: /usr/lib64/libcuda.so.545.23.06.backup
- CUDA libs: /mnt/cuda_install/backup/

SYSTEM INFO:
------------
- OS: XCP-ng (Xen 4.17)
- Kernel: 4.19-xen
- GPU: NVIDIA H100 PCIe
- CUDA: 12.3.52
- Driver: 545.23.06

================================================================================
                              END OF README
================================================================================

