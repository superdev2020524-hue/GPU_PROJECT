# Deployment Checklist: New VM Setup

Use this checklist when deploying the GOAL register to a new VM.

## Pre-Deployment

- [ ] Copy entire `phase3/GOAL/` directory to new VM
- [ ] Verify VGPU-STUB device exists: `lspci | grep -i nvidia`
- [ ] Verify GCC installed: `gcc --version`
- [ ] Verify root/sudo access available

## File Verification

- [ ] `SOURCE/` directory contains all .c and .h files
- [ ] `INCLUDE/` directory contains all protocol headers
- [ ] `BUILD/install.sh` exists and is executable
- [ ] `TEST_SCRIPTS/` contains all test programs

## Build and Install

- [ ] Run build script: `cd BUILD && sudo bash install.sh`
- [ ] Build completes without errors
- [ ] Libraries installed to `/usr/lib64/`
- [ ] Symlinks created: `libcuda.so.1`, `libnvidia-ml.so.1`
- [ ] Libraries registered with `ldconfig`

## Verification

- [ ] C test passes: `cd TEST_SCRIPTS && ./test_cuda_detection.sh`
- [ ] Python test passes: `cd TEST_SCRIPTS && python3 test_python_cuda.py`
- [ ] GPU detected (device count = 1)
- [ ] VGPU-STUB found at 0000:00:05.0

## Post-Installation

- [ ] Test with your application
- [ ] Verify no special configuration needed
- [ ] Confirm GPU mode works automatically

## Troubleshooting

If any step fails:
1. Check `VERIFICATION.md` for troubleshooting
2. Review `BUILD_INSTRUCTIONS.md` for build details
3. Check `TESTING_RESULTS.md` for expected behavior

---

**Status:** ✅ Ready for deployment
