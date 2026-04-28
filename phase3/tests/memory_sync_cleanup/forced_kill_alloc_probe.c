/*
 * Milestone 03 forced-kill probe.
 *
 * Build on VM:
 *   gcc -O2 -Wall -Wextra -std=c11 -o /tmp/forced_kill_alloc_probe forced_kill_alloc_probe.c -ldl
 *
 * Run on VM:
 *   LD_LIBRARY_PATH=/opt/vgpu/lib:$LD_LIBRARY_PATH /tmp/forced_kill_alloc_probe
 *
 * The process allocates mediated device memory, prints READY, and sleeps.
 * The gate kills it with SIGKILL to verify next-run behavior.
 */
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <unistd.h>

typedef int CUdevice;
typedef int CUresult;
typedef unsigned long long CUdeviceptr;
typedef void *CUcontext;

typedef CUresult (*cuInit_t)(unsigned int);
typedef CUresult (*cuDeviceGet_t)(CUdevice *, int);
typedef CUresult (*cuDevicePrimaryCtxRetain_t)(CUcontext *, CUdevice);
typedef CUresult (*cuCtxSetCurrent_t)(CUcontext);
typedef CUresult (*cuMemAlloc_v2_t)(CUdeviceptr *, size_t);

#define CHECK(name, expr)                                                     \
    do {                                                                      \
        CUresult rc = (expr);                                                 \
        if (rc != 0) {                                                        \
            fprintf(stderr, "%s failed rc=%d pid=%d\n", name, rc,            \
                    (int)getpid());                                           \
            return 2;                                                         \
        }                                                                     \
    } while (0)

int main(void)
{
    void *h = dlopen("libcuda.so.1", RTLD_NOW | RTLD_GLOBAL);
    if (!h) {
        fprintf(stderr, "dlopen failed: %s\n", dlerror());
        return 1;
    }

    cuInit_t cuInit = (cuInit_t)dlsym(h, "cuInit");
    cuDeviceGet_t cuDeviceGet = (cuDeviceGet_t)dlsym(h, "cuDeviceGet");
    cuDevicePrimaryCtxRetain_t cuDevicePrimaryCtxRetain =
        (cuDevicePrimaryCtxRetain_t)dlsym(h, "cuDevicePrimaryCtxRetain");
    cuCtxSetCurrent_t cuCtxSetCurrent =
        (cuCtxSetCurrent_t)dlsym(h, "cuCtxSetCurrent");
    cuMemAlloc_v2_t cuMemAlloc_v2 =
        (cuMemAlloc_v2_t)dlsym(h, "cuMemAlloc_v2");

    if (!cuInit || !cuDeviceGet || !cuDevicePrimaryCtxRetain ||
        !cuCtxSetCurrent || !cuMemAlloc_v2) {
        fprintf(stderr, "missing CUDA symbol\n");
        return 1;
    }

    CUdevice dev = 0;
    CUcontext ctx = 0;
    CUdeviceptr ptr = 0;

    CHECK("cuInit", cuInit(0));
    CHECK("cuDeviceGet", cuDeviceGet(&dev, 0));
    CHECK("cuDevicePrimaryCtxRetain", cuDevicePrimaryCtxRetain(&ctx, dev));
    CHECK("cuCtxSetCurrent", cuCtxSetCurrent(ctx));
    CHECK("cuMemAlloc_v2", cuMemAlloc_v2(&ptr, 64u * 1024u * 1024u));

    printf("READY pid=%d ptr=0x%llx\n", (int)getpid(), ptr);
    fflush(stdout);
    sleep(120);
    return 0;
}
