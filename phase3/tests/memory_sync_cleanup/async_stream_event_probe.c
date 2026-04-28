/*
 * Milestone 03 async/mixed stream-event probe.
 *
 * Build on VM:
 *   gcc -O2 -Wall -Wextra -std=c11 -o /tmp/async_stream_event_probe async_stream_event_probe.c -ldl
 *
 * Run on VM:
 *   LD_LIBRARY_PATH=/opt/vgpu/lib:$LD_LIBRARY_PATH /tmp/async_stream_event_probe
 */
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef int CUdevice;
typedef int CUresult;
typedef unsigned long long CUdeviceptr;
typedef void *CUcontext;
typedef void *CUstream;
typedef void *CUevent;

typedef CUresult (*cuInit_t)(unsigned int);
typedef CUresult (*cuDeviceGet_t)(CUdevice *, int);
typedef CUresult (*cuDevicePrimaryCtxRetain_t)(CUcontext *, CUdevice);
typedef CUresult (*cuCtxSetCurrent_t)(CUcontext);
typedef CUresult (*cuMemAlloc_v2_t)(CUdeviceptr *, size_t);
typedef CUresult (*cuMemFree_v2_t)(CUdeviceptr);
typedef CUresult (*cuMemcpyHtoDAsync_v2_t)(CUdeviceptr, const void *, size_t, CUstream);
typedef CUresult (*cuMemcpyDtoHAsync_v2_t)(void *, CUdeviceptr, size_t, CUstream);
typedef CUresult (*cuMemcpyDtoDAsync_v2_t)(CUdeviceptr, CUdeviceptr, size_t, CUstream);
typedef CUresult (*cuMemsetD8Async_t)(CUdeviceptr, unsigned char, size_t, CUstream);
typedef CUresult (*cuStreamCreate_t)(CUstream *, unsigned int);
typedef CUresult (*cuStreamSynchronize_t)(CUstream);
typedef CUresult (*cuStreamWaitEvent_t)(CUstream, CUevent, unsigned int);
typedef CUresult (*cuStreamDestroy_v2_t)(CUstream);
typedef CUresult (*cuEventCreate_t)(CUevent *, unsigned int);
typedef CUresult (*cuEventRecord_t)(CUevent, CUstream);
typedef CUresult (*cuEventSynchronize_t)(CUevent);
typedef CUresult (*cuEventQuery_t)(CUevent);
typedef CUresult (*cuEventDestroy_v2_t)(CUevent);

#define CHECK(name, expr)                                                     \
    do {                                                                      \
        CUresult rc_ = (expr);                                                \
        if (rc_ != 0) {                                                       \
            fprintf(stderr, "%s failed rc=%d\n", name, rc_);                 \
            return 2;                                                         \
        }                                                                     \
    } while (0)

#define LOAD(sym)                                                             \
    do {                                                                      \
        api.sym = (sym##_t)dlsym(h, #sym);                                    \
        if (!api.sym) {                                                       \
            fprintf(stderr, "missing symbol %s\n", #sym);                    \
            return 1;                                                         \
        }                                                                     \
    } while (0)

struct api {
    cuInit_t cuInit;
    cuDeviceGet_t cuDeviceGet;
    cuDevicePrimaryCtxRetain_t cuDevicePrimaryCtxRetain;
    cuCtxSetCurrent_t cuCtxSetCurrent;
    cuMemAlloc_v2_t cuMemAlloc_v2;
    cuMemFree_v2_t cuMemFree_v2;
    cuMemcpyHtoDAsync_v2_t cuMemcpyHtoDAsync_v2;
    cuMemcpyDtoHAsync_v2_t cuMemcpyDtoHAsync_v2;
    cuMemcpyDtoDAsync_v2_t cuMemcpyDtoDAsync_v2;
    cuMemsetD8Async_t cuMemsetD8Async;
    cuStreamCreate_t cuStreamCreate;
    cuStreamSynchronize_t cuStreamSynchronize;
    cuStreamWaitEvent_t cuStreamWaitEvent;
    cuStreamDestroy_v2_t cuStreamDestroy_v2;
    cuEventCreate_t cuEventCreate;
    cuEventRecord_t cuEventRecord;
    cuEventSynchronize_t cuEventSynchronize;
    cuEventQuery_t cuEventQuery;
    cuEventDestroy_v2_t cuEventDestroy_v2;
};

int main(void)
{
    enum { N = 4 * 1024 * 1024 };
    struct api api;
    memset(&api, 0, sizeof(api));

    void *h = dlopen("libcuda.so.1", RTLD_NOW | RTLD_GLOBAL);
    if (!h) {
        fprintf(stderr, "dlopen failed: %s\n", dlerror());
        return 1;
    }

    LOAD(cuInit);
    LOAD(cuDeviceGet);
    LOAD(cuDevicePrimaryCtxRetain);
    LOAD(cuCtxSetCurrent);
    LOAD(cuMemAlloc_v2);
    LOAD(cuMemFree_v2);
    LOAD(cuMemcpyHtoDAsync_v2);
    LOAD(cuMemcpyDtoHAsync_v2);
    LOAD(cuMemcpyDtoDAsync_v2);
    LOAD(cuMemsetD8Async);
    LOAD(cuStreamCreate);
    LOAD(cuStreamSynchronize);
    LOAD(cuStreamWaitEvent);
    LOAD(cuStreamDestroy_v2);
    LOAD(cuEventCreate);
    LOAD(cuEventRecord);
    LOAD(cuEventSynchronize);
    LOAD(cuEventQuery);
    LOAD(cuEventDestroy_v2);

    unsigned char *input = malloc(N);
    unsigned char *output = malloc(N);
    if (!input || !output) {
        fprintf(stderr, "host malloc failed\n");
        return 1;
    }
    for (int i = 0; i < N; i++) {
        input[i] = (unsigned char)((i * 17 + 3) & 0xff);
        output[i] = 0;
    }

    CUdevice dev = 0;
    CUcontext ctx = 0;
    CUdeviceptr a = 0, b = 0;
    CUstream stream = NULL;
    CUevent start = NULL, end = NULL;

    CHECK("cuInit", api.cuInit(0));
    CHECK("cuDeviceGet", api.cuDeviceGet(&dev, 0));
    CHECK("cuDevicePrimaryCtxRetain", api.cuDevicePrimaryCtxRetain(&ctx, dev));
    CHECK("cuCtxSetCurrent", api.cuCtxSetCurrent(ctx));
    CHECK("cuStreamCreate", api.cuStreamCreate(&stream, 0));
    CHECK("cuEventCreate(start)", api.cuEventCreate(&start, 0));
    CHECK("cuEventCreate(end)", api.cuEventCreate(&end, 0));
    CHECK("cuMemAlloc(a)", api.cuMemAlloc_v2(&a, N));
    CHECK("cuMemAlloc(b)", api.cuMemAlloc_v2(&b, N));

    CHECK("cuEventRecord(start)", api.cuEventRecord(start, stream));
    CHECK("cuMemcpyHtoDAsync", api.cuMemcpyHtoDAsync_v2(a, input, N, stream));
    CHECK("cuStreamSynchronize(after_htod)", api.cuStreamSynchronize(stream));
    CHECK("cuMemcpyDtoDAsync", api.cuMemcpyDtoDAsync_v2(b, a, N, stream));
    CHECK("cuEventRecord(end)", api.cuEventRecord(end, stream));
    CHECK("cuEventSynchronize(end)", api.cuEventSynchronize(end));
    CHECK("cuEventQuery(end)", api.cuEventQuery(end));
    CHECK("cuStreamWaitEvent", api.cuStreamWaitEvent(stream, end, 0));
    CHECK("cuMemcpyDtoHAsync", api.cuMemcpyDtoHAsync_v2(output, b, N, stream));

    if (memcmp(input, output, N) != 0) {
        fprintf(stderr, "pattern verification failed\n");
        return 3;
    }

    CHECK("cuMemsetD8Async", api.cuMemsetD8Async(b, 0x5a, N, stream));
    CHECK("cuStreamSynchronize(after_memset)", api.cuStreamSynchronize(stream));
    memset(output, 0, N);
    CHECK("cuMemcpyDtoHAsync(after_memset)", api.cuMemcpyDtoHAsync_v2(output, b, N, stream));
    for (int i = 0; i < N; i++) {
        if (output[i] != 0x5a) {
            fprintf(stderr, "memset verification failed at %d got=0x%02x\n", i, output[i]);
            return 4;
        }
    }

    CHECK("cuEventDestroy(start)", api.cuEventDestroy_v2(start));
    CHECK("cuEventDestroy(end)", api.cuEventDestroy_v2(end));
    CHECK("cuStreamDestroy", api.cuStreamDestroy_v2(stream));
    CHECK("cuMemFree(a)", api.cuMemFree_v2(a));
    CHECK("cuMemFree(b)", api.cuMemFree_v2(b));

    printf("ASYNC_STREAM_EVENT_PROBE PASS bytes=%d\n", N);
    free(input);
    free(output);
    return 0;
}
