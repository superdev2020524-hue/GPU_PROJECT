/*
 * Phase 3 Milestone 01 CUDA Runtime API probe.
 *
 * Build on VM:
 *   gcc -O2 -Wall -Wextra -std=c11 -o /tmp/runtime_api_probe runtime_api_probe.c -ldl
 *
 * Run on VM:
 *   LD_LIBRARY_PATH=/opt/vgpu/lib:$LD_LIBRARY_PATH /tmp/runtime_api_probe
 *
 * This avoids CUDA headers and validates the mediated libcudart.so.12 path.
 */
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef int cudaError_t;

#define cudaSuccess 0
#define cudaMemcpyHostToDevice 1
#define cudaMemcpyDeviceToHost 2

typedef struct {
    char name[256];
    size_t totalGlobalMem;
    size_t sharedMemPerBlock;
    int regsPerBlock;
    int warpSize;
    int memPitch;
    int maxThreadsPerBlock;
    int maxThreadsDim[3];
    int maxGridSize[3];
    int clockRate;
    int multiProcessorCount;
    int l2CacheSize;
    int maxThreadsPerMultiProcessor;
    int computeCapabilityMajor;
    int computeCapabilityMinor;
    int memoryClockRate;
    int memoryBusWidth;
    int totalConstMem;
    int major;
    int minor;
    int textureAlignment;
    int texturePitchAlignment;
    int deviceOverlap;
    int multiGpuBoardGroupID;
    int singleToDoublePrecisionPerfRatio;
    int pageableMemoryAccess;
    int concurrentKernels;
    int eccEnabled;
    int pciBusID;
    int pciDeviceID;
    int pciDomainID;
    int tccDriver;
    int asyncEngineCount;
    int unifiedAddressing;
    size_t sharedMemPerMultiprocessor;
    int sharedMemPerBlockOptin;
    int maxSharedMemoryPerMultiProcessor;
    int maxSharedMemoryPerBlockOptin;
    int maxSharedMemoryPerBlock;
    int hostNativeAtomicSupported;
    int pageableMemoryAccessUsesHostPageTables;
    int directManagedMemAccessFromHost;
    int maxBlocksPerMultiProcessor;
    int accessPolicyMaxWindowSize;
    int reservedSharedMemPerBlock;
    char _padding[512 - 0x1D0];
} cudaDeviceProp;

typedef cudaError_t (*cudaGetDeviceCount_t)(int *);
typedef cudaError_t (*cudaGetDeviceProperties_t)(cudaDeviceProp *, int);
typedef cudaError_t (*cudaSetDevice_t)(int);
typedef cudaError_t (*cudaMalloc_t)(void **, size_t);
typedef cudaError_t (*cudaFree_t)(void *);
typedef cudaError_t (*cudaMemcpy_t)(void *, const void *, size_t, int);
typedef cudaError_t (*cudaMemcpyAsync_t)(void *, const void *, size_t, int, void *);
typedef cudaError_t (*cudaDeviceSynchronize_t)(void);
typedef cudaError_t (*cudaStreamCreate_t)(void **);
typedef cudaError_t (*cudaStreamSynchronize_t)(void *);
typedef cudaError_t (*cudaStreamDestroy_t)(void *);
typedef cudaError_t (*cudaEventCreate_t)(void **);
typedef cudaError_t (*cudaEventRecord_t)(void *, void *);
typedef cudaError_t (*cudaEventSynchronize_t)(void *);
typedef cudaError_t (*cudaEventDestroy_t)(void *);

struct runtime_api {
    cudaGetDeviceCount_t cudaGetDeviceCount;
    cudaGetDeviceProperties_t cudaGetDeviceProperties;
    cudaSetDevice_t cudaSetDevice;
    cudaMalloc_t cudaMalloc;
    cudaFree_t cudaFree;
    cudaMemcpy_t cudaMemcpy;
    cudaMemcpyAsync_t cudaMemcpyAsync;
    cudaDeviceSynchronize_t cudaDeviceSynchronize;
    cudaStreamCreate_t cudaStreamCreate;
    cudaStreamSynchronize_t cudaStreamSynchronize;
    cudaStreamDestroy_t cudaStreamDestroy;
    cudaEventCreate_t cudaEventCreate;
    cudaEventRecord_t cudaEventRecord;
    cudaEventSynchronize_t cudaEventSynchronize;
    cudaEventDestroy_t cudaEventDestroy;
};

static int failed = 0;

static void check_rc(const char *case_name, const char *op, cudaError_t rc)
{
    printf("CASE %-28s OP %-28s RC %d\n", case_name, op, rc);
    if (rc != cudaSuccess) {
        failed = 1;
    }
}

static void *sym(void *lib, const char *name)
{
    void *p = dlsym(lib, name);
    if (!p) {
        fprintf(stderr, "MISSING_SYMBOL %s\n", name);
        failed = 1;
    }
    return p;
}

static int load_api(struct runtime_api *api)
{
    void *cuda = dlopen("libcuda.so.1", RTLD_NOW | RTLD_GLOBAL);
    void *cudart = dlopen("libcudart.so.12", RTLD_NOW | RTLD_GLOBAL);
    if (!cuda || !cudart) {
        fprintf(stderr, "DLERROR cuda=%p cudart=%p err=%s\n", cuda, cudart, dlerror());
        return 1;
    }

    memset(api, 0, sizeof(*api));
    api->cudaGetDeviceCount = (cudaGetDeviceCount_t)sym(cudart, "cudaGetDeviceCount");
    api->cudaGetDeviceProperties = (cudaGetDeviceProperties_t)sym(cudart, "cudaGetDeviceProperties");
    api->cudaSetDevice = (cudaSetDevice_t)sym(cudart, "cudaSetDevice");
    api->cudaMalloc = (cudaMalloc_t)sym(cudart, "cudaMalloc");
    api->cudaFree = (cudaFree_t)sym(cudart, "cudaFree");
    api->cudaMemcpy = (cudaMemcpy_t)sym(cudart, "cudaMemcpy");
    api->cudaMemcpyAsync = (cudaMemcpyAsync_t)sym(cudart, "cudaMemcpyAsync");
    api->cudaDeviceSynchronize = (cudaDeviceSynchronize_t)sym(cudart, "cudaDeviceSynchronize");
    api->cudaStreamCreate = (cudaStreamCreate_t)sym(cudart, "cudaStreamCreate");
    api->cudaStreamSynchronize = (cudaStreamSynchronize_t)sym(cudart, "cudaStreamSynchronize");
    api->cudaStreamDestroy = (cudaStreamDestroy_t)sym(cudart, "cudaStreamDestroy");
    api->cudaEventCreate = (cudaEventCreate_t)sym(cudart, "cudaEventCreate");
    api->cudaEventRecord = (cudaEventRecord_t)sym(cudart, "cudaEventRecord");
    api->cudaEventSynchronize = (cudaEventSynchronize_t)sym(cudart, "cudaEventSynchronize");
    api->cudaEventDestroy = (cudaEventDestroy_t)sym(cudart, "cudaEventDestroy");
    return failed ? 1 : 0;
}

int main(void)
{
    struct runtime_api api;
    cudaDeviceProp prop;
    enum { N = 128 };
    uint32_t input[N];
    uint32_t output[N];
    void *device = NULL;
    void *stream = NULL;
    void *event = NULL;
    int count = 0;

    if (load_api(&api) != 0) {
        return 2;
    }

    check_rc("runtime_device", "cudaGetDeviceCount", api.cudaGetDeviceCount(&count));
    printf("RUNTIME_DEVICE_COUNT %d\n", count);
    if (count < 1) {
        fprintf(stderr, "RUNTIME_NO_DEVICE\n");
        return 3;
    }
    check_rc("runtime_device", "cudaSetDevice", api.cudaSetDevice(0));
    memset(&prop, 0, sizeof(prop));
    check_rc("runtime_device", "cudaGetDeviceProperties", api.cudaGetDeviceProperties(&prop, 0));
    printf("RUNTIME_DEVICE_NAME %s\n", prop.name);
    printf("RUNTIME_DEVICE_CC %d.%d legacy=%d.%d\n",
           prop.computeCapabilityMajor, prop.computeCapabilityMinor, prop.major, prop.minor);

    for (uint32_t i = 0; i < N; i++) {
        input[i] = 0x1000u + i;
        output[i] = 0u;
    }

    check_rc("runtime_alloc_free", "cudaMalloc", api.cudaMalloc(&device, sizeof(input)));
    check_rc("runtime_copy", "cudaMemcpy HtoD",
             api.cudaMemcpy(device, input, sizeof(input), cudaMemcpyHostToDevice));
    check_rc("runtime_copy", "cudaMemcpy DtoH",
             api.cudaMemcpy(output, device, sizeof(output), cudaMemcpyDeviceToHost));
    if (memcmp(input, output, sizeof(input)) != 0) {
        fprintf(stderr, "RUNTIME_COPY_MISMATCH sync\n");
        failed = 1;
    } else {
        printf("RUNTIME_COPY_ROUNDTRIP_OK bytes=%zu\n", sizeof(input));
    }

    memset(output, 0, sizeof(output));
    check_rc("runtime_stream", "cudaStreamCreate", api.cudaStreamCreate(&stream));
    check_rc("runtime_event", "cudaEventCreate", api.cudaEventCreate(&event));
    check_rc("runtime_copy_async", "cudaMemcpyAsync HtoD",
             api.cudaMemcpyAsync(device, input, sizeof(input), cudaMemcpyHostToDevice, stream));
    check_rc("runtime_event", "cudaEventRecord", api.cudaEventRecord(event, stream));
    check_rc("runtime_event", "cudaEventSynchronize", api.cudaEventSynchronize(event));
    check_rc("runtime_stream", "cudaStreamSynchronize", api.cudaStreamSynchronize(stream));
    check_rc("runtime_copy_async", "cudaMemcpyAsync DtoH",
             api.cudaMemcpyAsync(output, device, sizeof(output), cudaMemcpyDeviceToHost, stream));
    check_rc("runtime_stream", "cudaStreamSynchronize", api.cudaStreamSynchronize(stream));
    if (memcmp(input, output, sizeof(input)) != 0) {
        fprintf(stderr, "RUNTIME_COPY_MISMATCH async\n");
        failed = 1;
    } else {
        printf("RUNTIME_ASYNC_COPY_ROUNDTRIP_OK bytes=%zu\n", sizeof(input));
    }

    check_rc("runtime_device", "cudaDeviceSynchronize", api.cudaDeviceSynchronize());
    if (event) {
        check_rc("runtime_event", "cudaEventDestroy", api.cudaEventDestroy(event));
    }
    if (stream) {
        check_rc("runtime_stream", "cudaStreamDestroy", api.cudaStreamDestroy(stream));
    }
    if (device) {
        check_rc("runtime_alloc_free", "cudaFree", api.cudaFree(device));
    }

    printf("OVERALL %s\n", failed ? "FAIL" : "PASS");
    return failed ? 1 : 0;
}
