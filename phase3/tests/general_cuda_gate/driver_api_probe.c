/*
 * Phase 3 Milestone 01 raw CUDA Driver API probe.
 *
 * Build on VM:
 *   gcc -O2 -Wall -Wextra -std=c11 -o /tmp/driver_api_probe driver_api_probe.c -ldl
 *
 * Run on VM:
 *   LD_LIBRARY_PATH=/opt/vgpu/lib:$LD_LIBRARY_PATH /tmp/driver_api_probe
 *
 * This intentionally uses dlopen/dlsym and local typedefs so the VM does not
 * need CUDA headers. It exercises the mediated libcuda.so.1 path.
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
typedef void *CUmodule;
typedef void *CUfunction;
typedef void *CUstream;
typedef void *CUevent;

#define CUDA_SUCCESS 0

typedef CUresult (*cuInit_t)(unsigned int);
typedef CUresult (*cuDeviceGetCount_t)(int *);
typedef CUresult (*cuDeviceGet_t)(CUdevice *, int);
typedef CUresult (*cuDeviceGetName_t)(char *, int, CUdevice);
typedef CUresult (*cuDeviceTotalMem_v2_t)(size_t *, CUdevice);
typedef CUresult (*cuDevicePrimaryCtxRetain_t)(CUcontext *, CUdevice);
typedef CUresult (*cuCtxSetCurrent_t)(CUcontext);
typedef CUresult (*cuCtxSynchronize_t)(void);
typedef CUresult (*cuMemAlloc_v2_t)(CUdeviceptr *, size_t);
typedef CUresult (*cuMemFree_v2_t)(CUdeviceptr);
typedef CUresult (*cuMemcpyHtoD_v2_t)(CUdeviceptr, const void *, size_t);
typedef CUresult (*cuMemcpyDtoH_v2_t)(void *, CUdeviceptr, size_t);
typedef CUresult (*cuStreamCreate_t)(CUstream *, unsigned int);
typedef CUresult (*cuStreamSynchronize_t)(CUstream);
typedef CUresult (*cuStreamDestroy_v2_t)(CUstream);
typedef CUresult (*cuEventCreate_t)(CUevent *, unsigned int);
typedef CUresult (*cuEventRecord_t)(CUevent, CUstream);
typedef CUresult (*cuEventSynchronize_t)(CUevent);
typedef CUresult (*cuEventDestroy_v2_t)(CUevent);
typedef CUresult (*cuModuleLoadData_t)(CUmodule *, const void *);
typedef CUresult (*cuModuleGetFunction_t)(CUfunction *, CUmodule, const char *);
typedef CUresult (*cuModuleUnload_t)(CUmodule);
typedef CUresult (*cuLaunchKernel_t)(CUfunction, unsigned int, unsigned int, unsigned int,
                                     unsigned int, unsigned int, unsigned int,
                                     unsigned int, CUstream, void **, void **);

struct cuda_api {
    cuInit_t cuInit;
    cuDeviceGetCount_t cuDeviceGetCount;
    cuDeviceGet_t cuDeviceGet;
    cuDeviceGetName_t cuDeviceGetName;
    cuDeviceTotalMem_v2_t cuDeviceTotalMem_v2;
    cuDevicePrimaryCtxRetain_t cuDevicePrimaryCtxRetain;
    cuCtxSetCurrent_t cuCtxSetCurrent;
    cuCtxSynchronize_t cuCtxSynchronize;
    cuMemAlloc_v2_t cuMemAlloc_v2;
    cuMemFree_v2_t cuMemFree_v2;
    cuMemcpyHtoD_v2_t cuMemcpyHtoD_v2;
    cuMemcpyDtoH_v2_t cuMemcpyDtoH_v2;
    cuStreamCreate_t cuStreamCreate;
    cuStreamSynchronize_t cuStreamSynchronize;
    cuStreamDestroy_v2_t cuStreamDestroy_v2;
    cuEventCreate_t cuEventCreate;
    cuEventRecord_t cuEventRecord;
    cuEventSynchronize_t cuEventSynchronize;
    cuEventDestroy_v2_t cuEventDestroy_v2;
    cuModuleLoadData_t cuModuleLoadData;
    cuModuleGetFunction_t cuModuleGetFunction;
    cuModuleUnload_t cuModuleUnload;
    cuLaunchKernel_t cuLaunchKernel;
};

static const char *ptx = 
".version 7.0\n"
".target sm_70\n"
".address_size 64\n"
".visible .entry add_one(\n"
"    .param .u64 data,\n"
"    .param .u32 n\n"
")\n"
"{\n"
"    .reg .pred %p;\n"
"    .reg .b32 %r<5>;\n"
"    .reg .b64 %rd<6>;\n"
"    ld.param.u64 %rd1, [data];\n"
"    ld.param.u32 %r1, [n];\n"
"    mov.u32 %r2, %tid.x;\n"
"    setp.ge.u32 %p, %r2, %r1;\n"
"    @%p bra DONE;\n"
"    mul.wide.u32 %rd2, %r2, 4;\n"
"    add.u64 %rd3, %rd1, %rd2;\n"
"    ld.global.u32 %r3, [%rd3];\n"
"    add.u32 %r4, %r3, 1;\n"
"    st.global.u32 [%rd3], %r4;\n"
"DONE:\n"
"    ret;\n"
"}\n"
".visible .entry scale_add(\n"
"    .param .u64 src,\n"
"    .param .u64 dst,\n"
"    .param .u32 n,\n"
"    .param .u32 scale\n"
")\n"
"{\n"
"    .reg .pred %p2;\n"
"    .reg .b32 %r<8>;\n"
"    .reg .b64 %rd<8>;\n"
"    ld.param.u64 %rd1, [src];\n"
"    ld.param.u64 %rd2, [dst];\n"
"    ld.param.u32 %r1, [n];\n"
"    ld.param.u32 %r2, [scale];\n"
"    mov.u32 %r3, %tid.x;\n"
"    setp.ge.u32 %p2, %r3, %r1;\n"
"    @%p2 bra SCALE_DONE;\n"
"    mul.wide.u32 %rd3, %r3, 4;\n"
"    add.u64 %rd4, %rd1, %rd3;\n"
"    add.u64 %rd5, %rd2, %rd3;\n"
"    ld.global.u32 %r4, [%rd4];\n"
"    mul.lo.u32 %r5, %r4, %r2;\n"
"    add.u32 %r6, %r5, %r3;\n"
"    st.global.u32 [%rd5], %r6;\n"
"SCALE_DONE:\n"
"    ret;\n"
"}\n";

static int failed = 0;

static void check_rc(const char *case_name, const char *op, CUresult rc)
{
    printf("CASE %-28s OP %-28s RC %d\n", case_name, op, rc);
    if (rc != CUDA_SUCCESS) {
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

static int load_api(struct cuda_api *api)
{
    void *lib = dlopen("libcuda.so.1", RTLD_NOW | RTLD_GLOBAL);
    if (!lib) {
        fprintf(stderr, "DLERROR %s\n", dlerror());
        return 1;
    }

    memset(api, 0, sizeof(*api));
    api->cuInit = (cuInit_t)sym(lib, "cuInit");
    api->cuDeviceGetCount = (cuDeviceGetCount_t)sym(lib, "cuDeviceGetCount");
    api->cuDeviceGet = (cuDeviceGet_t)sym(lib, "cuDeviceGet");
    api->cuDeviceGetName = (cuDeviceGetName_t)sym(lib, "cuDeviceGetName");
    api->cuDeviceTotalMem_v2 = (cuDeviceTotalMem_v2_t)sym(lib, "cuDeviceTotalMem_v2");
    api->cuDevicePrimaryCtxRetain = (cuDevicePrimaryCtxRetain_t)sym(lib, "cuDevicePrimaryCtxRetain");
    api->cuCtxSetCurrent = (cuCtxSetCurrent_t)sym(lib, "cuCtxSetCurrent");
    api->cuCtxSynchronize = (cuCtxSynchronize_t)sym(lib, "cuCtxSynchronize");
    api->cuMemAlloc_v2 = (cuMemAlloc_v2_t)sym(lib, "cuMemAlloc_v2");
    api->cuMemFree_v2 = (cuMemFree_v2_t)sym(lib, "cuMemFree_v2");
    api->cuMemcpyHtoD_v2 = (cuMemcpyHtoD_v2_t)sym(lib, "cuMemcpyHtoD_v2");
    api->cuMemcpyDtoH_v2 = (cuMemcpyDtoH_v2_t)sym(lib, "cuMemcpyDtoH_v2");
    api->cuStreamCreate = (cuStreamCreate_t)sym(lib, "cuStreamCreate");
    api->cuStreamSynchronize = (cuStreamSynchronize_t)sym(lib, "cuStreamSynchronize");
    api->cuStreamDestroy_v2 = (cuStreamDestroy_v2_t)sym(lib, "cuStreamDestroy_v2");
    api->cuEventCreate = (cuEventCreate_t)sym(lib, "cuEventCreate");
    api->cuEventRecord = (cuEventRecord_t)sym(lib, "cuEventRecord");
    api->cuEventSynchronize = (cuEventSynchronize_t)sym(lib, "cuEventSynchronize");
    api->cuEventDestroy_v2 = (cuEventDestroy_v2_t)sym(lib, "cuEventDestroy_v2");
    api->cuModuleLoadData = (cuModuleLoadData_t)sym(lib, "cuModuleLoadData");
    api->cuModuleGetFunction = (cuModuleGetFunction_t)sym(lib, "cuModuleGetFunction");
    api->cuModuleUnload = (cuModuleUnload_t)dlsym(lib, "cuModuleUnload");
    api->cuLaunchKernel = (cuLaunchKernel_t)sym(lib, "cuLaunchKernel");
    return failed ? 1 : 0;
}

int main(void)
{
    struct cuda_api api;
    CUdevice dev = 0;
    CUcontext ctx = NULL;
    CUdeviceptr dptr = 0;
    CUdeviceptr dptr_out = 0;
    CUstream stream = NULL;
    CUevent event = NULL;
    CUmodule module = NULL;
    CUfunction func = NULL;
    CUfunction scale_func = NULL;
    enum { N = 64 };
    uint32_t input[N];
    uint32_t output[N];
    size_t total_mem = 0;
    int count = 0;
    char name[128];

    if (load_api(&api) != 0) {
        return 2;
    }

    check_rc("device_discovery", "cuInit", api.cuInit(0));
    check_rc("device_discovery", "cuDeviceGetCount", api.cuDeviceGetCount(&count));
    printf("DEVICE_COUNT %d\n", count);
    if (count < 1) {
        fprintf(stderr, "NO_DEVICE\n");
        return 3;
    }
    check_rc("device_discovery", "cuDeviceGet", api.cuDeviceGet(&dev, 0));
    memset(name, 0, sizeof(name));
    check_rc("device_discovery", "cuDeviceGetName", api.cuDeviceGetName(name, (int)sizeof(name), dev));
    check_rc("device_discovery", "cuDeviceTotalMem_v2", api.cuDeviceTotalMem_v2(&total_mem, dev));
    printf("DEVICE_NAME %s\n", name);
    printf("DEVICE_TOTAL_MEM %zu\n", total_mem);

    check_rc("context", "cuDevicePrimaryCtxRetain", api.cuDevicePrimaryCtxRetain(&ctx, dev));
    check_rc("context", "cuCtxSetCurrent", api.cuCtxSetCurrent(ctx));

    for (uint32_t i = 0; i < N; i++) {
        input[i] = i * 3u;
        output[i] = 0u;
    }

    check_rc("alloc_free", "cuMemAlloc_v2", api.cuMemAlloc_v2(&dptr, sizeof(input)));
    check_rc("alloc_free", "cuMemAlloc_v2 out", api.cuMemAlloc_v2(&dptr_out, sizeof(input)));
    check_rc("copy_htod", "cuMemcpyHtoD_v2", api.cuMemcpyHtoD_v2(dptr, input, sizeof(input)));
    check_rc("copy_dtoh", "cuMemcpyDtoH_v2", api.cuMemcpyDtoH_v2(output, dptr, sizeof(output)));
    if (memcmp(input, output, sizeof(input)) != 0) {
        fprintf(stderr, "COPY_MISMATCH before kernel\n");
        failed = 1;
    } else {
        printf("COPY_ROUNDTRIP_OK bytes=%zu\n", sizeof(input));
    }

    check_rc("stream", "cuStreamCreate", api.cuStreamCreate(&stream, 0));
    check_rc("event", "cuEventCreate", api.cuEventCreate(&event, 0));
    check_rc("module", "cuModuleLoadData", api.cuModuleLoadData(&module, ptx));
    check_rc("module", "cuModuleGetFunction", api.cuModuleGetFunction(&func, module, "add_one"));
    check_rc("module", "cuModuleGetFunction scale", api.cuModuleGetFunction(&scale_func, module, "scale_add"));

    uint32_t n_param = N;
    void *params[] = { &dptr, &n_param, NULL };
    check_rc("kernel", "cuLaunchKernel",
             api.cuLaunchKernel(func, 1, 1, 1, N, 1, 1, 0, stream, params, NULL));
    check_rc("event", "cuEventRecord", api.cuEventRecord(event, stream));
    check_rc("event", "cuEventSynchronize", api.cuEventSynchronize(event));
    check_rc("stream", "cuStreamSynchronize", api.cuStreamSynchronize(stream));
    check_rc("context", "cuCtxSynchronize", api.cuCtxSynchronize());
    memset(output, 0, sizeof(output));
    check_rc("copy_dtoh_after_kernel", "cuMemcpyDtoH_v2", api.cuMemcpyDtoH_v2(output, dptr, sizeof(output)));
    for (uint32_t i = 0; i < N; i++) {
        if (output[i] != input[i] + 1u) {
            fprintf(stderr, "KERNEL_MISMATCH idx=%u got=%u expected=%u\n",
                    i, output[i], input[i] + 1u);
            failed = 1;
            break;
        }
    }
    if (!failed) {
        printf("KERNEL_RESULT_OK n=%d\n", N);
    }

    memset(output, 0, sizeof(output));
    uint32_t scale_param = 7u;
    void *scale_params[] = { &dptr, &dptr_out, &n_param, &scale_param, NULL };
    check_rc("kernel_second_shape", "cuLaunchKernel scale_add",
             api.cuLaunchKernel(scale_func, 1, 1, 1, N, 1, 1, 0, stream, scale_params, NULL));
    check_rc("stream", "cuStreamSynchronize second", api.cuStreamSynchronize(stream));
    check_rc("context", "cuCtxSynchronize second", api.cuCtxSynchronize());
    check_rc("copy_dtoh_second_kernel", "cuMemcpyDtoH_v2", api.cuMemcpyDtoH_v2(output, dptr_out, sizeof(output)));
    for (uint32_t i = 0; i < N; i++) {
        uint32_t expected = (input[i] + 1u) * scale_param + i;
        if (output[i] != expected) {
            fprintf(stderr, "SECOND_KERNEL_MISMATCH idx=%u got=%u expected=%u\n",
                    i, output[i], expected);
            failed = 1;
            break;
        }
    }
    if (!failed) {
        printf("SECOND_KERNEL_RESULT_OK n=%d scale=%u\n", N, scale_param);
    }

    if (api.cuModuleUnload && module) {
        check_rc("module", "cuModuleUnload", api.cuModuleUnload(module));
    }
    if (event) {
        check_rc("event", "cuEventDestroy_v2", api.cuEventDestroy_v2(event));
    }
    if (stream) {
        check_rc("stream", "cuStreamDestroy_v2", api.cuStreamDestroy_v2(stream));
    }
    if (dptr) {
        check_rc("alloc_free", "cuMemFree_v2", api.cuMemFree_v2(dptr));
    }
    if (dptr_out) {
        check_rc("alloc_free", "cuMemFree_v2 out", api.cuMemFree_v2(dptr_out));
    }

    printf("OVERALL %s\n", failed ? "FAIL" : "PASS");
    return failed ? 1 : 0;
}
