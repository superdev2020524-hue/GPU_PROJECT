/*
 * Preflight (no Ollama): cublasGemmEx at GGML-like sizes seen before SIGSEGV
 * (journal: m=2048 n=512 k=2048; then m=256 n=512 k=2048 ×2).
 *
 * SYSTEMATIC_ERROR_TRACKING_PLAN.md §6 Step 5b — wide FP16 tensor-op Gemm (E7):
 *   gcc ... -o /tmp/test_gemm_ex_vm test_gemm_ex_vm.c -ldl
 *   LD_LIBRARY_PATH=/opt/vgpu/lib:/usr/local/lib/ollama/cuda_v12:/usr/local/lib/ollama:$LD_LIBRARY_PATH \
 *     /tmp/test_gemm_ex_vm e7
 *
 * Step 5a burst-then-wide (ERROR_TRACKING_STATUS May 2026 journal): same FP16/OP_T/OP_N/tensor-op
 * as e7, replay m/n/k pattern seen before cublas_status=13 under Ollama, then wide 32000:
 *   ... /tmp/test_gemm_ex_vm e7seq
 *
 * GGML-fidelity (non-default stream + cublasSetWorkspace, Step 5a next increment):
 *   ... /tmp/test_gemm_ex_vm e7ws       — wide only (~same wall as e7)
 *   ... /tmp/test_gemm_ex_vm e7seqws    — burst×4 + wide (~same wall as e7seq)
 *
 * Two cublasHandle_t (multi-handle hypothesis vs E7):
 *   ... e7dual    — warmup Gemm on h1 then h0, wide on h0 (~e7 wall)
 *   ... e7seq2h   — e7seq bursts alternating h0/h1, wide on h0 (~e7seq wall)
 *
 * Dom0 native (no vgpu on LD_LIBRARY_PATH), same source:
 *   gcc ... && LD_LIBRARY_PATH=/usr/local/cuda-12.3/lib64:$LD_LIBRARY_PATH /tmp/test_gemm_ex_vm e7
 *
 * Build on VM:
 *   gcc -O2 -std=c11 -Wall -o /tmp/test_gemm_ex_vm test_gemm_ex_vm.c -ldl
 * Run (wrapper-equivalent search path):
 *   LD_LIBRARY_PATH=/opt/vgpu/lib:/usr/local/lib/ollama/cuda_v12:/usr/local/lib/ollama:$LD_LIBRARY_PATH \
 *     /tmp/test_gemm_ex_vm
 *
 * Exit 0 only if cublasGemmEx + cuCtxSynchronize succeed (matches mediated stack).
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <dlfcn.h>

typedef unsigned long long CUdeviceptr;
typedef int CUdevice;
typedef void *CUcontext;
typedef void *cublasHandle_t;

enum {
    CUBLAS_STATUS_SUCCESS = 0,
    CUBLAS_STATUS_EXECUTION_FAILED = 13,
};

enum { CUBLAS_OP_N = 0, CUBLAS_OP_T = 1 };
#ifndef CUDA_R_32F
#define CUDA_R_32F 0
#endif
#ifndef CUDA_R_16F
#define CUDA_R_16F 2
#endif
#define CUBLAS_COMPUTE_32F 68
#define CUBLAS_COMPUTE_16F 64
#define CUBLAS_GEMM_DEFAULT 0
#define CUBLAS_GEMM_DEFAULT_TENSOR_OP 99
/* IEEE half 1.0f */
#define HALF_ONE UINT16_C(0x3c00)

typedef int (*cuInit_t)(unsigned int);
typedef int (*cuDevicePrimaryCtxRetain_t)(CUcontext *, CUdevice);
typedef int (*cuCtxSetCurrent_t)(CUcontext);
typedef int (*cuCtxSynchronize_t)(void);
typedef int (*cuMemAlloc_v2_t)(CUdeviceptr *, size_t);
typedef int (*cuMemFree_v2_t)(CUdeviceptr);
typedef int (*cuMemcpyHtoD_v2_t)(CUdeviceptr, const void *, size_t);
typedef int (*cuStreamCreate_t)(void **phStream, unsigned int Flags);
typedef int (*cuStreamDestroy_t)(void *hStream);
typedef int (*cuStreamSynchronize_t)(void *hStream);

typedef int (*cublasCreate_v2_t)(cublasHandle_t *);
typedef int (*cublasDestroy_v2_t)(cublasHandle_t);
typedef int (*cublasGemmEx_t)(
    cublasHandle_t, int transa, int transb,
    int m, int n, int k,
    const void *alpha,
    const void *A, int Atype, int lda,
    const void *B, int Btype, int ldb,
    const void *beta,
    void *C, int Ctype, int ldc,
    int computeType, int algo);
typedef int (*cublasSetStream_t)(cublasHandle_t handle, void *stream);
typedef int (*cublasSetWorkspace_t)(cublasHandle_t handle, void *workspace, size_t workspaceSizeInBytes);

/* Typical GGML-style device workspace for cuBLAS 12 (harness); not a tuning optimum. */
#define E7_CUBLAS_WORKSPACE_BYTES ((size_t)32 * 1024 * 1024)

static int run_gemm(cublasGemmEx_t fn, cublasHandle_t h, cuCtxSynchronize_t sync,
                    int m, int n, int k,
                    CUdeviceptr da, CUdeviceptr db, CUdeviceptr dc,
                    int lda, int ldb, int ldc)
{
    float alpha = 1.f;
    float beta = 0.f;
    int st = fn(h, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k,
                &alpha,
                (const void *)(uintptr_t)da, CUDA_R_32F, lda,
                (const void *)(uintptr_t)db, CUDA_R_32F, ldb,
                &beta,
                (void *)(uintptr_t)dc, CUDA_R_32F, ldc,
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    printf("  cublasGemmEx m=%d n=%d k=%d lda=%d ldb=%d ldc=%d -> %d\n",
           m, n, k, lda, ldb, ldc, st);
    if (st != CUBLAS_STATUS_SUCCESS)
        return st;
    if (sync) {
        int e = sync();
        printf("  cuCtxSynchronize after GemmEx -> %d\n", e);
        if (e != 0)
            return CUBLAS_STATUS_EXECUTION_FAILED;
    }
    return CUBLAS_STATUS_SUCCESS;
}

/* Step 5b / E7: match ERROR_TRACKING_STATUS journal — OP_T, OP_N, m=32000, FP16 tensor op. */
static int run_e7_f16_tensor_wide(void)
{
    void *cudart __attribute__((unused)) = NULL, *cuda = NULL, *cublas = NULL;
    cublasHandle_t handle = NULL;
    CUcontext ctx = NULL;

    cudart = dlopen("/opt/vgpu/lib/libcudart.so.12", RTLD_NOW | RTLD_GLOBAL);
    if (!cudart)
        cudart = dlopen("libcudart.so.12", RTLD_NOW | RTLD_GLOBAL);
    cuda = dlopen("libcuda.so.1", RTLD_NOW | RTLD_GLOBAL);
    cublas = dlopen("libcublas.so.12", RTLD_NOW | RTLD_GLOBAL);
    if (!cuda || !cublas) {
        fprintf(stderr, "FAIL dlopen: %s\n", dlerror());
        return 1;
    }

    cuInit_t cuInit = (cuInit_t)dlsym(cuda, "cuInit");
    cuDevicePrimaryCtxRetain_t cuDevicePrimaryCtxRetain =
        (cuDevicePrimaryCtxRetain_t)dlsym(cuda, "cuDevicePrimaryCtxRetain");
    cuCtxSetCurrent_t cuCtxSetCurrent = (cuCtxSetCurrent_t)dlsym(cuda, "cuCtxSetCurrent");
    cuCtxSynchronize_t cuCtxSynchronize = (cuCtxSynchronize_t)dlsym(cuda, "cuCtxSynchronize");
    cuMemAlloc_v2_t cuMemAlloc_v2 = (cuMemAlloc_v2_t)dlsym(cuda, "cuMemAlloc_v2");
    cuMemFree_v2_t cuMemFree_v2 = (cuMemFree_v2_t)dlsym(cuda, "cuMemFree_v2");
    cuMemcpyHtoD_v2_t cuMemcpyHtoD_v2 = (cuMemcpyHtoD_v2_t)dlsym(cuda, "cuMemcpyHtoD_v2");
    cublasCreate_v2_t cublasCreate_v2 = (cublasCreate_v2_t)dlsym(cublas, "cublasCreate_v2");
    cublasDestroy_v2_t cublasDestroy_v2 = (cublasDestroy_v2_t)dlsym(cublas, "cublasDestroy_v2");
    cublasGemmEx_t cublasGemmEx = (cublasGemmEx_t)dlsym(cublas, "cublasGemmEx");

    if (!cuInit || !cuDevicePrimaryCtxRetain || !cuCtxSetCurrent || !cuMemAlloc_v2 ||
        !cuMemFree_v2 || !cuMemcpyHtoD_v2 || !cublasCreate_v2 || !cublasDestroy_v2 ||
        !cublasGemmEx || !cuCtxSynchronize) {
        fprintf(stderr, "FAIL dlsym\n");
        return 1;
    }

    printf("=== test_gemm_ex_vm e7 (Step 5b: m=32000 FP16 CUBLAS_COMPUTE_16F TENSOR_OP, OP_T/OP_N) ===\n");

    if (cuInit(0) != 0) {
        fprintf(stderr, "FAIL cuInit\n");
        return 1;
    }
    if (cuDevicePrimaryCtxRetain(&ctx, 0) != 0 || !ctx) {
        fprintf(stderr, "FAIL cuDevicePrimaryCtxRetain\n");
        return 1;
    }
    if (cuCtxSetCurrent(ctx) != 0) {
        fprintf(stderr, "FAIL cuCtxSetCurrent\n");
        return 1;
    }

    int st = cublasCreate_v2(&handle);
    if (st != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cublasCreate_v2 -> %d\n", st);
        return 1;
    }

    const int m = 32000;
    const int n = 512;
    const int k = 2048;
    const int lda = k;
    const int ldb = k;
    const int ldc = m;

    const size_t sz_a = (size_t)k * (size_t)m * 2u;
    const size_t sz_b = (size_t)k * (size_t)n * 2u;
    const size_t sz_c = (size_t)m * (size_t)n * 2u;

    uint16_t *ha = (uint16_t *)calloc(1, sz_a);
    uint16_t *hb = (uint16_t *)calloc(1, sz_b);
    uint16_t *hc = (uint16_t *)calloc(1, sz_c);
    if (!ha || !hb || !hc) {
        fprintf(stderr, "FAIL calloc e7 buffers\n");
        free(ha);
        free(hb);
        free(hc);
        cublasDestroy_v2(handle);
        return 1;
    }

    CUdeviceptr da = 0, db = 0, dc = 0;
    if (cuMemAlloc_v2(&da, sz_a) != 0 || cuMemAlloc_v2(&db, sz_b) != 0 ||
        cuMemAlloc_v2(&dc, sz_c) != 0) {
        fprintf(stderr, "FAIL cuMemAlloc e7 (need ~%zu bytes GPU)\n", sz_a + sz_b + sz_c);
        free(ha);
        free(hb);
        free(hc);
        cublasDestroy_v2(handle);
        return 1;
    }

    cuMemcpyHtoD_v2(da, ha, sz_a);
    cuMemcpyHtoD_v2(db, hb, sz_b);

    uint16_t alpha = HALF_ONE;
    uint16_t beta = 0;

    st = cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k,
                      &alpha,
                      (const void *)(uintptr_t)da, CUDA_R_16F, lda,
                      (const void *)(uintptr_t)db, CUDA_R_16F, ldb,
                      &beta,
                      (void *)(uintptr_t)dc, CUDA_R_16F, ldc,
                      CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    printf("  cublasGemmEx OP_T/OP_N m=%d n=%d k=%d FP16 compute=16F algo=TENSOR_OP -> status=%d\n",
           m, n, k, st);

    if (st == CUBLAS_STATUS_SUCCESS) {
        int e = cuCtxSynchronize();
        printf("  cuCtxSynchronize -> %d\n", e);
        if (e != 0)
            st = CUBLAS_STATUS_EXECUTION_FAILED;
    }

    cuMemFree_v2(dc);
    cuMemFree_v2(db);
    cuMemFree_v2(da);
    free(ha);
    free(hb);
    free(hc);
    cublasDestroy_v2(handle);

    if (st != CUBLAS_STATUS_SUCCESS) {
        printf("E7_WIDE_FAIL cublas_status=%d\n", st);
        return 2;
    }
    printf("E7_WIDE_OK GemmEx + sync passed\n");
    return 0;
}

/* One FP16 GemmEx OP_T/OP_N tensor-op; allocates, runs, syncs, frees. Returns cublas status. */
static int gemm_e7_f16_once(
    cublasGemmEx_t cublasGemmEx, cuCtxSynchronize_t cuCtxSynchronize,
    cuStreamSynchronize_t cuStreamSynchronize, void *stream,
    cuMemAlloc_v2_t cuMemAlloc_v2, cuMemFree_v2_t cuMemFree_v2,
    cuMemcpyHtoD_v2_t cuMemcpyHtoD_v2, cublasHandle_t handle,
    int m, int n, int k, const char *note)
{
    const int lda = k;
    const int ldb = k;
    const int ldc = m;
    const size_t sz_a = (size_t)k * (size_t)m * 2u;
    const size_t sz_b = (size_t)k * (size_t)n * 2u;
    const size_t sz_c = (size_t)m * (size_t)n * 2u;

    uint16_t *ha = (uint16_t *)calloc(1, sz_a);
    uint16_t *hb = (uint16_t *)calloc(1, sz_b);
    uint16_t *hc = (uint16_t *)calloc(1, sz_c);
    if (!ha || !hb || !hc) {
        free(ha);
        free(hb);
        free(hc);
        fprintf(stderr, "FAIL calloc %s m=%d n=%d k=%d\n", note, m, n, k);
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }

    CUdeviceptr da = 0, db = 0, dc = 0;
    if (cuMemAlloc_v2(&da, sz_a) != 0 || cuMemAlloc_v2(&db, sz_b) != 0 ||
        cuMemAlloc_v2(&dc, sz_c) != 0) {
        fprintf(stderr, "FAIL cuMemAlloc %s\n", note);
        free(ha);
        free(hb);
        free(hc);
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }

    cuMemcpyHtoD_v2(da, ha, sz_a);
    cuMemcpyHtoD_v2(db, hb, sz_b);

    uint16_t alpha = HALF_ONE;
    uint16_t beta = 0;

    int st = cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k,
                          &alpha,
                          (const void *)(uintptr_t)da, CUDA_R_16F, lda,
                          (const void *)(uintptr_t)db, CUDA_R_16F, ldb,
                          &beta,
                          (void *)(uintptr_t)dc, CUDA_R_16F, ldc,
                          CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    printf("  [%s] GemmEx OP_T/OP_N m=%d n=%d k=%d -> status=%d\n", note, m, n, k, st);

    if (st == CUBLAS_STATUS_SUCCESS) {
        int e;
        if (cuStreamSynchronize && stream)
            e = cuStreamSynchronize(stream);
        else
            e = cuCtxSynchronize();
        if (e != 0) {
            if (cuStreamSynchronize && stream)
                printf("  cuStreamSynchronize -> %d\n", e);
            else
                printf("  cuCtxSynchronize -> %d\n", e);
            st = CUBLAS_STATUS_EXECUTION_FAILED;
        }
    }

    cuMemFree_v2(dc);
    cuMemFree_v2(db);
    cuMemFree_v2(da);
    free(ha);
    free(hb);
    free(hc);
    return st;
}

/* Wide FP16 tensor Gemm with dedicated stream + optional cublas workspace (GGML-ish). */
static int run_e7_f16_tensor_wide_ws(void)
{
    void *cudart __attribute__((unused)) = NULL, *cuda = NULL, *cublas = NULL;
    cublasHandle_t handle = NULL;
    CUcontext ctx = NULL;
    void *stream = NULL;
    CUdeviceptr d_workspace = 0;

    cudart = dlopen("/opt/vgpu/lib/libcudart.so.12", RTLD_NOW | RTLD_GLOBAL);
    if (!cudart)
        cudart = dlopen("libcudart.so.12", RTLD_NOW | RTLD_GLOBAL);
    cuda = dlopen("libcuda.so.1", RTLD_NOW | RTLD_GLOBAL);
    cublas = dlopen("libcublas.so.12", RTLD_NOW | RTLD_GLOBAL);
    if (!cuda || !cublas) {
        fprintf(stderr, "FAIL dlopen: %s\n", dlerror());
        return 1;
    }

    cuInit_t cuInit = (cuInit_t)dlsym(cuda, "cuInit");
    cuDevicePrimaryCtxRetain_t cuDevicePrimaryCtxRetain =
        (cuDevicePrimaryCtxRetain_t)dlsym(cuda, "cuDevicePrimaryCtxRetain");
    cuCtxSetCurrent_t cuCtxSetCurrent = (cuCtxSetCurrent_t)dlsym(cuda, "cuCtxSetCurrent");
    cuCtxSynchronize_t cuCtxSynchronize = (cuCtxSynchronize_t)dlsym(cuda, "cuCtxSynchronize");
    cuMemAlloc_v2_t cuMemAlloc_v2 = (cuMemAlloc_v2_t)dlsym(cuda, "cuMemAlloc_v2");
    cuMemFree_v2_t cuMemFree_v2 = (cuMemFree_v2_t)dlsym(cuda, "cuMemFree_v2");
    cuMemcpyHtoD_v2_t cuMemcpyHtoD_v2 = (cuMemcpyHtoD_v2_t)dlsym(cuda, "cuMemcpyHtoD_v2");
    cuStreamCreate_t cuStreamCreate = (cuStreamCreate_t)dlsym(cuda, "cuStreamCreate");
    cuStreamDestroy_t cuStreamDestroy = (cuStreamDestroy_t)dlsym(cuda, "cuStreamDestroy");
    cuStreamSynchronize_t cuStreamSynchronize = (cuStreamSynchronize_t)dlsym(cuda, "cuStreamSynchronize");
    cublasCreate_v2_t cublasCreate_v2 = (cublasCreate_v2_t)dlsym(cublas, "cublasCreate_v2");
    cublasDestroy_v2_t cublasDestroy_v2 = (cublasDestroy_v2_t)dlsym(cublas, "cublasDestroy_v2");
    cublasGemmEx_t cublasGemmEx = (cublasGemmEx_t)dlsym(cublas, "cublasGemmEx");
    cublasSetStream_t cublasSetStream = (cublasSetStream_t)dlsym(cublas, "cublasSetStream");
    cublasSetWorkspace_t cublasSetWorkspace = (cublasSetWorkspace_t)dlsym(cublas, "cublasSetWorkspace");

    if (!cuInit || !cuDevicePrimaryCtxRetain || !cuCtxSetCurrent || !cuMemAlloc_v2 ||
        !cuMemFree_v2 || !cuMemcpyHtoD_v2 || !cublasCreate_v2 || !cublasDestroy_v2 ||
        !cublasGemmEx || !cuCtxSynchronize || !cuStreamCreate || !cuStreamDestroy ||
        !cuStreamSynchronize || !cublasSetStream) {
        fprintf(stderr, "FAIL dlsym (stream path requires cuStream* + cublasSetStream)\n");
        return 1;
    }

    printf("=== test_gemm_ex_vm e7ws (wide + non-default stream");
    if (cublasSetWorkspace)
        printf(" + cublasSetWorkspace %zu bytes", (size_t)E7_CUBLAS_WORKSPACE_BYTES);
    printf(") ===\n");

    if (cuInit(0) != 0) {
        fprintf(stderr, "FAIL cuInit\n");
        return 1;
    }
    if (cuDevicePrimaryCtxRetain(&ctx, 0) != 0 || !ctx) {
        fprintf(stderr, "FAIL cuDevicePrimaryCtxRetain\n");
        return 1;
    }
    if (cuCtxSetCurrent(ctx) != 0) {
        fprintf(stderr, "FAIL cuCtxSetCurrent\n");
        return 1;
    }

    int st = cublasCreate_v2(&handle);
    if (st != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cublasCreate_v2 -> %d\n", st);
        return 1;
    }

    if (cuStreamCreate(&stream, 0U) != 0 || !stream) {
        fprintf(stderr, "FAIL cuStreamCreate\n");
        cublasDestroy_v2(handle);
        return 1;
    }
    st = cublasSetStream(handle, stream);
    if (st != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cublasSetStream -> %d\n", st);
        cuStreamDestroy(stream);
        cublasDestroy_v2(handle);
        return 1;
    }

    if (cublasSetWorkspace) {
        if (cuMemAlloc_v2(&d_workspace, E7_CUBLAS_WORKSPACE_BYTES) != 0) {
            fprintf(stderr, "FAIL cuMemAlloc workspace\n");
            cuStreamDestroy(stream);
            cublasDestroy_v2(handle);
            return 1;
        }
        st = cublasSetWorkspace(handle, (void *)(uintptr_t)d_workspace, E7_CUBLAS_WORKSPACE_BYTES);
        if (st != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "cublasSetWorkspace -> %d\n", st);
            cuMemFree_v2(d_workspace);
            cuStreamDestroy(stream);
            cublasDestroy_v2(handle);
            return 1;
        }
    }

    const int m = 32000;
    const int n = 512;
    const int k = 2048;
    const int lda = k;
    const int ldb = k;
    const int ldc = m;

    const size_t sz_a = (size_t)k * (size_t)m * 2u;
    const size_t sz_b = (size_t)k * (size_t)n * 2u;
    const size_t sz_c = (size_t)m * (size_t)n * 2u;

    uint16_t *ha = (uint16_t *)calloc(1, sz_a);
    uint16_t *hb = (uint16_t *)calloc(1, sz_b);
    uint16_t *hc = (uint16_t *)calloc(1, sz_c);
    if (!ha || !hb || !hc) {
        fprintf(stderr, "FAIL calloc e7ws buffers\n");
        free(ha);
        free(hb);
        free(hc);
        if (d_workspace)
            cuMemFree_v2(d_workspace);
        cuStreamDestroy(stream);
        cublasDestroy_v2(handle);
        return 1;
    }

    CUdeviceptr da = 0, db = 0, dc = 0;
    if (cuMemAlloc_v2(&da, sz_a) != 0 || cuMemAlloc_v2(&db, sz_b) != 0 ||
        cuMemAlloc_v2(&dc, sz_c) != 0) {
        fprintf(stderr, "FAIL cuMemAlloc e7ws\n");
        free(ha);
        free(hb);
        free(hc);
        if (d_workspace)
            cuMemFree_v2(d_workspace);
        cuStreamDestroy(stream);
        cublasDestroy_v2(handle);
        return 1;
    }

    cuMemcpyHtoD_v2(da, ha, sz_a);
    cuMemcpyHtoD_v2(db, hb, sz_b);

    uint16_t alpha = HALF_ONE;
    uint16_t beta = 0;

    st = cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k,
                      &alpha,
                      (const void *)(uintptr_t)da, CUDA_R_16F, lda,
                      (const void *)(uintptr_t)db, CUDA_R_16F, ldb,
                      &beta,
                      (void *)(uintptr_t)dc, CUDA_R_16F, ldc,
                      CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    printf("  cublasGemmEx OP_T/OP_N m=%d n=%d k=%d (stream+ws) -> status=%d\n", m, n, k, st);

    if (st == CUBLAS_STATUS_SUCCESS) {
        int e = cuStreamSynchronize(stream);
        printf("  cuStreamSynchronize -> %d\n", e);
        if (e != 0)
            st = CUBLAS_STATUS_EXECUTION_FAILED;
    }

    cuMemFree_v2(dc);
    cuMemFree_v2(db);
    cuMemFree_v2(da);
    free(ha);
    free(hb);
    free(hc);

    cublasDestroy_v2(handle);
    if (d_workspace)
        cuMemFree_v2(d_workspace);
    cuStreamDestroy(stream);

    if (st != CUBLAS_STATUS_SUCCESS) {
        printf("E7WS_WIDE_FAIL cublas_status=%d\n", st);
        return 2;
    }
    printf("E7WS_WIDE_OK GemmEx + stream sync passed\n");
    return 0;
}

/*
 * Journal May 14 17:40:16 (FA-off E7): many cublas_status=0 with m in {256,2048,5632}, then m=32000 -> 13.
 * One logical "cycle" of ops before the failing line (approximate order from shim CALLED lines).
 */
static const int e7_journal_burst[][3] = {
    {5632, 512, 2048},
    {5632, 512, 2048},
    {2048, 512, 5632},
    {2048, 512, 2048},
    {256, 512, 2048},
    {256, 512, 2048},
    {2048, 512, 2048},
};

static int run_e7_burst_then_wide(void)
{
    void *cudart __attribute__((unused)) = NULL, *cuda = NULL, *cublas = NULL;
    cublasHandle_t handle = NULL;
    CUcontext ctx = NULL;

    cudart = dlopen("/opt/vgpu/lib/libcudart.so.12", RTLD_NOW | RTLD_GLOBAL);
    if (!cudart)
        cudart = dlopen("libcudart.so.12", RTLD_NOW | RTLD_GLOBAL);
    cuda = dlopen("libcuda.so.1", RTLD_NOW | RTLD_GLOBAL);
    cublas = dlopen("libcublas.so.12", RTLD_NOW | RTLD_GLOBAL);
    if (!cuda || !cublas) {
        fprintf(stderr, "FAIL dlopen: %s\n", dlerror());
        return 1;
    }

    cuInit_t cuInit = (cuInit_t)dlsym(cuda, "cuInit");
    cuDevicePrimaryCtxRetain_t cuDevicePrimaryCtxRetain =
        (cuDevicePrimaryCtxRetain_t)dlsym(cuda, "cuDevicePrimaryCtxRetain");
    cuCtxSetCurrent_t cuCtxSetCurrent = (cuCtxSetCurrent_t)dlsym(cuda, "cuCtxSetCurrent");
    cuCtxSynchronize_t cuCtxSynchronize = (cuCtxSynchronize_t)dlsym(cuda, "cuCtxSynchronize");
    cuMemAlloc_v2_t cuMemAlloc_v2 = (cuMemAlloc_v2_t)dlsym(cuda, "cuMemAlloc_v2");
    cuMemFree_v2_t cuMemFree_v2 = (cuMemFree_v2_t)dlsym(cuda, "cuMemFree_v2");
    cuMemcpyHtoD_v2_t cuMemcpyHtoD_v2 = (cuMemcpyHtoD_v2_t)dlsym(cuda, "cuMemcpyHtoD_v2");
    cublasCreate_v2_t cublasCreate_v2 = (cublasCreate_v2_t)dlsym(cublas, "cublasCreate_v2");
    cublasDestroy_v2_t cublasDestroy_v2 = (cublasDestroy_v2_t)dlsym(cublas, "cublasDestroy_v2");
    cublasGemmEx_t cublasGemmEx = (cublasGemmEx_t)dlsym(cublas, "cublasGemmEx");

    if (!cuInit || !cuDevicePrimaryCtxRetain || !cuCtxSetCurrent || !cuMemAlloc_v2 ||
        !cuMemFree_v2 || !cuMemcpyHtoD_v2 || !cublasCreate_v2 || !cublasDestroy_v2 ||
        !cublasGemmEx || !cuCtxSynchronize) {
        fprintf(stderr, "FAIL dlsym\n");
        return 1;
    }

    const int nburst = (int)(sizeof(e7_journal_burst) / sizeof(e7_journal_burst[0]));
    const int nrepeat = 4; /* ~ journal density before wide call */

    printf("=== test_gemm_ex_vm e7seq (burst %d shapes x%d, then wide m=32000) ===\n", nburst, nrepeat);

    if (cuInit(0) != 0) {
        fprintf(stderr, "FAIL cuInit\n");
        return 1;
    }
    if (cuDevicePrimaryCtxRetain(&ctx, 0) != 0 || !ctx) {
        fprintf(stderr, "FAIL cuDevicePrimaryCtxRetain\n");
        return 1;
    }
    if (cuCtxSetCurrent(ctx) != 0) {
        fprintf(stderr, "FAIL cuCtxSetCurrent\n");
        return 1;
    }

    int st = cublasCreate_v2(&handle);
    if (st != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cublasCreate_v2 -> %d\n", st);
        return 1;
    }

    int step = 0;
    for (int r = 0; r < nrepeat; r++) {
        for (int i = 0; i < nburst; i++) {
            int m = e7_journal_burst[i][0];
            int n = e7_journal_burst[i][1];
            int k = e7_journal_burst[i][2];
            char note[48];
            snprintf(note, sizeof(note), "burst r=%d i=%d", r, i);
            st = gemm_e7_f16_once(cublasGemmEx, cuCtxSynchronize, NULL, NULL,
                                  cuMemAlloc_v2, cuMemFree_v2, cuMemcpyHtoD_v2, handle,
                                  m, n, k, note);
            step++;
            if (st != CUBLAS_STATUS_SUCCESS) {
                printf("E7SEQ_BURST_FAIL step=%d cublas_status=%d\n", step, st);
                cublasDestroy_v2(handle);
                return 2;
            }
        }
    }

    st = gemm_e7_f16_once(cublasGemmEx, cuCtxSynchronize, NULL, NULL,
                          cuMemAlloc_v2, cuMemFree_v2, cuMemcpyHtoD_v2, handle,
                          32000, 512, 2048, "wide");
    cublasDestroy_v2(handle);

    if (st != CUBLAS_STATUS_SUCCESS) {
        printf("E7SEQ_WIDE_FAIL cublas_status=%d (burst-then-wide repro)\n", st);
        return 3;
    }
    printf("E7SEQ_OK burst+wide GemmEx + sync passed\n");
    return 0;
}

static int run_e7_burst_then_wide_ws(void)
{
    void *cudart __attribute__((unused)) = NULL, *cuda = NULL, *cublas = NULL;
    cublasHandle_t handle = NULL;
    CUcontext ctx = NULL;
    void *stream = NULL;
    CUdeviceptr d_workspace = 0;

    cudart = dlopen("/opt/vgpu/lib/libcudart.so.12", RTLD_NOW | RTLD_GLOBAL);
    if (!cudart)
        cudart = dlopen("libcudart.so.12", RTLD_NOW | RTLD_GLOBAL);
    cuda = dlopen("libcuda.so.1", RTLD_NOW | RTLD_GLOBAL);
    cublas = dlopen("libcublas.so.12", RTLD_NOW | RTLD_GLOBAL);
    if (!cuda || !cublas) {
        fprintf(stderr, "FAIL dlopen: %s\n", dlerror());
        return 1;
    }

    cuInit_t cuInit = (cuInit_t)dlsym(cuda, "cuInit");
    cuDevicePrimaryCtxRetain_t cuDevicePrimaryCtxRetain =
        (cuDevicePrimaryCtxRetain_t)dlsym(cuda, "cuDevicePrimaryCtxRetain");
    cuCtxSetCurrent_t cuCtxSetCurrent = (cuCtxSetCurrent_t)dlsym(cuda, "cuCtxSetCurrent");
    cuCtxSynchronize_t cuCtxSynchronize = (cuCtxSynchronize_t)dlsym(cuda, "cuCtxSynchronize");
    cuMemAlloc_v2_t cuMemAlloc_v2 = (cuMemAlloc_v2_t)dlsym(cuda, "cuMemAlloc_v2");
    cuMemFree_v2_t cuMemFree_v2 = (cuMemFree_v2_t)dlsym(cuda, "cuMemFree_v2");
    cuMemcpyHtoD_v2_t cuMemcpyHtoD_v2 = (cuMemcpyHtoD_v2_t)dlsym(cuda, "cuMemcpyHtoD_v2");
    cuStreamCreate_t cuStreamCreate = (cuStreamCreate_t)dlsym(cuda, "cuStreamCreate");
    cuStreamDestroy_t cuStreamDestroy = (cuStreamDestroy_t)dlsym(cuda, "cuStreamDestroy");
    cuStreamSynchronize_t cuStreamSynchronize = (cuStreamSynchronize_t)dlsym(cuda, "cuStreamSynchronize");
    cublasCreate_v2_t cublasCreate_v2 = (cublasCreate_v2_t)dlsym(cublas, "cublasCreate_v2");
    cublasDestroy_v2_t cublasDestroy_v2 = (cublasDestroy_v2_t)dlsym(cublas, "cublasDestroy_v2");
    cublasGemmEx_t cublasGemmEx = (cublasGemmEx_t)dlsym(cublas, "cublasGemmEx");
    cublasSetStream_t cublasSetStream = (cublasSetStream_t)dlsym(cublas, "cublasSetStream");
    cublasSetWorkspace_t cublasSetWorkspace = (cublasSetWorkspace_t)dlsym(cublas, "cublasSetWorkspace");

    if (!cuInit || !cuDevicePrimaryCtxRetain || !cuCtxSetCurrent || !cuMemAlloc_v2 ||
        !cuMemFree_v2 || !cuMemcpyHtoD_v2 || !cublasCreate_v2 || !cublasDestroy_v2 ||
        !cublasGemmEx || !cuCtxSynchronize || !cuStreamCreate || !cuStreamDestroy ||
        !cuStreamSynchronize || !cublasSetStream) {
        fprintf(stderr, "FAIL dlsym (e7seqws)\n");
        return 1;
    }

    const int nburst = (int)(sizeof(e7_journal_burst) / sizeof(e7_journal_burst[0]));
    const int nrepeat = 4;

    printf("=== test_gemm_ex_vm e7seqws (burst %d x%d + wide, stream", nburst, nrepeat);
    if (cublasSetWorkspace)
        printf(" + workspace %zu bytes", (size_t)E7_CUBLAS_WORKSPACE_BYTES);
    printf(") ===\n");

    if (cuInit(0) != 0) {
        fprintf(stderr, "FAIL cuInit\n");
        return 1;
    }
    if (cuDevicePrimaryCtxRetain(&ctx, 0) != 0 || !ctx) {
        fprintf(stderr, "FAIL cuDevicePrimaryCtxRetain\n");
        return 1;
    }
    if (cuCtxSetCurrent(ctx) != 0) {
        fprintf(stderr, "FAIL cuCtxSetCurrent\n");
        return 1;
    }

    int st = cublasCreate_v2(&handle);
    if (st != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cublasCreate_v2 -> %d\n", st);
        return 1;
    }

    if (cuStreamCreate(&stream, 0U) != 0 || !stream) {
        fprintf(stderr, "FAIL cuStreamCreate\n");
        cublasDestroy_v2(handle);
        return 1;
    }
    st = cublasSetStream(handle, stream);
    if (st != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cublasSetStream -> %d\n", st);
        cuStreamDestroy(stream);
        cublasDestroy_v2(handle);
        return 1;
    }

    if (cublasSetWorkspace) {
        if (cuMemAlloc_v2(&d_workspace, E7_CUBLAS_WORKSPACE_BYTES) != 0) {
            fprintf(stderr, "FAIL cuMemAlloc workspace\n");
            cuStreamDestroy(stream);
            cublasDestroy_v2(handle);
            return 1;
        }
        st = cublasSetWorkspace(handle, (void *)(uintptr_t)d_workspace, E7_CUBLAS_WORKSPACE_BYTES);
        if (st != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "cublasSetWorkspace -> %d\n", st);
            cuMemFree_v2(d_workspace);
            cuStreamDestroy(stream);
            cublasDestroy_v2(handle);
            return 1;
        }
    }

    int step = 0;
    for (int r = 0; r < nrepeat; r++) {
        for (int i = 0; i < nburst; i++) {
            int m = e7_journal_burst[i][0];
            int n = e7_journal_burst[i][1];
            int k = e7_journal_burst[i][2];
            char note[48];
            snprintf(note, sizeof(note), "burst r=%d i=%d", r, i);
            st = gemm_e7_f16_once(cublasGemmEx, cuCtxSynchronize, cuStreamSynchronize, stream,
                                  cuMemAlloc_v2, cuMemFree_v2, cuMemcpyHtoD_v2, handle,
                                  m, n, k, note);
            step++;
            if (st != CUBLAS_STATUS_SUCCESS) {
                printf("E7SEQWS_BURST_FAIL step=%d cublas_status=%d\n", step, st);
                cublasDestroy_v2(handle);
                if (d_workspace)
                    cuMemFree_v2(d_workspace);
                cuStreamDestroy(stream);
                return 2;
            }
        }
    }

    st = gemm_e7_f16_once(cublasGemmEx, cuCtxSynchronize, cuStreamSynchronize, stream,
                          cuMemAlloc_v2, cuMemFree_v2, cuMemcpyHtoD_v2, handle,
                          32000, 512, 2048, "wide");
    cublasDestroy_v2(handle);
    if (d_workspace)
        cuMemFree_v2(d_workspace);
    cuStreamDestroy(stream);

    if (st != CUBLAS_STATUS_SUCCESS) {
        printf("E7SEQWS_WIDE_FAIL cublas_status=%d\n", st);
        return 3;
    }
    printf("E7SEQWS_OK burst+wide + stream sync passed\n");
    return 0;
}

/* Two handles, same context: small FP16 Gemm on each then wide on h0 (bounded E7 multi-handle probe). */
static int run_e7_dual_wide(void)
{
    void *cudart __attribute__((unused)) = NULL, *cuda = NULL, *cublas = NULL;
    cublasHandle_t h0 = NULL, h1 = NULL;
    CUcontext ctx = NULL;

    cudart = dlopen("/opt/vgpu/lib/libcudart.so.12", RTLD_NOW | RTLD_GLOBAL);
    if (!cudart)
        cudart = dlopen("libcudart.so.12", RTLD_NOW | RTLD_GLOBAL);
    cuda = dlopen("libcuda.so.1", RTLD_NOW | RTLD_GLOBAL);
    cublas = dlopen("libcublas.so.12", RTLD_NOW | RTLD_GLOBAL);
    if (!cuda || !cublas) {
        fprintf(stderr, "FAIL dlopen: %s\n", dlerror());
        return 1;
    }

    cuInit_t cuInit = (cuInit_t)dlsym(cuda, "cuInit");
    cuDevicePrimaryCtxRetain_t cuDevicePrimaryCtxRetain =
        (cuDevicePrimaryCtxRetain_t)dlsym(cuda, "cuDevicePrimaryCtxRetain");
    cuCtxSetCurrent_t cuCtxSetCurrent = (cuCtxSetCurrent_t)dlsym(cuda, "cuCtxSetCurrent");
    cuCtxSynchronize_t cuCtxSynchronize = (cuCtxSynchronize_t)dlsym(cuda, "cuCtxSynchronize");
    cuMemAlloc_v2_t cuMemAlloc_v2 = (cuMemAlloc_v2_t)dlsym(cuda, "cuMemAlloc_v2");
    cuMemFree_v2_t cuMemFree_v2 = (cuMemFree_v2_t)dlsym(cuda, "cuMemFree_v2");
    cuMemcpyHtoD_v2_t cuMemcpyHtoD_v2 = (cuMemcpyHtoD_v2_t)dlsym(cuda, "cuMemcpyHtoD_v2");
    cublasCreate_v2_t cublasCreate_v2 = (cublasCreate_v2_t)dlsym(cublas, "cublasCreate_v2");
    cublasDestroy_v2_t cublasDestroy_v2 = (cublasDestroy_v2_t)dlsym(cublas, "cublasDestroy_v2");
    cublasGemmEx_t cublasGemmEx = (cublasGemmEx_t)dlsym(cublas, "cublasGemmEx");

    if (!cuInit || !cuDevicePrimaryCtxRetain || !cuCtxSetCurrent || !cuMemAlloc_v2 ||
        !cuMemFree_v2 || !cuMemcpyHtoD_v2 || !cublasCreate_v2 || !cublasDestroy_v2 ||
        !cublasGemmEx || !cuCtxSynchronize) {
        fprintf(stderr, "FAIL dlsym\n");
        return 1;
    }

    printf("=== test_gemm_ex_vm e7dual (two cublas handles: warmup h1, warmup h0, wide h0) ===\n");

    if (cuInit(0) != 0) {
        fprintf(stderr, "FAIL cuInit\n");
        return 1;
    }
    if (cuDevicePrimaryCtxRetain(&ctx, 0) != 0 || !ctx) {
        fprintf(stderr, "FAIL cuDevicePrimaryCtxRetain\n");
        return 1;
    }
    if (cuCtxSetCurrent(ctx) != 0) {
        fprintf(stderr, "FAIL cuCtxSetCurrent\n");
        return 1;
    }

    int st = cublasCreate_v2(&h0);
    if (st != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cublasCreate_v2 h0 -> %d\n", st);
        return 1;
    }
    st = cublasCreate_v2(&h1);
    if (st != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cublasCreate_v2 h1 -> %d\n", st);
        cublasDestroy_v2(h0);
        return 1;
    }

    st = gemm_e7_f16_once(cublasGemmEx, cuCtxSynchronize, NULL, NULL,
                          cuMemAlloc_v2, cuMemFree_v2, cuMemcpyHtoD_v2, h1,
                          256, 512, 2048, "warmup h1");
    if (st != CUBLAS_STATUS_SUCCESS) {
        printf("E7DUAL_WARMUP_FAIL h1 cublas_status=%d\n", st);
        cublasDestroy_v2(h1);
        cublasDestroy_v2(h0);
        return 2;
    }

    st = gemm_e7_f16_once(cublasGemmEx, cuCtxSynchronize, NULL, NULL,
                          cuMemAlloc_v2, cuMemFree_v2, cuMemcpyHtoD_v2, h0,
                          256, 512, 2048, "warmup h0");
    if (st != CUBLAS_STATUS_SUCCESS) {
        printf("E7DUAL_WARMUP_FAIL h0 cublas_status=%d\n", st);
        cublasDestroy_v2(h1);
        cublasDestroy_v2(h0);
        return 2;
    }

    st = gemm_e7_f16_once(cublasGemmEx, cuCtxSynchronize, NULL, NULL,
                          cuMemAlloc_v2, cuMemFree_v2, cuMemcpyHtoD_v2, h0,
                          32000, 512, 2048, "wide h0");

    cublasDestroy_v2(h1);
    cublasDestroy_v2(h0);

    if (st != CUBLAS_STATUS_SUCCESS) {
        printf("E7DUAL_WIDE_FAIL cublas_status=%d\n", st);
        return 3;
    }
    printf("E7DUAL_OK two-handle warmup + wide passed\n");
    return 0;
}

static int run_e7_burst_then_wide_2h(void)
{
    void *cudart __attribute__((unused)) = NULL, *cuda = NULL, *cublas = NULL;
    cublasHandle_t h0 = NULL, h1 = NULL;
    CUcontext ctx = NULL;

    cudart = dlopen("/opt/vgpu/lib/libcudart.so.12", RTLD_NOW | RTLD_GLOBAL);
    if (!cudart)
        cudart = dlopen("libcudart.so.12", RTLD_NOW | RTLD_GLOBAL);
    cuda = dlopen("libcuda.so.1", RTLD_NOW | RTLD_GLOBAL);
    cublas = dlopen("libcublas.so.12", RTLD_NOW | RTLD_GLOBAL);
    if (!cuda || !cublas) {
        fprintf(stderr, "FAIL dlopen: %s\n", dlerror());
        return 1;
    }

    cuInit_t cuInit = (cuInit_t)dlsym(cuda, "cuInit");
    cuDevicePrimaryCtxRetain_t cuDevicePrimaryCtxRetain =
        (cuDevicePrimaryCtxRetain_t)dlsym(cuda, "cuDevicePrimaryCtxRetain");
    cuCtxSetCurrent_t cuCtxSetCurrent = (cuCtxSetCurrent_t)dlsym(cuda, "cuCtxSetCurrent");
    cuCtxSynchronize_t cuCtxSynchronize = (cuCtxSynchronize_t)dlsym(cuda, "cuCtxSynchronize");
    cuMemAlloc_v2_t cuMemAlloc_v2 = (cuMemAlloc_v2_t)dlsym(cuda, "cuMemAlloc_v2");
    cuMemFree_v2_t cuMemFree_v2 = (cuMemFree_v2_t)dlsym(cuda, "cuMemFree_v2");
    cuMemcpyHtoD_v2_t cuMemcpyHtoD_v2 = (cuMemcpyHtoD_v2_t)dlsym(cuda, "cuMemcpyHtoD_v2");
    cublasCreate_v2_t cublasCreate_v2 = (cublasCreate_v2_t)dlsym(cublas, "cublasCreate_v2");
    cublasDestroy_v2_t cublasDestroy_v2 = (cublasDestroy_v2_t)dlsym(cublas, "cublasDestroy_v2");
    cublasGemmEx_t cublasGemmEx = (cublasGemmEx_t)dlsym(cublas, "cublasGemmEx");

    if (!cuInit || !cuDevicePrimaryCtxRetain || !cuCtxSetCurrent || !cuMemAlloc_v2 ||
        !cuMemFree_v2 || !cuMemcpyHtoD_v2 || !cublasCreate_v2 || !cublasDestroy_v2 ||
        !cublasGemmEx || !cuCtxSynchronize) {
        fprintf(stderr, "FAIL dlsym\n");
        return 1;
    }

    const int nburst = (int)(sizeof(e7_journal_burst) / sizeof(e7_journal_burst[0]));
    const int nrepeat = 4;

    printf("=== test_gemm_ex_vm e7seq2h (burst %d x%d alternating h0/h1, wide h0) ===\n", nburst, nrepeat);

    if (cuInit(0) != 0) {
        fprintf(stderr, "FAIL cuInit\n");
        return 1;
    }
    if (cuDevicePrimaryCtxRetain(&ctx, 0) != 0 || !ctx) {
        fprintf(stderr, "FAIL cuDevicePrimaryCtxRetain\n");
        return 1;
    }
    if (cuCtxSetCurrent(ctx) != 0) {
        fprintf(stderr, "FAIL cuCtxSetCurrent\n");
        return 1;
    }

    int st = cublasCreate_v2(&h0);
    if (st != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cublasCreate_v2 h0 -> %d\n", st);
        return 1;
    }
    st = cublasCreate_v2(&h1);
    if (st != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cublasCreate_v2 h1 -> %d\n", st);
        cublasDestroy_v2(h0);
        return 1;
    }

    int step = 0;
    for (int r = 0; r < nrepeat; r++) {
        for (int i = 0; i < nburst; i++) {
            int m = e7_journal_burst[i][0];
            int n = e7_journal_burst[i][1];
            int k = e7_journal_burst[i][2];
            char note[56];
            cublasHandle_t hh = (step % 2 == 0) ? h0 : h1;
            snprintf(note, sizeof(note), "burst r=%d i=%d h=%c", r, i, (step % 2 == 0) ? '0' : '1');
            st = gemm_e7_f16_once(cublasGemmEx, cuCtxSynchronize, NULL, NULL,
                                  cuMemAlloc_v2, cuMemFree_v2, cuMemcpyHtoD_v2, hh,
                                  m, n, k, note);
            step++;
            if (st != CUBLAS_STATUS_SUCCESS) {
                printf("E7SEQ2H_BURST_FAIL step=%d cublas_status=%d\n", step, st);
                cublasDestroy_v2(h1);
                cublasDestroy_v2(h0);
                return 2;
            }
        }
    }

    st = gemm_e7_f16_once(cublasGemmEx, cuCtxSynchronize, NULL, NULL,
                          cuMemAlloc_v2, cuMemFree_v2, cuMemcpyHtoD_v2, h0,
                          32000, 512, 2048, "wide h0");
    cublasDestroy_v2(h1);
    cublasDestroy_v2(h0);

    if (st != CUBLAS_STATUS_SUCCESS) {
        printf("E7SEQ2H_WIDE_FAIL cublas_status=%d\n", st);
        return 3;
    }
    printf("E7SEQ2H_OK burst+wide two-handle passed\n");
    return 0;
}

int main(int argc, char **argv)
{
    if (argc > 1 && strcmp(argv[1], "e7") == 0)
        return run_e7_f16_tensor_wide();
    if (argc > 1 && strcmp(argv[1], "e7ws") == 0)
        return run_e7_f16_tensor_wide_ws();
    if (argc > 1 && strcmp(argv[1], "e7seq") == 0)
        return run_e7_burst_then_wide();
    if (argc > 1 && strcmp(argv[1], "e7seqws") == 0)
        return run_e7_burst_then_wide_ws();
    if (argc > 1 && strcmp(argv[1], "e7dual") == 0)
        return run_e7_dual_wide();
    if (argc > 1 && strcmp(argv[1], "e7seq2h") == 0)
        return run_e7_burst_then_wide_2h();

    void *cuda = NULL, *cublas = NULL;
    cublasHandle_t handle = NULL;
    CUcontext ctx = NULL;

    cuda = dlopen("libcuda.so.1", RTLD_NOW | RTLD_GLOBAL);
    cublas = dlopen("libcublas.so.12", RTLD_NOW | RTLD_GLOBAL);
    if (!cuda || !cublas) {
        fprintf(stderr, "FAIL dlopen: %s\n", dlerror());
        return 1;
    }

    cuInit_t cuInit = (cuInit_t)dlsym(cuda, "cuInit");
    cuDevicePrimaryCtxRetain_t cuDevicePrimaryCtxRetain =
        (cuDevicePrimaryCtxRetain_t)dlsym(cuda, "cuDevicePrimaryCtxRetain");
    cuCtxSetCurrent_t cuCtxSetCurrent = (cuCtxSetCurrent_t)dlsym(cuda, "cuCtxSetCurrent");
    cuCtxSynchronize_t cuCtxSynchronize = (cuCtxSynchronize_t)dlsym(cuda, "cuCtxSynchronize");
    cuMemAlloc_v2_t cuMemAlloc_v2 = (cuMemAlloc_v2_t)dlsym(cuda, "cuMemAlloc_v2");
    cuMemFree_v2_t cuMemFree_v2 = (cuMemFree_v2_t)dlsym(cuda, "cuMemFree_v2");
    cuMemcpyHtoD_v2_t cuMemcpyHtoD_v2 = (cuMemcpyHtoD_v2_t)dlsym(cuda, "cuMemcpyHtoD_v2");
    cublasCreate_v2_t cublasCreate_v2 = (cublasCreate_v2_t)dlsym(cublas, "cublasCreate_v2");
    cublasDestroy_v2_t cublasDestroy_v2 = (cublasDestroy_v2_t)dlsym(cublas, "cublasDestroy_v2");
    cublasGemmEx_t cublasGemmEx = (cublasGemmEx_t)dlsym(cublas, "cublasGemmEx");

    if (!cuInit || !cuDevicePrimaryCtxRetain || !cuCtxSetCurrent || !cuMemAlloc_v2 ||
        !cuMemFree_v2 || !cuMemcpyHtoD_v2 || !cublasCreate_v2 || !cublasDestroy_v2 ||
        !cublasGemmEx) {
        fprintf(stderr, "FAIL dlsym\n");
        return 1;
    }

    printf("=== test_gemm_ex_vm (preflight, mediated cublasGemmEx) ===\n");

    if (cuInit(0) != 0) {
        fprintf(stderr, "FAIL cuInit\n");
        return 1;
    }
    if (cuDevicePrimaryCtxRetain(&ctx, 0) != 0 || !ctx) {
        fprintf(stderr, "FAIL cuDevicePrimaryCtxRetain\n");
        return 1;
    }
    if (cuCtxSetCurrent(ctx) != 0) {
        fprintf(stderr, "FAIL cuCtxSetCurrent\n");
        return 1;
    }

    int st = cublasCreate_v2(&handle);
    printf("cublasCreate_v2 -> %d handle=%p\n", st, (void *)handle);
    if (st != CUBLAS_STATUS_SUCCESS)
        return 1;

    /* Problem shape from long-run journal (libvgpu-cublas lines). */
    const int shapes[][3] = {
        { 2048, 512, 2048 },
        { 256, 512, 2048 },
        { 256, 512, 2048 },
    };
    const int nshape = (int)(sizeof(shapes) / sizeof(shapes[0]));

    for (int si = 0; si < nshape; si++) {
        int m = shapes[si][0];
        int n = shapes[si][1];
        int k = shapes[si][2];
        int lda = m;
        int ldb = k;
        int ldc = m;
        size_t sz_a = (size_t)m * (size_t)k * sizeof(float);
        size_t sz_b = (size_t)k * (size_t)n * sizeof(float);
        size_t sz_c = (size_t)m * (size_t)n * sizeof(float);

        float *ha = (float *)calloc(1, sz_a);
        float *hb = (float *)calloc(1, sz_b);
        float *hc = (float *)calloc(1, sz_c);
        if (!ha || !hb || !hc) {
            fprintf(stderr, "FAIL calloc shape %d\n", si);
            free(ha);
            free(hb);
            free(hc);
            cublasDestroy_v2(handle);
            return 1;
        }

        CUdeviceptr da = 0, db = 0, dc = 0;
        if (cuMemAlloc_v2(&da, sz_a) != 0 || cuMemAlloc_v2(&db, sz_b) != 0 ||
            cuMemAlloc_v2(&dc, sz_c) != 0) {
            fprintf(stderr, "FAIL cuMemAlloc shape %d\n", si);
            free(ha);
            free(hb);
            free(hc);
            cublasDestroy_v2(handle);
            return 1;
        }
        cuMemcpyHtoD_v2(da, ha, sz_a);
        cuMemcpyHtoD_v2(db, hb, sz_b);

        printf("-- shape %d/%d --\n", si + 1, nshape);
        st = run_gemm(cublasGemmEx, handle, cuCtxSynchronize, m, n, k, da, db, dc,
                      lda, ldb, ldc);

        cuMemFree_v2(dc);
        cuMemFree_v2(db);
        cuMemFree_v2(da);
        free(ha);
        free(hb);
        free(hc);

        if (st != CUBLAS_STATUS_SUCCESS) {
            printf("PREFLIGHT_FAIL status=%d\n", st);
            cublasDestroy_v2(handle);
            return 2;
        }
    }

    cublasDestroy_v2(handle);
    printf("PREFLIGHT_OK all GemmEx + sync passed\n");
    return 0;
}
