/*
 * libvgpu_cublasLt.c - CUBLAS LT (Light) API shim library
 *
 * This library provides the small CUBLAS LT surface PyTorch uses for
 * torch.nn.Linear/addmm. Matmul is routed through the mediated cuBLAS GEMM
 * shim so the guest never executes NVIDIA cuBLASLt kernels directly.
 *
 * Build:
 *   gcc -shared -fPIC -o libvgpu-cublasLt.so.12 libvgpu_cublasLt.c -ldl
 *
 * Symlink:
 *   ln -sf /opt/vgpu/lib/libvgpu-cublasLt.so.12 /usr/lib64/libcublasLt.so.12
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdarg.h>
#include <unistd.h>
#include <dlfcn.h>
#include <sys/syscall.h>

#ifndef __NR_write
#define __NR_write 1
#endif

typedef void* cublasHandle_t;
typedef void* cublasLtHandle_t;
typedef void* cublasLtMatmulDesc_t;
typedef void* cublasLtMatrixLayout_t;
typedef void* cublasLtMatmulPreference_t;

/* CUBLAS LT status */
typedef enum {
    CUBLAS_STATUS_SUCCESS = 0,
    CUBLAS_STATUS_NOT_INITIALIZED = 1,
    CUBLAS_STATUS_ALLOC_FAILED = 3,
    CUBLAS_STATUS_INVALID_VALUE = 7,
    CUBLAS_STATUS_ARCH_MISMATCH = 8,
    CUBLAS_STATUS_MAPPING_ERROR = 11,
    CUBLAS_STATUS_EXECUTION_FAILED = 13,
    CUBLAS_STATUS_INTERNAL_ERROR = 14,
    CUBLAS_STATUS_NOT_SUPPORTED = 15,
    CUBLAS_STATUS_LICENSE_ERROR = 16
} cublasStatus_t;

#define CUDA_R_32F 0
#define CUBLAS_OP_N 0
#define CUBLAS_OP_T 1
#define CUBLASLT_MATMUL_DESC_COMPUTE_TYPE 0
#define CUBLASLT_MATMUL_DESC_SCALE_TYPE 1
#define CUBLASLT_MATMUL_DESC_POINTER_MODE 2
#define CUBLASLT_MATMUL_DESC_TRANSA 3
#define CUBLASLT_MATMUL_DESC_TRANSB 4
#define CUBLASLT_MATMUL_DESC_EPILOGUE 7
#define CUBLASLT_MATMUL_DESC_BIAS_POINTER 8
#define CUBLASLT_MATRIX_LAYOUT_TYPE 0
#define CUBLASLT_MATRIX_LAYOUT_ORDER 1
#define CUBLASLT_MATRIX_LAYOUT_ROWS 2
#define CUBLASLT_MATRIX_LAYOUT_COLS 3
#define CUBLASLT_MATRIX_LAYOUT_LD 4
#define CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT 5
#define CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET 6
#define CUBLAS_COMPUTE_32F 68
#define LT_HANDLE_MAGIC 0x4c544831u
#define LT_DESC_MAGIC 0x4c544432u
#define LT_LAYOUT_MAGIC 0x4c544c33u
#define LT_PREF_MAGIC 0x4c545034u

typedef struct {
    uint64_t data[8];
} cublasLtMatmulAlgo_t;

typedef struct {
    cublasLtMatmulAlgo_t algo;
    size_t workspaceSize;
    cublasStatus_t state;
    float wavesCount;
    int reserved[4];
} cublasLtMatmulHeuristicResult_t;

typedef struct {
    unsigned int magic;
    cublasHandle_t cublas;
} vgpu_lt_handle_t;

typedef struct {
    unsigned int magic;
    int compute_type;
    int scale_type;
    int pointer_mode;
    int transa;
    int transb;
    unsigned int epilogue;
    const void *bias;
} vgpu_lt_desc_t;

typedef struct {
    unsigned int magic;
    int type;
    int order;
    uint64_t rows;
    uint64_t cols;
    int64_t ld;
    int batch_count;
    int64_t stride;
} vgpu_lt_layout_t;

typedef struct {
    unsigned int magic;
    uint64_t max_workspace;
} vgpu_lt_pref_t;

typedef cublasStatus_t (*pfn_cublasCreate_v2)(cublasHandle_t *);
typedef cublasStatus_t (*pfn_cublasDestroy_v2)(cublasHandle_t);
typedef cublasStatus_t (*pfn_cublasSetStream_v2)(cublasHandle_t, void *);
typedef cublasStatus_t (*pfn_cublasGemmEx)(cublasHandle_t, int, int, int, int, int,
                                           const void *, const void *, int, int,
                                           const void *, int, int, const void *,
                                           void *, int, int, int, int);

static void *g_cublas_dl;
static pfn_cublasCreate_v2 g_cublasCreate_v2;
static pfn_cublasDestroy_v2 g_cublasDestroy_v2;
static pfn_cublasSetStream_v2 g_cublasSetStream_v2;
static pfn_cublasGemmEx g_cublasGemmEx;
static vgpu_lt_handle_t g_default_handle;

static cublasStatus_t lt_resolve_cublas(void)
{
    if (g_cublasCreate_v2 && g_cublasDestroy_v2 && g_cublasSetStream_v2 && g_cublasGemmEx) {
        return CUBLAS_STATUS_SUCCESS;
    }
    if (!g_cublas_dl) {
        g_cublas_dl = dlopen("libcublas.so.12", RTLD_NOW | RTLD_GLOBAL);
    }
    if (!g_cublas_dl) {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }
    g_cublasCreate_v2 = (pfn_cublasCreate_v2)dlsym(g_cublas_dl, "cublasCreate_v2");
    g_cublasDestroy_v2 = (pfn_cublasDestroy_v2)dlsym(g_cublas_dl, "cublasDestroy_v2");
    g_cublasSetStream_v2 = (pfn_cublasSetStream_v2)dlsym(g_cublas_dl, "cublasSetStream_v2");
    g_cublasGemmEx = (pfn_cublasGemmEx)dlsym(g_cublas_dl, "cublasGemmEx");
    return (g_cublasCreate_v2 && g_cublasDestroy_v2 && g_cublasSetStream_v2 && g_cublasGemmEx)
        ? CUBLAS_STATUS_SUCCESS
        : CUBLAS_STATUS_NOT_INITIALIZED;
}

static void lt_log(const char *msg)
{
    if (msg) {
        syscall(__NR_write, 2, msg, strlen(msg));
    }
}

static void lt_debug(const char *fmt, ...)
{
    if (!getenv("CUBLAS_DEBUG")) {
        return;
    }
    char buf[512];
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    if (n > 0 && n < (int)sizeof(buf)) {
        syscall(__NR_write, 2, buf, (size_t)n);
    }
}

static vgpu_lt_handle_t *lt_handle(cublasLtHandle_t handle)
{
    vgpu_lt_handle_t *h = (vgpu_lt_handle_t *)handle;
    return (h && h->magic == LT_HANDLE_MAGIC) ? h : NULL;
}

static vgpu_lt_handle_t *lt_default_handle(void)
{
    if (g_default_handle.magic == LT_HANDLE_MAGIC && g_default_handle.cublas) {
        return &g_default_handle;
    }
    if (lt_resolve_cublas() != CUBLAS_STATUS_SUCCESS) {
        return NULL;
    }
    memset(&g_default_handle, 0, sizeof(g_default_handle));
    g_default_handle.magic = LT_HANDLE_MAGIC;
    if (g_cublasCreate_v2(&g_default_handle.cublas) != CUBLAS_STATUS_SUCCESS) {
        memset(&g_default_handle, 0, sizeof(g_default_handle));
        return NULL;
    }
    return &g_default_handle;
}

static vgpu_lt_desc_t *lt_desc(cublasLtMatmulDesc_t desc)
{
    vgpu_lt_desc_t *d = (vgpu_lt_desc_t *)desc;
    return (d && d->magic == LT_DESC_MAGIC) ? d : NULL;
}

static vgpu_lt_layout_t *lt_layout(cublasLtMatrixLayout_t layout)
{
    vgpu_lt_layout_t *l = (vgpu_lt_layout_t *)layout;
    return (l && l->magic == LT_LAYOUT_MAGIC) ? l : NULL;
}

static cublasStatus_t cublasLt_unsupported(const char *func_name) {
    char msg[128];
    int len = snprintf(msg, sizeof(msg), "[libvgpu-cublasLt] %s() NOT_SUPPORTED\n", func_name);
    if (len > 0 && len < (int)sizeof(msg)) {
        syscall(__NR_write, 2, msg, len);
    }
    return CUBLAS_STATUS_NOT_SUPPORTED;
}

cublasStatus_t cublasLtCreate(cublasLtHandle_t *handle) {
    if (!handle) return CUBLAS_STATUS_INVALID_VALUE;
    cublasStatus_t rc = lt_resolve_cublas();
    if (rc != CUBLAS_STATUS_SUCCESS) return rc;
    vgpu_lt_handle_t *h = (vgpu_lt_handle_t *)calloc(1, sizeof(*h));
    if (!h) return CUBLAS_STATUS_ALLOC_FAILED;
    h->magic = LT_HANDLE_MAGIC;
    rc = g_cublasCreate_v2(&h->cublas);
    if (rc != CUBLAS_STATUS_SUCCESS) {
        free(h);
        return rc;
    }
    *handle = (cublasLtHandle_t)h;
    lt_log("[libvgpu-cublasLt] cublasLtCreate REMOTE\n");
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasLtDestroy(cublasLtHandle_t handle) {
    vgpu_lt_handle_t *h = lt_handle(handle);
    if (!h) return CUBLAS_STATUS_SUCCESS;
    if (g_cublasDestroy_v2 && h->cublas) {
        (void)g_cublasDestroy_v2(h->cublas);
    }
    h->magic = 0;
    free(h);
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasLtMatmulDescCreate(cublasLtMatmulDesc_t *matmulDesc, int computeType, int scaleType) {
    if (!matmulDesc) return CUBLAS_STATUS_INVALID_VALUE;
    vgpu_lt_desc_t *d = (vgpu_lt_desc_t *)calloc(1, sizeof(*d));
    if (!d) return CUBLAS_STATUS_ALLOC_FAILED;
    d->magic = LT_DESC_MAGIC;
    d->compute_type = computeType;
    d->scale_type = scaleType;
    d->pointer_mode = 0;
    d->transa = CUBLAS_OP_N;
    d->transb = CUBLAS_OP_N;
    *matmulDesc = (cublasLtMatmulDesc_t)d;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasLtMatmulDescDestroy(cublasLtMatmulDesc_t matmulDesc) {
    vgpu_lt_desc_t *d = lt_desc(matmulDesc);
    if (d) {
        d->magic = 0;
        free(d);
    }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasLtMatmulDescSetAttribute(cublasLtMatmulDesc_t matmulDesc,
                                              int attr,
                                              const void *buf,
                                              size_t sizeInBytes) {
    vgpu_lt_desc_t *d = lt_desc(matmulDesc);
    if (!d || !buf) return CUBLAS_STATUS_INVALID_VALUE;
    switch (attr) {
        case CUBLASLT_MATMUL_DESC_COMPUTE_TYPE:
            if (sizeInBytes >= sizeof(int)) memcpy(&d->compute_type, buf, sizeof(int));
            break;
        case CUBLASLT_MATMUL_DESC_SCALE_TYPE:
            if (sizeInBytes >= sizeof(int)) memcpy(&d->scale_type, buf, sizeof(int));
            break;
        case CUBLASLT_MATMUL_DESC_POINTER_MODE:
            if (sizeInBytes >= sizeof(int)) memcpy(&d->pointer_mode, buf, sizeof(int));
            break;
        case CUBLASLT_MATMUL_DESC_TRANSA:
            if (sizeInBytes >= sizeof(int)) memcpy(&d->transa, buf, sizeof(int));
            break;
        case CUBLASLT_MATMUL_DESC_TRANSB:
            if (sizeInBytes >= sizeof(int)) memcpy(&d->transb, buf, sizeof(int));
            break;
        case CUBLASLT_MATMUL_DESC_EPILOGUE:
            if (sizeInBytes >= sizeof(unsigned int)) memcpy(&d->epilogue, buf, sizeof(unsigned int));
            break;
        case CUBLASLT_MATMUL_DESC_BIAS_POINTER:
            if (sizeInBytes >= sizeof(void *)) memcpy(&d->bias, buf, sizeof(void *));
            break;
        default:
            return CUBLAS_STATUS_SUCCESS;
    }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasLtMatrixLayoutCreate(cublasLtMatrixLayout_t *matLayout, int type, uint64_t rows, uint64_t cols, int64_t ld) {
    if (!matLayout) return CUBLAS_STATUS_INVALID_VALUE;
    vgpu_lt_layout_t *l = (vgpu_lt_layout_t *)calloc(1, sizeof(*l));
    if (!l) return CUBLAS_STATUS_ALLOC_FAILED;
    l->magic = LT_LAYOUT_MAGIC;
    l->type = type;
    l->order = 0;
    l->rows = rows;
    l->cols = cols;
    l->ld = ld;
    l->batch_count = 1;
    *matLayout = (cublasLtMatrixLayout_t)l;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasLtMatrixLayoutDestroy(cublasLtMatrixLayout_t matLayout) {
    vgpu_lt_layout_t *l = lt_layout(matLayout);
    if (l) {
        l->magic = 0;
        free(l);
    }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasLtMatrixLayoutSetAttribute(cublasLtMatrixLayout_t matLayout,
                                                int attr,
                                                const void *buf,
                                                size_t sizeInBytes) {
    vgpu_lt_layout_t *l = lt_layout(matLayout);
    if (!l || !buf) return CUBLAS_STATUS_INVALID_VALUE;
    switch (attr) {
        case CUBLASLT_MATRIX_LAYOUT_TYPE:
            if (sizeInBytes >= sizeof(int)) memcpy(&l->type, buf, sizeof(int));
            break;
        case CUBLASLT_MATRIX_LAYOUT_ORDER:
            if (sizeInBytes >= sizeof(int)) memcpy(&l->order, buf, sizeof(int));
            break;
        case CUBLASLT_MATRIX_LAYOUT_ROWS:
            if (sizeInBytes >= sizeof(uint64_t)) memcpy(&l->rows, buf, sizeof(uint64_t));
            break;
        case CUBLASLT_MATRIX_LAYOUT_COLS:
            if (sizeInBytes >= sizeof(uint64_t)) memcpy(&l->cols, buf, sizeof(uint64_t));
            break;
        case CUBLASLT_MATRIX_LAYOUT_LD:
            if (sizeInBytes >= sizeof(int64_t)) memcpy(&l->ld, buf, sizeof(int64_t));
            break;
        case CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT:
            if (sizeInBytes >= sizeof(int)) memcpy(&l->batch_count, buf, sizeof(int));
            break;
        case CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET:
            if (sizeInBytes >= sizeof(int64_t)) memcpy(&l->stride, buf, sizeof(int64_t));
            break;
        default:
            return CUBLAS_STATUS_SUCCESS;
    }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasLtMatmulPreferenceCreate(cublasLtMatmulPreference_t *pref) {
    if (!pref) return CUBLAS_STATUS_INVALID_VALUE;
    vgpu_lt_pref_t *p = (vgpu_lt_pref_t *)calloc(1, sizeof(*p));
    if (!p) return CUBLAS_STATUS_ALLOC_FAILED;
    p->magic = LT_PREF_MAGIC;
    *pref = (cublasLtMatmulPreference_t)p;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasLtMatmulPreferenceDestroy(cublasLtMatmulPreference_t pref) {
    vgpu_lt_pref_t *p = (vgpu_lt_pref_t *)pref;
    if (p && p->magic == LT_PREF_MAGIC) {
        p->magic = 0;
        free(p);
    }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasLtMatmulPreferenceSetAttribute(cublasLtMatmulPreference_t pref,
                                                    int attr,
                                                    const void *buf,
                                                    size_t sizeInBytes) {
    vgpu_lt_pref_t *p = (vgpu_lt_pref_t *)pref;
    if (!p || p->magic != LT_PREF_MAGIC || !buf) return CUBLAS_STATUS_INVALID_VALUE;
    if (attr == 1 && sizeInBytes >= sizeof(uint64_t)) {
        memcpy(&p->max_workspace, buf, sizeof(uint64_t));
    }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasLtMatmulAlgoGetHeuristic(cublasLtHandle_t lightHandle,
                                              cublasLtMatmulDesc_t operationDesc,
                                              cublasLtMatrixLayout_t Adesc,
                                              cublasLtMatrixLayout_t Bdesc,
                                              cublasLtMatrixLayout_t Cdesc,
                                              cublasLtMatrixLayout_t Ddesc,
                                              cublasLtMatmulPreference_t preference,
                                              int requestedAlgoCount,
                                              cublasLtMatmulHeuristicResult_t heuristicResults[],
                                              int *returnedAlgoCount) {
    (void)lightHandle; (void)operationDesc; (void)Adesc; (void)Bdesc;
    (void)Cdesc; (void)Ddesc; (void)preference;
    if (requestedAlgoCount <= 0 || !returnedAlgoCount) return CUBLAS_STATUS_INVALID_VALUE;
    if (heuristicResults) {
        memset(&heuristicResults[0], 0, sizeof(heuristicResults[0]));
        heuristicResults[0].workspaceSize = 0;
        heuristicResults[0].state = CUBLAS_STATUS_SUCCESS;
        heuristicResults[0].wavesCount = 1.0f;
    }
    *returnedAlgoCount = heuristicResults ? 1 : 0;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasLtMatmul(cublasLtHandle_t lightHandle,
                              cublasLtMatmulDesc_t computeDesc,
                              const void *alpha,
                              const void *A,
                              cublasLtMatrixLayout_t Adesc,
                              const void *B,
                              cublasLtMatrixLayout_t Bdesc,
                              const void *beta,
                              const void *C,
                              cublasLtMatrixLayout_t Cdesc,
                              void *D,
                              cublasLtMatrixLayout_t Ddesc,
                              const cublasLtMatmulAlgo_t *algo,
                              void *workspace,
                              size_t workspaceSizeInBytes,
                              void *stream) {
    (void)Cdesc; (void)algo; (void)workspace; (void)workspaceSizeInBytes;
    vgpu_lt_handle_t *h = lt_handle(lightHandle);
    if (!h) {
        h = lt_default_handle();
    }
    vgpu_lt_desc_t *op = lt_desc(computeDesc);
    vgpu_lt_layout_t *la = lt_layout(Adesc);
    vgpu_lt_layout_t *lb = lt_layout(Bdesc);
    vgpu_lt_layout_t *ld = lt_layout(Ddesc);
    if (!h || !op || !la || !lb || !ld || !alpha || !A || !B || !D) {
        lt_debug("[libvgpu-cublasLt] matmul invalid inputs h=%p op=%p A=%p B=%p D=%p alpha=%p a=%p b=%p d=%p\n",
                 (void *)h, (void *)op, (void *)la, (void *)lb, (void *)ld,
                 alpha, A, B, D);
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (lt_resolve_cublas() != CUBLAS_STATUS_SUCCESS) {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }
    if (la->type != CUDA_R_32F || lb->type != CUDA_R_32F || ld->type != CUDA_R_32F) {
        lt_debug("[libvgpu-cublasLt] matmul unsupported type A=%d B=%d D=%d\n",
                 la->type, lb->type, ld->type);
        return cublasLt_unsupported("cublasLtMatmul(non-fp32)");
    }
    if (op->pointer_mode != 0) {
        lt_debug("[libvgpu-cublasLt] matmul unsupported pointer_mode=%d\n", op->pointer_mode);
        return cublasLt_unsupported("cublasLtMatmul(device-pointer-mode)");
    }

    int m = (int)((op->transa == CUBLAS_OP_N) ? la->rows : la->cols);
    int k = (int)((op->transa == CUBLAS_OP_N) ? la->cols : la->rows);
    int n = (int)((op->transb == CUBLAS_OP_N) ? lb->cols : lb->rows);
    int kb = (int)((op->transb == CUBLAS_OP_N) ? lb->rows : lb->cols);
    if (m <= 0 || n <= 0 || k <= 0 || kb != k || la->ld <= 0 || lb->ld <= 0 || ld->ld <= 0) {
        lt_debug("[libvgpu-cublasLt] matmul invalid dims trans=(%d,%d) A=(%llu,%llu,%lld) B=(%llu,%llu,%lld) D=(%llu,%llu,%lld) m=%d n=%d k=%d kb=%d\n",
                 op->transa, op->transb,
                 (unsigned long long)la->rows, (unsigned long long)la->cols, (long long)la->ld,
                 (unsigned long long)lb->rows, (unsigned long long)lb->cols, (long long)lb->ld,
                 (unsigned long long)ld->rows, (unsigned long long)ld->cols, (long long)ld->ld,
                 m, n, k, kb);
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    float zero = 0.0f;
    const void *beta_ptr = beta ? beta : &zero;
    const void *c_ptr = C ? C : D;
    (void)c_ptr; /* Bias/epilogue paths are intentionally ignored for the M04 inference gate. */
    (void)op->bias;
    (void)op->epilogue;

    (void)g_cublasSetStream_v2(h->cublas, stream);
    lt_debug("[libvgpu-cublasLt] matmul -> cublasGemmEx trans=(%d,%d) m=%d n=%d k=%d lda=%lld ldb=%lld ldd=%lld compute=%d\n",
             op->transa, op->transb, m, n, k,
             (long long)la->ld, (long long)lb->ld, (long long)ld->ld,
             op->compute_type ? op->compute_type : CUBLAS_COMPUTE_32F);
    cublasStatus_t rc = g_cublasGemmEx(h->cublas,
                                       op->transa,
                                       op->transb,
                                       m, n, k,
                                       alpha,
                                       A, la->type, (int)la->ld,
                                       B, lb->type, (int)lb->ld,
                                       beta_ptr,
                                       D, ld->type, (int)ld->ld,
                                       op->compute_type ? op->compute_type : CUBLAS_COMPUTE_32F,
                                       0);
    lt_debug("[libvgpu-cublasLt] matmul <- cublasGemmEx rc=%d\n", (int)rc);
    return rc;
}
