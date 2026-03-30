/* Native dom0 sanity check: cublasGemmBatchedEx (no mediator, no guest).
 *
 * Finding (H100 + CUDA 12.x cuBLAS on dom0): cublasSgemm + sync OK;
 * cublasGemmEx (typed single GEMM) + sync OK in probe builds;
 * cublasGemmBatchedEx returns success but cudaDeviceSynchronize -> 700
 * (illegal memory access) with the same device buffers and N=32.
 * So guest/mediator remoting is not the root cause of that sync failure.
 *
 * Build: gcc -O2 -std=c11 -o /tmp/host_test_gemm_batched_native host_test_gemm_batched_native.c \
 *        -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcudart -lcublas -lcuda
 * Run:  /tmp/host_test_gemm_batched_native
 */
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void)
{
    int N = 32;
    size_t sz = (size_t)N * (size_t)N * sizeof(float);
    float *ha = (float *)malloc(sz);
    float *hb = (float *)malloc(sz);
    float *hc = (float *)calloc(N * N, sizeof(float));
    if (!ha || !hb || !hc) return 1;
    for (int j = 0; j < N; j++)
        for (int i = 0; i < N; i++) {
            ha[i + j * N] = (i == j) ? 1.f : 0.f;
            hb[i + j * N] = (i == j) ? 1.f : 0.f;
        }

    float *da = NULL, *db = NULL, *dc = NULL;
    cudaError_t e;
    e = cudaMalloc((void **)&da, sz);
    if (e != cudaSuccess) { fprintf(stderr, "cudaMalloc a %s\n", cudaGetErrorString(e)); return 2; }
    e = cudaMalloc((void **)&db, sz);
    if (e != cudaSuccess) { fprintf(stderr, "cudaMalloc b %s\n", cudaGetErrorString(e)); return 2; }
    e = cudaMalloc((void **)&dc, sz);
    if (e != cudaSuccess) { fprintf(stderr, "cudaMalloc c %s\n", cudaGetErrorString(e)); return 2; }
    cudaMemcpy(da, ha, sz, cudaMemcpyHostToDevice);
    cudaMemcpy(db, hb, sz, cudaMemcpyHostToDevice);

    cublasHandle_t h;
    cublasStatus_t st = cublasCreate(&h);
    if (st != CUBLAS_STATUS_SUCCESS) { fprintf(stderr, "cublasCreate %d\n", st); return 3; }

    float alpha = 1.f, beta = 0.f;
    const void *Aa[1] = { da };
    const void *Ba[1] = { db };
    void *Ca[1] = { dc };

    st = cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N,
                     &alpha, da, N, db, N, &beta, dc, N);
    printf("cublasSgemm status=%d\n", st);
    e = cudaDeviceSynchronize();
    printf("after cublasSgemm sync -> %d (%s)\n", (int)e, cudaGetErrorString(e));
    if (e != cudaSuccess) return 5;

    cudaMemcpy(dc, hc, sz, cudaMemcpyHostToDevice); /* reset C */

    st = cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N,
                      &alpha, da, CUDA_R_32F, N,
                      db, CUDA_R_32F, N,
                      &beta, dc, CUDA_R_32F, N,
                      CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    printf("cublasGemmEx COMPUTE_32F+DEFAULT status=%d\n", st);
    e = cudaDeviceSynchronize();
    printf("after cublasGemmEx sync -> %d (%s)\n", (int)e, cudaGetErrorString(e));
    if (e != cudaSuccess) return 6;

    cudaMemcpy(dc, hc, sz, cudaMemcpyHostToDevice); /* reset C */

    st = cublasGemmBatchedEx(h, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N,
                             &alpha, Aa, CUDA_R_32F, N,
                             Ba, CUDA_R_32F, N,
                             &beta, Ca, CUDA_R_32F, N,
                             1, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    printf("cublasGemmBatchedEx COMPUTE_32F+DEFAULT status=%d\n", st);
    e = cudaDeviceSynchronize();
    printf("after cublasGemmBatchedEx COMPUTE_32F+DEFAULT sync -> %d (%s)\n",
           (int)e, cudaGetErrorString(e));
    cudaError_t e_default = e;
    cublasStatus_t st_default = st;

    /* Second probe: pedantic FP32 (no Tensor Core shortcuts) — if this passes
     * while DEFAULT fails, E4 is likely compute-path / TC selection on this stack. */
    cudaMemcpy(dc, hc, sz, cudaMemcpyHostToDevice);
    st = cublasGemmBatchedEx(h, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N,
                             &alpha, Aa, CUDA_R_32F, N,
                             Ba, CUDA_R_32F, N,
                             &beta, Ca, CUDA_R_32F, N,
                             1, CUBLAS_COMPUTE_32F_PEDANTIC, CUBLAS_GEMM_ALGO0);
    printf("cublasGemmBatchedEx COMPUTE_32F_PEDANTIC+ALGO0 status=%d\n", st);
    cudaError_t e2 = cudaDeviceSynchronize();
    printf("after cublasGemmBatchedEx PEDANTIC sync -> %d (%s)\n",
           (int)e2, cudaGetErrorString(e2));
    printf("summary: DEFAULT ok=%d  PEDANTIC ok=%d\n",
           (int)(st_default == CUBLAS_STATUS_SUCCESS && e_default == cudaSuccess),
           (int)(st == CUBLAS_STATUS_SUCCESS && e2 == cudaSuccess));

    cudaFree(da); cudaFree(db); cudaFree(dc);
    cublasDestroy(h);
    free(ha); free(hb); free(hc);
    /* Exit 0 if pedantic batched path is healthy (primary E4 workaround signal). */
    return (st == CUBLAS_STATUS_SUCCESS && e2 == cudaSuccess) ? 0 : 4;
}
