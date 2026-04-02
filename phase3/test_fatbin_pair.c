#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

static void *read_file(const char *path, size_t *size_out)
{
    FILE *f = fopen(path, "rb");
    void *buf = NULL;
    long n = 0;

    if (!f) {
        perror(path);
        return NULL;
    }
    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        return NULL;
    }
    n = ftell(f);
    if (n < 0) {
        fclose(f);
        return NULL;
    }
    if (fseek(f, 0, SEEK_SET) != 0) {
        fclose(f);
        return NULL;
    }

    buf = malloc((size_t)n);
    if (!buf) {
        fclose(f);
        return NULL;
    }
    if (fread(buf, 1, (size_t)n, f) != (size_t)n) {
        free(buf);
        fclose(f);
        return NULL;
    }
    fclose(f);
    *size_out = (size_t)n;
    return buf;
}

static void print_rc(const char *label, CUresult rc, size_t size)
{
    const char *name = NULL;
    const char *msg = NULL;
    cuGetErrorName(rc, &name);
    cuGetErrorString(rc, &msg);
    printf("%s rc=%d name=%s msg=%s size=%zu\n",
           label, (int)rc, name ? name : "?", msg ? msg : "?", size);
}

int main(int argc, char **argv)
{
    CUdevice dev;
    CUcontext ctx;
    CUmodule mod1 = NULL;
    CUmodule mod2 = NULL;
    void *buf1 = NULL;
    void *buf2 = NULL;
    size_t size1 = 0;
    size_t size2 = 0;
    CUresult rc;

    if (argc != 3) {
        fprintf(stderr, "usage: %s <fatbin1> <fatbin2>\n", argv[0]);
        return 2;
    }

    buf1 = read_file(argv[1], &size1);
    buf2 = read_file(argv[2], &size2);
    if (!buf1 || !buf2) {
        free(buf1);
        free(buf2);
        return 3;
    }

    rc = cuInit(0);
    if (rc != CUDA_SUCCESS) {
        print_rc("cuInit", rc, 0);
        return 10;
    }
    rc = cuDeviceGet(&dev, 0);
    if (rc != CUDA_SUCCESS) {
        print_rc("cuDeviceGet", rc, 0);
        return 11;
    }
    rc = cuDevicePrimaryCtxRetain(&ctx, dev);
    if (rc != CUDA_SUCCESS) {
        print_rc("cuDevicePrimaryCtxRetain", rc, 0);
        return 12;
    }
    rc = cuCtxSetCurrent(ctx);
    if (rc != CUDA_SUCCESS) {
        print_rc("cuCtxSetCurrent", rc, 0);
        return 13;
    }

    rc = cuModuleLoadFatBinary(&mod1, buf1);
    print_rc("load_first", rc, size1);

    if (rc == CUDA_SUCCESS) {
        CUresult rc2 = cuModuleLoadFatBinary(&mod2, buf2);
        print_rc("load_second_with_first_still_loaded", rc2, size2);
        if (rc2 == CUDA_SUCCESS) {
            cuModuleUnload(mod2);
            mod2 = NULL;
        }
        cuModuleUnload(mod1);
        mod1 = NULL;
    }

    rc = cuModuleLoadFatBinary(&mod2, buf2);
    print_rc("load_second_alone_after_first_unloaded", rc, size2);
    if (rc == CUDA_SUCCESS) {
        cuModuleUnload(mod2);
    }

    cuCtxSetCurrent(NULL);
    cuDevicePrimaryCtxRelease(dev);
    free(buf1);
    free(buf2);
    return 0;
}
