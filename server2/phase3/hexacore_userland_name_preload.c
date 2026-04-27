#define _GNU_SOURCE

#include <dlfcn.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#define HEXACORE_NAME "HEXACORE vH100 CAP"
#define CUDA_SUCCESS 0
#define CUDA_ERROR_UNKNOWN 999
#define NVML_SUCCESS 0
#define NVML_ERROR_UNKNOWN 999

typedef int CUresult;
typedef int CUdevice;
typedef int cudaError_t;
typedef int nvmlReturn_t;
typedef struct nvmlDevice_st *nvmlDevice_t;

typedef struct {
    char name[256];
} hexacore_cuda_device_prop_prefix;

static int branding_enabled(void) {
    const char *bypass = getenv("HEXACORE_BRAND_BYPASS");
    return !(bypass && strcmp(bypass, "1") == 0);
}

static void overwrite_name(char *dst, size_t len) {
    if (!dst || len == 0 || !branding_enabled()) {
        return;
    }

    strncpy(dst, HEXACORE_NAME, len - 1);
    dst[len - 1] = '\0';
}

static void overwrite_device_prop_name(void *prop) {
    hexacore_cuda_device_prop_prefix *name_only_prop;

    if (!prop || !branding_enabled()) {
        return;
    }

    name_only_prop = (hexacore_cuda_device_prop_prefix *)prop;
    overwrite_name(name_only_prop->name, sizeof(name_only_prop->name));
}

static void *lookup_symbol(const char *symbol, const char *soname) {
    void *resolved;
    void *handle;

    resolved = dlsym(RTLD_NEXT, symbol);
    if (resolved) {
        return resolved;
    }

    handle = dlopen(soname, RTLD_LAZY | RTLD_LOCAL | RTLD_NOLOAD);
    if (!handle) {
        handle = dlopen(soname, RTLD_LAZY | RTLD_LOCAL);
    }
    if (!handle) {
        return NULL;
    }

    return dlsym(handle, symbol);
}

CUresult cuDeviceGetName(char *name, int len, CUdevice dev) {
    typedef CUresult (*real_fn_t)(char *, int, CUdevice);
    static real_fn_t real_fn;
    CUresult rc;

    if (!real_fn) {
        real_fn = (real_fn_t)lookup_symbol("cuDeviceGetName", "libcuda.so.1");
    }
    if (!real_fn) {
        return CUDA_ERROR_UNKNOWN;
    }

    rc = real_fn(name, len, dev);
    if (rc == CUDA_SUCCESS && len > 0) {
        overwrite_name(name, (size_t)len);
    }

    return rc;
}

cudaError_t cudaGetDeviceProperties(void *prop, int device) {
    typedef cudaError_t (*real_fn_t)(void *, int);
    static real_fn_t real_fn;
    cudaError_t rc;

    if (!real_fn) {
        real_fn = (real_fn_t)lookup_symbol("cudaGetDeviceProperties", "libcudart.so.12");
    }
    if (!real_fn) {
        return CUDA_ERROR_UNKNOWN;
    }

    rc = real_fn(prop, device);
    if (rc == CUDA_SUCCESS) {
        overwrite_device_prop_name(prop);
    }

    return rc;
}

cudaError_t cudaGetDeviceProperties_v2(void *prop, int device) {
    typedef cudaError_t (*real_fn_t)(void *, int);
    static real_fn_t real_fn;
    cudaError_t rc;

    if (!real_fn) {
        real_fn = (real_fn_t)lookup_symbol("cudaGetDeviceProperties_v2", "libcudart.so.12");
    }
    if (!real_fn) {
        return CUDA_ERROR_UNKNOWN;
    }

    rc = real_fn(prop, device);
    if (rc == CUDA_SUCCESS) {
        overwrite_device_prop_name(prop);
    }

    return rc;
}

nvmlReturn_t nvmlDeviceGetName(nvmlDevice_t device, char *name, unsigned int length) {
    typedef nvmlReturn_t (*real_fn_t)(nvmlDevice_t, char *, unsigned int);
    static real_fn_t real_fn;
    nvmlReturn_t rc;

    if (!real_fn) {
        real_fn = (real_fn_t)lookup_symbol("nvmlDeviceGetName", "libnvidia-ml.so.1");
    }
    if (!real_fn) {
        return NVML_ERROR_UNKNOWN;
    }

    rc = real_fn(device, name, length);
    if (rc == NVML_SUCCESS && length > 0) {
        overwrite_name(name, (size_t)length);
    }

    return rc;
}
