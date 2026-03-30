#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

int main() {
    void *handle;
    int (*cuInit)(unsigned int);
    int (*cuDeviceGetCount)(int *);
    char *error;
    int count = 0;
    
    // Load CUDA library (should load our shim via system library path)
    handle = dlopen("libcuda.so.1", RTLD_LAZY);
    if (!handle) {
        fprintf(stderr, "ERROR: Cannot load libcuda.so.1: %s\n", dlerror());
        return 1;
    }
    
    // Get function pointers
    cuInit = (int (*)(unsigned int))dlsym(handle, "cuInit");
    cuDeviceGetCount = (int (*)(int *))dlsym(handle, "cuDeviceGetCount");
    
    if ((error = dlerror()) != NULL) {
        fprintf(stderr, "ERROR: dlsym failed: %s\n", error);
        dlclose(handle);
        return 1;
    }
    
    // Initialize CUDA
    if (cuInit(0) != 0) {
        fprintf(stderr, "ERROR: cuInit failed\n");
        dlclose(handle);
        return 1;
    }
    
    // Get device count
    if (cuDeviceGetCount(&count) != 0) {
        fprintf(stderr, "ERROR: cuDeviceGetCount failed\n");
        dlclose(handle);
        return 1;
    }
    
    printf("SUCCESS: CUDA initialized, device count = %d\n", count);
    
    if (count > 0) {
        printf("✓✓✓ GPU DETECTED: vGPU shim works as system library!\n");
    } else {
        printf("✗ No GPU detected\n");
    }
    
    dlclose(handle);
    return (count > 0) ? 0 : 1;
}
