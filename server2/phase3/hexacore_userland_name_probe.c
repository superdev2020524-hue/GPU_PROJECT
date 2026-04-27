#include <stdio.h>
#include <string.h>

typedef int CUresult;
typedef int CUdevice;
typedef int nvmlReturn_t;
typedef struct nvmlDevice_st *nvmlDevice_t;

extern CUresult cuInit(unsigned int flags);
extern CUresult cuDeviceGetName(char *name, int len, CUdevice dev);

extern nvmlReturn_t nvmlInit_v2(void);
extern nvmlReturn_t nvmlDeviceGetCount_v2(unsigned int *count);
extern nvmlReturn_t nvmlDeviceGetHandleByIndex_v2(unsigned int index, nvmlDevice_t *device);
extern nvmlReturn_t nvmlDeviceGetName(nvmlDevice_t device, char *name, unsigned int length);

int main(void) {
    char cuda_name[256] = {0};
    char nvml_name[256] = {0};
    unsigned int count = 0;
    nvmlDevice_t device = NULL;
    CUresult cu_rc;
    nvmlReturn_t nvml_rc;

    cu_rc = cuInit(0);
    printf("cuInit=%d\n", cu_rc);
    cu_rc = cuDeviceGetName(cuda_name, (int)sizeof(cuda_name), 0);
    printf("cuDeviceGetName_rc=%d\n", cu_rc);
    printf("cuDeviceGetName_name=%s\n", cuda_name);

    nvml_rc = nvmlInit_v2();
    printf("nvmlInit_v2=%d\n", nvml_rc);
    nvml_rc = nvmlDeviceGetCount_v2(&count);
    printf("nvmlDeviceGetCount_v2=%d count=%u\n", nvml_rc, count);
    nvml_rc = nvmlDeviceGetHandleByIndex_v2(0, &device);
    printf("nvmlDeviceGetHandleByIndex_v2=%d\n", nvml_rc);
    nvml_rc = nvmlDeviceGetName(device, nvml_name, (unsigned int)sizeof(nvml_name));
    printf("nvmlDeviceGetName_rc=%d\n", nvml_rc);
    printf("nvmlDeviceGetName_name=%s\n", nvml_name);

    return 0;
}
