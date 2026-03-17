/* vGPU self-test: ring CUDA_CALL_INIT, poll BAR0 STATUS. Use for MMIO correlation. */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <dirent.h>
#include <sys/mman.h>
#include <time.h>

#define VGPU_VENDOR_ID  0x10DE
#define VGPU_DEVICE_ID  0x2331
#define VGPU_CLASS      0x030200
#define VGPU_CLASS_MASK 0xFFFF00
#define BAR0_SIZE       4096
#define REG_STATUS          0x004
#define REG_VM_ID           0x010
#define REG_PROTOCOL_VER    0x020
#define REG_CAPABILITIES    0x024
#define REG_CUDA_OP         0x080
#define REG_CUDA_SEQ        0x084
#define REG_CUDA_NUM_ARGS   0x088
#define REG_CUDA_DATA_LEN   0x08C
#define REG_CUDA_DOORBELL   0x0A8
#define REG_CUDA_RESULT_STATUS 0x0F0
#define STATUS_DONE  0x02
#define STATUS_ERROR 0x03
#define CUDA_CALL_INIT 0x0001
#define REG32(base, off) (*(volatile uint32_t *)((volatile char *)(base) + (off)))

static int find_bar0(char *out, size_t sz) {
    DIR *dir = opendir("/sys/bus/pci/devices");
    if (!dir) return -1;
    struct dirent *e;
    while ((e = readdir(dir)) != NULL) {
        if (e->d_name[0] == '.') continue;
        char p[512];
        unsigned vendor = 0, device = 0, cls = 0;
        FILE *f;
        snprintf(p, sizeof(p), "/sys/bus/pci/devices/%s/vendor", e->d_name);
        f = fopen(p, "r"); if (!f) continue;
        fscanf(f, "%x", &vendor); fclose(f);
        snprintf(p, sizeof(p), "/sys/bus/pci/devices/%s/device", e->d_name);
        f = fopen(p, "r"); if (!f) continue;
        fscanf(f, "%x", &device); fclose(f);
        snprintf(p, sizeof(p), "/sys/bus/pci/devices/%s/class", e->d_name);
        f = fopen(p, "r"); if (!f) continue;
        fscanf(f, "%x", &cls); fclose(f);
        int class_ok = ((cls & VGPU_CLASS_MASK) == VGPU_CLASS);
        int exact  = class_ok && (vendor == VGPU_VENDOR_ID) && (device == VGPU_DEVICE_ID);
        int legacy = class_ok && (vendor == 0x1234 || vendor == 0x1AF4);
        if (!exact && !legacy) continue;
        fprintf(stderr, "[probe] PCI: %s vm_id will read from BAR0\n", e->d_name);
        snprintf(out, sz, "/sys/bus/pci/devices/%s/resource0", e->d_name);
        closedir(dir);
        return 0;
    }
    closedir(dir);
    return -1;
}

int main(void) {
    char res0[512];
    if (find_bar0(res0, sizeof(res0)) != 0) {
        fprintf(stderr, "[probe] FAIL: no VGPU-STUB device\n");
        return 1;
    }
    int fd = open(res0, O_RDWR | O_SYNC);
    if (fd < 0) { perror("[probe] open BAR0"); return 1; }
    volatile void *bar0 = mmap(NULL, BAR0_SIZE, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
    if (bar0 == MAP_FAILED) { perror("[probe] mmap"); close(fd); return 1; }
    uint32_t vm_id = REG32(bar0, REG_VM_ID);
    fprintf(stderr, "[probe] vm_id=%u ringing doorbell, polling BAR0 STATUS (5s max)\n", (unsigned)vm_id);
    REG32(bar0, REG_CUDA_OP) = CUDA_CALL_INIT;
    REG32(bar0, REG_CUDA_SEQ) = 0xDEAD;
    REG32(bar0, REG_CUDA_NUM_ARGS) = 0;
    REG32(bar0, REG_CUDA_DATA_LEN) = 0;
    REG32(bar0, REG_CUDA_DOORBELL) = 1;
    time_t start = time(NULL);
    uint32_t st = 0xFF;
    int timed_out = 0;
    while (1) {
        st = REG32(bar0, REG_STATUS);
        if (st == STATUS_DONE || st == STATUS_ERROR) break;
        if (time(NULL) - start >= 5) { timed_out = 1; break; }
        usleep(50000);
    }
    uint32_t res = REG32(bar0, REG_CUDA_RESULT_STATUS);
    munmap((void *)bar0, BAR0_SIZE);
    close(fd);
    fprintf(stderr, "[probe] result: status=0x%02x timeout=%d cuda_result=%u\n", (unsigned)st, timed_out, (unsigned)res);
    if (!timed_out && st == STATUS_DONE) return 0;
    if (!timed_out && st == STATUS_ERROR) return 2;
    return 1;
}
