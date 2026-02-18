/*
 * Phase 3 Mediator — extends the Phase 2 mediator_enhanced.c with:
 *   - Weighted Fair Queuing scheduler (replaces priority linked list)
 *   - Per-VM token-bucket rate limiter with back-pressure
 *   - Watchdog with per-job timeout and auto-quarantine
 *   - Metrics collector with p50/p95/p99 and Prometheus export
 *   - NVML GPU health polling (dlopen, graceful fallback)
 *   - Admin socket for vgpu-admin CLI commands
 *
 * Communication channel: MMIO PCI BAR0 + Unix domain socket
 *     (same as Phase 2 — one socket per QEMU chroot)
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <pthread.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/stat.h>
#include <sys/uio.h>
#include <signal.h>
#include <errno.h>
#include <fcntl.h>
#include <sys/select.h>
#include <dirent.h>

/* Phase 2 shared protocol & CUDA interface */
#include "vgpu_protocol.h"
#include "cuda_vector_add.h"

/* Phase 3 modules */
#include "scheduler_wfq.h"
#include "rate_limiter.h"
#include "metrics.h"
#include "watchdog.h"
#include "nvml_monitor.h"

/* Phase 3 DB config library (for VM weight / rate-limit lookups) */
#include "vgpu_config.h"

/* ====================================================================
 * Constants
 * ==================================================================== */

#define MAX_CONNECTIONS      32
#define SOCKET_BACKLOG       10
#define MAX_SERVER_SOCKETS   16
#define ADMIN_BUF_SIZE       (64 * 1024)   /* 64 KiB for admin responses */

/* ====================================================================
 * Global state
 * ==================================================================== */

/* Server sockets — one per QEMU chroot */
static int    g_server_fds[MAX_SERVER_SOCKETS];
static char   g_socket_paths[MAX_SERVER_SOCKETS][512];
static int    g_num_servers = 0;

/* Admin socket */
static int    g_admin_fd = -1;

/* Shutdown flag */
static volatile int g_shutdown = 0;

/* Phase 3 subsystems */
static wfq_scheduler_t g_scheduler;
static rate_limiter_t  g_rate_limiter;
static metrics_t       g_metrics;
static watchdog_t      g_watchdog;

/* Legacy stats (kept for backward compat with Phase 2 output) */
static uint64_t g_total_processed = 0;
static uint64_t g_pool_a_processed = 0;
static uint64_t g_pool_b_processed = 0;

/* CUDA busy flag — single-GPU, one job at a time */
static int  g_cuda_busy = 0;
static pthread_mutex_t g_cuda_lock = PTHREAD_MUTEX_INITIALIZER;

/* Currently executing entry (for watchdog tracking) */
static wfq_entry_t g_current_job;
static int g_has_current_job = 0;

/* DB connection for looking up VM configs */
static sqlite3 *g_db = NULL;

/* ====================================================================
 * Forward declarations
 * ==================================================================== */
static int  setup_socket_server(const char *socket_path);
static int  setup_admin_socket(void);
static void handle_client_connection(int client_fd);
static void handle_admin_connection(int client_fd);
static void dispatch_next_job(void);
static void execute_job(wfq_entry_t *entry);

/* ====================================================================
 * Signal handler
 * ==================================================================== */
static void signal_handler(int sig)
{
    printf("\n[SHUTDOWN] Received signal %d, shutting down gracefully...\n", sig);
    g_shutdown = 1;
}

/* ====================================================================
 * Auto-discover all QEMU chroot directories by scanning /proc
 * for vgpu-stub processes.  (Copied from Phase 2.)
 * ==================================================================== */
static int discover_all_qemu_chroots(char *chroots[], int max_chroots)
{
    DIR *proc_dir;
    struct dirent *entry;
    char cmdline_path[256];
    char cmdline[4096];
    int count = 0;

    proc_dir = opendir("/proc");
    if (!proc_dir) return 0;

    while ((entry = readdir(proc_dir)) != NULL && count < max_chroots) {
        if (entry->d_name[0] < '0' || entry->d_name[0] > '9')
            continue;

        snprintf(cmdline_path, sizeof(cmdline_path), "/proc/%s/cmdline",
                 entry->d_name);
        FILE *f = fopen(cmdline_path, "r");
        if (!f) continue;

        size_t len = fread(cmdline, 1, sizeof(cmdline) - 1, f);
        fclose(f);
        if (len == 0) continue;
        cmdline[len] = '\0';

        for (size_t i = 0; i < len; i++) {
            if (cmdline[i] == '\0') cmdline[i] = ' ';
        }

        if (strstr(cmdline, "vgpu-stub") == NULL)
            continue;

        char *chroot_arg = strstr(cmdline, "-chroot ");
        if (chroot_arg) {
            chroot_arg += strlen("-chroot ");
            while (*chroot_arg == ' ') chroot_arg++;
            char *end = strchr(chroot_arg, ' ');
            size_t path_len = end ? (size_t)(end - chroot_arg)
                                  : strlen(chroot_arg);

            /* Check for duplicates */
            int duplicate = 0;
            for (int i = 0; i < count; i++) {
                if (strlen(chroots[i]) == path_len &&
                    memcmp(chroots[i], chroot_arg, path_len) == 0) {
                    duplicate = 1;
                    break;
                }
            }
            if (duplicate) continue;

            chroots[count] = malloc(path_len + 1);
            if (chroots[count]) {
                memcpy(chroots[count], chroot_arg, path_len);
                chroots[count][path_len] = '\0';
                count++;
            }
        }
    }

    closedir(proc_dir);
    return count;
}

/* ====================================================================
 * Setup a filesystem Unix domain socket server
 * ==================================================================== */
static int setup_socket_server(const char *socket_path)
{
    struct sockaddr_un addr;
    int fd;

    unlink(socket_path);

    fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) { perror("socket"); return -1; }

    int reuse = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));

    int flags = fcntl(fd, F_GETFL, 0);
    if (flags < 0 || fcntl(fd, F_SETFL, flags | O_NONBLOCK) < 0) {
        perror("fcntl"); close(fd); return -1;
    }

    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, socket_path, sizeof(addr.sun_path) - 1);

    if (bind(fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        perror("bind"); close(fd); return -1;
    }

    chmod(socket_path, 0777);

    if (listen(fd, SOCKET_BACKLOG) < 0) {
        perror("listen"); close(fd); return -1;
    }

    printf("[SOCKET] Listening on %s\n", socket_path);
    return fd;
}

/* ====================================================================
 * Setup the admin socket for vgpu-admin CLI
 * ==================================================================== */
static int setup_admin_socket(void)
{
    /* Ensure the directory exists */
    mkdir("/var/vgpu", 0755);

    int fd = setup_socket_server(VGPU_ADMIN_SOCKET_PATH);
    if (fd >= 0) {
        printf("[ADMIN] Admin socket listening on %s\n", VGPU_ADMIN_SOCKET_PATH);
    }
    return fd;
}

/* ====================================================================
 * Parse VGPURequest payload (same as Phase 2)
 * ==================================================================== */
static int parse_vgpu_request(const uint8_t *payload, uint16_t payload_len,
                              int *num1, int *num2)
{
    if (payload_len < VGPU_REQUEST_HEADER_SIZE)
        return -1;

    VGPURequest *req = (VGPURequest *)payload;

    if (req->version != VGPU_PROTOCOL_VERSION) {
        fprintf(stderr, "[ERROR] Invalid protocol version: 0x%08x\n",
                req->version);
        return -1;
    }
    if (req->opcode != VGPU_OP_CUDA_KERNEL) {
        fprintf(stderr, "[ERROR] Unsupported opcode: 0x%04x\n", req->opcode);
        return -1;
    }
    if (req->param_count != 2) {
        fprintf(stderr, "[ERROR] Expected 2 parameters, got %u\n",
                req->param_count);
        return -1;
    }
    if (payload_len < VGPU_REQUEST_HEADER_SIZE + 2 * sizeof(uint32_t))
        return -1;

    uint32_t *params = (uint32_t *)(payload + VGPU_REQUEST_HEADER_SIZE);
    *num1 = (int)params[0];
    *num2 = (int)params[1];
    return 0;
}

/* ====================================================================
 * Send response to client via socket (same wire format as Phase 2)
 * ==================================================================== */
static int send_response(int client_fd, uint32_t vm_id, uint32_t request_id,
                         char pool_id, uint8_t priority, int result,
                         uint32_t exec_time_us)
{
    VGPUSocketHeader hdr;
    VGPUResponse resp;
    uint32_t result_value = (uint32_t)result;
    struct iovec iov[3];
    struct msghdr msg;

    memset(&hdr, 0, sizeof(hdr));
    hdr.magic       = VGPU_SOCKET_MAGIC;
    hdr.msg_type    = VGPU_MSG_RESPONSE;
    hdr.vm_id       = vm_id;
    hdr.request_id  = request_id;
    hdr.pool_id     = pool_id;
    hdr.priority    = priority;
    hdr.payload_len = VGPU_RESPONSE_HEADER_SIZE + sizeof(uint32_t);

    memset(&resp, 0, sizeof(resp));
    resp.version      = VGPU_PROTOCOL_VERSION;
    resp.status       = 0;
    resp.result_count = 1;
    resp.data_offset  = VGPU_RESPONSE_HEADER_SIZE + sizeof(uint32_t);
    resp.data_length  = 0;
    resp.exec_time_us = exec_time_us;

    iov[0].iov_base = &hdr;
    iov[0].iov_len  = VGPU_SOCKET_HDR_SIZE;
    iov[1].iov_base = &resp;
    iov[1].iov_len  = VGPU_RESPONSE_HEADER_SIZE;
    iov[2].iov_base = &result_value;
    iov[2].iov_len  = sizeof(uint32_t);

    memset(&msg, 0, sizeof(msg));
    msg.msg_iov    = iov;
    msg.msg_iovlen = 3;

    ssize_t sent = sendmsg(client_fd, &msg, MSG_NOSIGNAL);
    if (sent < 0) {
        fprintf(stderr, "[ERROR] sendmsg failed: %s\n", strerror(errno));
        return -1;
    }
    return 0;
}

/* ====================================================================
 * Send a short rejection response (BUSY or QUARANTINED)
 * ==================================================================== */
static void send_rejection(int client_fd, uint32_t vm_id, uint32_t request_id,
                           char pool_id, uint8_t priority, uint32_t msg_type)
{
    VGPUSocketHeader hdr;
    memset(&hdr, 0, sizeof(hdr));
    hdr.magic       = VGPU_SOCKET_MAGIC;
    hdr.msg_type    = msg_type;
    hdr.vm_id       = vm_id;
    hdr.request_id  = request_id;
    hdr.pool_id     = pool_id;
    hdr.priority    = priority;
    hdr.payload_len = 0;

    write(client_fd, &hdr, VGPU_SOCKET_HDR_SIZE);
    close(client_fd);
}

/* ====================================================================
 * CUDA completion callback (called from CUDA worker thread)
 * ==================================================================== */
static void cuda_result_callback(int result, void *user_data)
{
    wfq_entry_t *entry = (wfq_entry_t *)user_data;

    /* Compute execution time */
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    double elapsed = (now.tv_sec - entry->enqueue_time.tv_sec)
                   + (now.tv_nsec - entry->enqueue_time.tv_nsec) / 1e9;
    uint32_t exec_time_us = (uint32_t)(elapsed * 1e6);

    printf("[RESULT] Pool %c: vm=%u, req=%u, result=%d, time=%u us\n",
           entry->pool_id, entry->vm_id, entry->request_id,
           result, exec_time_us);

    /* Send response to client */
    if (send_response(entry->client_fd, entry->vm_id, entry->request_id,
                      entry->pool_id, entry->priority, result,
                      exec_time_us) == 0) {
        printf("[RESPONSE] Sent to vm%u (req=%u): %d\n",
               entry->vm_id, entry->request_id, result);
    } else {
        fprintf(stderr, "[ERROR] Failed to send response to vm%u\n",
                entry->vm_id);
    }

    /* Close the client socket */
    close(entry->client_fd);

    /* Record metrics */
    uint64_t latency_us = (uint64_t)exec_time_us;
    metrics_record_job(&g_metrics, entry->vm_id, latency_us, latency_us);

    /* Notify scheduler that this VM's job completed */
    wfq_complete(&g_scheduler, entry->vm_id, exec_time_us);

    /* Notify watchdog */
    wd_job_completed(&g_watchdog, entry->vm_id, entry->request_id);

    /* Update legacy stats */
    g_total_processed++;
    if (entry->pool_id == 'A') g_pool_a_processed++;
    else                        g_pool_b_processed++;

    /* Mark CUDA idle */
    pthread_mutex_lock(&g_cuda_lock);
    g_cuda_busy = 0;
    g_has_current_job = 0;
    pthread_mutex_unlock(&g_cuda_lock);

    /* Free the entry copy */
    free(entry);

    /* Dispatch next job from the WFQ scheduler */
    dispatch_next_job();
}

/* ====================================================================
 * Execute a job (send to CUDA)
 * ==================================================================== */
static void execute_job(wfq_entry_t *entry)
{
    pthread_mutex_lock(&g_cuda_lock);
    if (g_cuda_busy) {
        /* Shouldn't happen — means dispatch logic has a bug */
        fprintf(stderr, "[WARNING] CUDA busy, cannot execute job\n");
        pthread_mutex_unlock(&g_cuda_lock);
        return;
    }
    g_cuda_busy = 1;
    memcpy(&g_current_job, entry, sizeof(g_current_job));
    g_has_current_job = 1;
    pthread_mutex_unlock(&g_cuda_lock);

    printf("[PROCESS] Pool %c: vm=%u, req=%u, prio=%u, w=%d, %d+%d\n",
           entry->pool_id, entry->vm_id, entry->request_id,
           entry->priority, entry->weight, entry->num1, entry->num2);

    /* Track context switches in metrics */
    /* (WFQ scheduler already tracks internally; mirror to metrics) */
    uint64_t cs = wfq_context_switches(&g_scheduler);
    static uint64_t last_cs = 0;
    if (cs > last_cs) {
        for (uint64_t i = last_cs; i < cs; i++)
            metrics_record_context_switch(&g_metrics);
        last_cs = cs;
    }

    /* Notify watchdog */
    wd_job_started(&g_watchdog, entry->vm_id, entry->request_id);

    /* Allocate a copy of the entry for the callback (since `entry` is stack) */
    wfq_entry_t *cb_entry = (wfq_entry_t *)malloc(sizeof(wfq_entry_t));
    if (!cb_entry) {
        fprintf(stderr, "[ERROR] malloc failed for cb_entry\n");
        pthread_mutex_lock(&g_cuda_lock);
        g_cuda_busy = 0;
        g_has_current_job = 0;
        pthread_mutex_unlock(&g_cuda_lock);
        close(entry->client_fd);
        return;
    }
    memcpy(cb_entry, entry, sizeof(*cb_entry));

    /* Submit to CUDA */
    if (cuda_vector_add_async(entry->num1, entry->num2,
                              cuda_result_callback, cb_entry) != 0) {
        fprintf(stderr, "[ERROR] cuda_vector_add_async failed for vm%u\n",
                entry->vm_id);
        wd_job_failed(&g_watchdog, entry->vm_id, entry->request_id);
        metrics_record_error(&g_metrics, entry->vm_id);

        pthread_mutex_lock(&g_cuda_lock);
        g_cuda_busy = 0;
        g_has_current_job = 0;
        pthread_mutex_unlock(&g_cuda_lock);

        close(entry->client_fd);
        free(cb_entry);
    }
}

/* ====================================================================
 * Dispatch the next job from the WFQ scheduler (if CUDA is idle)
 * ==================================================================== */
static void dispatch_next_job(void)
{
    pthread_mutex_lock(&g_cuda_lock);
    if (g_cuda_busy) {
        pthread_mutex_unlock(&g_cuda_lock);
        return;
    }
    pthread_mutex_unlock(&g_cuda_lock);

    wfq_entry_t entry;
    if (wfq_dequeue(&g_scheduler, &entry) == 0) {
        execute_job(&entry);
    }
}

/* ====================================================================
 * Handle a new client connection from a vgpu-stub (QEMU chroot socket)
 *
 * Wire format is identical to Phase 2.  The difference is:
 *   1. We check the rate limiter and watchdog before accepting
 *   2. We enqueue into the WFQ scheduler instead of the linked list
 * ==================================================================== */
static void handle_client_connection(int client_fd)
{
    uint8_t rx_buf[VGPU_SOCKET_HDR_SIZE + VGPU_SOCKET_MAX_PAYLOAD];
    ssize_t n;

    /* Read socket header */
    n = read(client_fd, rx_buf, VGPU_SOCKET_HDR_SIZE);
    if (n < (ssize_t)VGPU_SOCKET_HDR_SIZE) {
        if (n > 0) fprintf(stderr, "[ERROR] Incomplete header (%zd bytes)\n", n);
        close(client_fd);
        return;
    }

    VGPUSocketHeader *hdr = (VGPUSocketHeader *)rx_buf;

    /* Validate magic */
    if (hdr->magic != VGPU_SOCKET_MAGIC) {
        fprintf(stderr, "[ERROR] Invalid magic: 0x%08x\n", hdr->magic);
        close(client_fd);
        return;
    }

    /* Handle PING */
    if (hdr->msg_type == VGPU_MSG_PING) {
        VGPUSocketHeader pong;
        memset(&pong, 0, sizeof(pong));
        pong.magic    = VGPU_SOCKET_MAGIC;
        pong.msg_type = VGPU_MSG_PONG;
        pong.vm_id    = hdr->vm_id;
        write(client_fd, &pong, VGPU_SOCKET_HDR_SIZE);
        close(client_fd);
        return;
    }

    /* Only accept REQUEST messages from vgpu-stub */
    if (hdr->msg_type != VGPU_MSG_REQUEST) {
        close(client_fd);
        return;
    }

    /* Read payload */
    if (hdr->payload_len > 0) {
        n = read(client_fd, rx_buf + VGPU_SOCKET_HDR_SIZE, hdr->payload_len);
        if (n < 0 || (uint16_t)n < hdr->payload_len) {
            fprintf(stderr, "[ERROR] Incomplete payload\n");
            close(client_fd);
            return;
        }
    }

    /* Parse request */
    int num1, num2;
    if (parse_vgpu_request(rx_buf + VGPU_SOCKET_HDR_SIZE,
                           hdr->payload_len, &num1, &num2) != 0) {
        fprintf(stderr, "[ERROR] Failed to parse request from vm%u\n",
                hdr->vm_id);
        close(client_fd);
        return;
    }

    /* Validate pool and priority */
    if (hdr->pool_id != 'A' && hdr->pool_id != 'B') {
        fprintf(stderr, "[ERROR] Invalid pool_id: '%c'\n", hdr->pool_id);
        close(client_fd);
        return;
    }
    if (hdr->priority > VGPU_PRIORITY_HIGH) {
        fprintf(stderr, "[ERROR] Invalid priority: %u\n", hdr->priority);
        close(client_fd);
        return;
    }

    /* === Phase 3 isolation checks === */

    /* 1. Check quarantine */
    if (wd_is_quarantined(&g_watchdog, hdr->vm_id)) {
        printf("[REJECT] vm=%u quarantined\n", hdr->vm_id);
        metrics_record_rejection(&g_metrics, hdr->vm_id);
        send_rejection(client_fd, hdr->vm_id, hdr->request_id,
                       hdr->pool_id, hdr->priority, VGPU_MSG_QUARANTINED);
        return;
    }

    /* 2. Check rate limiter */
    int vm_qd = wfq_vm_queue_depth(&g_scheduler, hdr->vm_id);
    int rl_result = rl_check(&g_rate_limiter, hdr->vm_id, vm_qd);
    if (rl_result != RL_ALLOW) {
        printf("[REJECT] vm=%u rate-limited (code=%d)\n",
               hdr->vm_id, rl_result);
        metrics_record_rejection(&g_metrics, hdr->vm_id);
        send_rejection(client_fd, hdr->vm_id, hdr->request_id,
                       hdr->pool_id, hdr->priority, VGPU_MSG_BUSY);
        return;
    }

    /* 3. Look up VM config from DB (for weight) */
    int weight = 50;  /* default */
    if (g_db) {
        vgpu_vm_config_t vm_cfg;
        if (vgpu_get_vm_config_by_id(g_db, (int)hdr->vm_id, &vm_cfg) == VGPU_OK) {
            weight = (vm_cfg.weight > 0) ? vm_cfg.weight : 50;
        }
    }

    /* 4. Enqueue into WFQ scheduler */
    wfq_entry_t entry;
    memset(&entry, 0, sizeof(entry));
    entry.vm_id      = hdr->vm_id;
    entry.request_id = hdr->request_id;
    entry.pool_id    = hdr->pool_id;
    entry.priority   = hdr->priority;
    entry.weight     = weight;
    entry.num1       = num1;
    entry.num2       = num2;
    entry.client_fd  = client_fd;  /* Keep open until response */

    /* Copy raw payload for pass-through if needed */
    if (hdr->payload_len > 0 && hdr->payload_len <= sizeof(entry.payload)) {
        memcpy(entry.payload, rx_buf + VGPU_SOCKET_HDR_SIZE,
               hdr->payload_len);
        entry.payload_len = hdr->payload_len;
    }

    if (wfq_enqueue(&g_scheduler, &entry) != 0) {
        fprintf(stderr, "[ERROR] WFQ queue full for vm%u\n", hdr->vm_id);
        metrics_record_rejection(&g_metrics, hdr->vm_id);
        send_rejection(client_fd, hdr->vm_id, hdr->request_id,
                       hdr->pool_id, hdr->priority, VGPU_MSG_BUSY);
        return;
    }

    /* Try to dispatch immediately if CUDA is idle */
    dispatch_next_job();
}

/* ====================================================================
 * Handle an admin socket connection (from vgpu-admin CLI)
 * ==================================================================== */
static void handle_admin_connection(int client_fd)
{
    VGPUAdminRequest req;
    ssize_t n = read(client_fd, &req, sizeof(req));
    if (n < (ssize_t)sizeof(req)) {
        close(client_fd);
        return;
    }

    if (req.magic != VGPU_SOCKET_MAGIC) {
        close(client_fd);
        return;
    }

    char *buf = (char *)malloc(ADMIN_BUF_SIZE);
    if (!buf) { close(client_fd); return; }

    int data_len = 0;
    VGPUAdminResponse resp;
    memset(&resp, 0, sizeof(resp));
    resp.magic  = VGPU_SOCKET_MAGIC;
    resp.status = 0;

    switch (req.command) {
    case VGPU_ADMIN_SHOW_METRICS:
        if (req.param1 == 0) {
            /* Full summary */
            data_len = metrics_export_summary(&g_metrics, buf, ADMIN_BUF_SIZE);
        } else {
            /* Prometheus format */
            data_len = metrics_export_prometheus(&g_metrics, buf, ADMIN_BUF_SIZE);
        }
        break;

    case VGPU_ADMIN_SHOW_HEALTH: {
        nvml_health_t health;
        nvml_poll(&health);
        if (health.available) {
            data_len = snprintf(buf, ADMIN_BUF_SIZE,
                "=== GPU Health ===\n"
                "Temperature:   %u °C\n"
                "GPU Util:      %u%%\n"
                "Memory Util:   %u%%\n"
                "Memory Used:   %lu / %lu MB\n"
                "Power Draw:    %u W\n"
                "ECC Errors:    %lu\n"
                "Needs Reset:   %s\n"
                "\n"
                "=== Watchdog ===\n"
                "GPU Resets:    %lu\n"
                "Scheduler Queue: %d\n"
                "Context Switches: %lu\n",
                health.temperature_c,
                health.gpu_utilization,
                health.memory_utilization,
                (unsigned long)health.memory_used_mb,
                (unsigned long)health.memory_total_mb,
                health.power_watts,
                (unsigned long)health.ecc_errors,
                health.needs_reset ? "YES" : "no",
                (unsigned long)wd_total_resets(&g_watchdog),
                wfq_queue_len(&g_scheduler),
                (unsigned long)wfq_context_switches(&g_scheduler));
        } else {
            data_len = snprintf(buf, ADMIN_BUF_SIZE,
                "=== GPU Health ===\n"
                "NVML: not available (GPU monitoring disabled)\n"
                "\n"
                "=== Watchdog ===\n"
                "GPU Resets:    %lu\n"
                "Scheduler Queue: %d\n"
                "Context Switches: %lu\n",
                (unsigned long)wd_total_resets(&g_watchdog),
                wfq_queue_len(&g_scheduler),
                (unsigned long)wfq_context_switches(&g_scheduler));
        }
        break;
    }

    case VGPU_ADMIN_RELOAD_CONFIG:
        /* Re-read DB config and push to rate limiter */
        if (g_db) {
            vgpu_vm_config_t configs[64];
            int count = 0;
            if (vgpu_list_vms(g_db, 0, -1, configs, &count, 64) == VGPU_OK) {
                for (int i = 0; i < count; i++) {
                    rl_configure_vm(&g_rate_limiter,
                                    (uint32_t)configs[i].vm_id,
                                    configs[i].max_jobs_per_sec,
                                    configs[i].max_queue_depth);

                    if (configs[i].quarantined) {
                        /* Sync quarantine state from DB */
                        /* (watchdog tracks its own state, but DB is source of truth
                           for admin-set quarantines) */
                    }
                }
                data_len = snprintf(buf, ADMIN_BUF_SIZE,
                    "Config reloaded: %d VM(s) updated\n", count);
            } else {
                data_len = snprintf(buf, ADMIN_BUF_SIZE,
                    "ERROR: Failed to read VM configs from DB\n");
                resp.status = 1;
            }
        } else {
            data_len = snprintf(buf, ADMIN_BUF_SIZE,
                "ERROR: Database not connected\n");
            resp.status = 1;
        }
        break;

    default:
        data_len = snprintf(buf, ADMIN_BUF_SIZE,
            "ERROR: Unknown admin command 0x%02x\n", req.command);
        resp.status = 1;
        break;
    }

    resp.data_len = (uint32_t)data_len;

    /* Send response header + text */
    write(client_fd, &resp, sizeof(resp));
    if (data_len > 0) {
        write(client_fd, buf, data_len);
    }

    free(buf);
    close(client_fd);
}

/* ====================================================================
 * Load VM configs from DB and push to rate limiter on startup
 * ==================================================================== */
static void load_vm_configs(void)
{
    if (!g_db) return;

    vgpu_vm_config_t configs[64];
    int count = 0;
    if (vgpu_list_vms(g_db, 0, -1, configs, &count, 64) != VGPU_OK) {
        fprintf(stderr, "[CONFIG] WARNING: Could not read VM configs from DB\n");
        return;
    }

    printf("[CONFIG] Loaded %d VM(s) from database\n", count);
    for (int i = 0; i < count; i++) {
        rl_configure_vm(&g_rate_limiter,
                        (uint32_t)configs[i].vm_id,
                        configs[i].max_jobs_per_sec,
                        configs[i].max_queue_depth);

        printf("[CONFIG]   vm_id=%d uuid=%s weight=%d rate=%d/s queue=%d %s\n",
               configs[i].vm_id, configs[i].vm_uuid,
               configs[i].weight, configs[i].max_jobs_per_sec,
               configs[i].max_queue_depth,
               configs[i].quarantined ? "(quarantined)" : "");
    }
}

/* ====================================================================
 * Print statistics (legacy, runs every 60s)
 * ==================================================================== */
static void print_stats(void)
{
    printf("\n[MEDIATOR STATS]\n");
    printf("  Total processed:  %lu\n", (unsigned long)g_total_processed);
    printf("  Pool A processed: %lu\n", (unsigned long)g_pool_a_processed);
    printf("  Pool B processed: %lu\n", (unsigned long)g_pool_b_processed);
    printf("  WFQ queue depth:  %d\n",  wfq_queue_len(&g_scheduler));
    printf("  Context switches: %lu\n",
           (unsigned long)wfq_context_switches(&g_scheduler));
    printf("  CUDA busy:        %s\n",  g_cuda_busy ? "yes" : "no");

    /* Quick NVML summary if available */
    nvml_health_t health;
    nvml_poll(&health);
    if (health.available) {
        printf("  GPU temp:         %u°C\n", health.temperature_c);
        printf("  GPU util:         %u%%\n",  health.gpu_utilization);
    }
    printf("\n");
}

/* ====================================================================
 * Print usage
 * ==================================================================== */
static void print_usage(const char *prog)
{
    printf("Usage: %s [OPTIONS]\n\n", prog);
    printf("Options:\n");
    printf("  --socket-path <path>   Override socket path (host filesystem)\n");
    printf("  --no-db                Run without database\n");
    printf("  --no-nvml              Disable NVML GPU health monitoring\n");
    printf("  --help                 Show this help\n");
    printf("\nIf --socket-path is not given, the mediator auto-discovers\n");
    printf("the QEMU chroot directory and creates the socket inside it.\n");
    printf("Fallback: %s\n", VGPU_SOCKET_PATH);
}

/* ====================================================================
 * Main event loop
 * ==================================================================== */
static void run_mediator(void)
{
    time_t last_stats = time(NULL);
    fd_set read_fds;
    int max_fd;
    struct timeval timeout;

    printf("[MEDIATOR] Starting main loop...\n");
    printf("[MEDIATOR] Listening on %d socket(s) for VM connections\n",
           g_num_servers);
    if (g_admin_fd >= 0)
        printf("[MEDIATOR] Admin socket active on %s\n",
               VGPU_ADMIN_SOCKET_PATH);

    while (!g_shutdown) {
        FD_ZERO(&read_fds);
        max_fd = -1;

        /* Add all QEMU chroot sockets */
        for (int i = 0; i < g_num_servers; i++) {
            FD_SET(g_server_fds[i], &read_fds);
            if (g_server_fds[i] > max_fd) max_fd = g_server_fds[i];
        }

        /* Add admin socket */
        if (g_admin_fd >= 0) {
            FD_SET(g_admin_fd, &read_fds);
            if (g_admin_fd > max_fd) max_fd = g_admin_fd;
        }

        timeout.tv_sec  = 1;
        timeout.tv_usec = 0;

        int ret = select(max_fd + 1, &read_fds, NULL, NULL, &timeout);
        if (ret < 0) {
            if (errno == EINTR) continue;
            perror("select");
            break;
        }

        /* Accept connections on QEMU chroot sockets */
        for (int i = 0; i < g_num_servers; i++) {
            if (FD_ISSET(g_server_fds[i], &read_fds)) {
                struct sockaddr_un client_addr;
                socklen_t client_len = sizeof(client_addr);
                int client_fd = accept(g_server_fds[i],
                                       (struct sockaddr *)&client_addr,
                                       &client_len);
                if (client_fd >= 0) {
                    printf("[SOCKET] New connection on %s (fd=%d)\n",
                           g_socket_paths[i], client_fd);
                    handle_client_connection(client_fd);
                }
            }
        }

        /* Accept connections on admin socket */
        if (g_admin_fd >= 0 && FD_ISSET(g_admin_fd, &read_fds)) {
            struct sockaddr_un client_addr;
            socklen_t client_len = sizeof(client_addr);
            int client_fd = accept(g_admin_fd,
                                   (struct sockaddr *)&client_addr,
                                   &client_len);
            if (client_fd >= 0) {
                handle_admin_connection(client_fd);
            }
        }

        /* Try to dispatch if CUDA is idle (in case a previous callback
           completed while we were in select) */
        dispatch_next_job();

        /* Periodic stats */
        time_t now = time(NULL);
        if (now - last_stats >= 60) {
            print_stats();
            last_stats = now;
        }
    }

    /* ---- Shutdown ---- */
    printf("[MEDIATOR] Shutting down...\n");

    /* Stop watchdog thread */
    wd_stop(&g_watchdog);

    /* Wait for CUDA to finish */
    if (g_cuda_busy) {
        printf("[MEDIATOR] Waiting for CUDA to finish...\n");
        cuda_sync();
    }

    /* Drain and close remaining queued requests */
    wfq_entry_t entry;
    while (wfq_dequeue(&g_scheduler, &entry) == 0) {
        if (entry.client_fd >= 0) close(entry.client_fd);
    }

    /* Close server sockets */
    for (int i = 0; i < g_num_servers; i++) {
        close(g_server_fds[i]);
        if (g_socket_paths[i][0]) unlink(g_socket_paths[i]);
    }

    /* Close admin socket */
    if (g_admin_fd >= 0) {
        close(g_admin_fd);
        unlink(VGPU_ADMIN_SOCKET_PATH);
    }
}

/* ====================================================================
 * main()
 * ==================================================================== */
int main(int argc, char *argv[])
{
    const char *override_path = NULL;
    int no_db   = 0;
    int no_nvml = 0;

    printf("===========================================================\n");
    printf("  MEDIATOR DAEMON v3.0 — Phase 3: WFQ + Isolation + Metrics\n");
    printf("  Communication: MMIO PCI BAR0 + Unix domain socket\n");
    printf("===========================================================\n\n");

    /* Parse arguments */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--socket-path") == 0 && i + 1 < argc) {
            override_path = argv[++i];
        } else if (strcmp(argv[i], "--no-db") == 0) {
            no_db = 1;
        } else if (strcmp(argv[i], "--no-nvml") == 0) {
            no_nvml = 1;
        } else if (strcmp(argv[i], "--help") == 0 ||
                   strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown argument: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

    /* Signal handlers */
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    /* ---- Initialize Phase 3 subsystems ---- */

    /* 1. WFQ scheduler */
    wfq_init(&g_scheduler);
    printf("[INIT] WFQ scheduler ready\n");

    /* 2. Rate limiter */
    rl_init(&g_rate_limiter);
    printf("[INIT] Rate limiter ready\n");

    /* 3. Metrics */
    metrics_init(&g_metrics);
    printf("[INIT] Metrics collector ready\n");

    /* 4. Watchdog (start thread) */
    wd_init(&g_watchdog);
    if (wd_start(&g_watchdog) != 0) {
        fprintf(stderr, "[ERROR] Failed to start watchdog thread\n");
        return 1;
    }
    printf("[INIT] Watchdog started (timeout=%ds, threshold=%d)\n",
           g_watchdog.job_timeout_sec, g_watchdog.fault_threshold);

    /* 5. NVML (optional) */
    if (!no_nvml) {
        if (nvml_init() == 0) {
            printf("[INIT] NVML GPU health monitoring active\n");
        } else {
            printf("[INIT] NVML not available — running without GPU health monitoring\n");
        }
    }

    /* 6. Database (optional) */
    if (!no_db) {
        if (vgpu_db_init(&g_db) == VGPU_OK) {
            vgpu_db_init_schema(g_db);
            printf("[INIT] Database connected (%s)\n", VGPU_DB_PATH);
            load_vm_configs();
        } else {
            fprintf(stderr, "[WARN] Could not open database — "
                    "running without persistent config\n");
            g_db = NULL;
        }
    }

    /* 7. CUDA */
    if (cuda_init() != 0) {
        fprintf(stderr, "[ERROR] Failed to initialize CUDA\n");
        return 1;
    }
    printf("[INIT] CUDA ready\n");

    /* ---- Setup sockets ---- */
    g_num_servers = 0;

    if (override_path) {
        snprintf(g_socket_paths[0], sizeof(g_socket_paths[0]),
                 "%s", override_path);
        g_server_fds[0] = setup_socket_server(g_socket_paths[0]);
        if (g_server_fds[0] < 0) {
            fprintf(stderr, "[ERROR] Failed to setup socket at %s\n",
                    override_path);
            cuda_cleanup();
            return 1;
        }
        g_num_servers = 1;
        printf("[CONFIG] Using user-specified socket: %s\n",
               g_socket_paths[0]);
    } else {
        char *chroots[MAX_SERVER_SOCKETS];
        int num_chroots = discover_all_qemu_chroots(chroots,
                                                     MAX_SERVER_SOCKETS);

        if (num_chroots > 0) {
            printf("[CONFIG] Found %d QEMU VM(s) with vgpu-stub:\n",
                   num_chroots);
            for (int i = 0; i < num_chroots; i++) {
                char tmp_dir[512];
                snprintf(tmp_dir, sizeof(tmp_dir), "%s/tmp", chroots[i]);
                mkdir(tmp_dir, 0755);

                snprintf(g_socket_paths[g_num_servers],
                         sizeof(g_socket_paths[g_num_servers]),
                         "%s%s", chroots[i], VGPU_SOCKET_PATH);

                g_server_fds[g_num_servers] =
                    setup_socket_server(g_socket_paths[g_num_servers]);
                if (g_server_fds[g_num_servers] >= 0) {
                    printf("  [%d] %s -> %s\n", g_num_servers + 1,
                           chroots[i],
                           g_socket_paths[g_num_servers]);
                    g_num_servers++;
                } else {
                    fprintf(stderr, "[WARN] Failed to setup socket in %s\n",
                            chroots[i]);
                }
                free(chroots[i]);
            }
        } else {
            snprintf(g_socket_paths[0], sizeof(g_socket_paths[0]),
                     "%s", VGPU_SOCKET_PATH);
            g_server_fds[0] = setup_socket_server(g_socket_paths[0]);
            if (g_server_fds[0] >= 0) g_num_servers = 1;
            printf("[CONFIG] No QEMU chroot found, using fallback: %s\n",
                   g_socket_paths[0]);
        }
    }

    if (g_num_servers == 0) {
        fprintf(stderr, "[ERROR] No server sockets created\n");
        cuda_cleanup();
        return 1;
    }

    /* Admin socket */
    g_admin_fd = setup_admin_socket();

    /* ---- Run main loop ---- */
    run_mediator();

    /* ---- Cleanup ---- */
    nvml_shutdown();
    wd_destroy(&g_watchdog);
    wfq_destroy(&g_scheduler);
    rl_destroy(&g_rate_limiter);
    metrics_destroy(&g_metrics);
    cuda_cleanup();
    if (g_db) vgpu_db_close(g_db);

    printf("[MEDIATOR] Exited cleanly\n");
    return 0;
}
