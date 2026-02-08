/*
 * vGPU Administration CLI Tool
 * Configuration & Management Interface (Step 2-4)
 *
 * Usage: vgpu-admin <command> [options]
 */

#define _POSIX_C_SOURCE 200809L  /* For popen/pclose */

#include "vgpu_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <ctype.h>
#include <sys/wait.h>

#define MAX_VMS 1000
#define MAX_LINE 1024

/* Forward declarations */
static int confirm_apply_settings(const char *vm_uuid, char old_pool, char new_pool,
                                  int old_priority, int new_priority,
                                  int old_vm_id, int new_vm_id);
static int vm_is_running(const char *vm_uuid);
static int vm_stop(const char *vm_uuid);
static int vm_stop_and_verify(const char *vm_uuid);
static int vm_start(const char *vm_uuid);
static int update_device_model_args(const char *vm_uuid, char pool_id, int priority, int vm_id);
static int get_device_model_args(const char *vm_uuid, char *args, size_t args_size);
static int scan_xcpng_vms(void);
static int get_vm_name(const char *vm_uuid, char *name, size_t name_size);
static int get_vm_uuid_from_name(const char *vm_name, char *uuid, size_t uuid_size);
static int get_vm_power_state(const char *vm_uuid, char *state, size_t state_size);
static void print_help(void);
static void print_vm_config(const vgpu_vm_config_t *config);
static void print_pool_info(const vgpu_pool_info_t *pool);
static const char *priority_str(int priority);
static int parse_priority(const char *str);
static char parse_pool_id(const char *str);

/* ============================================================================
 * Main
 * ============================================================================ */

int main(int argc, char *argv[]) {
    sqlite3 *db;
    int rc;
    
    if (argc < 2) {
        print_help();
        return 1;
    }
    
    /* Initialize database */
    rc = vgpu_db_init(&db);
    if (rc != VGPU_OK) {
        fprintf(stderr, "Error: Failed to initialize database\n");
        return 1;
    }
    
    /* Ensure schema exists */
    vgpu_db_init_schema(db);
    
    /* Parse command */
    const char *cmd = argv[1];
    
    if (strcmp(cmd, "register-vm") == 0) {
        /* vgpu-admin register-vm (--vm-uuid=<uuid> | --vm-name=<name>) [--pool=<A|B>] [--priority=<low|medium|high>] [--vm-id=<id>] */
        const char *vm_uuid = NULL;
        const char *vm_name_param = NULL;
        const char *vm_name = NULL;
        char pool_id = 0;
        int priority = -1;
        int vm_id = 0;
        char resolved_uuid[64] = {0};
        
        for (int i = 2; i < argc; i++) {
            if (strncmp(argv[i], "--vm-uuid=", 10) == 0) {
                vm_uuid = argv[i] + 10;
            } else if (strncmp(argv[i], "--vm-name=", 10) == 0) {
                vm_name_param = argv[i] + 10;
            } else if (strncmp(argv[i], "--pool=", 7) == 0) {
                pool_id = parse_pool_id(argv[i] + 7);
            } else if (strncmp(argv[i], "--priority=", 11) == 0) {
                priority = parse_priority(argv[i] + 11);
            } else if (strncmp(argv[i], "--vm-id=", 8) == 0) {
                vm_id = atoi(argv[i] + 8);
            }
        }
        
        /* Validate that either UUID or name is provided, but not both */
        if (!vm_uuid && !vm_name_param) {
            fprintf(stderr, "Error: Either --vm-uuid or --vm-name is required\n");
            fprintf(stderr, "Usage: vgpu-admin register-vm (--vm-uuid=<uuid> | --vm-name=<name>) [options]\n");
            vgpu_db_close(db);
            return 1;
        }
        
        if (vm_uuid && vm_name_param) {
            fprintf(stderr, "Error: Cannot specify both --vm-uuid and --vm-name. Use one or the other.\n");
            vgpu_db_close(db);
            return 1;
        }
        
        /* If VM name provided, resolve it to UUID */
        if (vm_name_param) {
            if (get_vm_uuid_from_name(vm_name_param, resolved_uuid, sizeof(resolved_uuid)) != 0) {
                fprintf(stderr, "Error: VM with name '%s' not found\n", vm_name_param);
                vgpu_db_close(db);
                return 1;
            }
            vm_uuid = resolved_uuid;
            vm_name = vm_name_param;  /* Use provided name for database */
        }
        
        /* Determine final values (with defaults) */
        char final_pool = (pool_id != 0) ? pool_id : 'A';
        int final_priority = (priority >= 0) ? priority : 1;  /* medium */
        int final_vm_id = vm_id;
        
        /* If vm_id not provided, get next available (but don't register yet) */
        if (final_vm_id == 0) {
            final_vm_id = vgpu_get_next_vm_id(db);
        } else {
            /* Check if VM ID is already in use */
            if (vgpu_vm_id_in_use(db, final_vm_id)) {
                fprintf(stderr, "Error: VM ID %d is already in use\n", final_vm_id);
                vgpu_db_close(db);
                return 1;
            }
        }
        
        /* Check if VM is running */
        int was_running = vm_is_running(vm_uuid);
        
        if (was_running) {
            printf("VM is currently running. Stopping VM...\n");
            if (vm_stop_and_verify(vm_uuid) != 0) {
                fprintf(stderr, "Error: Failed to stop VM within 5 seconds\n");
                vgpu_db_close(db);
                return 1;
            }
            printf("VM stopped successfully.\n");
        }
        
        /* Set device-model-args */
        printf("Configuring vGPU-stub device...\n");
        if (update_device_model_args(vm_uuid, final_pool, final_priority, final_vm_id) != 0) {
            fprintf(stderr, "Error: Failed to set device-model-args\n");
            if (was_running) {
                fprintf(stderr, "Note: VM was stopped but configuration failed. You may need to start it manually.\n");
            }
            vgpu_db_close(db);
            return 1;
        }
        
        /* Verify device-model-args were set correctly */
        char device_args[512];
        if (get_device_model_args(vm_uuid, device_args, sizeof(device_args)) == 0) {
            /* Convert priority to string for verification */
            const char *priority_str_expected;
            switch (final_priority) {
                case 0: priority_str_expected = "low"; break;
                case 1: priority_str_expected = "medium"; break;
                case 2: priority_str_expected = "high"; break;
                default: priority_str_expected = "medium"; break;
            }
            
            /* Check if the device-model-args contain our vgpu-stub configuration */
            char pool_str[8], priority_check[16], vm_id_str[16];
            snprintf(pool_str, sizeof(pool_str), "pool_id=%c", final_pool);
            snprintf(priority_check, sizeof(priority_check), "priority=%s", priority_str_expected);
            snprintf(vm_id_str, sizeof(vm_id_str), "vm_id=%d", final_vm_id);
            
            if (strstr(device_args, "vgpu-stub") == NULL ||
                strstr(device_args, pool_str) == NULL ||
                strstr(device_args, priority_check) == NULL ||
                strstr(device_args, vm_id_str) == NULL) {
                fprintf(stderr, "Error: Device-model-args verification failed\n");
                fprintf(stderr, "Expected to contain: vgpu-stub, pool_id=%c, priority=%s, vm_id=%d\n",
                        final_pool, priority_str_expected, final_vm_id);
                fprintf(stderr, "Got: %s\n", device_args);
                if (was_running) {
                    fprintf(stderr, "Note: VM was stopped but configuration verification failed. You may need to start it manually.\n");
                }
                vgpu_db_close(db);
                return 1;
            }
            printf("Device-model-args verified successfully.\n");
        } else {
            fprintf(stderr, "Warning: Could not verify device-model-args, but continuing...\n");
        }
        
        /* Now register in database */
        rc = vgpu_register_vm(db, vm_uuid, vm_name, final_pool, final_priority, final_vm_id);
        if (rc == VGPU_OK) {
            vgpu_vm_config_t config;
            if (vgpu_get_vm_config(db, vm_uuid, &config) == VGPU_OK) {
                printf("\nVM registered successfully:\n");
                print_vm_config(&config);
                
                if (was_running) {
                    printf("\nStarting VM...\n");
                    vm_start(vm_uuid);
                }
            }
        } else if (rc == VGPU_ERROR) {
            fprintf(stderr, "Error: VM ID already in use or database error\n");
            if (was_running) {
                fprintf(stderr, "Note: VM was stopped and configured, but database registration failed. You may need to start it manually.\n");
            }
        } else {
            fprintf(stderr, "Error: Failed to register VM in database\n");
            if (was_running) {
                fprintf(stderr, "Note: VM was stopped and configured, but database registration failed. You may need to start it manually.\n");
            }
        }
        
    } else if (strcmp(cmd, "scan-vms") == 0) {
        /* vgpu-admin scan-vms */
        scan_xcpng_vms();
        
    } else if (strcmp(cmd, "show-vm") == 0) {
        /* vgpu-admin show-vm --vm-uuid=<uuid> */
        const char *vm_uuid = NULL;
        
        for (int i = 2; i < argc; i++) {
            if (strncmp(argv[i], "--vm-uuid=", 10) == 0) {
                vm_uuid = argv[i] + 10;
            }
        }
        
        if (!vm_uuid) {
            fprintf(stderr, "Error: --vm-uuid is required\n");
            vgpu_db_close(db);
            return 1;
        }
        
        vgpu_vm_config_t config;
        rc = vgpu_get_vm_config(db, vm_uuid, &config);
        if (rc == VGPU_OK) {
            print_vm_config(&config);
        } else {
            fprintf(stderr, "Error: VM not found in database\n");
        }
        
    } else if (strcmp(cmd, "list-vms") == 0) {
        /* vgpu-admin list-vms [--pool=<A|B>] [--priority=<low|medium|high>] */
        char pool_id = 0;
        int priority = -1;
        
        for (int i = 2; i < argc; i++) {
            if (strncmp(argv[i], "--pool=", 7) == 0) {
                pool_id = parse_pool_id(argv[i] + 7);
            } else if (strncmp(argv[i], "--priority=", 11) == 0) {
                priority = parse_priority(argv[i] + 11);
            }
        }
        
        vgpu_vm_config_t configs[MAX_VMS];
        int count = 0;
        rc = vgpu_list_vms(db, pool_id, priority, configs, &count, MAX_VMS);
        if (rc == VGPU_OK) {
            if (count == 0) {
                printf("No VMs found\n");
            } else {
                for (int i = 0; i < count; i++) {
                    print_vm_config(&configs[i]);
                    if (i < count - 1) printf("\n");
                }
            }
        }
        
    } else if (strcmp(cmd, "list-pools") == 0) {
        /* vgpu-admin list-pools */
        vgpu_pool_info_t pools[2];
        int count = 0;
        rc = vgpu_list_pools(db, pools, &count);
        if (rc == VGPU_OK) {
            for (int i = 0; i < count; i++) {
                print_pool_info(&pools[i]);
                if (i < count - 1) printf("\n");
            }
        }
        
    } else if (strcmp(cmd, "show-pool") == 0) {
        /* vgpu-admin show-pool --pool-id=<A|B> */
        char pool_id = 0;
        
        for (int i = 2; i < argc; i++) {
            if (strncmp(argv[i], "--pool-id=", 10) == 0) {
                pool_id = parse_pool_id(argv[i] + 10);
            }
        }
        
        if (pool_id == 0) {
            fprintf(stderr, "Error: --pool-id is required (A or B)\n");
            vgpu_db_close(db);
            return 1;
        }
        
        vgpu_pool_info_t pool;
        rc = vgpu_get_pool_info(db, pool_id, &pool);
        if (rc == VGPU_OK) {
            print_pool_info(&pool);
            
            /* List VMs in this pool */
            vgpu_vm_config_t configs[MAX_VMS];
            int count = 0;
            vgpu_list_vms(db, pool_id, -1, configs, &count, MAX_VMS);
            if (count > 0) {
                printf("\nVMs in this pool (%d):\n", count);
                for (int i = 0; i < count; i++) {
                    printf("  - %s (ID: %d, Priority: %s)\n",
                           configs[i].vm_name[0] ? configs[i].vm_name : configs[i].vm_uuid,
                           configs[i].vm_id, priority_str(configs[i].priority));
                }
            }
        } else {
            fprintf(stderr, "Error: Pool not found\n");
        }
        
    } else if (strcmp(cmd, "set-pool") == 0) {
        /* vgpu-admin set-pool --vm-uuid=<uuid> --pool=<A|B> */
        const char *vm_uuid = NULL;
        char new_pool = 0;
        
        for (int i = 2; i < argc; i++) {
            if (strncmp(argv[i], "--vm-uuid=", 10) == 0) {
                vm_uuid = argv[i] + 10;
            } else if (strncmp(argv[i], "--pool=", 7) == 0) {
                new_pool = parse_pool_id(argv[i] + 7);
            }
        }
        
        if (!vm_uuid || new_pool == 0) {
            fprintf(stderr, "Error: --vm-uuid and --pool are required\n");
            vgpu_db_close(db);
            return 1;
        }
        
        vgpu_vm_config_t old_config;
        rc = vgpu_get_vm_config(db, vm_uuid, &old_config);
        if (rc != VGPU_OK) {
            fprintf(stderr, "Error: VM not found in database\n");
            vgpu_db_close(db);
            return 1;
        }
        
        if (old_config.pool_id == new_pool) {
            printf("VM is already in Pool %c. No changes needed.\n", new_pool);
            vgpu_db_close(db);
            return 0;
        }
        
        /* Show confirmation */
        if (!confirm_apply_settings(vm_uuid, old_config.pool_id, new_pool,
                                     old_config.priority, old_config.priority,
                                     old_config.vm_id, old_config.vm_id)) {
            printf("Operation cancelled.\n");
            vgpu_db_close(db);
            return 0;
        }
        
        /* Stop VM if running */
        if (vm_is_running(vm_uuid)) {
            printf("Stopping VM...\n");
            if (vm_stop(vm_uuid) != 0) {
                fprintf(stderr, "Error: Failed to stop VM\n");
                vgpu_db_close(db);
                return 1;
            }
        }
        
        /* Update database */
        rc = vgpu_set_vm_pool(db, vm_uuid, new_pool);
        if (rc != VGPU_OK) {
            fprintf(stderr, "Error: Failed to update database\n");
            vgpu_db_close(db);
            return 1;
        }
        
        /* Update device-model-args */
        vgpu_vm_config_t new_config;
        vgpu_get_vm_config(db, vm_uuid, &new_config);
        if (update_device_model_args(vm_uuid, new_config.pool_id, new_config.priority, new_config.vm_id) != 0) {
            fprintf(stderr, "Error: Failed to update device-model-args\n");
            vgpu_db_close(db);
            return 1;
        }
        
        /* Start VM */
        printf("Starting VM...\n");
        vm_start(vm_uuid);
        
        printf("VM pool updated successfully.\n");
        
    } else if (strcmp(cmd, "set-priority") == 0) {
        /* vgpu-admin set-priority --vm-uuid=<uuid> --priority=<low|medium|high> */
        const char *vm_uuid = NULL;
        int new_priority = -1;
        
        for (int i = 2; i < argc; i++) {
            if (strncmp(argv[i], "--vm-uuid=", 10) == 0) {
                vm_uuid = argv[i] + 10;
            } else if (strncmp(argv[i], "--priority=", 11) == 0) {
                new_priority = parse_priority(argv[i] + 11);
            }
        }
        
        if (!vm_uuid || new_priority < 0) {
            fprintf(stderr, "Error: --vm-uuid and --priority are required\n");
            vgpu_db_close(db);
            return 1;
        }
        
        vgpu_vm_config_t old_config;
        rc = vgpu_get_vm_config(db, vm_uuid, &old_config);
        if (rc != VGPU_OK) {
            fprintf(stderr, "Error: VM not found in database\n");
            vgpu_db_close(db);
            return 1;
        }
        
        if (old_config.priority == new_priority) {
            printf("VM priority is already %s. No changes needed.\n", priority_str(new_priority));
            vgpu_db_close(db);
            return 0;
        }
        
        /* Show confirmation */
        if (!confirm_apply_settings(vm_uuid, old_config.pool_id, old_config.pool_id,
                                     old_config.priority, new_priority,
                                     old_config.vm_id, old_config.vm_id)) {
            printf("Operation cancelled.\n");
            vgpu_db_close(db);
            return 0;
        }
        
        /* Stop VM if running */
        if (vm_is_running(vm_uuid)) {
            printf("Stopping VM...\n");
            if (vm_stop(vm_uuid) != 0) {
                fprintf(stderr, "Error: Failed to stop VM\n");
                vgpu_db_close(db);
                return 1;
            }
        }
        
        /* Update database */
        rc = vgpu_set_vm_priority(db, vm_uuid, new_priority);
        if (rc != VGPU_OK) {
            fprintf(stderr, "Error: Failed to update database\n");
            vgpu_db_close(db);
            return 1;
        }
        
        /* Update device-model-args */
        vgpu_vm_config_t new_config;
        vgpu_get_vm_config(db, vm_uuid, &new_config);
        if (update_device_model_args(vm_uuid, new_config.pool_id, new_config.priority, new_config.vm_id) != 0) {
            fprintf(stderr, "Error: Failed to update device-model-args\n");
            vgpu_db_close(db);
            return 1;
        }
        
        /* Start VM */
        printf("Starting VM...\n");
        vm_start(vm_uuid);
        
        printf("VM priority updated successfully.\n");
        
    } else if (strcmp(cmd, "set-vm-id") == 0) {
        /* vgpu-admin set-vm-id --vm-uuid=<uuid> --vm-id=<id> */
        const char *vm_uuid = NULL;
        int new_vm_id = 0;
        
        for (int i = 2; i < argc; i++) {
            if (strncmp(argv[i], "--vm-uuid=", 10) == 0) {
                vm_uuid = argv[i] + 10;
            } else if (strncmp(argv[i], "--vm-id=", 8) == 0) {
                new_vm_id = atoi(argv[i] + 8);
            }
        }
        
        if (!vm_uuid || new_vm_id <= 0) {
            fprintf(stderr, "Error: --vm-uuid and --vm-id are required\n");
            vgpu_db_close(db);
            return 1;
        }
        
        vgpu_vm_config_t old_config;
        rc = vgpu_get_vm_config(db, vm_uuid, &old_config);
        if (rc != VGPU_OK) {
            fprintf(stderr, "Error: VM not found in database\n");
            vgpu_db_close(db);
            return 1;
        }
        
        if (old_config.vm_id == new_vm_id) {
            printf("VM ID is already %d. No changes needed.\n", new_vm_id);
            vgpu_db_close(db);
            return 0;
        }
        
        /* Show confirmation */
        if (!confirm_apply_settings(vm_uuid, old_config.pool_id, old_config.pool_id,
                                     old_config.priority, old_config.priority,
                                     old_config.vm_id, new_vm_id)) {
            printf("Operation cancelled.\n");
            vgpu_db_close(db);
            return 0;
        }
        
        /* Stop VM if running */
        if (vm_is_running(vm_uuid)) {
            printf("Stopping VM...\n");
            if (vm_stop(vm_uuid) != 0) {
                fprintf(stderr, "Error: Failed to stop VM\n");
                vgpu_db_close(db);
                return 1;
            }
        }
        
        /* Update database */
        rc = vgpu_set_vm_id(db, vm_uuid, new_vm_id);
        if (rc == VGPU_ERROR) {
            fprintf(stderr, "Error: VM ID %d is already in use\n", new_vm_id);
            vgpu_db_close(db);
            return 1;
        } else if (rc != VGPU_OK) {
            fprintf(stderr, "Error: Failed to update database\n");
            vgpu_db_close(db);
            return 1;
        }
        
        /* Update device-model-args */
        vgpu_vm_config_t new_config;
        vgpu_get_vm_config(db, vm_uuid, &new_config);
        if (update_device_model_args(vm_uuid, new_config.pool_id, new_config.priority, new_config.vm_id) != 0) {
            fprintf(stderr, "Error: Failed to update device-model-args\n");
            vgpu_db_close(db);
            return 1;
        }
        
        /* Start VM */
        printf("Starting VM...\n");
        vm_start(vm_uuid);
        
        printf("VM ID updated successfully.\n");
        
    } else if (strcmp(cmd, "update-vm") == 0) {
        /* vgpu-admin update-vm --vm-uuid=<uuid> [--pool=<A|B>] [--priority=<low|medium|high>] [--vm-id=<id>] */
        const char *vm_uuid = NULL;
        char new_pool = 0;
        int new_priority = -1;
        int new_vm_id = 0;
        
        for (int i = 2; i < argc; i++) {
            if (strncmp(argv[i], "--vm-uuid=", 10) == 0) {
                vm_uuid = argv[i] + 10;
            } else if (strncmp(argv[i], "--pool=", 7) == 0) {
                new_pool = parse_pool_id(argv[i] + 7);
            } else if (strncmp(argv[i], "--priority=", 11) == 0) {
                new_priority = parse_priority(argv[i] + 11);
            } else if (strncmp(argv[i], "--vm-id=", 8) == 0) {
                new_vm_id = atoi(argv[i] + 8);
            }
        }
        
        if (!vm_uuid) {
            fprintf(stderr, "Error: --vm-uuid is required\n");
            vgpu_db_close(db);
            return 1;
        }
        
        vgpu_vm_config_t old_config;
        rc = vgpu_get_vm_config(db, vm_uuid, &old_config);
        if (rc != VGPU_OK) {
            fprintf(stderr, "Error: VM not found in database\n");
            vgpu_db_close(db);
            return 1;
        }
        
        char final_pool = (new_pool != 0) ? new_pool : old_config.pool_id;
        int final_priority = (new_priority >= 0) ? new_priority : old_config.priority;
        int final_vm_id = (new_vm_id > 0) ? new_vm_id : old_config.vm_id;
        
        /* Check if anything changed */
        if (final_pool == old_config.pool_id && 
            final_priority == old_config.priority && 
            final_vm_id == old_config.vm_id) {
            printf("No changes specified. VM configuration unchanged.\n");
            vgpu_db_close(db);
            return 0;
        }
        
        /* Show confirmation */
        if (!confirm_apply_settings(vm_uuid, old_config.pool_id, final_pool,
                                     old_config.priority, final_priority,
                                     old_config.vm_id, final_vm_id)) {
            printf("Operation cancelled.\n");
            vgpu_db_close(db);
            return 0;
        }
        
        /* Stop VM if running */
        if (vm_is_running(vm_uuid)) {
            printf("Stopping VM...\n");
            if (vm_stop(vm_uuid) != 0) {
                fprintf(stderr, "Error: Failed to stop VM\n");
                vgpu_db_close(db);
                return 1;
            }
        }
        
        /* Update database */
        rc = vgpu_update_vm(db, vm_uuid, new_pool, new_priority, new_vm_id);
        if (rc == VGPU_ERROR) {
            fprintf(stderr, "Error: VM ID already in use\n");
            vgpu_db_close(db);
            return 1;
        } else if (rc != VGPU_OK) {
            fprintf(stderr, "Error: Failed to update database\n");
            vgpu_db_close(db);
            return 1;
        }
        
        /* Update device-model-args */
        vgpu_vm_config_t new_config;
        vgpu_get_vm_config(db, vm_uuid, &new_config);
        if (update_device_model_args(vm_uuid, new_config.pool_id, new_config.priority, new_config.vm_id) != 0) {
            fprintf(stderr, "Error: Failed to update device-model-args\n");
            vgpu_db_close(db);
            return 1;
        }
        
        /* Start VM */
        printf("Starting VM...\n");
        vm_start(vm_uuid);
        
        printf("VM configuration updated successfully.\n");
        
    } else if (strcmp(cmd, "remove-vm") == 0) {
        /* vgpu-admin remove-vm (--vm-uuid=<uuid> | --vm-name=<name>) */
        const char *vm_uuid = NULL;
        const char *vm_name_param = NULL;
        char resolved_uuid[64] = {0};
        
        for (int i = 2; i < argc; i++) {
            if (strncmp(argv[i], "--vm-uuid=", 10) == 0) {
                vm_uuid = argv[i] + 10;
            } else if (strncmp(argv[i], "--vm-name=", 10) == 0) {
                vm_name_param = argv[i] + 10;
            }
        }
        
        /* Validate that either UUID or name is provided, but not both */
        if (!vm_uuid && !vm_name_param) {
            fprintf(stderr, "Error: Either --vm-uuid or --vm-name is required\n");
            fprintf(stderr, "Usage: vgpu-admin remove-vm (--vm-uuid=<uuid> | --vm-name=<name>)\n");
            vgpu_db_close(db);
            return 1;
        }
        
        if (vm_uuid && vm_name_param) {
            fprintf(stderr, "Error: Cannot specify both --vm-uuid and --vm-name. Use one or the other.\n");
            vgpu_db_close(db);
            return 1;
        }
        
        /* If VM name provided, resolve it to UUID */
        if (vm_name_param) {
            if (get_vm_uuid_from_name(vm_name_param, resolved_uuid, sizeof(resolved_uuid)) != 0) {
                fprintf(stderr, "Error: VM with name '%s' not found\n", vm_name_param);
                vgpu_db_close(db);
                return 1;
            }
            vm_uuid = resolved_uuid;
        }
        
        /* Check if VM is registered in database */
        vgpu_vm_config_t config;
        rc = vgpu_get_vm_config(db, vm_uuid, &config);
        if (rc != VGPU_OK) {
            fprintf(stderr, "Error: VM not found in database\n");
            vgpu_db_close(db);
            return 1;
        }
        
        /* Check if VM is running */
        int was_running = vm_is_running(vm_uuid);
        
        if (was_running) {
            printf("VM is currently running. Stopping VM...\n");
            if (vm_stop_and_verify(vm_uuid) != 0) {
                fprintf(stderr, "Error: Failed to stop VM within 5 seconds\n");
                vgpu_db_close(db);
                return 1;
            }
            printf("VM stopped successfully.\n");
        }
        
        /* Remove vGPU-stub device (detach device-model-args) */
        printf("Removing vGPU-stub device...\n");
        char cmd[512];
        snprintf(cmd, sizeof(cmd), "xe vm-param-remove uuid=%s param-name=platform param-key=device-model-args 2>/dev/null", vm_uuid);
        int remove_rc = system(cmd);
        
        if (remove_rc != 0) {
            fprintf(stderr, "Warning: Failed to remove device-model-args (VM may not have vGPU-stub attached)\n");
            /* Continue anyway - maybe device wasn't attached */
        } else {
            printf("vGPU-stub device removed successfully.\n");
        }
        
        /* Remove from database */
        rc = vgpu_remove_vm(db, vm_uuid);
        if (rc == VGPU_OK) {
            printf("VM removed from database successfully.\n");
        } else {
            fprintf(stderr, "Error: Failed to remove VM from database\n");
            vgpu_db_close(db);
            return 1;
        }
        
    } else if (strcmp(cmd, "status") == 0) {
        /* vgpu-admin status */
        printf("=== vGPU Configuration Status ===\n\n");
        
        vgpu_pool_info_t pools[2];
        int pool_count = 0;
        vgpu_list_pools(db, pools, &pool_count);
        
        for (int i = 0; i < pool_count; i++) {
            print_pool_info(&pools[i]);
            
            vgpu_vm_config_t configs[MAX_VMS];
            int vm_count = 0;
            vgpu_list_vms(db, pools[i].pool_id, -1, configs, &vm_count, MAX_VMS);
            
            if (vm_count > 0) {
                printf("  VMs (%d):\n", vm_count);
                for (int j = 0; j < vm_count; j++) {
                    char vm_name[256];
                    get_vm_name(configs[j].vm_uuid, vm_name, sizeof(vm_name));
                    printf("    - %s (ID: %d, Priority: %s)\n",
                           vm_name[0] ? vm_name : configs[j].vm_uuid,
                           configs[j].vm_id, priority_str(configs[j].priority));
                }
            }
            printf("\n");
        }
        
        /* Show unregistered VMs */
        printf("=== Unregistered VMs ===\n");
        scan_xcpng_vms();
        
    } else if (strcmp(cmd, "help") == 0 || strcmp(cmd, "--help") == 0 || strcmp(cmd, "-h") == 0) {
        print_help();
    } else {
        fprintf(stderr, "Error: Unknown command: %s\n", cmd);
        fprintf(stderr, "Run 'vgpu-admin help' for usage information\n");
        vgpu_db_close(db);
        return 1;
    }
    
    vgpu_db_close(db);
    return 0;
}

/* ============================================================================
 * Helper Functions
 * ============================================================================ */

static int confirm_apply_settings(const char *vm_uuid, char old_pool, char new_pool,
                                  int old_priority, int new_priority,
                                  int old_vm_id, int new_vm_id) {
    char vm_name[256];
    get_vm_name(vm_uuid, vm_name, sizeof(vm_name));
    
    printf("\nCurrent configuration:\n");
    printf("  Pool: %c\n", old_pool);
    printf("  Priority: %s (%d)\n", priority_str(old_priority), old_priority);
    printf("  VM ID: %d\n", old_vm_id);
    
    printf("\nNew configuration:\n");
    printf("  Pool: %c\n", new_pool);
    printf("  Priority: %s (%d)\n", priority_str(new_priority), new_priority);
    printf("  VM ID: %d\n", new_vm_id);
    
    printf("\nThis change requires stopping and restarting the VM.\n");
    printf("Do you want to apply these settings? (yes/no): ");
    
    char answer[10];
    if (fgets(answer, sizeof(answer), stdin) == NULL) {
        return 0;
    }
    
    /* Remove newline */
    size_t len = strlen(answer);
    if (len > 0 && answer[len-1] == '\n') {
        answer[len-1] = '\0';
    }
    
    /* Convert to lowercase */
    for (int i = 0; answer[i]; i++) {
        answer[i] = tolower(answer[i]);
    }
    
    return (strcmp(answer, "yes") == 0 || strcmp(answer, "y") == 0);
}

static int vm_is_running(const char *vm_uuid) {
    char state[32];
    if (get_vm_power_state(vm_uuid, state, sizeof(state)) == 0) {
        return (strcmp(state, "running") == 0);
    }
    return 0;
}

static int vm_stop(const char *vm_uuid) {
    char cmd[512];
    snprintf(cmd, sizeof(cmd), "xe vm-shutdown uuid=%s 2>/dev/null", vm_uuid);
    int rc = system(cmd);
    
    if (rc != 0) {
        /* Try force shutdown */
        snprintf(cmd, sizeof(cmd), "xe vm-shutdown uuid=%s force=true 2>/dev/null", vm_uuid);
        rc = system(cmd);
    }
    
    /* Wait for VM to stop (max 30 seconds) */
    for (int i = 0; i < 30; i++) {
        if (!vm_is_running(vm_uuid)) {
            return 0;
        }
        sleep(1);
    }
    
    return (vm_is_running(vm_uuid) ? -1 : 0);
}

static int vm_stop_and_verify(const char *vm_uuid) {
    char cmd[512];
    snprintf(cmd, sizeof(cmd), "xe vm-shutdown uuid=%s 2>/dev/null", vm_uuid);
    int rc = system(cmd);
    
    if (rc != 0) {
        /* Try force shutdown */
        snprintf(cmd, sizeof(cmd), "xe vm-shutdown uuid=%s force=true 2>/dev/null", vm_uuid);
        rc = system(cmd);
    }
    
    /* Wait 2-5 seconds and verify VM is stopped */
    for (int i = 0; i < 5; i++) {
        sleep(1);
        if (!vm_is_running(vm_uuid)) {
            return 0;  /* Successfully stopped */
        }
    }
    
    /* Check one more time after 5 seconds */
    if (!vm_is_running(vm_uuid)) {
        return 0;
    }
    
    return -1;  /* Failed to stop */
}

static int vm_start(const char *vm_uuid) {
    char cmd[512];
    snprintf(cmd, sizeof(cmd), "xe vm-start uuid=%s 2>/dev/null", vm_uuid);
    return system(cmd);
}

static int update_device_model_args(const char *vm_uuid, char pool_id, int priority, int vm_id) {
    char cmd[1024];
    const char *priority_str;
    
    /* Convert priority integer to string (vgpu-stub expects string, not integer) */
    switch (priority) {
        case 0: priority_str = "low"; break;
        case 1: priority_str = "medium"; break;
        case 2: priority_str = "high"; break;
        default: priority_str = "medium"; break;  /* Default to medium if invalid */
    }
    
    snprintf(cmd, sizeof(cmd),
             "xe vm-param-set uuid=%s platform:device-model-args=\"-device vgpu-stub,pool_id=%c,priority=%s,vm_id=%d\" 2>/dev/null",
             vm_uuid, pool_id, priority_str, vm_id);
    return system(cmd);
}

static int get_device_model_args(const char *vm_uuid, char *args, size_t args_size) {
    char cmd[512];
    FILE *fp;
    
    snprintf(cmd, sizeof(cmd), "xe vm-param-get uuid=%s param-name=platform param-key=device-model-args 2>/dev/null", vm_uuid);
    
    fp = popen(cmd, "r");
    if (!fp) {
        return -1;
    }
    
    if (fgets(args, args_size, fp) == NULL) {
        pclose(fp);
        return -1;
    }
    
    pclose(fp);
    
    /* Remove newline */
    size_t len = strlen(args);
    if (len > 0 && args[len-1] == '\n') {
        args[len-1] = '\0';
    }
    
    return 0;
}

static int scan_xcpng_vms(void) {
    FILE *fp;
    char line[MAX_LINE];
    sqlite3 *db;
    
    if (vgpu_db_init(&db) != VGPU_OK) {
        fprintf(stderr, "Error: Failed to initialize database\n");
        return 1;
    }
    
    vgpu_db_init_schema(db);
    
    /* Get all VM UUIDs from XCP-ng (one per line) */
    fp = popen("xe vm-list is-control-domain=false params=uuid --minimal 2>/dev/null | tr ',' '\\n'", "r");
    if (!fp) {
        fprintf(stderr, "Error: Failed to query XCP-ng\n");
        vgpu_db_close(db);
        return 1;
    }
    
    /* Group VMs by pool */
    vgpu_vm_config_t pool_a_vms[MAX_VMS];
    vgpu_vm_config_t pool_b_vms[MAX_VMS];
    char unregistered_vms[MAX_VMS][64];
    int pool_a_count = 0;
    int pool_b_count = 0;
    int unregistered_count = 0;
    
    while (fgets(line, sizeof(line), fp)) {
        /* Remove newline and whitespace */
        size_t len = strlen(line);
        if (len > 0 && line[len-1] == '\n') {
            line[len-1] = '\0';
            len--;
        }
        /* Trim leading/trailing whitespace */
        while (len > 0 && (line[len-1] == ' ' || line[len-1] == '\t')) {
            line[len-1] = '\0';
            len--;
        }
        char *uuid = line;
        while (*uuid == ' ' || *uuid == '\t') uuid++;
        
        /* Skip empty lines */
        if (strlen(uuid) == 0) continue;
        
        /* Validate UUID format (should be 36 characters with dashes) */
        if (strlen(uuid) < 36) continue;
        
        /* Extract just the UUID part (first 36 characters) */
        char clean_uuid[64];
        strncpy(clean_uuid, uuid, 36);
        clean_uuid[36] = '\0';
        
        /* Check if VM is registered */
        vgpu_vm_config_t config;
        if (vgpu_get_vm_config(db, clean_uuid, &config) == VGPU_OK) {
            if (config.pool_id == 'A') {
                if (pool_a_count < MAX_VMS) {
                    pool_a_vms[pool_a_count++] = config;
                }
            } else if (config.pool_id == 'B') {
                if (pool_b_count < MAX_VMS) {
                    pool_b_vms[pool_b_count++] = config;
                }
            }
        } else {
            /* Unregistered VM */
            if (unregistered_count < MAX_VMS) {
                strncpy(unregistered_vms[unregistered_count++], clean_uuid, 63);
                unregistered_vms[unregistered_count-1][63] = '\0';
            }
        }
    }
    
    pclose(fp);
    
    /* Print Pool A VMs */
    if (pool_a_count > 0) {
        printf("\n=== Pool A: %d VMs ===\n", pool_a_count);
        for (int i = 0; i < pool_a_count; i++) {
            char vm_name[256];
            char device_args[512];
            get_vm_name(pool_a_vms[i].vm_uuid, vm_name, sizeof(vm_name));
            get_device_model_args(pool_a_vms[i].vm_uuid, device_args, sizeof(device_args));
            
            printf("UUID: %.12s... | Name: %s | Pool: A | Priority: %s (%d) | VM ID: %d | %s\n",
                   pool_a_vms[i].vm_uuid,
                   vm_name[0] ? vm_name : pool_a_vms[i].vm_uuid,
                   priority_str(pool_a_vms[i].priority),
                   pool_a_vms[i].priority,
                   pool_a_vms[i].vm_id,
                   device_args[0] ? "✓ Configured" : "⚠ Not configured");
        }
    }
    
    /* Print Pool B VMs */
    if (pool_b_count > 0) {
        printf("\n=== Pool B: %d VMs ===\n", pool_b_count);
        for (int i = 0; i < pool_b_count; i++) {
            char vm_name[256];
            char device_args[512];
            get_vm_name(pool_b_vms[i].vm_uuid, vm_name, sizeof(vm_name));
            get_device_model_args(pool_b_vms[i].vm_uuid, device_args, sizeof(device_args));
            
            printf("UUID: %.12s... | Name: %s | Pool: B | Priority: %s (%d) | VM ID: %d | %s\n",
                   pool_b_vms[i].vm_uuid,
                   vm_name[0] ? vm_name : pool_b_vms[i].vm_uuid,
                   priority_str(pool_b_vms[i].priority),
                   pool_b_vms[i].priority,
                   pool_b_vms[i].vm_id,
                   device_args[0] ? "✓ Configured" : "⚠ Not configured");
        }
    }
    
    /* Print unregistered VMs */
    if (unregistered_count > 0) {
        printf("\n=== Unregistered VMs: %d ===\n", unregistered_count);
        for (int i = 0; i < unregistered_count; i++) {
            char vm_name[256];
            get_vm_name(unregistered_vms[i], vm_name, sizeof(vm_name));
            
            printf("UUID: %.12s... | Name: %s | Status: ⚠ Not registered | Action: Run 'vgpu-admin register-vm'\n",
                   unregistered_vms[i],
                   vm_name[0] ? vm_name : unregistered_vms[i]);
        }
    } else if (pool_a_count == 0 && pool_b_count == 0) {
        printf("\nNo VMs found.\n");
    }
    
    vgpu_db_close(db);
    return 0;
}

static int get_vm_name(const char *vm_uuid, char *name, size_t name_size) {
    char cmd[512];
    FILE *fp;
    
    snprintf(cmd, sizeof(cmd), "xe vm-param-get uuid=%s param-name=name-label 2>/dev/null", vm_uuid);
    
    fp = popen(cmd, "r");
    if (!fp) {
        name[0] = '\0';
        return -1;
    }
    
    if (fgets(name, name_size, fp) == NULL) {
        pclose(fp);
        name[0] = '\0';
        return -1;
    }
    
    pclose(fp);
    
    /* Remove newline */
    size_t len = strlen(name);
    if (len > 0 && name[len-1] == '\n') {
        name[len-1] = '\0';
    }
    
    return 0;
}

static int get_vm_uuid_from_name(const char *vm_name, char *uuid, size_t uuid_size) {
    char cmd[512];
    FILE *fp;
    char line[256];
    int uuid_count = 0;
    char first_uuid[64] = {0};
    
    /* Query for VM by name */
    snprintf(cmd, sizeof(cmd), "xe vm-list name-label=\"%s\" params=uuid --minimal 2>/dev/null", vm_name);
    
    fp = popen(cmd, "r");
    if (!fp) {
        return -1;
    }
    
    if (fgets(line, sizeof(line), fp) == NULL) {
        pclose(fp);
        return -1;
    }
    
    pclose(fp);
    
    /* Remove newline */
    size_t len = strlen(line);
    if (len > 0 && line[len-1] == '\n') {
        line[len-1] = '\0';
        len--;
    }
    
    /* Trim whitespace */
    while (len > 0 && (line[len-1] == ' ' || line[len-1] == '\t')) {
        line[len-1] = '\0';
        len--;
    }
    
    if (len == 0) {
        return -1; /* No UUID found */
    }
    
    /* Count commas to detect multiple VMs with same name */
    for (int i = 0; line[i]; i++) {
        if (line[i] == ',') {
            uuid_count++;
        }
    }
    
    if (uuid_count > 0) {
        /* Multiple VMs with same name - error */
        fprintf(stderr, "Error: Multiple VMs found with name '%s'\n", vm_name);
        fprintf(stderr, "Found UUIDs: %s\n", line);
        fprintf(stderr, "Please use --vm-uuid to specify which VM to register.\n");
        return -1;
    }
    
    /* Extract UUID (should be exactly one) */
    strncpy(first_uuid, line, 36);
    first_uuid[36] = '\0';
    
    /* Validate UUID format (basic check - 36 chars with dashes) */
    if (strlen(first_uuid) < 36) {
        return -1;
    }
    
    strncpy(uuid, first_uuid, uuid_size - 1);
    uuid[uuid_size - 1] = '\0';
    
    return 0;
}

static int get_vm_power_state(const char *vm_uuid, char *state, size_t state_size) {
    char cmd[512];
    FILE *fp;
    
    snprintf(cmd, sizeof(cmd), "xe vm-param-get uuid=%s param-name=power-state 2>/dev/null", vm_uuid);
    
    fp = popen(cmd, "r");
    if (!fp) {
        state[0] = '\0';
        return -1;
    }
    
    if (fgets(state, state_size, fp) == NULL) {
        pclose(fp);
        state[0] = '\0';
        return -1;
    }
    
    pclose(fp);
    
    /* Remove newline */
    size_t len = strlen(state);
    if (len > 0 && state[len-1] == '\n') {
        state[len-1] = '\0';
    }
    
    return 0;
}

static void print_help(void) {
    printf("vGPU Administration Tool\n");
    printf("Usage: vgpu-admin <command> [options]\n\n");
    printf("Commands:\n");
    printf("  scan-vms                      Scan VMs, grouped by pool (A, B, then unregistered)\n");
    printf("  register-vm                   Register a new VM\n");
    printf("    (--vm-uuid=<uuid> | --vm-name=<name>)  VM identifier (required, use one)\n");
    printf("    [--pool=<A|B>]              Pool assignment (default: A)\n");
    printf("    [--priority=<low|medium|high>] Priority (default: medium)\n");
    printf("    [--vm-id=<id>]              VM ID (default: auto-assign)\n");
    printf("    Examples:\n");
    printf("      vgpu-admin register-vm --vm-uuid=abc123...\n");
    printf("      vgpu-admin register-vm --vm-name=\"My VM\"\n");
    printf("  show-vm --vm-uuid=<uuid>      Show VM configuration\n");
    printf("  list-vms [--pool=<A|B>] [--priority=<low|medium|high>]  List VMs\n");
    printf("  set-pool --vm-uuid=<uuid> --pool=<A|B>  Change VM pool (requires restart)\n");
    printf("  set-priority --vm-uuid=<uuid> --priority=<low|medium|high>  Change priority (requires restart)\n");
    printf("  set-vm-id --vm-uuid=<uuid> --vm-id=<id>  Change VM ID (requires restart)\n");
    printf("  update-vm --vm-uuid=<uuid> [--pool=<A|B>] [--priority=<low|medium|high>] [--vm-id=<id>]  Update multiple settings (requires restart)\n");
    printf("  remove-vm (--vm-uuid=<uuid> | --vm-name=<name>)  Remove VM (stops if running, detaches vGPU-stub, removes from database)\n");
    printf("  list-pools                    List all pools\n");
    printf("  show-pool --pool-id=<A|B>     Show pool details\n");
    printf("  status                        Show system status\n");
    printf("  help                          Show this help message\n\n");
    printf("Note: Commands that change pool/priority/vm_id require VM restart and will ask for confirmation.\n");
}

static void print_vm_config(const vgpu_vm_config_t *config) {
    printf("VM Configuration:\n");
    printf("  UUID: %s\n", config->vm_uuid);
    if (config->vm_name[0]) {
        printf("  Name: %s\n", config->vm_name);
    }
    printf("  VM ID: %d\n", config->vm_id);
    printf("  Pool: %c\n", config->pool_id);
    printf("  Priority: %s (%d)\n", priority_str(config->priority), config->priority);
    printf("  Created: %s\n", config->created_at);
    printf("  Updated: %s\n", config->updated_at);
}

static void print_pool_info(const vgpu_pool_info_t *pool) {
    printf("Pool %c: %s\n", pool->pool_id, pool->pool_name);
    if (pool->description[0]) {
        printf("  Description: %s\n", pool->description);
    }
    printf("  Status: %s\n", pool->enabled ? "Enabled" : "Disabled");
    printf("  VMs: %d\n", pool->vm_count);
    printf("  Created: %s\n", pool->created_at);
    printf("  Updated: %s\n", pool->updated_at);
}

static const char *priority_str(int priority) {
    switch (priority) {
        case 0: return "low";
        case 1: return "medium";
        case 2: return "high";
        default: return "unknown";
    }
}

static int parse_priority(const char *str) {
    if (strcmp(str, "low") == 0) return 0;
    if (strcmp(str, "medium") == 0) return 1;
    if (strcmp(str, "high") == 0) return 2;
    return -1;
}

static char parse_pool_id(const char *str) {
    if (strlen(str) == 0) return 0;
    char c = toupper(str[0]);
    if (c == 'A' || c == 'B') return c;
    return 0;
}
