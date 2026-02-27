# Manual Constructor Fix Instructions

## Date: 2026-02-26

## Problem

The constructor in `libvgpu_cuda.c` only initializes if it detects an application process via LD_PRELOAD. When the shim is loaded via symlinks in the runner subprocess (which doesn't have LD_PRELOAD), the constructor doesn't initialize, so device count functions return 0.

## Fix Required

In `~/phase3/guest-shim/libvgpu_cuda.c`, around line 2237, replace:

```c
        } else {
            /* Try the normal check as fallback */
            if (is_safe_to_check_process()) {
                is_app = is_application_process();
                if (is_app) {
                    const char *app_msg = "[libvgpu-cuda] constructor: Application process detected (via normal check)\n";
                    syscall(__NR_write, 2, app_msg, strlen(app_msg));
                }
            }
        }
```

With:

```c
        } else {
            /* Check for OLLAMA environment variables - runner processes have these */
            const char *ollama_lib = getenv("OLLAMA_LLM_LIBRARY");
            const char *ollama_path = getenv("OLLAMA_LIBRARY_PATH");
            if (ollama_lib || ollama_path) {
                /* OLLAMA environment variables present - this is likely Ollama/runner process */
                is_app = 1;
                const char *ollama_msg = "[libvgpu-cuda] constructor: Ollama process detected (via OLLAMA env vars), initializing\n";
                syscall(__NR_write, 2, ollama_msg, strlen(ollama_msg));
            } else {
                /* Try the normal check as fallback */
                if (is_safe_to_check_process()) {
                    is_app = is_application_process();
                    if (is_app) {
                        const char *app_msg = "[libvgpu-cuda] constructor: Application process detected (via normal check)\n";
                        syscall(__NR_write, 2, app_msg, strlen(app_msg));
                    }
                }
            }
        }
```

## Manual Application Steps

1. SSH to VM: `ssh test-10@10.25.33.110`
2. Edit file: `cd ~/phase3/guest-shim && nano libvgpu_cuda.c`
3. Go to line 2237 (or search for "Try the normal check as fallback")
4. Replace the section as shown above
5. Save and exit
6. Rebuild: `sudo bash install.sh`
7. Restart Ollama: `sudo systemctl restart ollama`

## Why This Works

Runner subprocesses have `OLLAMA_LLM_LIBRARY` and `OLLAMA_LIBRARY_PATH` environment variables set. When the shim is loaded via symlinks in the runner, the constructor will detect these variables and initialize, ensuring device count functions return 1.
