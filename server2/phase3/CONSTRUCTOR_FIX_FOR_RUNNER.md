# Constructor Fix for Runner Processes

## Date: 2026-02-26

## Problem

The constructor in `libvgpu_cuda.c` only initializes if it detects an application process. When the shim is loaded via symlinks in the runner subprocess (which doesn't have LD_PRELOAD), the constructor doesn't initialize, so device count functions return 0.

## Fix Applied

Modified the constructor to check for OLLAMA environment variables (`OLLAMA_LLM_LIBRARY` or `OLLAMA_LIBRARY_PATH`). If these are set, treat it as an Ollama/runner process and initialize.

### Code Change

**Before:**
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

**After:**
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

## Why This Works

Runner subprocesses have `OLLAMA_LLM_LIBRARY` and `OLLAMA_LIBRARY_PATH` environment variables set (from discovery logs). When the shim is loaded via symlinks in the runner, the constructor will detect these variables and initialize, ensuring device count functions return 1.

## Next Steps

1. Copy fixed `libvgpu_cuda.c` to VM
2. Rebuild `libvgpu-cuda.so`
3. Restart Ollama
4. Verify constructor initializes for runner
5. Check if device count > 0
