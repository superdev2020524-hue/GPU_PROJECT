# Ollama patch: pass LD_PRELOAD to runner

## Why

The Ollama server starts a **runner** subprocess for GPU discovery and inference. On some builds (e.g. snap or older code), the runner is started with an environment that does **not** include `LD_PRELOAD` or `LD_LIBRARY_PATH`, so the runner never loads our vGPU shims and discovery reports CPU.

## Fix

Patch `llm/server.go` so that right before starting the runner we **ensure** `LD_PRELOAD`, `LD_LIBRARY_PATH`, `OLLAMA_LIBRARY_PATH`, and `OLLAMA_LLM_LIBRARY` from the current process are in `cmd.Env`. The patch is in `phase3/patches/ollama_runner_ld_preload.patch`. Passing `OLLAMA_LIBRARY_PATH` (e.g. `/usr/local/lib/ollama:/usr/local/lib/ollama/cuda_v12`) is required so the runner loads `libggml-cuda.so` from `cuda_v12` and discovery can see the GPU via our shims.

## Apply and build (on a machine with Go)

1. **Clone Ollama**
   ```bash
   git clone https://github.com/ollama/ollama.git
   cd ollama
   ```

2. **Apply the patch**
   ```bash
   # From the gpu repo root (or adjust path)
   patch -p1 < /path/to/gpu/phase3/patches/ollama_runner_ld_preload.patch
   ```
   If the patch reports offset or fuzz, the Ollama tree may have changed; apply the same logic by hand in `llm/server.go` in `StartRunner`, right before `slog.Info("starting runner", ...)`.

3. **Build**
   ```bash
   go build -o ollama .
   ```

4. **Install on the VM**
   - Copy the built `ollama` binary to the VM (e.g. as `/usr/local/bin/ollama.bin`).
   - Keep the existing `ollama.service.d/vgpu.conf` (LD_PRELOAD, LD_LIBRARY_PATH, etc.).
   - Restart: `sudo systemctl restart ollama.service`.

5. **Verify**
   - After restart, trigger discovery (e.g. `ollama run llama3.2:1b 'Hi'` or just list models).
   - Check logs: `journalctl -u ollama.service -n 30 --no-pager | grep -E 'inference compute|total_vram|library='`
   - You should see `library=cuda` (or similar) and `total_vram` non-zero if the runner now loads the shims.

### Optional: confirm runner has LD_PRELOAD (on VM)

The runner process is short-lived (starts at server boot for discovery, then exits). To try to capture its environment:

1. SSH to the VM and run a poll loop in the background that dumps any process with `runner` in cmdline:
   ```bash
   ( for i in $(seq 1 20); do
     for p in $(sudo pgrep -f ollama.bin); do
       sudo cat /proc/$p/cmdline 2>/dev/null | tr '\0' ' ' | grep -q " runner " || continue
       echo "Runner PID=$p"
       sudo cat /proc/$p/environ 2>/dev/null | tr '\0' '\n' | grep -E "LD_PRELOAD|LD_LIBRARY"
       exit 0
     done
     sleep 0.5
   done
   echo "No runner found in 10s" ) &
   ```
2. In another terminal (or after a short sleep), restart the service so a new runner is spawned:
   ```bash
   sudo systemctl restart ollama.service
   ```
3. Wait for the loop to finish, then check the output. If you see `LD_PRELOAD=/opt/vgpu/lib/...`, the patch is passing env to the runner.

## Patch content (manual apply)

If `patch` fails, add this block in `llm/server.go` inside `StartRunner`, after the `for k, done := range extraEnvsDone { ... }` block and before `slog.Info("starting runner", ...)`:

```go
	// Ensure runner inherits LD_PRELOAD and LD_LIBRARY_PATH from current process
	// (e.g. for vGPU shims in guest VM; some builds filter env and drop these)
	for _, key := range []string{"LD_PRELOAD", "LD_LIBRARY_PATH"} {
		if v, ok := os.LookupEnv(key); ok && v != "" {
			found := false
			for i := range cmd.Env {
				if strings.HasPrefix(cmd.Env[i], key+"=") {
					cmd.Env[i] = key + "=" + v
					found = true
					break
				}
			}
			if !found {
				cmd.Env = append(cmd.Env, key+"="+v)
			}
		}
	}
```

## Note

Upstream `llm/server.go` already sets `cmd.Env = os.Environ()`, so in theory the runner gets the server’s env. Packaged or snap builds may use an older or patched tree that filters env; this patch forces `LD_PRELOAD` and `LD_LIBRARY_PATH` through regardless.

## Applied in this repo

**`transfer_ollama_go_patches.py`** applies this logic when patching `llm/server.go`: it injects the "ensure runner inherits LD_PRELOAD and LD_LIBRARY_PATH" block (step 3 in `patch_server_go`). Run that script with an Ollama tree at `phase3/ollama-src` (with `ml/device.go`, `llm/server.go`, `discover/runner.go`) to transfer patched files to the VM, build `ollama.bin`, and install. After that, the runner will receive `LD_PRELOAD` from the server, so the vGPU shim's `write()` interceptor can capture runner stderr to `/tmp/ollama_errors_full.log` for error tracking.
