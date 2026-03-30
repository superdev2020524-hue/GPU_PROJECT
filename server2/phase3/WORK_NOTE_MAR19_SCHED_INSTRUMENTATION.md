# Work note: Sched instrumentation and verification (Mar 19, 2026)

## What was done

1. **Applied** `patch_sched_after_loading_first_model.py` on the VM to add a file write in `server/sched.go` immediately after the "loading first model" log line. **No existing code or Phase3 logic was removed or overwritten** — only one new line was inserted.

2. **Rebuilt** ollama on the VM with `/usr/local/go/bin/go build -o ollama.bin .` in `/home/test-4/ollama`. **No change** to service config, env, or other patches (device.go, server.go, discover/runner.go, etc.).

3. **Installed** the new binary as `/usr/local/bin/ollama.bin.new` (the service’s existing target) and restarted ollama.

4. **Anti-regression check:** After restart, confirmed Ollama is still in **GPU mode**: journal shows `inference compute` with `library=CUDA`, `description="NVIDIA H100 80GB HBM3"`, `total="80.0 GiB"`, and no "filtering device which didn't fully initialize". **No destructive impact** on discovery or existing behavior.

5. **Triggered** one generate (tinyllama, 90s timeout) and then checked Phase3 instrumentation files and vgpu call sequence.

## Result

- **phase3_sched_load_entered.txt** — present (sched load entered).
- **phase3_sched_after_loading_first_model.txt** — present (execution reaches the line after "loading first model").
- **phase3_before_llama_load.txt** — present, multiple entries (sched reaches the line before `llama.Load()` and calls it).
- **phase3_load_path.txt** — present with `llama_load` and `ollama_load` (both server Load paths invoked).
- **vgpu_call_sequence.log** — **763** lines matching 0x0030|0x0032 (alloc/HtoD), so the load path is issuing CUDA alloc and HtoD via the shims.

So the previous “block before llama.Load()” is no longer observed in this run: the scheduler reaches the load path and we see alloc/HtoD. The run still ended with **"timed out waiting for llama runner to start: context canceled"** in the journal (sched.go:575). So the current failure is the **runner start/health timeout**, not “sched never reaches llama.Load()”.

## Next focus

- **Runner start/health:** Investigate why the server times out waiting for the llama runner to start (e.g. `waitUntilRunnerLaunched` / `getServerStatus`, runner HTTP not ready, or port/process mismatch). The load path and alloc/HtoD are active; the blocker has moved to runner readiness/health check.

## Anti-destructive verification

- **GPU mode:** Still on (library=CUDA, 80 GiB in journal).
- **Service:** Unchanged (ExecStart=/usr/local/bin/ollama.bin.new serve; LD_LIBRARY_PATH with cuda_v12 and /opt/vgpu/lib).
- **Patches:** Only one line added in sched.go; no existing Phase3 instrumentation or logic removed.
