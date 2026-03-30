# Work note: Runner never reports Ready — loadModel blocker (Mar 19, 2026)

## What was done

1. **Instrumented** the runner’s load path so we can see where it blocks:
   - **ollamarunner/runner.go:** Added slog + file writes at start of `loadModel()` and after `Backend().Load()` returns.
   - **llamarunner/runner.go:** For tinyllama (GGUF) the **llama** runner is used, not ollama-engine. Added file writes at the start of `loadModel()` and immediately after `llama.LoadModelFromFile(mpath, params)` returns.

2. **Rebuilt** ollama on the VM, installed as `ollama.bin.new`, restarted the service.

3. **Triggered** one generate (tinyllama, 55s client timeout) and then checked the marker files on the VM.

## Result

- **`/tmp/phase3_loadmodel_started.txt`** — **present**, content `llama\n`. So the **llama** runner’s `loadModel()` goroutine **is** entered (first line runs).
- **`/tmp/phase3_loadmodel_returned.txt`** — **absent**. So **`llama.LoadModelFromFile(mpath, params)`** has **not** returned within 55 seconds.

So the runner stays in “loading model” and never reports Ready because it **blocks inside the C call `llama.LoadModelFromFile()`**. That call is the GGML/llama.cpp load over the vGPU path (alloc/HtoD, etc.). It either never returns or takes longer than the client timeout.

## Conclusion

- **Blocker:** The **llama** runner’s `loadModel()` blocks in **`llama.LoadModelFromFile()`** (CGo). Until that call returns, the runner never sets `s.status = llm.ServerStatusReady`, so the server never sees Ready and keeps logging “waiting for server to become available” status="llm server loading model".
- **Cause:** Either (1) the C/GGML load over vGPU is very slow (e.g. HtoD or host work), (2) something in that path blocks indefinitely (e.g. transport wait, host module load failure, or deadlock), or (3) the call eventually returns but only after a long time (e.g. >40m).

## 20-minute run (Mar 19)

- Generate with **client timeout 1200s** (20 min): **phase3_loadmodel_returned.txt** still **absent**; curl exited 28 (timeout). So LoadModelFromFile did **not** return within 20 min.
- **Host mediator:** For vm=9, HtoD progress reached **1215+ MB** and was still advancing (cuMemAlloc SUCCESS, HtoD progress lines). Module-load lines in log showed **rc=0** (CUDA_SUCCESS) for an earlier run.
- So either HtoD is very slow and the full model needs >20 min, or something after HtoD (e.g. another sync or call) blocks. No response body was written (long_gen.json empty).

## Next step (recommended)

- **In the C/GGML/vGPU path:** Inspect what happens during model load: HtoD progress, host `cuModuleLoadFatBinary` (e.g. INVALID_IMAGE if no sm_90), and any blocking wait in the transport or host response. See CURRENT_STATE_AND_DIRECTION.md (module load INVALID_IMAGE) and BUILD_LIBGGML_CUDA_HOPPER.md.
- **Optional:** Run with client timeout **40m** (or set OLLAMA_LOAD_TIMEOUT=40m and ensure client does not disconnect) to see if LoadModelFromFile eventually returns once HtoD completes.

## Anti-destructive verification

- Only additive instrumentation (file writes and existing slog in ollamarunner). No existing logic removed. GPU mode and service config unchanged.
