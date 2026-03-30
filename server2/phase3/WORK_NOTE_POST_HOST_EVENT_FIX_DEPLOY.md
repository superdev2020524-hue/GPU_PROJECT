# Post-deploy check (Mar 19): host executor fix + VM probe

## Host (user terminal) — OK

- `scp` of `src/cuda_executor.c` to `root@10.25.33.10:/root/phase3/src/` succeeded.
- `grep CUDA_ERROR_INVALID_HANDLE` matched expected lines in `cuda_executor.c`.
- `make mediator_phase3` succeeded (cuda_executor.o rebuilt, mediator linked).
- Mediator restarted: `pgrep` shows PID **563312** `./mediator_phase3`.
- `mediator.log` tail: heartbeats, **1** server socket: `/var/xen/qemu/root-235/tmp/vgpu-mediator.sock`.

## Note: single socket

Earlier logs sometimes showed **two** sockets (e.g. root-232 and root-235). After this restart only **root-235** appears. If a second VM/domain uses 232, ensure that guest is running or expect only one active vGPU path until both QEMU instances expose their sockets.

## VM (assistant `connect_vm.py`)

- `ollama`: **active**; `GET /api/tags` → **200**.
- **2 min** `tinyllama` generate: client ended with **HTTP=000** (typical when load does not finish before `curl -m 120`).
- `sudo` read of shim logs:
  - `/tmp/vgpu_call_sequence.log` ends with many **0x0071 cuEventCreateWithFlags** and **0x0030 cuMemAlloc_v2** (expected for GGML load).
  - `/tmp/vgpu_current_call.txt`: **cuMemAlloc_v2 seq=6** (snapshot while blocked or after partial run).

## Journal

- Still see **llama runner terminated / exit status 2** with **rip …969fc** (same **0x969fc** instruction offset as prior crash, different ASLR base). That points to a **native crash in the same code** (likely GGML/CUDA `.so`), not fixed by returning proper CUDA errors for bad handles on the host.

## Next steps

1. **Long client timeout** again (e.g. 40 min) after host fix; watch for **CUDA_INVALID_HANDLE** in guest journal or `vgpu_debug.txt` if the silent-success path was hit.
2. **Coredump + gdb** on the VM (`CRASH_SYMBOLICATION_AND_COREDUMPS.md`) to map **rip+0x969fc** to a symbol in `libggml-cuda` / `libllama` / runner.
3. Optional: on host during load, `grep -E 'INVALID_HANDLE|cuda-executor.*rc=' /tmp/mediator.log` to confirm the executor is not now returning success on bad event handles (would show as explicit errors if triggered).
