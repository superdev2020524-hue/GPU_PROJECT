# FATBIN long-run trace — poll record

**Purpose:** Append-only log of VM + host snapshots taken **every ~10 minutes** during a long generate / load test.

**Timeouts (VM `vgpu.conf`):** `OLLAMA_LOAD_TIMEOUT=2h`, `CUDA_TRANSPORT_TIMEOUT_SEC=7200`.

**Do not edit** past sample blocks — append only.

## Session — 2026-03-23 (assistant)

- **VM:** `OLLAMA_LOAD_TIMEOUT=2h`, `CUDA_TRANSPORT_TIMEOUT_SEC=7200` in `vgpu.conf`; backup `vgpu.conf.bak.fatbin-*`; Ollama restarted (`systemctl kill` + `start` pattern).
- **Long generate:** `nohup curl -m 7200` → **`/tmp/fatbin_long_gen.out`**, PID file **`/tmp/fatbin_long_gen.pid`** (curl PID **455026** on VM at start).
- **Polling:** `fatbin_trace_poll_loop.sh` — **13** samples, **10 min** apart (~**2 h** total), appends below.

### Confirmed — TINY model + test live (2026-03-23)

- **Model:** **`tinyllama:latest`** (small / “TINY” stack for this project).
- **Request:** `POST /api/generate`, `prompt=trace`, **`num_predict=16`**, **`curl -m 7200`** (2 h client cap).
- **VM process:** `curl` **PID 455026** still running; **`/tmp/fatbin_long_gen.out`** (response body — often **0 bytes** until completion).
- **Polling:** Background **`fatbin_trace_poll_loop.sh`** continues appending **Sample 2…13** here every **~10 min**.
- **Note:** “Perfect” tracing depends on the stack (network, mediator, driver); this run captures **trend** lines for HtoD / `401312` / errors.

---
### Note: poll loop restarted 2026-03-23T09:19:43Z after fixing script line endings

---
## Poll loop started 2026-03-23T09:19:43Z (local)


### Sample 1 — 2026-03-23T09:19:44Z UTC
#### VM — ollama journal (filtered tail)
Connecting to test-4@10.25.33.12... (attempt 1/3)
Sending password...
Connected successfully!
Running command: 
sudo journalctl -u ollama -n 80 --no-pager 2>/dev/null | grep -E "model load progress|load_tensors|ERROR|cuda-transport|runner terminated|context canceled|llama runner|sched.go" | tail -35
echo "---VM unfiltered tail (last 8 lines)---"
sudo journalctl -u ollama -n 8 --no-pager 2>/dev/null

Output:
 for test-4: 
Mar 23 05:17:35 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: [cuda-transport] mmap shmem 256 MB failed: Resource temporarily unavailable
Mar 23 05:17:35 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: [cuda-transport] Cannot resolve GPA for shmem (need CAP_SYS_ADMIN or /proc/self/pagemap access) — using BAR1
Mar 23 05:17:35 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: [cuda-transport] Connected (vm_id=9) data_path=BAR1 status_from=BAR1
Mar 23 05:17:35 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: [cuda-transport] poll call_id=0x0030 seq=1 iter=1 status=0x01 from=BAR1
Mar 23 05:17:35 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: load_tensors: offloading 22 repeating layers to GPU
Mar 23 05:17:35 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: load_tensors: offloading output layer to GPU
Mar 23 05:17:35 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: load_tensors: offloaded 23/23 layers to GPU
Mar 23 05:17:35 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: load_tensors:        CUDA0 model buffer size =   571.37 MiB
Mar 23 05:17:35 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: load_tensors:    CUDA_Host model buffer size =    35.16 MiB
Mar 23 05:17:35 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: [cuda-transport] poll call_id=0x0032 seq=7 iter=1 status=0x01 from=BAR1
Mar 23 05:17:35 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:17:35.181-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.00"
Mar 23 05:17:43 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: [cuda-transport] poll call_id=0x0032 seq=8 iter=1 status=0x01 from=BAR1
---VM unfiltered tail (last 8 lines)---
Mar 23 05:17:35 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: load_tensors: offloading output layer to GPU
Mar 23 05:17:35 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: load_tensors: offloaded 23/23 layers to GPU
Mar 23 05:17:35 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: load_tensors:        CUDA0 model buffer size =   571.37 MiB
Mar 23 05:17:35 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: load_tensors:    CUDA_Host model buffer size =    35.16 MiB
Mar 23 05:17:35 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: load_all_data: using async uploads for device CUDA0, buffer type CUDA0, backend CUDA0
Mar 23 05:17:35 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: [cuda-transport] poll call_id=0x0032 seq=7 iter=1 status=0x01 from=BAR1
Mar 23 05:17:35 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:17:35.181-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.00"
Mar 23 05:17:43 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: [cuda-transport] poll call_id=0x0032 seq=8 iter=1 status=0x01 from=BAR1


Remote command exit code: 0

#### Host — mediator.log (line count + HtoD / module / 401312)
 ( 
> echo "wc:"; wc -l /tmp/mediator.log 2>/dev/null
> grep -E "HtoD progress|401312|INVALID_IMAGE|module-load|CUDA_ERROR" /tmp/me
<1312|INVALID_IMAGE|module-load|CUDA_ERROR" /tmp/med                         iator.log 2>/dev/null | t

<load|CUDA_ERROR" /tmp/mediator.log 2>/dev/null | ta                         il -30
>  ); __rc=$?; printf '\n__CONNECT_HOST_DONE__:%s\n' "$__rc"
wc:
370 /tmp/mediator.log
[cuda-executor] HtoD progress: 10 MB total (vm=9)



--- end sample 1 ---

### Sample 2 — 2026-03-23T09:30:00Z UTC
#### VM — ollama journal (filtered tail)
Connecting to test-4@10.25.33.12... (attempt 1/3)
Sending password...
Connected successfully!
Running command: 
sudo journalctl -u ollama -n 80 --no-pager 2>/dev/null | grep -E "model load progress|load_tensors|ERROR|cuda-transport|runner terminated|context canceled|llama runner|sched.go" | tail -35
echo "---VM unfiltered tail (last 8 lines)---"
sudo journalctl -u ollama -n 8 --no-pager 2>/dev/null

Output:
 for test-4: 
Mar 23 05:17:35 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: [cuda-transport] mmap shmem 256 MB failed: Resource temporarily unavailable
Mar 23 05:17:35 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: [cuda-transport] Cannot resolve GPA for shmem (need CAP_SYS_ADMIN or /proc/self/pagemap access) — using BAR1
Mar 23 05:17:35 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: [cuda-transport] Connected (vm_id=9) data_path=BAR1 status_from=BAR1
Mar 23 05:17:35 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: [cuda-transport] poll call_id=0x0030 seq=1 iter=1 status=0x01 from=BAR1
Mar 23 05:17:35 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: load_tensors: offloading 22 repeating layers to GPU
Mar 23 05:17:35 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: load_tensors: offloading output layer to GPU
Mar 23 05:17:35 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: load_tensors: offloaded 23/23 layers to GPU
Mar 23 05:17:35 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: load_tensors:        CUDA0 model buffer size =   571.37 MiB
Mar 23 05:17:35 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: load_tensors:    CUDA_Host model buffer size =    35.16 MiB
Mar 23 05:17:35 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: [cuda-transport] poll call_id=0x0032 seq=7 iter=1 status=0x01 from=BAR1
Mar 23 05:17:35 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:17:35.181-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.00"
Mar 23 05:17:43 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: [cuda-transport] poll call_id=0x0032 seq=8 iter=1 status=0x01 from=BAR1
Mar 23 05:25:15 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:25:15.203-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.08"
Mar 23 05:25:35 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:25:35.567-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.09"
Mar 23 05:25:38 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:25:38.081-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.09"
Mar 23 05:25:40 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:25:40.343-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.09"
Mar 23 05:26:00 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:26:00.706-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.09"
Mar 23 05:26:00 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:26:00.957-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.09"
Mar 23 05:26:57 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:26:57.516-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.10"
Mar 23 05:27:54 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:27:54.073-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.11"
Mar 23 05:28:50 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:28:50.121-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.12"
Mar 23 05:29:10 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:29:10.732-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.13"
Mar 23 05:29:13 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:29:13.247-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.13"
Mar 23 05:29:15 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:29:15.761-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.13"
Mar 23 05:29:36 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:29:36.373-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.13"
---VM unfiltered tail (last 8 lines)---
Mar 23 05:26:00 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:26:00.957-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.09"
Mar 23 05:26:57 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:26:57.516-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.10"
Mar 23 05:27:54 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:27:54.073-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.11"
Mar 23 05:28:50 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:28:50.121-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.12"
Mar 23 05:29:10 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:29:10.732-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.13"
Mar 23 05:29:13 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:29:13.247-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.13"
Mar 23 05:29:15 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:29:15.761-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.13"
Mar 23 05:29:36 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:29:36.373-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.13"


Remote command exit code: 0

#### Host — mediator.log (line count + HtoD / module / 401312)
 ( 
> echo "wc:"; wc -l /tmp/mediator.log 2>/dev/null
> grep -E "HtoD progress|401312|INVALID_IMAGE|module-load|CUDA_ERROR" /tmp/me<1312|INVALID_IMAGE|module-load|CUDA_ERROR" /tmp/med                         iator.log 2>/dev/null | t<load|CUDA_ERROR" /tmp/mediator.log 2>/dev/null | ta                         il -30
>  ); __rc=$?; printf '\n__CONNECT_HOST_DONE__:%s\n' "$__rc"
wc:
691 /tmp/mediator.log
[cuda-executor] HtoD progress: 10 MB total (vm=9)
[cuda-executor] HtoD progress: 20 MB total (vm=9)
[cuda-executor] HtoD progress: 30 MB total (vm=9)
[cuda-executor] HtoD progress: 40 MB total (vm=9)
[cuda-executor] HtoD progress: 50 MB total (vm=9)
[cuda-executor] HtoD progress: 60 MB total (vm=9)
[cuda-executor] HtoD progress: 70 MB total (vm=9)
[cuda-executor] HtoD progress: 80 MB total (vm=9)



--- end sample 2 ---

### Sample 3 — 2026-03-23T09:40:13Z UTC
#### VM — ollama journal (filtered tail)
Connecting to test-4@10.25.33.12... (attempt 1/3)
Sending password...
Connected successfully!
Running command: 
sudo journalctl -u ollama -n 80 --no-pager 2>/dev/null | grep -E "model load progress|load_tensors|ERROR|cuda-transport|runner terminated|context canceled|llama runner|sched.go" | tail -35
echo "---VM unfiltered tail (last 8 lines)---"
sudo journalctl -u ollama -n 8 --no-pager 2>/dev/null

Output:
 for test-4: 
Mar 23 05:26:00 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:26:00.706-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.09"
Mar 23 05:26:00 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:26:00.957-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.09"
Mar 23 05:26:57 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:26:57.516-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.10"
Mar 23 05:27:54 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:27:54.073-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.11"
Mar 23 05:28:50 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:28:50.121-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.12"
Mar 23 05:29:10 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:29:10.732-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.13"
Mar 23 05:29:13 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:29:13.247-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.13"
Mar 23 05:29:15 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:29:15.761-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.13"
Mar 23 05:29:36 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:29:36.373-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.13"
Mar 23 05:30:32 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:30:32.942-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.14"
Mar 23 05:31:28 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: [cuda-transport] poll call_id=0x0032 seq=114 iter=1 status=0x01 from=BAR1
Mar 23 05:31:28 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:31:28.753-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.15"
Mar 23 05:32:24 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:32:24.308-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.16"
Mar 23 05:32:44 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:32:44.667-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.17"
Mar 23 05:32:47 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:32:47.180-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.17"
Mar 23 05:32:49 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:32:49.945-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.17"
Mar 23 05:33:10 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:33:10.306-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.17"
Mar 23 05:34:06 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:34:06.863-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.18"
Mar 23 05:35:02 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:35:02.923-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.19"
Mar 23 05:35:59 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:35:59.980-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.20"
Mar 23 05:36:20 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:36:20.587-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.21"
Mar 23 05:36:23 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:36:23.099-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.21"
Mar 23 05:36:25 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:36:25.613-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.21"
Mar 23 05:36:45 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:36:45.975-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.21"
Mar 23 05:36:46 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:36:46.226-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.21"
Mar 23 05:37:42 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: [cuda-transport] poll call_id=0x0032 seq=169 iter=1 status=0x01 from=BAR1
Mar 23 05:37:42 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:37:42.287-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.22"
Mar 23 05:38:38 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:38:38.337-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.23"
Mar 23 05:39:34 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:39:34.628-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.24"
Mar 23 05:39:54 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:39:54.238-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.24"
Mar 23 05:39:56 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:39:56.751-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.24"
Mar 23 05:39:59 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:39:59.265-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.25"
Mar 23 05:40:19 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:40:19.367-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.25"
Mar 23 05:40:19 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: [cuda-transport] poll call_id=0x0032 seq=193 iter=1 status=0x01 from=BAR1
Mar 23 05:40:19 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:40:19.618-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.25"
---VM unfiltered tail (last 8 lines)---
Mar 23 05:38:38 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:38:38.337-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.23"
Mar 23 05:39:34 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:39:34.628-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.24"
Mar 23 05:39:54 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:39:54.238-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.24"
Mar 23 05:39:56 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:39:56.751-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.24"
Mar 23 05:39:59 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:39:59.265-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.25"
Mar 23 05:40:19 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:40:19.367-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.25"
Mar 23 05:40:19 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: [cuda-transport] poll call_id=0x0032 seq=193 iter=1 status=0x01 from=BAR1
Mar 23 05:40:19 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:40:19.618-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.25"


Remote command exit code: 0

#### Host — mediator.log (line count + HtoD / module / 401312)
 ( 
> echo "wc:"; wc -l /tmp/mediator.log 2>/dev/null
> grep -E "HtoD progress|401312|INVALID_IMAGE|module-load|CUDA_ERROR" /tmp/me<1312|INVALID_IMAGE|module-load|CUDA_ERROR" /tmp/med                         iator.log 2>/dev/null | t<load|CUDA_ERROR" /tmp/mediator.log 2>/dev/null | ta                         il -30
>  ); __rc=$?; printf '\n__CONNECT_HOST_DONE__:%s\n' "$__rc"
wc:
1030 /tmp/mediator.log
[cuda-executor] HtoD progress: 10 MB total (vm=9)
[cuda-executor] HtoD progress: 20 MB total (vm=9)
[cuda-executor] HtoD progress: 30 MB total (vm=9)
[cuda-executor] HtoD progress: 40 MB total (vm=9)
[cuda-executor] HtoD progress: 50 MB total (vm=9)
[cuda-executor] HtoD progress: 60 MB total (vm=9)
[cuda-executor] HtoD progress: 70 MB total (vm=9)
[cuda-executor] HtoD progress: 80 MB total (vm=9)
[cuda-executor] HtoD progress: 91 MB total (vm=9)
[cuda-executor] HtoD progress: 101 MB total (vm=9)
[cuda-executor] HtoD progress: 111 MB total (vm=9)
[cuda-executor] HtoD progress: 122 MB total (vm=9)
[cuda-executor] HtoD progress: 132 MB total (vm=9)
[cuda-executor] HtoD progress: 142 MB total (vm=9)



--- end sample 3 ---

### Sample 4 — 2026-03-23T09:50:28Z UTC
#### VM — ollama journal (filtered tail)
Connecting to test-4@10.25.33.12... (attempt 1/3)
Sending password...
Connected successfully!
Running command: 
sudo journalctl -u ollama -n 80 --no-pager 2>/dev/null | grep -E "model load progress|load_tensors|ERROR|cuda-transport|runner terminated|context canceled|llama runner|sched.go" | tail -35
echo "---VM unfiltered tail (last 8 lines)---"
sudo journalctl -u ollama -n 8 --no-pager 2>/dev/null

Output:
 for test-4: 
Mar 23 05:36:20 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:36:20.587-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.21"
Mar 23 05:36:23 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:36:23.099-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.21"
Mar 23 05:36:25 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:36:25.613-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.21"
Mar 23 05:36:45 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:36:45.975-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.21"
Mar 23 05:36:46 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:36:46.226-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.21"
Mar 23 05:37:42 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: [cuda-transport] poll call_id=0x0032 seq=169 iter=1 status=0x01 from=BAR1
Mar 23 05:37:42 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:37:42.287-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.22"
Mar 23 05:38:38 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:38:38.337-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.23"
Mar 23 05:39:34 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:39:34.628-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.24"
Mar 23 05:39:54 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:39:54.238-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.24"
Mar 23 05:39:56 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:39:56.751-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.24"
Mar 23 05:39:59 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:39:59.265-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.25"
Mar 23 05:40:19 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:40:19.367-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.25"
Mar 23 05:40:19 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: [cuda-transport] poll call_id=0x0032 seq=193 iter=1 status=0x01 from=BAR1
Mar 23 05:40:19 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:40:19.618-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.25"
Mar 23 05:41:15 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:41:15.932-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.26"
Mar 23 05:42:09 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:42:09.982-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.27"
Mar 23 05:43:03 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:43:03.264-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.28"
Mar 23 05:43:23 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:43:23.629-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.28"
Mar 23 05:43:26 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:43:26.143-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.28"
Mar 23 05:43:28 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:43:28.657-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.28"
Mar 23 05:43:48 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:43:48.519-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.29"
Mar 23 05:44:44 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:44:44.823-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.30"
Mar 23 05:45:40 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:45:40.117-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.31"
Mar 23 05:46:34 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:46:34.657-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.32"
Mar 23 05:46:55 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:46:55.268-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.32"
Mar 23 05:46:57 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:46:57.782-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.32"
Mar 23 05:47:00 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:47:00.295-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.32"
Mar 23 05:47:20 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:47:20.648-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.33"
Mar 23 05:47:20 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:47:20.899-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.33"
Mar 23 05:48:16 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:48:16.946-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.34"
Mar 23 05:49:13 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:49:13.005-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.35"
Mar 23 05:50:09 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:50:09.320-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.36"
Mar 23 05:50:29 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:50:29.929-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.36"
Mar 23 05:50:32 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:50:32.441-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.36"
---VM unfiltered tail (last 8 lines)---
Mar 23 05:47:00 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:47:00.295-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.32"
Mar 23 05:47:20 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:47:20.648-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.33"
Mar 23 05:47:20 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:47:20.899-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.33"
Mar 23 05:48:16 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:48:16.946-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.34"
Mar 23 05:49:13 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:49:13.005-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.35"
Mar 23 05:50:09 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:50:09.320-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.36"
Mar 23 05:50:29 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:50:29.929-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.36"
Mar 23 05:50:32 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:50:32.441-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.36"


Remote command exit code: 0

#### Host — mediator.log (line count + HtoD / module / 401312)
 ( 
> echo "wc:"; wc -l /tmp/mediator.log 2>/dev/null
> grep -E "HtoD progress|401312|INVALID_IMAGE|module-load|CUDA_ERROR" /tmp/me<1312|INVALID_IMAGE|module-load|CUDA_ERROR" /tmp/med                         iator.log 2>/dev/null | t<load|CUDA_ERROR" /tmp/mediator.log 2>/dev/null | ta                         il -30
>  ); __rc=$?; printf '\n__CONNECT_HOST_DONE__:%s\n' "$__rc"
wc:
1358 /tmp/mediator.log
[cuda-executor] HtoD progress: 10 MB total (vm=9)
[cuda-executor] HtoD progress: 20 MB total (vm=9)
[cuda-executor] HtoD progress: 30 MB total (vm=9)
[cuda-executor] HtoD progress: 40 MB total (vm=9)
[cuda-executor] HtoD progress: 50 MB total (vm=9)
[cuda-executor] HtoD progress: 60 MB total (vm=9)
[cuda-executor] HtoD progress: 70 MB total (vm=9)
[cuda-executor] HtoD progress: 80 MB total (vm=9)
[cuda-executor] HtoD progress: 91 MB total (vm=9)
[cuda-executor] HtoD progress: 101 MB total (vm=9)
[cuda-executor] HtoD progress: 111 MB total (vm=9)
[cuda-executor] HtoD progress: 122 MB total (vm=9)
[cuda-executor] HtoD progress: 132 MB total (vm=9)
[cuda-executor] HtoD progress: 142 MB total (vm=9)
[cuda-executor] HtoD progress: 152 MB total (vm=9)
[cuda-executor] HtoD progress: 163 MB total (vm=9)
[cuda-executor] HtoD progress: 173 MB total (vm=9)
[cuda-executor] HtoD progress: 183 MB total (vm=9)
[cuda-executor] HtoD progress: 194 MB total (vm=9)
[cuda-executor] HtoD progress: 204 MB total (vm=9)
[cuda-executor] HtoD progress: 214 MB total (vm=9)



--- end sample 4 ---

### Sample 5 — 2026-03-23T10:00:40Z UTC
#### VM — ollama journal (filtered tail)
Connecting to test-4@10.25.33.12... (attempt 1/3)
Sending password...
Connected successfully!
Running command: 
sudo journalctl -u ollama -n 80 --no-pager 2>/dev/null | grep -E "model load progress|load_tensors|ERROR|cuda-transport|runner terminated|context canceled|llama runner|sched.go" | tail -35
echo "---VM unfiltered tail (last 8 lines)---"
sudo journalctl -u ollama -n 8 --no-pager 2>/dev/null

Output:
 for test-4: 
Mar 23 05:43:28 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:43:28.657-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.28"
Mar 23 05:43:48 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:43:48.519-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.29"
Mar 23 05:44:44 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:44:44.823-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.30"
Mar 23 05:45:40 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:45:40.117-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.31"
Mar 23 05:46:34 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:46:34.657-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.32"
Mar 23 05:46:55 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:46:55.268-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.32"
Mar 23 05:46:57 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:46:57.782-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.32"
Mar 23 05:47:00 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:47:00.295-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.32"
Mar 23 05:47:20 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:47:20.648-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.33"
Mar 23 05:47:20 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:47:20.899-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.33"
Mar 23 05:48:16 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:48:16.946-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.34"
Mar 23 05:49:13 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:49:13.005-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.35"
Mar 23 05:50:09 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:50:09.320-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.36"
Mar 23 05:50:29 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:50:29.929-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.36"
Mar 23 05:50:32 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:50:32.441-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.36"
Mar 23 05:50:35 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:50:35.205-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.36"
Mar 23 05:50:55 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:50:55.564-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.37"
Mar 23 05:51:51 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:51:51.857-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.38"
Mar 23 05:52:00 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: [cuda-transport] poll call_id=0x0032 seq=294 iter=1 status=0x01 from=BAR1
Mar 23 05:52:47 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:52:47.909-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.39"
Mar 23 05:53:42 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:53:42.958-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.40"
Mar 23 05:54:03 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:54:03.569-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.40"
Mar 23 05:54:06 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:54:06.083-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.40"
Mar 23 05:54:08 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:54:08.597-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.40"
Mar 23 05:54:29 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:54:29.209-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.40"
Mar 23 05:55:25 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:55:25.005-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.41"
Mar 23 05:56:20 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:56:20.551-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.43"
Mar 23 05:57:16 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:57:16.352-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.44"
Mar 23 05:57:16 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:57:16.604-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.44"
Mar 23 05:57:37 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:57:37.215-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.44"
Mar 23 05:57:39 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:57:39.730-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.44"
Mar 23 05:57:42 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:57:42.244-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.44"
Mar 23 05:58:02 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:58:02.352-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.44"
Mar 23 05:58:58 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:58:58.155-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.45"
Mar 23 05:59:54 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:59:54.466-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.46"
---VM unfiltered tail (last 8 lines)---
Mar 23 05:57:16 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:57:16.352-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.44"
Mar 23 05:57:16 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:57:16.604-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.44"
Mar 23 05:57:37 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:57:37.215-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.44"
Mar 23 05:57:39 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:57:39.730-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.44"
Mar 23 05:57:42 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:57:42.244-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.44"
Mar 23 05:58:02 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:58:02.352-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.44"
Mar 23 05:58:58 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:58:58.155-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.45"
Mar 23 05:59:54 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:59:54.466-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.46"


Remote command exit code: 0

#### Host — mediator.log (line count + HtoD / module / 401312)
 ( 
> echo "wc:"; wc -l /tmp/mediator.log 2>/dev/null
> grep -E "HtoD progress|401312|INVALID_IMAGE|module-load|CUDA_ERROR" /tmp/me<1312|INVALID_IMAGE|module-load|CUDA_ERROR" /tmp/med                         iator.log 2>/dev/null | t<load|CUDA_ERROR" /tmp/mediator.log 2>/dev/null | ta                         il -30
>  ); __rc=$?; printf '\n__CONNECT_HOST_DONE__:%s\n' "$__rc"
wc:
1687 /tmp/mediator.log
[cuda-executor] HtoD progress: 10 MB total (vm=9)
[cuda-executor] HtoD progress: 20 MB total (vm=9)
[cuda-executor] HtoD progress: 30 MB total (vm=9)
[cuda-executor] HtoD progress: 40 MB total (vm=9)
[cuda-executor] HtoD progress: 50 MB total (vm=9)
[cuda-executor] HtoD progress: 60 MB total (vm=9)
[cuda-executor] HtoD progress: 70 MB total (vm=9)
[cuda-executor] HtoD progress: 80 MB total (vm=9)
[cuda-executor] HtoD progress: 91 MB total (vm=9)
[cuda-executor] HtoD progress: 101 MB total (vm=9)
[cuda-executor] HtoD progress: 111 MB total (vm=9)
[cuda-executor] HtoD progress: 122 MB total (vm=9)
[cuda-executor] HtoD progress: 132 MB total (vm=9)
[cuda-executor] HtoD progress: 142 MB total (vm=9)
[cuda-executor] HtoD progress: 152 MB total (vm=9)
[cuda-executor] HtoD progress: 163 MB total (vm=9)
[cuda-executor] HtoD progress: 173 MB total (vm=9)
[cuda-executor] HtoD progress: 183 MB total (vm=9)
[cuda-executor] HtoD progress: 194 MB total (vm=9)
[cuda-executor] HtoD progress: 204 MB total (vm=9)
[cuda-executor] HtoD progress: 214 MB total (vm=9)
[cuda-executor] HtoD progress: 224 MB total (vm=9)
[cuda-executor] HtoD progress: 235 MB total (vm=9)
[cuda-executor] HtoD progress: 245 MB total (vm=9)
[cuda-executor] HtoD progress: 255 MB total (vm=9)
[cuda-executor] HtoD progress: 266 MB total (vm=9)
[cuda-executor] HtoD progress: 276 MB total (vm=9)
[cuda-executor] HtoD progress: 286 MB total (vm=9)



--- end sample 5 ---

### Sample 6 — 2026-03-23T10:10:54Z UTC
#### VM — ollama journal (filtered tail)
Connecting to test-4@10.25.33.12... (attempt 1/3)
Sending password...
Connected successfully!
Running command: 
sudo journalctl -u ollama -n 80 --no-pager 2>/dev/null | grep -E "model load progress|load_tensors|ERROR|cuda-transport|runner terminated|context canceled|llama runner|sched.go" | tail -35
echo "---VM unfiltered tail (last 8 lines)---"
sudo journalctl -u ollama -n 8 --no-pager 2>/dev/null

Output:
 for test-4: 
Mar 23 05:56:20 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:56:20.551-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.43"
Mar 23 05:57:16 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:57:16.352-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.44"
Mar 23 05:57:16 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:57:16.604-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.44"
Mar 23 05:57:37 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:57:37.215-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.44"
Mar 23 05:57:39 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:57:39.730-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.44"
Mar 23 05:57:42 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:57:42.244-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.44"
Mar 23 05:58:02 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:58:02.352-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.44"
Mar 23 05:58:58 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:58:58.155-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.45"
Mar 23 05:59:54 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T05:59:54.466-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.46"
Mar 23 06:00:50 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T06:00:50.261-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.47"
Mar 23 06:01:10 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T06:01:10.628-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.48"
Mar 23 06:01:13 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T06:01:13.393-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.48"
Mar 23 06:01:15 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T06:01:15.653-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.48"
Mar 23 06:01:36 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T06:01:36.265-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.48"
Mar 23 06:02:32 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T06:02:32.065-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.49"
Mar 23 06:03:28 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T06:03:28.123-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.50"
Mar 23 06:03:37 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: [cuda-transport] poll call_id=0x0032 seq=394 iter=1 status=0x01 from=BAR1
Mar 23 06:04:23 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T06:04:23.678-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.51"
Mar 23 06:04:44 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T06:04:44.041-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.52"
Mar 23 06:04:46 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T06:04:46.554-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.52"
Mar 23 06:04:49 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T06:04:49.319-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.52"
Mar 23 06:05:08 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T06:05:08.423-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.52"
Mar 23 06:05:08 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T06:05:08.675-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.52"
Mar 23 06:06:04 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T06:06:04.476-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.53"
Mar 23 06:07:00 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T06:07:00.526-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.54"
Mar 23 06:07:45 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: [cuda-transport] poll call_id=0x0032 seq=429 iter=1 status=0x01 from=BAR1
Mar 23 06:07:56 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T06:07:56.831-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.55"
Mar 23 06:07:57 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T06:07:57.083-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.55"
Mar 23 06:08:17 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T06:08:17.693-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.56"
Mar 23 06:08:20 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T06:08:20.207-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.56"
Mar 23 06:08:22 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T06:08:22.720-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.56"
Mar 23 06:08:43 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T06:08:43.339-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.56"
Mar 23 06:09:10 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: [cuda-transport] poll call_id=0x0032 seq=444 iter=1 status=0x01 from=BAR1
Mar 23 06:09:39 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T06:09:39.399-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.57"
Mar 23 06:10:35 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T06:10:35.692-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.58"
---VM unfiltered tail (last 8 lines)---
Mar 23 06:07:57 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T06:07:57.083-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.55"
Mar 23 06:08:17 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T06:08:17.693-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.56"
Mar 23 06:08:20 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T06:08:20.207-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.56"
Mar 23 06:08:22 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T06:08:22.720-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.56"
Mar 23 06:08:43 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T06:08:43.339-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.56"
Mar 23 06:09:10 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: [cuda-transport] poll call_id=0x0032 seq=444 iter=1 status=0x01 from=BAR1
Mar 23 06:09:39 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T06:09:39.399-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.57"
Mar 23 06:10:35 test4-HVM-domU ollama_vgpu_wrapper.sh[454955]: time=2026-03-23T06:10:35.692-04:00 level=DEBUG source=server.go:1474 msg="model load progress 0.58"


Remote command exit code: 0

#### Host — mediator.log (line count + HtoD / module / 401312)
 ( 
> echo "wc:"; wc -l /tmp/mediator.log 2>/dev/null
> grep -E "HtoD progress|401312|INVALID_IMAGE|module-load|CUDA_ERROR" /tmp/me<1312|INVALID_IMAGE|module-load|CUDA_ERROR" /tmp/med                         iator.log 2>/dev/null | t<load|CUDA_ERROR" /tmp/mediator.log 2>/dev/null | ta                         il -30
>  ); __rc=$?; printf '\n__CONNECT_HOST_DONE__:%s\n' "$__rc"
wc:
2013 /tmp/mediator.log
[cuda-executor] HtoD progress: 50 MB total (vm=9)
[cuda-executor] HtoD progress: 60 MB total (vm=9)
[cuda-executor] HtoD progress: 70 MB total (vm=9)
[cuda-executor] HtoD progress: 80 MB total (vm=9)
[cuda-executor] HtoD progress: 91 MB total (vm=9)
[cuda-executor] HtoD progress: 101 MB total (vm=9)
[cuda-executor] HtoD progress: 111 MB total (vm=9)
[cuda-executor] HtoD progress: 122 MB total (vm=9)
[cuda-executor] HtoD progress: 132 MB total (vm=9)
[cuda-executor] HtoD progress: 142 MB total (vm=9)
[cuda-executor] HtoD progress: 152 MB total (vm=9)
[cuda-executor] HtoD progress: 163 MB total (vm=9)
[cuda-executor] HtoD progress: 173 MB total (vm=9)
[cuda-executor] HtoD progress: 183 MB total (vm=9)
[cuda-executor] HtoD progress: 194 MB total (vm=9)
[cuda-executor] HtoD progress: 204 MB total (vm=9)
[cuda-executor] HtoD progress: 214 MB total (vm=9)
[cuda-executor] HtoD progress: 224 MB total (vm=9)
[cuda-executor] HtoD progress: 235 MB total (vm=9)
[cuda-executor] HtoD progress: 245 MB total (vm=9)
[cuda-executor] HtoD progress: 255 MB total (vm=9)
[cuda-executor] HtoD progress: 266 MB total (vm=9)
[cuda-executor] HtoD progress: 276 MB total (vm=9)
[cuda-executor] HtoD progress: 286 MB total (vm=9)
[cuda-executor] HtoD progress: 296 MB total (vm=9)
[cuda-executor] HtoD progress: 307 MB total (vm=9)
[cuda-executor] HtoD progress: 317 MB total (vm=9)
[cuda-executor] HtoD progress: 327 MB total (vm=9)
[cuda-executor] HtoD progress: 337 MB total (vm=9)
[cuda-executor] HtoD progress: 348 MB total (vm=9)



--- end sample 6 ---
