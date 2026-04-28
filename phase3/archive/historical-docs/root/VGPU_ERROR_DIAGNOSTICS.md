# vGPU Error Diagnostics — Accurate Error Tracking

Every failure writes a debug report. Use `/tmp/vgpu_debug.txt` for exact diagnosis.

## Primary: Debug report

**`/tmp/vgpu_debug.txt`** — Full diagnostic report written on any failure. Contains:

- **WHAT FAILED** — Error code
- **FAILING CALL** — CUDA call name (e.g. cuDevicePrimaryCtxRetain) and call_id
- **TRANSPORT ERR** — Transport-level error code
- **RECENT CALL HISTORY** — Last ~24 transport calls with result (OK/ERR)
- **CHECKPOINT TRAIL** — Init phases reached (INIT_START → DEVICE_FOUND → … → FIRST_CALL)
- **LIKELY CAUSE** — Actionable hint

Example: `cat /tmp/vgpu_debug.txt`

## Secondary files

| File | Purpose |
|------|---------|
| `/tmp/vgpu_last_error` | Single-line canonical format for scripts |
| `/tmp/vgpu_checkpoint` | Last checkpoint phase |
| `/tmp/vgpu_status` | CUBLAS mode: STUB or REAL |

Files are cleared at the start of each runner's `ensure_connected()`.

## Error codes (`vgpu_last_error`)

| Code | Meaning |
|------|---------|
| `DEVICE_NOT_FOUND` | VGPU-STUB not in /sys/bus/pci/devices |
| `BAR0_OPEN_FAILED` | Cannot open resource0 |
| `BAR0_MMAP_FAILED` | Cannot mmap BAR0 |
| `TRANSPORT_TIMEOUT` | Poll timed out waiting for host response |
| `MEDIATOR_UNAVAIL` | Host mediator not running or not responding |
| `QUEUE_FULL` | Host queue full |
| `RATE_LIMITED` | Host rate-limited this VM |
| `VM_QUARANTINED` | VM quarantined by host |
| `CUDA_CALL_FAILED` | Transport OK but host returned CUDA error (see `transport_err`) |
| `CUBLAS_CREATE_FAILED` | Real CUBLAS rejected context (from CUBLAS shim) |

## Checkpoint phases (for hang diagnosis)

| Phase | Meaning |
|-------|---------|
| `INIT_START` | About to scan for device |
| `DEVICE_FOUND` | PCI device found |
| `BAR0_OPENED` | resource0 opened |
| `TRANSPORT_READY` | BAR0 mapped, transport ready |
| `FIRST_CALL` | First RPC sent to host |

If the process hangs: the last checkpoint shows where it stopped.

## Outcome determination

1. **Success (host path)**  
   - No `vgpu_last_error` (or stale timestamp)  
   - `vgpu_status` has `VGPU_CUBLAS_MODE=REAL`  
   - Ollama returns 200

2. **Success (VM stub path)**  
   - No `vgpu_last_error`  
   - `vgpu_status` has `VGPU_CUBLAS_MODE=STUB`  
   - Degraded mode, no host used

3. **Failure**  
   - `cat /tmp/vgpu_last_error` → exact code, call_id, detail

4. **Hang**  
   - `cat /tmp/vgpu_checkpoint` → last phase reached before stall
