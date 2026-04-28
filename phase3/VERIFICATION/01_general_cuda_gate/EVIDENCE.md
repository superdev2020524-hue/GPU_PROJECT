# Evidence - Milestone 01 General CUDA Gate

## Evidence Rule

Every runtime conclusion must include correlated host and VM evidence from the
same session.

## Current Session Evidence

Date: 2026-04-27 host/VM local time, requested 2026-04-28 KST.

Milestone: `01_general_cuda_gate`.

Plan A report: `/tmp/phase1_milestone_gate_before_m01.json`.

Plan A result: `overall_pass=True`.

VM `/api/ps` after Plan A: `{"models":[]}`.

### Baseline Commands

```bash
# Plan A canary
python3 phase1_milestone_gate.py \
  --base-url http://10.25.33.110:11434 \
  --timeout-sec 600 \
  --output /tmp/phase1_milestone_gate_before_m01.json
```

### Host Evidence To Capture

```bash
pgrep -a mediator_phase3
stat /tmp/mediator.log
grep -E 'cuLaunchKernel SUCCESS|sync FAILED|CUDA_ERROR_ILLEGAL_ADDRESS|result.status=801|FAILED' /tmp/mediator.log | tail -40
```

### VM Evidence To Capture

```bash
systemctl is-active ollama
curl -s http://127.0.0.1:11434/api/ps
lspci -nn | grep -i '10de\|nvidia'
ls -la /opt/vgpu/lib
```

## First Gate Evidence

First run:

- Report: `/tmp/phase3_general_cuda_gate_report.json`
- Result: fail
- Build on VM: pass
- First run: fail at `cuLaunchKernel`

Rerun after correcting the probe's `kernelParams` array to be NULL-terminated:

- Report: `/tmp/phase3_general_cuda_gate_report_rerun1.json`
- Result: fail
- Build on VM: pass
- First run: fail at `cuLaunchKernel`

Passing cases before the failure:

```text
CASE device_discovery             OP cuInit                       RC 0
CASE device_discovery             OP cuDeviceGetCount             RC 0
DEVICE_COUNT 1
CASE device_discovery             OP cuDeviceGet                  RC 0
CASE device_discovery             OP cuDeviceGetName              RC 0
CASE device_discovery             OP cuDeviceTotalMem_v2          RC 0
DEVICE_NAME HEXACORE vH100 CAP
DEVICE_TOTAL_MEM 85899345920
CASE context                      OP cuDevicePrimaryCtxRetain     RC 0
CASE context                      OP cuCtxSetCurrent              RC 0
CASE alloc_free                   OP cuMemAlloc_v2                RC 0
CASE copy_htod                    OP cuMemcpyHtoD_v2              RC 0
CASE copy_dtoh                    OP cuMemcpyDtoH_v2              RC 0
COPY_ROUNDTRIP_OK bytes=256
CASE stream                       OP cuStreamCreate               RC 0
CASE event                        OP cuEventCreate                RC 0
CASE module                       OP cuModuleLoadData             RC 0
CASE module                       OP cuModuleGetFunction          RC 0
```

Failing case:

```text
CASE kernel                       OP cuLaunchKernel               RC 1
KERNEL_MISMATCH idx=0 got=0 expected=1
```

VM failure signature:

```text
[cuda-transport] STATUS_ERROR: call=cuFuncGetParamInfo(0x00bc) seq=9 err=0x00000005(CUDA_ERROR) vm_id=10
[cuda-transport] STATUS_ERROR host-cuda: call=cuFuncGetParamInfo(0x00bc) seq=9 host_status=0x00000321
[cuda-transport] STATUS_ERROR: call=cuLaunchKernel(0x0050) seq=10 err=0x00000005(CUDA_ERROR) vm_id=10
[cuda-transport] STATUS_ERROR host-cuda: call=cuLaunchKernel(0x0050) seq=10 host_status=0x00000001
```

Host failure signature:

```text
[cuda-executor] vm_id=10 module-load done call_id=0x0040 rc=0 name=CUDA_SUCCESS detail=no error module=0x1ff4c0f0
[cuda-executor] cuLaunchKernel launch FAILED: rc=1 func=0x1ffa4ac0 name=add_one mod=0x1ff4c0f0 grid=(1,1,1) block=(64,1,1) shared=0 params=0 vm=10
[cuda-executor] call FAILED: vm=10 call=cuLaunchKernel(0x0050) rc=1(CUDA_ERROR_INVALID_VALUE) detail=invalid argument
```

## Fix Attempt 1

File changed:

- `guest-shim/libvgpu_cuda.c`

Change:

- In `cuLaunchKernel`, if the first `cuFuncGetParamInfo` call fails for a
  generic function, keep the scanned `kernelParams` count and use the legacy
  parameter path instead of serializing zero parameters.

Status:

- Initial deployment with `python3 transfer_libvgpu_cuda.py` failed during
  base64 chunk transfer through `connect_vm.py`, before build/install.
- Follow-up deployment used `scp`, built `/tmp/libvgpu-cuda.so.1`, installed it
  into the VM library locations, and restarted `ollama`.
- VM service after install: `active`.

Verification:

- Raw CUDA gate report:
  `/tmp/phase3_general_cuda_gate_after_m01_e1_fix.json`.
- Result: `overall_pass=True`.
- Repetitions: 2/2 passed.
- Plan A recheck report:
  `/tmp/phase1_milestone_gate_after_m01_e1_fix.json`.
- Plan A result: `overall_pass=True`.

Closure evidence:

```text
CASE kernel                       OP cuLaunchKernel               RC 0
KERNEL_RESULT_OK n=64
OVERALL PASS
```

Host closure evidence:

```text
[cuda-executor] cuLaunchKernel SUCCESS: kernel executed on physical GPU (func=0x1ffa4ac0 name=add_one mod=0x1ff4c0f0 vm=10)
[cuda-executor] cuLaunchKernel SUCCESS: kernel executed on physical GPU (func=0x1ffa4ac0 name=add_one mod=0x1ff4c0f0 vm=10)
```

## Runtime API Expansion

Files added:

- `tests/general_cuda_gate/runtime_api_probe.c`

Runner updated:

- `tests/general_cuda_gate/run_general_cuda_gate.py`

Initial expanded gate:

- Report: `/tmp/phase3_general_cuda_gate_runtime_expansion.json`
- Result: fail
- Driver API probe: pass, 2/2
- Runtime API probe: build pass, run fail
- First failure: missing `cudaEventCreate` symbol.

Fix:

- Added `cudaEventCreate()` wrapper in `guest-shim/libvgpu_cudart.c`.
- Deployed rebuilt Runtime shim to `/opt/vgpu/lib/libvgpu-cudart.so`.

Expanded gate after Runtime fix:

- Report:
  `/tmp/phase3_general_cuda_gate_runtime_expansion_after_event_fix.json`
- Result: `overall_pass=True`
- Driver API probe: pass, 2/2
- Runtime API probe: pass, 2/2

Plan A regression after Runtime deployment:

- Report: `/tmp/phase1_milestone_gate_after_m01_runtime_expansion.json`
- Result: `overall_pass=False`
- Symptom: all Plan A generate cases returned HTTP 500 in about 10-16 seconds.
- Root cause: deployment copied the vGPU Runtime shim through Ollama's
  `libcudart.so.12` symlink and overwrote
  `/usr/local/lib/ollama/cuda_v12/libcudart.so.12.8.90`.

Recovery:

- Preserved overwritten file:
  `/usr/local/lib/ollama/cuda_v12/libcudart.so.12.8.90.bad-vgpu-cudart-20260427-1751`
- Restored `/usr/local/lib/ollama/cuda_v12/libcudart.so.12.8.90` from the
  existing local CUDA 13 runtime copy.
- Restarted `ollama`.

Recovery proof:

- Plan A report:
  `/tmp/phase1_milestone_gate_after_runtime_restore_attempt.json`
- Plan A result: `overall_pass=True`
- Expanded gate report:
  `/tmp/phase3_general_cuda_gate_runtime_expansion_after_planA_recovery.json`
- Expanded gate result: `overall_pass=True`

Passing expanded gate cases:

```text
copy_driver_api_probe_to_vm      pass
build_driver_api_probe           pass
driver_api_probe_run_1           pass
driver_api_probe_run_2           pass
copy_runtime_api_probe_to_vm     pass
build_runtime_api_probe          pass
runtime_api_probe_run_1          pass
runtime_api_probe_run_2          pass
```

## Hidden-Risk Sweep

Path isolation audit:

```text
/opt/vgpu/lib/libcudart.so.12 -> /opt/vgpu/lib/libvgpu-cudart.so
/usr/local/lib/ollama/cuda_v12/libcudart.so.12 -> libcudart.so.12.8.90
/usr/local/lib/ollama/cuda_v12/libcudart.so.12.8.90 size=704288
/usr/local/lib/ollama/cuda_v12/libcuda.so.1 -> /opt/vgpu/lib/libcuda.so.1
/usr/lib64/libcudart.so.12 -> libvgpu-cudart.so
```

Interpretation:

- Standalone probes can use the vGPU Runtime shim through `/opt/vgpu/lib`.
- Ollama's CUDA v12 runtime path is separated again and no longer points to the
  45 KB vGPU Runtime shim.
- `/usr/lib64/libcudart.so.12` still points to the vGPU Runtime shim and remains
  a candidate deployment-scope risk for future non-Ollama applications.

Symbol audit:

```text
libvgpu-cuda.so.1 exports cuInit, cuDeviceGetCount, cuDeviceGetName,
cuDeviceTotalMem_v2, cuDevicePrimaryCtxRetain, cuCtxSetCurrent,
cuMemAlloc_v2, cuMemcpyHtoD_v2, cuMemcpyDtoH_v2, cuStreamCreate,
cuEventCreate, cuModuleLoadData, cuModuleGetFunction, cuLaunchKernel.

libvgpu-cudart.so exports cudaGetDeviceCount, cudaGetDeviceProperties,
cudaSetDevice, cudaMalloc, cudaFree, cudaMemcpy, cudaMemcpyAsync,
cudaDeviceSynchronize, cudaStreamCreate, cudaStreamDestroy,
cudaStreamSynchronize, cudaEventCreate, cudaEventDestroy, cudaEventRecord,
cudaEventSynchronize.
```

Second kernel shape:

- Added `scale_add` to `driver_api_probe.c`.
- It uses four parameters: source pointer, destination pointer, element count,
  and scale.
- Initial hidden-risk run failed because the expected value did not account for
  the earlier `add_one` mutation of the source device buffer.
- Corrected expectation to `(input[i] + 1) * scale + i`.

Passing hidden-risk sweep:

- Report: `/tmp/phase3_general_cuda_gate_hidden_risk_sweep_rerun1.json`
- Result: `overall_pass=True`
- Driver API probe: 5/5 pass
- Runtime API probe: 5/5 pass

Plan A after hidden-risk sweep:

- Report: `/tmp/phase1_milestone_gate_after_m01_hidden_risk_sweep.json`
- Result: `overall_pass=True`

## Candidate Evidence

Candidate `0x00bc` remains candidate-only. It was present in host evidence
during the pre-Milestone Plan A window, but Plan A completed successfully.

Host evidence after Plan A:

```text
mediator_lines 496773
KEY cuLaunchKernel SUCCESS: kernel executed on physical GPU count=86928
KEY sync FAILED count=0
KEY CUDA_ERROR_ILLEGAL_ADDRESS count=0
KEY result.status=801 count=17
KEY FAILED count=93
```

Representative host candidate lines:

```text
[MEDIATOR] CUDA result sent vm_id=10 request_id=149890 call_id=0xbc result.status=801 -> stub sets DONE
[cuda-executor] call FAILED: vm=10 call=cuFuncGetParamInfo(0x00bc) rc=801(CUDA_ERROR_NOT_SUPPORTED) detail=operation not supported
```

VM evidence after Plan A:

```text
{"models":[]}
```
