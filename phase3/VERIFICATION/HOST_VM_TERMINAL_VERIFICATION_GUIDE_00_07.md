# Host/VM Terminal Verification Guide: Milestones 00-07

This guide is for a client-facing video where commands are run from the host
terminal or directly inside VM terminals. It avoids the appearance that the
verification depends on a local development workstation.

Use three terminals:

1. Host terminal: `root@10.25.33.10`
2. VM-10 terminal: `test-10@10.25.33.110`
3. VM-6 terminal: `test-6@10.25.33.16`

Do not show passwords in the video.

## Video Framing

Use this short explanation before running commands:

> I am running these checks from the actual Server 1 host and the guest VMs. The
> goal is to show that the virtual GPU path works from inside the VM, through the
> host mediator, and back to the VM with verified results.

## Host Preflight

Run on the host:

```bash
hostname
date
pgrep -af mediator_phase3
ls -la /var/xen/qemu/root-*/tmp/vgpu-mediator.sock
```

Expected result:

- The host name is shown.
- `mediator_phase3` is running.
- At least the VM-10 and VM-6 mediator sockets are present.

## Host Mediator Log Proof

This is one of the most important parts of the video. The VM terminal proves that
the workload succeeded from inside the guest. The host mediator log proves that
the request crossed the virtualization boundary and was handled by the Server 1
GPU mediation layer.

### Real-Time Mediator Log View

Run this on the **host terminal** and leave it running while you execute CUDA,
PyTorch, CuPy, or TensorFlow commands inside VM-10 or VM-6:

```bash
tail -n 0 -F /tmp/mediator.log | grep --line-buffered -E \
  'vm_id=10|vm_id=6|CUDA result sent|cuLaunchKernel SUCCESS|CUDA process cleanup|Pool A processed|Pool B processed|sync FAILED|CUDA_ERROR_ILLEGAL_ADDRESS|Recovering primary context|RATE-LIMIT|WATCHDOG'
```

Use this when you want the video to show the host mediator reacting live to VM
workload activity. Start the command first on the host, then switch to VM-10 or
VM-6 and run the workload. New mediator log lines should appear in the host
terminal as the VM sends GPU work through the mediator.

Current Ollama warning:

The earlier VM-10 Ollama check answered correctly but did **not** produce new
host mediator CUDA traffic. A patched debug build proved that Ollama can reach
CUDA discovery and host mediator traffic, but the runner still terminates before
a successful response. The live service was restored to the previous functional
binary for demonstration safety. Do not present the current Ollama Plan A command
as a closed live host-mediated GPU proof. Use raw CUDA, PyTorch, CuPy, or
TensorFlow for closed mediator-log proof until `M00-E1` is fully closed.

If you want to show recent history first and then continue live, use:

```bash
tail -n 200 -F /tmp/mediator.log | grep --line-buffered -E \
  'vm_id=10|vm_id=6|CUDA result sent|cuLaunchKernel SUCCESS|CUDA process cleanup|Pool A processed|Pool B processed|sync FAILED|CUDA_ERROR_ILLEGAL_ADDRESS|Recovering primary context|RATE-LIMIT|WATCHDOG'
```

For an unfiltered raw live view, use:

```bash
tail -n 0 -F /tmp/mediator.log
```

What to say:

> This host terminal is following the mediator log in real time. When I run a
> verified CUDA, PyTorch, CuPy, or TensorFlow workload inside VM-10 or VM-6, the
> host log shows the VM-owned CUDA requests arriving at the mediator and being
> executed through the physical GPU path.

Expected live signals:

- VM-10 work appears with `vm_id=10`.
- VM-6 work appears with `vm_id=6`.
- Successful kernel execution appears as `cuLaunchKernel SUCCESS`.
- Cleanup appears as `CUDA process cleanup`.
- A clean demonstration should not show new `sync FAILED`,
  `CUDA_ERROR_ILLEGAL_ADDRESS`, or `Recovering primary context` lines.

Run on the host before starting the milestone checks:

```bash
LOG=/tmp/mediator.log

test -f "$LOG" && echo "mediator log exists: $LOG"
wc -l "$LOG"
pgrep -af mediator_phase3
```

Then show a short live summary of the important mediator signals:

```bash
LOG=/tmp/mediator.log

python3 - <<'PY'
from pathlib import Path

log = Path("/tmp/mediator.log")
lines = log.read_text(errors="replace").splitlines() if log.exists() else []

keys = [
    "CUDA result sent vm_id=10",
    "CUDA result sent vm_id=6",
    "cuLaunchKernel SUCCESS",
    "CUDA process cleanup",
    "Pool A processed",
    "Pool B processed",
    "sync FAILED",
    "CUDA_ERROR_ILLEGAL_ADDRESS",
    "Recovering primary context",
]

print("mediator_log_lines =", len(lines))
for key in keys:
    count = sum(1 for line in lines if key in line)
    print(f"{key}: {count}")
PY
```

For the video, also show a few recent mediator lines after running a VM workload:

```bash
LOG=/tmp/mediator.log

python3 - <<'PY'
from pathlib import Path

log = Path("/tmp/mediator.log")
lines = log.read_text(errors="replace").splitlines() if log.exists() else []

interesting = [
    line for line in lines
    if "vm_id=10" in line
    or "vm_id=6" in line
    or "cuLaunchKernel SUCCESS" in line
    or "CUDA process cleanup" in line
]

for line in interesting[-25:]:
    print(line)
PY
```

What to say:

> The VM output shows the application-level pass. The host mediator log is the
> server-side proof: it shows VM-owned CUDA requests reaching the mediator,
> successful physical GPU execution, cleanup, and separate VM ownership through
> `vm_id=10` and `vm_id=6`.

Expected result:

- The mediator log exists and has entries.
- `mediator_phase3` is still running.
- VM-10 activity appears as `vm_id=10`.
- VM-6 activity appears as `vm_id=6` after VM-6 checks.
- `cuLaunchKernel SUCCESS` increases after framework or CUDA workloads.
- `sync FAILED`, `CUDA_ERROR_ILLEGAL_ADDRESS`, and
  `Recovering primary context` should stay at `0` for a clean demonstration
  segment.

Recommended recording pattern:

1. Show this mediator summary before the workload.
2. Run the VM workload.
3. Return to the host terminal and show the mediator summary again.
4. Point out that the VM-side pass and host-side mediator activity line up.

## VM-10 Preflight

Run on VM-10:

```bash
hostname
date
lspci -nn | grep -i nvidia
ls -la /opt/vgpu/lib/libcuda.so.1 /opt/vgpu/lib/libvgpu-cuda.so.1
curl -s http://127.0.0.1:11434/api/ps
```

Expected result:

- VM-10 shows the vGPU PCI device.
- The vGPU CUDA shim exists.
- Ollama API responds.

## VM-6 Preflight

Run on VM-6:

```bash
hostname
date
lspci -nn | grep -i nvidia
ls -la /opt/vgpu/lib/libcuda.so.1 /opt/vgpu/lib/libvgpu-cuda.so.1
test -x /home/test-6/m06-cupy-venv/bin/python && echo "VM-6 CuPy venv is ready"
```

Expected result:

- VM-6 shows the vGPU PCI device.
- The vGPU CUDA shim exists.
- The VM-6 CuPy venv is available.

## 00 - Ollama Baseline

Run on VM-10:

```bash
cd /tmp/may4_reverify

python3 phase1_milestone_gate.py \
  --suite phase1_milestone_test_suite.json \
  --output /tmp/client_m00_planA.json

python3 phase1_plan_b_tiny_gate.py \
  --output /tmp/client_m00_planB.json

python3 phase1_plan_c_client_gate.py \
  --output /tmp/client_m00_planC.json

python3 - <<'PY'
import json
for path in [
    "/tmp/client_m00_planA.json",
    "/tmp/client_m00_planB.json",
    "/tmp/client_m00_planC.json",
]:
    report = json.load(open(path))
    print(path, "overall_pass =", report.get("overall_pass"))
PY
```

What to say:

> This proves the original Ollama API canary still answers correctly. This does
> not currently prove Ollama GPU mediation. The current live mediator-log proof
> should be shown with raw CUDA, PyTorch, CuPy, or TensorFlow until `M00-E1` is
> closed.

Expected result:

- All three reports print `overall_pass = True`.
- On the patched baseline, mediator CUDA traffic may appear, but `M00-E1` is not
  closed unless the same run also returns a successful Ollama response and
  resident GPU state.

## 01 - Raw CUDA

Run on VM-10:

```bash
export LD_LIBRARY_PATH=/opt/vgpu/lib:/usr/local/lib/ollama/cuda_v12:/usr/local/lib/ollama:${LD_LIBRARY_PATH:-}

for i in 1 2 3; do
  echo "Driver API run $i"
  /tmp/phase3_general_cuda_gate/driver_api_probe
done

for i in 1 2 3; do
  echo "Runtime API run $i"
  /tmp/phase3_general_cuda_gate/runtime_api_probe
done
```

What to say:

> This checks CUDA below any specific framework. If this fails, PyTorch,
> TensorFlow, CuPy, and Ollama results are not enough.

Expected result:

- The Driver API and Runtime API probe runs complete successfully.

## 02 - API Coverage And Executor Consistency

Run on the host:

```bash
cd /root/phase3

python3 - <<'PY'
import json
import re
from pathlib import Path

root = Path("/root/phase3")
proto = (root / "include/cuda_protocol.h").read_text(errors="replace")
executor = (root / "src/cuda_executor.c").read_text(errors="replace")

protocol_ids = {
    m.group(1)
    for m in re.finditer(
        r"^\s*#define\s+(CUDA_CALL_[A-Z0-9_]+)\s+(0x[0-9a-fA-F]+|\d+)\b",
        proto,
        re.M,
    )
    if m.group(1) != "CUDA_CALL_MAX"
}
executor_cases = set(re.findall(r"case\s+(CUDA_CALL_[A-Z0-9_]+)\s*:", executor))
missing = sorted(protocol_ids - executor_cases)

report = {
    "protocol_ids": len(protocol_ids),
    "executor_cases": len(executor_cases & protocol_ids),
    "missing_cases": missing,
    "overall_pass": len(protocol_ids) > 0 and not missing,
}
print(json.dumps(report, indent=2, sort_keys=True))
raise SystemExit(0 if report["overall_pass"] else 1)
PY
```

What to say:

> This confirms the host mediator has a defined behavior for every CUDA protocol
> call currently exposed by the vGPU layer.

Expected result:

- `overall_pass` is `true`.
- `missing_cases` is empty.

## 03 - Memory, Synchronization, And Cleanup

Run on VM-10:

```bash
export LD_LIBRARY_PATH=/opt/vgpu/lib:${LD_LIBRARY_PATH:-}

/tmp/20260504_reverify_async_stream_event_probe | tee /tmp/client_m03_async.log

(
  /tmp/20260504_reverify_forced_kill_alloc_probe \
    > /tmp/client_m03_kill_child.log 2>&1 &
  echo $! > /tmp/client_m03_kill_child.pid
)

for i in 1 2 3 4 5 6 7 8 9 10; do
  grep -q READY /tmp/client_m03_kill_child.log && break
  sleep 1
done

grep READY /tmp/client_m03_kill_child.log
kill -9 "$(cat /tmp/client_m03_kill_child.pid)" || true
sleep 2

/tmp/20260504_reverify_async_stream_event_probe \
  > /tmp/client_m03_post_kill_async.log
grep PASS /tmp/client_m03_post_kill_async.log
```

What to say:

> This proves memory copies, stream/event synchronization, and recovery after a
> killed GPU process.

Expected result:

- The async probe prints `ASYNC_STREAM_EVENT_PROBE PASS`.
- The killed child reaches `READY`.
- The post-kill async probe also prints `PASS`.

## 04 - PyTorch

Run on VM-10:

```bash
export LD_LIBRARY_PATH=/opt/vgpu/lib:${LD_LIBRARY_PATH:-}

/mnt/m04-pytorch/venv/bin/python /mnt/m04-pytorch/pytorch_probe.py \
  > /tmp/client_m04_pytorch.json

python3 - <<'PY'
import json
report = json.load(open("/tmp/client_m04_pytorch.json"))
print("PyTorch overall_pass =", report["overall_pass"])
print("device_name =", report.get("device_name"))
PY
```

What to say:

> This proves a real CUDA framework can allocate tensors, move data, run
> operations, and produce verified results through the mediated GPU path.

Expected result:

- `PyTorch overall_pass = True`

## 05 - CuPy And TensorFlow

Run on VM-10:

```bash
export LD_LIBRARY_PATH=/opt/vgpu/lib:${LD_LIBRARY_PATH:-}

/mnt/m04-pytorch/venv/bin/python /mnt/m04-pytorch/cupy_probe.py \
  > /tmp/client_m05_cupy.json

python3 - <<'PY'
import json
report = json.load(open("/tmp/client_m05_cupy.json"))
print("CuPy overall_pass =", report["overall_pass"])
PY
```

Then run TensorFlow on VM-10:

```bash
TFV=/mnt/m04-pytorch/tf123-venv
NLIB=$(find "$TFV/lib/python3.10/site-packages/nvidia" -type d -name lib 2>/dev/null | paste -sd: -)

LD_LIBRARY_PATH="$NLIB:/opt/vgpu/lib:${LD_LIBRARY_PATH:-}" \
TF_FORCE_GPU_ALLOW_GROWTH=true \
"$TFV/bin/python" /mnt/m04-pytorch/tensorflow_mnist_probe.py \
  > /tmp/client_m05_tensorflow.json

python3 - <<'PY'
import json
report = json.load(open("/tmp/client_m05_tensorflow.json"))
print("TensorFlow overall_pass =", report["overall_pass"])
print("used_gpu_for_training =", report.get("used_gpu_for_training"))
print("logical_gpus =", report.get("logical_gpus"))
PY
```

What to say:

> CuPy and TensorFlow exercise different CUDA and library-loading paths. Passing
> both is stronger than passing only one framework.

Expected result:

- `CuPy overall_pass = True`
- `TensorFlow overall_pass = True`
- `used_gpu_for_training = True`

## 06 - Multi-Process And Multi-VM

First run same-VM concurrency on VM-10:

```bash
export LD_LIBRARY_PATH=/opt/vgpu/lib:${LD_LIBRARY_PATH:-}

/mnt/m04-pytorch/venv/bin/python /mnt/m04-pytorch/two_process_cupy_probe.py \
  > /tmp/client_m06_two_process_cupy.json

/mnt/m04-pytorch/venv/bin/python /mnt/m04-pytorch/mixed_pytorch_cupy_probe.py \
  > /tmp/client_m06_mixed_pytorch_cupy.json

python3 - <<'PY'
import json
for path in [
    "/tmp/client_m06_two_process_cupy.json",
    "/tmp/client_m06_mixed_pytorch_cupy.json",
]:
    report = json.load(open(path))
    print(path, "overall_pass =", report["overall_pass"])
PY
```

Then run the VM-6 CuPy check on VM-6:

```bash
TFV=/home/test-6/m06-cupy-venv
NLIB=$(find "$TFV/lib/python3.10/site-packages/nvidia" -type d -name lib 2>/dev/null | paste -sd: -)

LD_LIBRARY_PATH="/opt/vgpu/lib:$NLIB:${LD_LIBRARY_PATH:-}" \
"$TFV/bin/python" /tmp/may4_reverify/cupy_probe.py \
  > /tmp/client_m06_vm6_cupy.json

python3 - <<'PY'
import json
report = json.load(open("/tmp/client_m06_vm6_cupy.json"))
print("VM-6 CuPy overall_pass =", report["overall_pass"])
PY
```

For a cross-VM demonstration, run these two commands at the same time in two VM
terminals:

VM-10:

```bash
export LD_LIBRARY_PATH=/opt/vgpu/lib:${LD_LIBRARY_PATH:-}
/mnt/m04-pytorch/venv/bin/python /mnt/m04-pytorch/cupy_probe.py \
  > /tmp/client_m06_cross_vm10_cupy.json
python3 -c 'import json; print(json.load(open("/tmp/client_m06_cross_vm10_cupy.json"))["overall_pass"])'
```

VM-6:

```bash
TFV=/home/test-6/m06-cupy-venv
NLIB=$(find "$TFV/lib/python3.10/site-packages/nvidia" -type d -name lib 2>/dev/null | paste -sd: -)
LD_LIBRARY_PATH="/opt/vgpu/lib:$NLIB:${LD_LIBRARY_PATH:-}" \
"$TFV/bin/python" /tmp/may4_reverify/cupy_probe.py \
  > /tmp/client_m06_cross_vm6_cupy.json
python3 -c 'import json; print(json.load(open("/tmp/client_m06_cross_vm6_cupy.json"))["overall_pass"])'
```

What to say:

> VM-10 is the full validation VM. VM-6 proves the second mediated VM path and
> cross-VM GPU sharing behavior.

Expected result:

- Same-VM VM-10 reports print `overall_pass = True`.
- VM-6 CuPy prints `overall_pass = True`.
- Both cross-VM CuPy commands print `True`.

## 07 - Security And Isolation

Run on the host:

```bash
SOCK=$(ls /var/xen/qemu/root-2/tmp/vgpu-mediator.sock 2>/dev/null || ls /var/xen/qemu/root-*/tmp/vgpu-mediator.sock | sed -n '1p')

python3 /tmp/20260504_reverify_malformed_socket_probe.py \
  --socket "$SOCK" \
  --vm-id 10 \
  --output /tmp/client_m07_malformed_socket.json

python3 - <<'PY'
import json
report = json.load(open("/tmp/client_m07_malformed_socket.json"))
print("M07 malformed socket overall_pass =", report["overall_pass"])
print("cases =", sorted(report["cases"]))
PY

pgrep -af mediator_phase3
```

Then run known-good checks after the malformed traffic.

On VM-10:

```bash
export LD_LIBRARY_PATH=/opt/vgpu/lib:${LD_LIBRARY_PATH:-}
/mnt/m04-pytorch/venv/bin/python /mnt/m04-pytorch/cupy_probe.py \
  > /tmp/client_m07_post_vm10_cupy.json
python3 -c 'import json; print(json.load(open("/tmp/client_m07_post_vm10_cupy.json"))["overall_pass"])'
```

On VM-6:

```bash
TFV=/home/test-6/m06-cupy-venv
NLIB=$(find "$TFV/lib/python3.10/site-packages/nvidia" -type d -name lib 2>/dev/null | paste -sd: -)
LD_LIBRARY_PATH="/opt/vgpu/lib:$NLIB:${LD_LIBRARY_PATH:-}" \
"$TFV/bin/python" /tmp/may4_reverify/cupy_probe.py \
  > /tmp/client_m07_post_vm6_cupy.json
python3 -c 'import json; print(json.load(open("/tmp/client_m07_post_vm6_cupy.json"))["overall_pass"])'
```

What to say:

> The mediator rejects malformed requests, stays alive, and continues to serve
> known-good GPU work from both VMs afterward.

Expected result:

- M07 malformed socket `overall_pass = True`
- `mediator_phase3` is still running
- VM-10 post-check prints `True`
- VM-6 post-check prints `True`

## Closing Statement For The Video

Use this wording:

> This demonstration verifies the current GPU virtualization layer on Server 1.
> VM-10 shows the complete validated path: Ollama, raw CUDA, PyTorch, CuPy,
> TensorFlow, memory cleanup, and same-VM concurrency. VM-6 shows the second VM
> path for mediated GPU use, cross-VM behavior, and post-security validation. The
> host mediator remains running and safely rejects malformed input.

Avoid saying:

> Every VM and every possible CUDA workload is fully supported.

Say this instead:

> The system passes the defined Milestone 00-07 validation scope, and the next
> hardening step is to promote additional VMs through the same full gate set.
