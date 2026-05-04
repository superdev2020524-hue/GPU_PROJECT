# Self-Verification Guide: Phase 3 Milestones 00-07

Use this guide when you want to verify the system yourself and record a video for
the client report.

The goal is simple: run each gate in order, show the result, and only move to the
next gate when the current one passes.

## Before Recording

Open three terminals if possible:

1. Workstation terminal in `/home/david/Downloads/gpu`
2. VM-10 SSH terminal: `test-10@10.25.33.110`
3. Host SSH terminal: `root@10.25.33.10`

Do not show passwords in the video. Set them before recording, or blur that part.

Recommended workstation setup:

```bash
cd /home/david/Downloads/gpu
export SSHPASS='<password>'
export TS="$(date +%Y%m%d_%H%M%S)_self_verify"
export SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20 -o PreferredAuthentications=password -o PubkeyAuthentication=no"
```

In the video, explain this once:

> I am going to run the verification gates from 00 to 07 in order. A gate passes
> only when its JSON report says `overall_pass: true` or the probe prints its
> documented PASS line.

## 00 - Ollama Baseline

Purpose: prove the original Ollama path still works before testing broader CUDA
work.

Run from the workstation:

```bash
sshpass -e ssh $SSH_OPTS test-10@10.25.33.110 'mkdir -p /tmp/self_verify'
sshpass -e scp $SSH_OPTS \
  phase3/phase1_milestone_gate.py \
  phase3/phase1_milestone_test_suite.json \
  phase3/phase1_plan_b_tiny_gate.py \
  phase3/phase1_plan_c_client_gate.py \
  test-10@10.25.33.110:/tmp/self_verify/

sshpass -e ssh $SSH_OPTS test-10@10.25.33.110 "
  cd /tmp/self_verify &&
  python3 phase1_milestone_gate.py \
    --suite phase1_milestone_test_suite.json \
    --output /tmp/${TS}_m00_planA.json &&
  python3 phase1_plan_b_tiny_gate.py \
    --output /tmp/${TS}_m00_planB.json &&
  python3 phase1_plan_c_client_gate.py \
    --output /tmp/${TS}_m00_planC.json
"
```

What to show:

```bash
sshpass -e ssh $SSH_OPTS test-10@10.25.33.110 "
  python3 - <<PY
import json
for p in ['/tmp/${TS}_m00_planA.json','/tmp/${TS}_m00_planB.json','/tmp/${TS}_m00_planC.json']:
    r=json.load(open(p))
    print(p, r.get('overall_pass'))
PY
"
```

Expected result: all three lines print `True`.

## 01 - Raw CUDA

Purpose: prove CUDA works below PyTorch, CuPy, TensorFlow, or Ollama.

Run from the workstation:

```bash
python3 phase3/tests/general_cuda_gate/run_general_cuda_gate.py \
  --output /tmp/${TS}_m01_general_cuda.json \
  --repetitions 5
```

What to show:

```bash
python3 - <<PY
import json
r=json.load(open('/tmp/${TS}_m01_general_cuda.json'))
print('overall_pass =', r['overall_pass'])
print('case_count =', len(r['cases']))
PY
```

Expected result: `overall_pass = True`.

## 02 - API Coverage Audit

Purpose: prove the protocol header and host executor still match.

Run from the workstation:

```bash
python3 - <<'PY'
import json, re, time
from pathlib import Path

root = Path('/home/david/Downloads/gpu/phase3')
proto = (root / 'include/cuda_protocol.h').read_text(errors='replace')
executor = (root / 'src/cuda_executor.c').read_text(errors='replace')

proto_names = {
    m.group(1)
    for m in re.finditer(r'^\s*#define\s+(CUDA_CALL_[A-Z0-9_]+)\s+(0x[0-9a-fA-F]+|\d+)\b', proto, re.M)
    if m.group(1) != 'CUDA_CALL_MAX'
}
case_names = set(re.findall(r'case\s+(CUDA_CALL_[A-Z0-9_]+)\s*:', executor))
missing = sorted(proto_names - case_names)

report = {
    'gate': 'phase3_api_coverage_audit_self_check',
    'started_at': int(time.time()),
    'protocol_ids': len(proto_names),
    'executor_case_ids': len(case_names & proto_names),
    'missing_cases': missing,
    'matrix_exists': (root / 'VERIFICATION/02_api_coverage_audit/API_COVERAGE_MATRIX.md').exists(),
    'gap_list_exists': (root / 'VERIFICATION/02_api_coverage_audit/GAP_LIST.md').exists(),
}
report['overall_pass'] = (
    report['protocol_ids'] > 0
    and not missing
    and report['matrix_exists']
    and report['gap_list_exists']
)
Path('/tmp/self_verify_m02_api_coverage.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
raise SystemExit(0 if report['overall_pass'] else 1)
PY
```

Expected result: `overall_pass` is `true`, and `missing_cases` is empty.

## 03 - Memory, Sync, And Cleanup

Purpose: prove memory copies, streams, events, and post-kill recovery still work.

Run from the workstation:

```bash
sshpass -e scp $SSH_OPTS \
  phase3/tests/memory_sync_cleanup/async_stream_event_probe.c \
  phase3/tests/memory_sync_cleanup/forced_kill_alloc_probe.c \
  test-10@10.25.33.110:/tmp/self_verify/

sshpass -e ssh $SSH_OPTS test-10@10.25.33.110 "
  set -e
  cd /tmp/self_verify
  gcc -O2 -Wall -Wextra -std=c11 -o /tmp/${TS}_async_stream_event_probe async_stream_event_probe.c -ldl
  LD_LIBRARY_PATH=/opt/vgpu/lib:\${LD_LIBRARY_PATH:-} /tmp/${TS}_async_stream_event_probe | tee /tmp/${TS}_m03_async.log

  gcc -O2 -Wall -Wextra -std=c11 -o /tmp/${TS}_forced_kill_alloc_probe forced_kill_alloc_probe.c -ldl
  (LD_LIBRARY_PATH=/opt/vgpu/lib:\${LD_LIBRARY_PATH:-} /tmp/${TS}_forced_kill_alloc_probe > /tmp/${TS}_m03_kill_child.log 2>&1 & echo \$! > /tmp/${TS}_m03_kill_child.pid)
  for i in 1 2 3 4 5 6 7 8 9 10; do grep -q READY /tmp/${TS}_m03_kill_child.log && break; sleep 1; done
  grep READY /tmp/${TS}_m03_kill_child.log
  kill -9 \$(cat /tmp/${TS}_m03_kill_child.pid) || true
  sleep 2
  LD_LIBRARY_PATH=/opt/vgpu/lib:\${LD_LIBRARY_PATH:-} /tmp/${TS}_async_stream_event_probe > /tmp/${TS}_m03_post_kill_async.log
  grep PASS /tmp/${TS}_m03_post_kill_async.log
"
```

Expected result: both async probes print `ASYNC_STREAM_EVENT_PROBE PASS`.

## 04 - PyTorch

Purpose: prove PyTorch can use the mediated GPU path.

Run from the workstation:

```bash
sshpass -e scp $SSH_OPTS phase3/tests/pytorch_gate/pytorch_probe.py test-10@10.25.33.110:/tmp/self_verify/
sshpass -e ssh $SSH_OPTS test-10@10.25.33.110 "
  cp /tmp/self_verify/pytorch_probe.py /mnt/m04-pytorch/pytorch_probe.py
  LD_LIBRARY_PATH=/opt/vgpu/lib:\${LD_LIBRARY_PATH:-} \
    /mnt/m04-pytorch/venv/bin/python /mnt/m04-pytorch/pytorch_probe.py \
    > /tmp/${TS}_m04_pytorch.json
  python3 -c \"import json; print(json.load(open('/tmp/${TS}_m04_pytorch.json'))['overall_pass'])\"
"
```

Expected result: `True`.

## 05 - CuPy And TensorFlow

Purpose: prove a second CUDA framework path and the TensorFlow GPU path.

Run from the workstation:

```bash
sshpass -e scp $SSH_OPTS \
  phase3/tests/second_framework_gate/cupy_probe.py \
  phase3/tests/second_framework_gate/tensorflow_mnist_probe.py \
  test-10@10.25.33.110:/tmp/self_verify/

sshpass -e ssh $SSH_OPTS test-10@10.25.33.110 "
  cp /tmp/self_verify/cupy_probe.py /mnt/m04-pytorch/cupy_probe.py
  cp /tmp/self_verify/tensorflow_mnist_probe.py /mnt/m04-pytorch/tensorflow_mnist_probe.py

  LD_LIBRARY_PATH=/opt/vgpu/lib:\${LD_LIBRARY_PATH:-} \
    /mnt/m04-pytorch/venv/bin/python /mnt/m04-pytorch/cupy_probe.py \
    > /tmp/${TS}_m05_cupy.json
  python3 -c \"import json; print('CuPy', json.load(open('/tmp/${TS}_m05_cupy.json'))['overall_pass'])\"

  TFV=/mnt/m04-pytorch/tf123-venv
  NLIB=\$(find \"\$TFV/lib/python3.10/site-packages/nvidia\" -type d -name lib 2>/dev/null | paste -sd: -)
  LD_LIBRARY_PATH=\"\$NLIB:/opt/vgpu/lib:\${LD_LIBRARY_PATH:-}\" TF_FORCE_GPU_ALLOW_GROWTH=true \
    \$TFV/bin/python /mnt/m04-pytorch/tensorflow_mnist_probe.py \
    > /tmp/${TS}_m05_tensorflow.json
  python3 -c \"import json; r=json.load(open('/tmp/${TS}_m05_tensorflow.json')); print('TensorFlow', r['overall_pass'], r.get('used_gpu_for_training'))\"
"
```

Expected result:

- `CuPy True`
- `TensorFlow True True`

## 06 - Multi-Process And Multi-VM

Purpose: prove the mediator can serve more than one workload.

Run same-VM checks on VM-10:

```bash
sshpass -e scp $SSH_OPTS \
  phase3/tests/multiprocess_multivm/two_process_cupy_probe.py \
  phase3/tests/multiprocess_multivm/mixed_pytorch_cupy_probe.py \
  test-10@10.25.33.110:/tmp/self_verify/

sshpass -e ssh $SSH_OPTS test-10@10.25.33.110 "
  cp /tmp/self_verify/two_process_cupy_probe.py /mnt/m04-pytorch/two_process_cupy_probe.py
  cp /tmp/self_verify/mixed_pytorch_cupy_probe.py /mnt/m04-pytorch/mixed_pytorch_cupy_probe.py
  LD_LIBRARY_PATH=/opt/vgpu/lib:\${LD_LIBRARY_PATH:-} /mnt/m04-pytorch/venv/bin/python /mnt/m04-pytorch/two_process_cupy_probe.py > /tmp/${TS}_m06_two_process_cupy.json
  LD_LIBRARY_PATH=/opt/vgpu/lib:\${LD_LIBRARY_PATH:-} /mnt/m04-pytorch/venv/bin/python /mnt/m04-pytorch/mixed_pytorch_cupy_probe.py > /tmp/${TS}_m06_mixed_pytorch_cupy.json
  python3 - <<PY
import json
for p in ['/tmp/${TS}_m06_two_process_cupy.json','/tmp/${TS}_m06_mixed_pytorch_cupy.json']:
    print(p, json.load(open(p))['overall_pass'])
PY
"
```

For Test-6, first make sure a CuPy venv exists:

```bash
sshpass -e ssh $SSH_OPTS test-6@10.25.33.16 "
  test -x /home/test-6/m06-cupy-venv/bin/python || (
    python3 -m venv /home/test-6/m06-cupy-venv &&
    /home/test-6/m06-cupy-venv/bin/python -m pip install --upgrade pip &&
    /home/test-6/m06-cupy-venv/bin/python -m pip install cupy-cuda12x nvidia-cuda-nvrtc-cu12 nvidia-cuda-runtime-cu12 numpy
  )
"
```

Then run the single Test-6 check:

```bash
sshpass -e scp $SSH_OPTS phase3/tests/second_framework_gate/cupy_probe.py test-6@10.25.33.16:/tmp/self_verify_cupy_probe.py
sshpass -e ssh $SSH_OPTS test-6@10.25.33.16 "
  TFV=/home/test-6/m06-cupy-venv
  NLIB=\$(find \"\$TFV/lib/python3.10/site-packages/nvidia\" -type d -name lib 2>/dev/null | paste -sd: -)
  LD_LIBRARY_PATH=\"/opt/vgpu/lib:\$NLIB:\${LD_LIBRARY_PATH:-}\" \
    \$TFV/bin/python /tmp/self_verify_cupy_probe.py \
    > /tmp/${TS}_m06_test6_cupy.json
  python3 -c \"import json; print(json.load(open('/tmp/${TS}_m06_test6_cupy.json'))['overall_pass'])\"
"
```

Expected result: every M06 report prints `True`.

For the cross-VM check, start one CuPy probe on VM-10 and one CuPy probe on
Test-6 at the same time:

```bash
python3 - <<'PY'
import json, os, shutil, subprocess, time
from pathlib import Path

ts = os.environ['TS']
sshpass = shutil.which('sshpass')
opts = os.environ['SSH_OPTS'].split()
env = {**os.environ}

cmd10 = (
    f"LD_LIBRARY_PATH=/opt/vgpu/lib:${{LD_LIBRARY_PATH:-}} "
    f"/mnt/m04-pytorch/venv/bin/python /mnt/m04-pytorch/cupy_probe.py "
    f"> /tmp/{ts}_m06_cross_vm_test10_cupy.json && "
    f"python3 -c \"import json; print(json.load(open('/tmp/{ts}_m06_cross_vm_test10_cupy.json'))['overall_pass'])\""
)
cmd6 = (
    "TFV=/home/test-6/m06-cupy-venv; "
    "NLIB=$(find \"$TFV/lib/python3.10/site-packages/nvidia\" -type d -name lib 2>/dev/null | paste -sd: -); "
    f"LD_LIBRARY_PATH=\"/opt/vgpu/lib:$NLIB:${{LD_LIBRARY_PATH:-}}\" "
    f"$TFV/bin/python /tmp/self_verify_cupy_probe.py "
    f"> /tmp/{ts}_m06_cross_vm_test6_cupy.json && "
    f"python3 -c \"import json; print(json.load(open('/tmp/{ts}_m06_cross_vm_test6_cupy.json'))['overall_pass'])\""
)

children = [
    ('test10', 'test-10@10.25.33.110', cmd10),
    ('test6', 'test-6@10.25.33.16', cmd6),
]
procs = []
for name, target, cmd in children:
    procs.append((name, subprocess.Popen([sshpass, '-e', 'ssh', *opts, target, cmd],
                                         stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                         text=True, env=env), time.time()))

results = []
for name, proc, started in procs:
    out, err = proc.communicate(timeout=420)
    results.append({
        'name': name,
        'rc': proc.returncode,
        'stdout': out.strip(),
        'elapsed_sec': round(time.time() - started, 3),
        'pass': proc.returncode == 0 and 'True' in out,
    })

report = {'gate': 'm06_cross_vm_concurrent_cupy', 'children': results}
report['overall_pass'] = all(item['pass'] for item in results)
Path(f'/tmp/{ts}_m06_cross_vm_cupy.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
raise SystemExit(0 if report['overall_pass'] else 1)
PY
```

Expected result: the cross-VM report prints `"overall_pass": true`.

## 07 - Security And Isolation

Purpose: prove malformed mediator socket traffic is rejected and does not poison
good workloads.

Run from the workstation:

```bash
sshpass -e scp $SSH_OPTS phase3/tests/security_isolation/malformed_socket_probe.py root@10.25.33.10:/tmp/${TS}_malformed_socket_probe.py
sshpass -e ssh $SSH_OPTS root@10.25.33.10 "
  SOCK=\$(ls /var/xen/qemu/root-2/tmp/vgpu-mediator.sock 2>/dev/null || ls /var/xen/qemu/root-*/tmp/vgpu-mediator.sock | sed -n '1p')
  python3 /tmp/${TS}_malformed_socket_probe.py --socket \"\$SOCK\" --vm-id 10 --output /tmp/${TS}_m07_malformed_socket.json
  python3 -c \"import json; print(json.load(open('/tmp/${TS}_m07_malformed_socket.json'))['overall_pass'])\"
  pgrep -af mediator_phase3
"
```

Then rerun known-good CuPy probes on both VMs:

```bash
sshpass -e ssh $SSH_OPTS test-10@10.25.33.110 "
  LD_LIBRARY_PATH=/opt/vgpu/lib:\${LD_LIBRARY_PATH:-} \
    /mnt/m04-pytorch/venv/bin/python /mnt/m04-pytorch/cupy_probe.py \
    > /tmp/${TS}_m07_post_test10_cupy.json
  python3 -c \"import json; print(json.load(open('/tmp/${TS}_m07_post_test10_cupy.json'))['overall_pass'])\"
"

sshpass -e ssh $SSH_OPTS test-6@10.25.33.16 "
  TFV=/home/test-6/m06-cupy-venv
  NLIB=\$(find \"\$TFV/lib/python3.10/site-packages/nvidia\" -type d -name lib 2>/dev/null | paste -sd: -)
  LD_LIBRARY_PATH=\"/opt/vgpu/lib:\$NLIB:\${LD_LIBRARY_PATH:-}\" \
    \$TFV/bin/python /tmp/self_verify_cupy_probe.py \
    > /tmp/${TS}_m07_post_test6_cupy.json
  python3 -c \"import json; print(json.load(open('/tmp/${TS}_m07_post_test6_cupy.json'))['overall_pass'])\"
"
```

Expected result:

- The malformed socket probe prints `True`.
- The mediator process is still running.
- Both post-probe CuPy checks print `True`.

## How To Explain The Result In The Video

Use this wording:

> I verified the system from Milestone 00 through Milestone 07 in order. The
> Ollama baseline passed first. Then raw CUDA, API consistency, memory and sync,
> PyTorch, CuPy, TensorFlow, multi-process, multi-VM, and security rejection
> checks all passed. After the malformed security probe, I reran known-good GPU
> framework checks on both VMs, and they still passed.

Do not say:

> This proves every CUDA program will work.

Say this instead:

> This proves the current implementation passes the defined Milestone 00-07 gate
> scope on the live system.
