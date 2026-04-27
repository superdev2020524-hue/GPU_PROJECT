# Client Startup and Verification Guide

This guide describes the exact operator procedure for bringing the current Phase 3 runtime online and verifying that it is working correctly.

The procedure starts at the host mediator, continues through the VM `ollama` service, and ends with direct prompt validation and a full milestone canary run.

This document is written for field use. Follow it in order. Do not skip steps.

---

## 1. Purpose

Use this guide when you need to:

- start the host mediator cleanly,
- restart the VM `ollama` service cleanly,
- confirm that Ollama is running on the GPU-backed path rather than a CPU-only fallback,
- validate direct prompts,
- and prove that the validated baseline still passes the full canary gate.

If every step in this guide passes, the current Phase 1 serving baseline is operating as intended.

---

## 2. Preconditions

You need:

- access to the host mediator machine,
- access to the VM,
- the current `phase3` workspace,
- the current validated model baseline,
- and permission to restart the mediator and `ollama`.

Current validated baseline:

- model: `qwen2.5:0.5b`
- gate file: `phase3/phase1_milestone_test_suite.json`
- gate runner: `phase3/phase1_milestone_gate.py`

---

## 3. Startup order

Always use this order:

1. restart the host mediator,
2. verify the host mediator is running,
3. restart VM `ollama`,
4. verify VM API health,
5. verify GPU mode,
6. run direct prompt checks,
7. run the full default canary gate.

This order prevents ambiguous failures caused by mixed old and new runtime state.

---

## 4. Step 1: restart the host mediator

### Recommended method from the workstation

From the machine where this repository is checked out:

```bash
cd /home/david/Downloads/gpu/phase3
export SSHPASS='<HOST_ROOT_PASSWORD>'
MEDIATOR_TRUNCATE_LOG=1 ./host_restart_mediator.sh
```

Expected result:

```text
(truncated /tmp/mediator.log)
mediator_pid=<pid>
<pid> ./mediator_phase3
OK: mediator_phase3 running
```

### Manual method on the host

If you are already logged into the host as `root`:

```bash
cd /root/phase3
killall -9 mediator_phase3 2>/dev/null || true
sleep 1
: > /tmp/mediator.log
nohup ./mediator_phase3 >> /tmp/mediator.log 2>&1 &
sleep 2
pgrep -a mediator_phase3
```

Acceptance criteria:

- `mediator_phase3` is running,
- `/tmp/mediator.log` is fresh,
- there is no immediate crash or restart loop.

---

## 5. Step 2: restart VM `ollama`

Log into the VM and restart `ollama`:

```bash
sudo systemctl restart ollama
systemctl is-active ollama
```

Expected result:

```text
active
```

Then confirm the local API is reachable:

```bash
curl -s http://127.0.0.1:11434/api/tags
```

Expected result:

- HTTP success,
- model list includes `qwen2.5:0.5b`.

---

## 6. Step 3: verify that the service is configured for GPU mode

Inside the VM:

```bash
systemctl cat ollama --no-pager | grep -E 'OLLAMA_NUM_GPU|OLLAMA_LLM_LIBRARY|LD_LIBRARY_PATH|OLLAMA_LIBRARY_PATH'
```

Expected result should include:

```text
Environment=OLLAMA_NUM_GPU=1
Environment=OLLAMA_LLM_LIBRARY=cuda_v12
Environment=LD_LIBRARY_PATH=/opt/vgpu/lib:/usr/local/lib/ollama/cuda_v12:/usr/local/lib/ollama
Environment=OLLAMA_LIBRARY_PATH=/opt/vgpu/lib:/usr/local/lib/ollama/cuda_v12:/usr/local/lib/ollama
```

Interpretation:

- this proves the live service is configured for the CUDA/vGPU-backed path,
- it does not by itself prove that the current response used the GPU,
- so you must still complete the next verification steps.

---

## 7. Step 4: direct prompt verification

Run these prompts inside the VM exactly as written.

### 7.1 Arithmetic check: 2 + 9

```bash
curl -s http://127.0.0.1:11434/api/generate \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen2.5:0.5b","prompt":"What is 2 + 9? Reply with digits only.","stream":false,"keep_alive":-1,"options":{"temperature":0,"seed":1234,"num_predict":8}}' \
| python3 -c 'import sys,json; print(json.load(sys.stdin)["response"])'
```

Expected output:

```text
11
```

### 7.2 Arithmetic check: 13 + 9

```bash
curl -s http://127.0.0.1:11434/api/generate \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen2.5:0.5b","prompt":"What is 13 + 9? Reply with digits only.","stream":false,"keep_alive":-1,"options":{"temperature":0,"seed":1234,"num_predict":8}}' \
| python3 -c 'import sys,json; print(json.load(sys.stdin)["response"])'
```

Expected output:

```text
22
```

### 7.3 Exact token check

```bash
curl -s http://127.0.0.1:11434/api/generate \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen2.5:0.5b","prompt":"Return exactly this token and nothing else: PHASE1_OK_314159","stream":false,"keep_alive":-1,"options":{"temperature":0,"seed":1234,"num_predict":24}}' \
| python3 -c 'import sys,json; print(json.load(sys.stdin)["response"])'
```

Expected output:

```text
PHASE1_OK_314159
```

If these three direct checks pass, the user-visible behavior is correct on the current runtime.

---

## 8. Step 5: prove that the current successful requests used the GPU path

This step answers the most important operational question:

"Did the good answer come from the real GPU-backed path, or from an accidental CPU fallback?"

### 8.1 Host mediator proof

On the host:

```bash
python3 - <<'PY'
from pathlib import Path
p = Path('/tmp/mediator.log')
lines = p.read_text(errors='replace').splitlines() if p.exists() else []
for line in [l for l in lines if 'cuLaunchKernel SUCCESS: kernel executed on physical GPU' in l][-40:]:
    print(line)
PY
```

Expected result:

- you should see recent `cuLaunchKernel SUCCESS: kernel executed on physical GPU ... vm=10` lines.

Interpretation:

- those lines are strong proof that the restarted request path is traversing the host mediator and executing real GPU kernels on the physical GPU.

### 8.2 Optional VM-side transport proof

Inside the VM:

```bash
python3 - <<'PY'
from pathlib import Path
p = Path('/var/log/ollama-stderr.log')
lines = p.read_text(errors='replace').splitlines() if p.exists() else []
for line in lines[-400:]:
    if any(k in line for k in ('libvgpu-cuda', 'cuda-transport', 'cuMemcpyHtoDAsync')):
        print(line)
PY
```

Expected result:

- live CUDA/vGPU-related lines should appear during or after active requests.

Interpretation:

- this is supporting proof from the VM side,
- the host mediator proof is usually easier to interpret in field operation.

---

## 9. Step 6: run the full default canary gate

From the host workspace:

```bash
cd /home/david/Downloads/gpu/phase3
python3 phase1_milestone_gate.py \
  --base-url http://10.25.33.110:11434 \
  --timeout-sec 240 \
  --output /tmp/phase1_milestone_gate_report.json
```

Expected terminal summary:

```text
overall_pass=True
```

What this proves:

- accuracy passes,
- speed passes,
- residency passes,
- and the validated baseline still holds after restart.

---

## 10. Current known-good reference result

The current validated baseline has already been rechecked after:

- host mediator restart,
- VM `ollama` restart,
- direct prompt retest,
- and a full default gate rerun.

Reference results from that restart verification:

- direct prompt: `2 + 9` -> `11`
- direct prompt: `13 + 9` -> `22`
- direct prompt: exact token -> `PHASE1_OK_314159`

Full canary result:

- `A1_exact_string`: pass
- `A2_arithmetic`: pass
- `A3_json_shape`: pass
- `cold`: pass
- `warm`: pass
- `keep_loaded`: pass
- `force_unload`: pass
- `overall_pass=True`

This is the current deployment reference.

---

## 11. Failure handling

If something fails, classify it before making changes.

### If the mediator does not restart

Check:

- host process exists,
- `/tmp/mediator.log` for immediate crash lines,
- correct working directory (`/root/phase3`),
- correct host CUDA library path.

Do not continue to VM verification until mediator is healthy.

### If `ollama` does not restart cleanly

Check:

- `systemctl status ollama --no-pager`
- `journalctl -u ollama -n 100 --no-pager`

Do not continue to prompt verification until `/api/tags` returns successfully.

### If prompts work but GPU-path proof is missing

Treat that as a deployment risk.

Do not claim success until:

- host mediator shows real GPU kernel execution,
- or equivalent live GPU-path proof is captured.

### If the direct prompts pass but the full canary fails

Treat the canary as authoritative.

The direct prompt checks are smoke tests. The full gate is the real acceptance condition.

---

## 12. Field rule

At a client site, do not rely on memory and do not rely on "it worked earlier."

Use this sequence every time:

1. restart mediator,
2. restart `ollama`,
3. verify health,
4. verify GPU mode,
5. run direct prompts,
6. confirm mediator GPU execution,
7. run the full canary.

If all seven steps pass, you are on solid ground.
