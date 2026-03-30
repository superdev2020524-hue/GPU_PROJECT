# Operational Checklist and Commands

This file is the fastest path back into execution.

Use it when the goal is to act, not to re-study the whole Phase 3 tree.

## 1. Current execution stance

As of 2026-03-28:

- local artifact exists at `phase3/out/libggml-cuda.so`
- local artifact SHA256 matches the VM-deployed `/usr/local/lib/ollama/cuda_v12/libggml-cuda.so`
- Docker is not currently usable from this user account without elevation
- therefore the fastest next live action is a bounded full-load retest using the currently deployed artifact
- but every retest must also capture whether the transport path is `shmem` or `BAR1`

If Docker later becomes available, rebuild from the known graphs-off Hopper path and redeploy.

## 2. Step 0: quick baseline

Run these first from the workstation repo root:

```bash
python3 phase3/connect_vm.py "systemctl is-active ollama"
python3 phase3/connect_vm.py "journalctl -u ollama -b --no-pager | python3 -c \"import sys,re; lines=sys.stdin.read().splitlines(); hits=[l for l in lines if re.search(r'inference compute', l)]; print('\n'.join(hits[-3:]))\""
sshpass -p 'Calvin@123' ssh -n -o StrictHostKeyChecking=no -o PreferredAuthentications=password -o PubkeyAuthentication=no root@10.25.33.10 "python3 -c \"import pathlib,re; lines=pathlib.Path('/tmp/mediator.log').read_text(errors='replace').splitlines(); hits=[l for l in lines if re.search(r'401312|INVALID_IMAGE|rc=700|CUDA_ERROR_ILLEGAL_ADDRESS|module-load', l)]; print('\n'.join(hits[-20:]))\""
bash phase3/run_preflight_gemm_ex_vm.sh
```

Expected:

- Ollama `active`
- latest `inference compute` shows `library=CUDA compute=9.0`
- host log does not suddenly reintroduce E1 or E4
- preflight ends with `PREFLIGHT_OK`

Also run this VM path check:

```bash
python3 phase3/connect_vm.py "journalctl -u ollama -b --no-pager | python3 -c \"import sys,re; lines=sys.stdin.read().splitlines(); hits=[l for l in lines if re.search(r'shmem|using BAR1|data_path=BAR1|data_path=SHMEM|Cannot resolve GPA|mmap shmem', l)]; print('\n'.join(hits[-20:]))\""
```

Interpretation:

- `shmem registered` / SHMEM path = intended fast-path candidate
- `using BAR1` = fallback path, and load-performance conclusions must be treated as BAR1 results

## 3. Step 1: bounded full-load retest

Launch the existing bounded trace on the VM.

Workstation command:

```bash
sshpass -p 'Calvin@123' ssh -n -o StrictHostKeyChecking=no -o PreferredAuthentications=password -o PubkeyAuthentication=no test-4@10.25.33.12 'E3_CURL_TIMEOUT_SEC=7200 bash -s' < phase3/e3_bounded_trace_launch.sh
```

Expected immediate output:

- `E3_TRACE_BASE=/tmp/e3_bounded_<timestamp>`
- `journal_follow_pid=...`
- `curl_pid=...`

This does not wait 2 hours on the terminal. It starts the bounded run and returns the artifact path.

## 4. Step 2: monitor the bounded run

Check every 3-5 minutes.

VM monitor:

```bash
python3 phase3/connect_vm.py "journalctl -u ollama -n 40 --no-pager | python3 -c \"import sys,re; lines=sys.stdin.read().splitlines(); hits=[l for l in lines if re.search(r'model load progress|mmq_x_best|mmq\\.cuh|ggml_backend_cuda_graph_reserve|exit status|error loading llama server|SIGABRT|SIGSEGV', l)]; print('\n'.join(hits[-40:]))\""
```

Host monitor:

```bash
sshpass -p 'Calvin@123' ssh -n -o StrictHostKeyChecking=no -o PreferredAuthentications=password -o PubkeyAuthentication=no root@10.25.33.10 "python3 -c \"import pathlib,re; lines=pathlib.Path('/tmp/mediator.log').read_text(errors='replace').splitlines(); hits=[l for l in lines if re.search(r'HtoD progress|401312|INVALID_IMAGE|rc=700|CUDA_ERROR_ILLEGAL_ADDRESS|FAILED|call_id=0xb5|call_id=0x26', l)]; print('\n'.join(hits[-40:]))\""
```

Transmission monitor:

```bash
python3 phase3/connect_vm.py "journalctl -u ollama -n 80 --no-pager | python3 -c \"import sys,re; lines=sys.stdin.read().splitlines(); hits=[l for l in lines if re.search(r'mmap shmem|using BAR1|data_path=BAR1|poll call_id=0x0032|model load progress', l)]; print('\n'.join(hits[-80:]))\""
```

Stop conditions:

- no `model load progress` movement for 15+ minutes
- `mmq_x_best=0`
- `mmq.cuh:3884`
- `error loading llama server`
- `exit status 2`
- `SIGABRT` or `SIGSEGV`
- new `401312 INVALID_IMAGE`
- new `rc=700`

Performance conclusions to record during monitoring:

- whether the run is `BAR1` or `shmem`
- whether `0x0032` / HtoD appears highly serialized
- whether model-load progress is advancing too slowly for the observed transferred bytes

If any of those appear, stop the run and capture the artifacts.

## 5. Step 3: stop and capture if it fails

Stop the active curl and journal followers on the VM after locating the base path:

```bash
python3 phase3/connect_vm.py "BASE=\$(cat /tmp/e3_trace_latest.txt); echo BASE=\$BASE; cat \$BASE/curl.pid; cat \$BASE/journal_follow.pid; kill \$(cat \$BASE/curl.pid) \$(cat \$BASE/journal_follow.pid) 2>/dev/null || true; echo stopped"
```

Capture the most useful evidence:

```bash
python3 phase3/connect_vm.py "BASE=\$(cat /tmp/e3_trace_latest.txt); echo BASE=\$BASE; ls -l \$BASE; tail -n 40 \$BASE/journal_follow.log 2>/dev/null || true; tail -n 40 \$BASE/curl_generate.stderr.log 2>/dev/null || true; tail -n 40 \$BASE/curl_generate.stdout.log 2>/dev/null || true"
python3 phase3/connect_vm.py "coredumpctl list --no-pager || true"
sshpass -p 'Calvin@123' ssh -n -o StrictHostKeyChecking=no -o PreferredAuthentications=password -o PubkeyAuthentication=no root@10.25.33.10 "python3 -c \"import pathlib; lines=pathlib.Path('/tmp/mediator.log').read_text(errors='replace').splitlines(); print('\n'.join(lines[-80:]))\""
```

## 6. Step 4: if it fails again with MMQ

Treat the result as confirmation that the currently deployed library still does not avoid the active E5 path.

Next action then becomes:

1. obtain Docker access or another valid CUDA build path
2. rebuild from `build_libggml_cuda_hopper_docker.sh`
3. redeploy using `deploy_libggml_cuda_hopper.py`
4. rerun the same bounded retest
5. continue the transmission track separately until `shmem` and load-performance are explained

## 7. Rebuild path once Docker is available

Fastest existing flow:

```bash
cd phase3
./run_libggml_docker_build_and_deploy.sh
```

Or, if Docker requires elevation:

```bash
cd phase3
sudo -E ./run_libggml_docker_build_and_deploy.sh
```

That path is intended to:

- apply the Phase 3 CC patch
- build Hopper `sm_90`
- set `GGML_CUDA_GRAPHS=OFF`
- produce `out/libggml-cuda.so`
- deploy it to the VM

## 8. Decision summary

If you need the shortest practical sequence:

1. baseline
2. launch bounded retest now
3. record `shmem` vs `BAR1` during the same run
4. if MMQ repeats, stop
5. rebuild/deploy with confirmed graphs-off access
6. rerun bounded retest
