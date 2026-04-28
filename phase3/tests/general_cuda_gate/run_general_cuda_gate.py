#!/usr/bin/env python3
"""
Phase 3 Milestone 01 general CUDA gate runner.

Runs from the workstation. It deploys a small Driver API probe to the VM, builds
small probes with gcc, executes them through the VM's mediated CUDA libraries,
and writes a JSON report.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

SCRIPT_DIR = Path(__file__).resolve().parent
PHASE3_DIR = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(PHASE3_DIR))

from vm_config import MEDIATOR_HOST, MEDIATOR_PASSWORD, MEDIATOR_USER, VM_HOST, VM_PASSWORD, VM_USER


SSH_OPTS = [
    "-o",
    "StrictHostKeyChecking=no",
    "-o",
    "ConnectTimeout=20",
    "-o",
    "PreferredAuthentications=password",
    "-o",
    "PubkeyAuthentication=no",
]


def run_local(argv: List[str], *, env: Dict[str, str] | None = None, input_text: str | None = None, timeout: int = 120) -> Dict[str, Any]:
    start = time.time()
    proc = subprocess.run(
        argv,
        input=input_text,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
        check=False,
    )
    return {
        "argv": argv,
        "rc": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "wall_sec": round(time.time() - start, 3),
    }


def ssh_bash(user: str, host: str, password: str, script: str, *, timeout: int = 180) -> Dict[str, Any]:
    sshpass = shutil.which("sshpass")
    if not sshpass:
        raise RuntimeError("sshpass is required for this runner")
    env = {**os.environ, "SSHPASS": password}
    argv = [sshpass, "-e", "ssh", *SSH_OPTS, f"{user}@{host}", "bash", "-s"]
    return run_local(argv, env=env, input_text=script, timeout=timeout)


def scp_to_vm(local_path: Path, remote_path: str, *, timeout: int = 120) -> Dict[str, Any]:
    sshpass = shutil.which("sshpass")
    if not sshpass:
        raise RuntimeError("sshpass is required for this runner")
    env = {**os.environ, "SSHPASS": VM_PASSWORD}
    argv = [
        sshpass,
        "-e",
        "scp",
        *SSH_OPTS,
        str(local_path),
        f"{VM_USER}@{VM_HOST}:{remote_path}",
    ]
    return run_local(argv, env=env, timeout=timeout)


def summarize_host_evidence() -> Dict[str, Any]:
    script = r"""
python3 - <<'PY'
from pathlib import Path
import json
p = Path('/tmp/mediator.log')
lines = p.read_text(errors='replace').splitlines() if p.exists() else []
keys = [
    'cuLaunchKernel SUCCESS: kernel executed on physical GPU',
    'module-load',
    'sync FAILED',
    'CUDA_ERROR_ILLEGAL_ADDRESS',
    'result.status=801',
    'FAILED',
]
out = {'exists': p.exists(), 'line_count': len(lines), 'keys': {}}
for key in keys:
    hits = [line for line in lines if key in line]
    out['keys'][key] = {'count': len(hits), 'tail': hits[-5:]}
print(json.dumps(out, indent=2))
PY
"""
    result = ssh_bash(MEDIATOR_USER, MEDIATOR_HOST, MEDIATOR_PASSWORD, script, timeout=120)
    try:
        parsed = json.loads(result["stdout"])
    except Exception:
        parsed = {"parse_error": True, "raw": result["stdout"] + result["stderr"]}
    return {"command": "host mediator evidence summary", "result": result, "parsed": parsed}


def run_vm_command(command: str, *, timeout: int = 180) -> Dict[str, Any]:
    return ssh_bash(VM_USER, VM_HOST, VM_PASSWORD, command, timeout=timeout)


def case_result(name: str, result: Dict[str, Any], pass_rc: int = 0) -> Dict[str, Any]:
    return {
        "name": name,
        "rc": result["rc"],
        "wall_sec": result["wall_sec"],
        "pass": result["rc"] == pass_rc,
        "stdout_preview": result["stdout"][-4000:],
        "stderr_preview": result["stderr"][-4000:],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--remote-dir", default="/tmp/phase3_general_cuda_gate")
    parser.add_argument("--output", default="/tmp/phase3_general_cuda_gate_report.json")
    parser.add_argument("--repetitions", type=int, default=5)
    args = parser.parse_args()

    probes = [
        {
            "name": "driver_api_probe",
            "source": SCRIPT_DIR / "driver_api_probe.c",
            "remote_source": f"{args.remote_dir}/driver_api_probe.c",
            "remote_bin": f"{args.remote_dir}/driver_api_probe",
        },
        {
            "name": "runtime_api_probe",
            "source": SCRIPT_DIR / "runtime_api_probe.c",
            "remote_source": f"{args.remote_dir}/runtime_api_probe.c",
            "remote_bin": f"{args.remote_dir}/runtime_api_probe",
        },
    ]
    report: Dict[str, Any] = {
        "gate": "phase3_general_cuda_gate",
        "started_at": int(time.time()),
        "vm": f"{VM_USER}@{VM_HOST}",
        "host": f"{MEDIATOR_USER}@{MEDIATOR_HOST}",
        "remote_dir": args.remote_dir,
        "cases": [],
        "host_evidence_before": None,
        "host_evidence_after": None,
        "overall_pass": False,
    }

    report["host_evidence_before"] = summarize_host_evidence()

    prep = run_vm_command(f"mkdir -p {args.remote_dir}", timeout=60)
    report["cases"].append(case_result("prepare_remote_dir", prep))
    if prep["rc"] != 0:
        return finish(report, args.output)

    env_prefix = "LD_LIBRARY_PATH=/opt/vgpu/lib:/usr/local/lib/ollama/cuda_v12:/usr/local/lib/ollama:${LD_LIBRARY_PATH:-}"
    for probe in probes:
        copied = scp_to_vm(probe["source"], probe["remote_source"])
        report["cases"].append(case_result(f"copy_{probe['name']}_to_vm", copied))
        if copied["rc"] != 0:
            return finish(report, args.output)

        build_cmd = f"gcc -O2 -Wall -Wextra -std=c11 -o {probe['remote_bin']} {probe['remote_source']} -ldl"
        built = run_vm_command(build_cmd, timeout=120)
        report["cases"].append(case_result(f"build_{probe['name']}", built))
        if built["rc"] != 0:
            return finish(report, args.output)

        for idx in range(args.repetitions):
            run = run_vm_command(f"{env_prefix} {probe['remote_bin']}", timeout=240)
            report["cases"].append(case_result(f"{probe['name']}_run_{idx + 1}", run))
            if run["rc"] != 0:
                report["host_evidence_after"] = summarize_host_evidence()
                return finish(report, args.output)

    report["host_evidence_after"] = summarize_host_evidence()
    return finish(report, args.output)


def finish(report: Dict[str, Any], output: str) -> int:
    report["finished_at"] = int(time.time())
    report["overall_pass"] = all(case.get("pass") for case in report["cases"])
    output_path = Path(output)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "overall_pass": report["overall_pass"],
        "output": str(output_path),
        "cases": [(case["name"], case["pass"], case["rc"]) for case in report["cases"]],
    }, indent=2))
    return 0 if report["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
