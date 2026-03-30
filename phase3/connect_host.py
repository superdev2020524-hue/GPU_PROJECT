#!/usr/bin/env python3
"""
Connect to mediator host using password authentication.

Primary path: sshpass + subprocess (reliable, same as manual SSH from the repo).
Fallback: pexpect + ssh (when sshpass is not installed).

Config from vm_config.py: MEDIATOR_HOST, MEDIATOR_USER, MEDIATOR_PASSWORD.

Environment:
  CONNECT_HOST_COMMAND_TIMEOUT_SEC — max seconds for remote command (default 300).
  CONNECT_HOST_FORCE_PEXPECT=1 — use pexpect even if sshpass exists.
"""
import sys
import os
import re
import shlex
import shutil
import subprocess
import pexpect
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from vm_config import MEDIATOR_HOST, MEDIATOR_USER, MEDIATOR_PASSWORD

CONNECT_TIMEOUT_SEC = int(os.environ.get("CONNECT_HOST_CONNECT_TIMEOUT_SEC", "30"))
PROMPT_TIMEOUT_SEC = int(os.environ.get("CONNECT_HOST_PROMPT_TIMEOUT_SEC", "45"))
COMMAND_TIMEOUT_SEC = int(os.environ.get("CONNECT_HOST_COMMAND_TIMEOUT_SEC", "300"))
DONE_MARKER = "__CONNECT_HOST_DONE__"


def _ssh_base_args():
    return [
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "ConnectTimeout=25",
        "-o",
        "PreferredAuthentications=password",
        "-o",
        "PubkeyAuthentication=no",
        "-o",
        "ServerAliveInterval=15",
        "-o",
        "ServerAliveCountMax=3",
    ]


def connect_and_run_sshpass(command: str):
    """Run remote command via sshpass; return (stdout text, exit code)."""
    sshpass = shutil.which("sshpass")
    if not sshpass:
        return None

    # Build remote script: run user command, then print marker with exit code.
    # Use bash -lc with a single quoted string body to survive ';' and most shell metachars.
    # Do not start with `set +e;` — on some bash builds the remote can mis-parse and run
    # bare `set`, dumping the environment. Default bash -c has errexit off; `__rc` captures status.
    # Note: user one-liners often use `grep` (exit 1 when no match); chain with `|| true` if needed.
    inner = f"{command}; __rc=$?; printf '\\n{DONE_MARKER}:%s\\n' \"$__rc\""
    remote = ["bash", "-c", inner]

    env = {**os.environ, "SSHPASS": MEDIATOR_PASSWORD}
    argv = [
        sshpass,
        "-e",
        "ssh",
        "-n",
        *_ssh_base_args(),
        f"{MEDIATOR_USER}@{MEDIATOR_HOST}",
        *remote,
    ]
    try:
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=COMMAND_TIMEOUT_SEC,
            env=env,
        )
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or "") + (e.stderr or "")
        return out, -124

    combined = (proc.stdout or "") + (proc.stderr or "")
    m = re.search(rf"{re.escape(DONE_MARKER)}:(-?\d+)\s*$", combined, re.MULTILINE)
    if m:
        exit_code = int(m.group(1))
        # Strip marker line from displayed output
        out = re.sub(rf"\n{DONE_MARKER}:-?\d+\s*$", "", combined, count=1).rstrip()
        return out, exit_code
    return combined, proc.returncode if proc.returncode is not None else -1


def connect_and_run_pexpect(command):
    """Legacy pexpect path (StrictHostKeyChecking + host-key yes + marker)."""
    ssh_cmd = (
        "ssh -n "
        + " ".join(shlex.quote(x) for x in _ssh_base_args())
        + f" {shlex.quote(MEDIATOR_USER + '@' + MEDIATOR_HOST)}"
    )
    wrapped = f"( {command} ); __rc=$?; printf '\\n{DONE_MARKER}:%s\\n' \"$__rc\""

    child = pexpect.spawn(ssh_cmd, timeout=CONNECT_TIMEOUT_SEC, encoding="utf-8")

    try:
        patterns = [
            "Are you sure you want to continue connecting",
            "password:",
            "Password:",
            r"\$",
            "#",
            pexpect.EOF,
            pexpect.TIMEOUT,
        ]
        index = child.expect(patterns, timeout=PROMPT_TIMEOUT_SEC)
        if index == 0:
            child.sendline("yes")
            index = child.expect(
                ["password:", "Password:", r"\$", "#", pexpect.EOF, pexpect.TIMEOUT],
                timeout=PROMPT_TIMEOUT_SEC,
            )
        if index in [1, 2]:
            child.sendline(MEDIATOR_PASSWORD)
            child.expect(
                [r"\[.*[#\$]", r"\$", "#", pexpect.EOF, pexpect.TIMEOUT],
                timeout=PROMPT_TIMEOUT_SEC,
            )
        elif index in [3, 4]:
            pass  # already at shell
        elif index in [5, 6]:
            return child.before or "", -1

        if child.isalive():
            child.sendline(wrapped)
            deadline = time.time() + COMMAND_TIMEOUT_SEC
            while time.time() < deadline:
                try:
                    idx = child.expect(
                        [
                            rf"{DONE_MARKER}:(-?\d+)",
                            "password:",
                            "Password:",
                            pexpect.EOF,
                            pexpect.TIMEOUT,
                        ],
                        timeout=10,
                    )
                    if idx == 0:
                        exit_code = int(child.match.group(1))
                        output = child.before
                        child.sendline("exit")
                        child.close()
                        return output, exit_code
                    if idx in [1, 2]:
                        child.sendline(MEDIATOR_PASSWORD)
                except pexpect.TIMEOUT:
                    continue
        return child.before or "", -1
    finally:
        try:
            if child.isalive():
                child.close(force=True)
        except Exception:
            pass


def connect_and_run(command: str):
    if os.environ.get("CONNECT_HOST_FORCE_PEXPECT") == "1":
        return connect_and_run_pexpect(command)
    result = connect_and_run_sshpass(command)
    if result is not None:
        return result
    return connect_and_run_pexpect(command)


if __name__ == "__main__":
    cmd = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "echo Connected; hostname"
    out, rc = connect_and_run(cmd)
    print(out)
    sys.exit(0 if rc == 0 else 1)
