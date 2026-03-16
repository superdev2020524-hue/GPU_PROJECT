#!/usr/bin/env python3
"""
Transfer the modified host-side files needed for mediator rebuild
to the mediator host, then rebuild mediator_phase3 and (if you use it) the
QEMU stub. Does not transfer any other PHASE3 files.

Usage:
  1. Set MEDIATOR_HOST (and optionally MEDIATOR_USER, REMOTE_PHASE3) in env or below.
  2. Run: python3 transfer_cuda_executor_to_host.py
  3. On the mediator host, run the printed build and restart commands.

Requires: scp/ssh access to the host where mediator_phase3 runs (often dom0).
"""
import os
import subprocess
import sys

# --- Configure mediator host (where mediator_phase3 runs) ---
# Not the VM — the host that has the GPU (e.g. 10.25.33.10).
MEDIATOR_HOST = os.environ.get("MEDIATOR_HOST", "YOUR_MEDIATOR_HOST")
MEDIATOR_USER = os.environ.get("MEDIATOR_USER", "root")
REMOTE_PHASE3 = os.environ.get("REMOTE_PHASE3_HOST", "/root/phase3")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FILES = [
    ("Makefile", "Makefile"),
    ("include/cuda_protocol.h", "include/cuda_protocol.h"),
    ("src/cuda_executor.c", "src/cuda_executor.c"),
]


def main():
    if "YOUR_MEDIATOR" in MEDIATOR_HOST or not MEDIATOR_HOST:
        print("Configure MEDIATOR_HOST (and optionally MEDIATOR_USER, REMOTE_PHASE3_HOST).", file=sys.stderr)
        print("  Example: export MEDIATOR_HOST=10.25.33.10", file=sys.stderr)
        print("  Or edit this script and set MEDIATOR_HOST at the top.", file=sys.stderr)
        sys.exit(1)

    for local_rel, remote_rel in FILES:
        local_file = os.path.join(SCRIPT_DIR, local_rel)
        if not os.path.isfile(local_file):
            print(f"Error: {local_file} not found.", file=sys.stderr)
            sys.exit(1)
        remote_file = f"{REMOTE_PHASE3}/{remote_rel}"
        dest = f"{MEDIATOR_USER}@{MEDIATOR_HOST}:{remote_file}"
        print(f"Transferring: {local_rel} -> {dest}")
        r = subprocess.run(
            ["scp", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=15", local_file, dest],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if r.returncode != 0:
            print(f"scp failed: {r.stderr or r.stdout}", file=sys.stderr)
            sys.exit(1)
    print("Transfer OK.\n")
    print("On the mediator host, run in this order:")
    print("")
    print("  1. Detach vGPU from test-3 (stops VM and removes from vGPU DB):")
    print("     vgpu-admin remove-vm --vm-name=test-3")
    print("")
    print("  2. Rebuild mediator and (if you use it) QEMU with the new stub:")
    print(f"     cd {REMOTE_PHASE3}")
    print("     make")
    print("     # If the vGPU stub is built via QEMU RPM/script, run that rebuild too.")
    print("")
    print("  3. Restart the mediator:")
    print("     pkill -f mediator_phase3  # or stop however you run it")
    print("     ./mediator_phase3 2>/tmp/mediator.log &")
    print("")
    print("  4. Reattach vGPU to test-3 and start the VM:")
    print("     vgpu-admin register-vm --vm-name=test-3   # add --pool=A etc. if you use them")
    print("     xe vm-start uuid=<test-3-uuid>   # or: xe vm-start name-label=test-3")
    print("")
    print("Manual scp (if you prefer not to use this script):")
    for local_rel, remote_rel in FILES:
        local_file = os.path.join(SCRIPT_DIR, local_rel)
        remote_file = f"{REMOTE_PHASE3}/{remote_rel}"
        print(f"  scp -o StrictHostKeyChecking=no {local_file} {MEDIATOR_USER}@{MEDIATOR_HOST}:{remote_file}")


if __name__ == "__main__":
    main()
