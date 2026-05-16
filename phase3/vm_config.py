# VM connection config — change these to point at the target VM.
import os

# VM: test-6@10.25.33.16 — active Phase 3 guest (see deploy_to_test3.py).
# (Older checklist used test-4@10.25.33.12; change here if targeting that VM.)
VM_USER = "test-6"
VM_HOST = "10.25.33.16"
# Override without editing the file: `VM_PASSWORD='…' python3 connect_vm.py '…'`
# If `connect_vm.py` reports Permission denied but `ssh test-6@10.25.33.16` works, the
# default fallback below does not match the guest — export VM_PASSWORD to the same secret
# you type at the ssh prompt (see ERROR_TRACKING_STATUS.md rolling log).
VM_PASSWORD = os.environ.get("VM_PASSWORD", "Calvin@123")
REMOTE_HOME = f"/home/{VM_USER}"
REMOTE_PHASE3 = f"{REMOTE_HOME}/phase3"

# Mediator host (dom0 / GPU host) — for fetching mediator logs
MEDIATOR_HOST = os.environ.get("MEDIATOR_HOST", "10.25.33.10")
MEDIATOR_USER = os.environ.get("MEDIATOR_USER", "root")
MEDIATOR_PASSWORD = os.environ.get("MEDIATOR_PASSWORD", VM_PASSWORD)
