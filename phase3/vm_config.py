# VM connection config — change these to point at the target VM.
import os

# VM default: test-10@10.25.33.110 (override: VM_USER, VM_HOST, VM_PASSWORD).
VM_USER = os.environ.get("VM_USER", "test-10")
VM_HOST = os.environ.get("VM_HOST", "10.25.33.110")
VM_PASSWORD = os.environ.get("VM_PASSWORD", "Calvin@123")
REMOTE_HOME = f"/home/{VM_USER}"
REMOTE_PHASE3 = f"{REMOTE_HOME}/phase3"

# Mediator host (dom0 / GPU host) — for fetching mediator logs
MEDIATOR_HOST = os.environ.get("MEDIATOR_HOST", "10.25.33.10")
MEDIATOR_USER = os.environ.get("MEDIATOR_USER", "root")
MEDIATOR_PASSWORD = os.environ.get("MEDIATOR_PASSWORD", VM_PASSWORD)
