# VM connection config — change these to point at the target VM.
import os

# VM: test-4@10.25.33.12 (username has hyphen, not underscore)
VM_USER = "test-4"
VM_HOST = "10.25.33.12"
VM_PASSWORD = "Calvin@123"
REMOTE_HOME = f"/home/{VM_USER}"
REMOTE_PHASE3 = f"{REMOTE_HOME}/phase3"

# Mediator host (dom0 / GPU host) — for fetching mediator logs
MEDIATOR_HOST = os.environ.get("MEDIATOR_HOST", "10.25.33.10")
MEDIATOR_USER = os.environ.get("MEDIATOR_USER", "root")
MEDIATOR_PASSWORD = os.environ.get("MEDIATOR_PASSWORD", VM_PASSWORD)
