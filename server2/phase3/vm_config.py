# VM connection config - change these to point at the target VM.
import os

# Current Server 2 target VM. Keep these defaults inside the Server 2 registry
# only; do not mirror them back into the root phase3 tree.
VM_USER = os.environ.get("VM_USER", "root")
VM_HOST = os.environ.get("VM_HOST", "10.25.33.21")
VM_PASSWORD = os.environ.get("VM_PASSWORD", "Calvin@123")
REMOTE_HOME = os.environ.get("REMOTE_HOME", "/root" if VM_USER == "root" else f"/home/{VM_USER}")
REMOTE_PHASE3 = os.environ.get("REMOTE_PHASE3", f"{REMOTE_HOME}/phase3")

# Mediator host (dom0 / GPU host) — Host 2 registry: second pool master / mediator
MEDIATOR_HOST = os.environ.get("MEDIATOR_HOST", "10.25.33.20")
MEDIATOR_USER = os.environ.get("MEDIATOR_USER", "root")
MEDIATOR_PASSWORD = os.environ.get("MEDIATOR_PASSWORD", VM_PASSWORD)
