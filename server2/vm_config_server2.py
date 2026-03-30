# Copy or symlink to vm_config.py when working exclusively against server2 (second dom0).
# Defaults: mediator @ 10.25.33.20; VM targets unchanged from repo — override via env.
import os

VM_USER = os.environ.get("VM_USER", "test-4")
VM_HOST = os.environ.get("VM_HOST", "10.25.33.12")
VM_PASSWORD = os.environ.get("VM_PASSWORD", "Calvin@123")
REMOTE_HOME = f"/home/{VM_USER}"
REMOTE_PHASE3 = f"{REMOTE_HOME}/phase3"

MEDIATOR_HOST = os.environ.get("MEDIATOR_HOST", "10.25.33.20")
MEDIATOR_USER = os.environ.get("MEDIATOR_USER", "root")
MEDIATOR_PASSWORD = os.environ.get("MEDIATOR_PASSWORD", VM_PASSWORD)
