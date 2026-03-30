#!/usr/bin/env python3
"""Fix pci.ids on VM to show HEXACORE vH100 CAP for device 2331."""
import base64
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from vm_config import VM_HOST, VM_USER, VM_PASSWORD
from connect_vm import connect_and_run_command

PY_FIX = """
import os
for p in ['/usr/share/hwdata/pci.ids', '/usr/share/misc/pci.ids']:
    if os.path.isfile(p):
        with open(p) as f:
            lines = f.readlines()
        out = []
        added_in_nvidia = False
        for i, L in enumerate(lines):
            # Remove wrongly placed 2331 (in Intel block) - line with 2331 but not HEXACORE
            if L.startswith('\\t') and '2331' in L and 'HEXACORE' not in L:
                continue  # skip wrong entry
            # Remove our correct one too if re-running, we'll add fresh
            if '2331' in L and 'HEXACORE' in L:
                continue
            out.append(L)
            # Add 2331 right after "10de  NVIDIA Corporation" (no leading tab = vendor)
            if not added_in_nvidia and L.startswith('10de') and 'NVIDIA' in L and not L.startswith('\\t'):
                out.append('\\t2331  HEXACORE vH100 CAP\\n')
                added_in_nvidia = True
        open(p, 'w').writelines(out)
        print('Updated', p, 'added_in_nvidia=', added_in_nvidia)
        break
"""

def main():
    b64 = base64.b64encode(PY_FIX.encode()).decode()

    print("=== Step 1: Diagnose pci.ids ===")
    out1 = connect_and_run_command(
        "grep -n '2331\\|10de' /usr/share/hwdata/pci.ids 2>/dev/null | head -20 || "
        "grep -n '2331\\|10de' /usr/share/misc/pci.ids 2>/dev/null | head -20"
    )
    print(out1 or "(no output)")

    print("\n=== Step 2: Apply fix ===")
    cmd2 = "echo '%s' | base64 -d | sudo python3" % b64
    out2 = connect_and_run_command(cmd2)
    print(out2 or "(no output)")

    print("\n=== Step 3: Verify lspci ===")
    out3 = connect_and_run_command("lspci | grep 00:05")
    print(out3 or "(no output)")

    if out3 and "HEXACORE" in out3:
        print("\n*** SUCCESS: HEXACORE vH100 CAP now appears in lspci ***")
    elif out3:
        print("\n*** lspci output above - HEXACORE may need manual pci.ids edit ***")

    return 0

if __name__ == "__main__":
    sys.exit(main())
