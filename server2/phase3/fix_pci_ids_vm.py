#!/usr/bin/env python3
"""Apply Server 2 passthrough branding inside the guest VM.

This helper keeps the real passthrough/NVIDIA driver path intact and only
changes guest-facing branding:

- patch `pci.ids` so plain `lspci` shows `HEXACORE vH100 CAP`
- install a thin `nvidia-smi` wrapper at the real `/usr/bin/nvidia-smi` path
  so existing shells also pick up the branding without needing `hash -r`
- build and enable a tiny CUDA/NVML name-only preload shim so user-space
  frameworks such as TensorFlow, PyTorch, and Ollama can report `HEXACORE`
  without changing the real compute path
"""
import base64
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from vm_config import VM_HOST, VM_USER, VM_PASSWORD
from connect_vm import connect_and_run_command


def _read_asset(name: str) -> str:
    with open(os.path.join(SCRIPT_DIR, name), encoding="utf-8") as handle:
        return handle.read()


PY_FIX = """
import os
TARGET_IDS = ('2331', '2321')
for p in ['/usr/share/hwdata/pci.ids', '/usr/share/misc/pci.ids']:
    if os.path.isfile(p):
        with open(p) as f:
            lines = f.readlines()
        out = []
        added_in_nvidia = False
        for i, L in enumerate(lines):
            # Remove any existing Server 2 GPU aliases before re-adding them
            if L.startswith('\\t') and any(dev_id in L for dev_id in TARGET_IDS):
                continue
            out.append(L)
            # Add Server 2 aliases right after "10de  NVIDIA Corporation"
            if not added_in_nvidia and L.startswith('10de') and 'NVIDIA' in L and not L.startswith('\\t'):
                out.append('\\t2321  HEXACORE vH100 CAP\\n')
                out.append('\\t2331  HEXACORE vH100 CAP\\n')
                added_in_nvidia = True
        open(p, 'w').writelines(out)
        print('Updated', p, 'added_in_nvidia=', added_in_nvidia)
        break
"""

NVIDIA_SMI_WRAPPER = """#!/usr/bin/env python3
import os
import re
import signal
import subprocess
import sys
import threading

LIVE_NVIDIA_SMI = "/usr/bin/nvidia-smi"
REAL_NVIDIA_SMI = "/usr/bin/nvidia-smi.real"
HEXACORE_NAME = "HEXACORE vH100 CAP"
REAL_NAMES = (
    "NVIDIA H100 NVL",
    "NVIDIA H100 80GB HBM3",
    "NVIDIA H100L 94GB",
    "NVIDIA H100 PCIe",
)
GPU_ROW_RE = re.compile(r"^(\\|\\s*\\d+\\s+)(.+?)(\\s{2,}(?:On|Off)\\s+\\|.*)$")
PERSIST_RE = re.compile(r"^(\\s+)(On|Off)(\\s+\\|.*)$")


def replace_known_names(text):
    for old_name in REAL_NAMES:
        text = text.replace(old_name, HEXACORE_NAME)
    return text


def rewrite_table_row(text):
    newline = "\\n" if text.endswith("\\n") else ""
    body = text[:-1] if newline else text
    row_match = GPU_ROW_RE.match(body)
    if not row_match:
        return None
    prefix, detected_name, suffix = row_match.groups()
    if detected_name not in REAL_NAMES and detected_name != HEXACORE_NAME:
        return None
    persist_match = PERSIST_RE.match(suffix)
    if not persist_match:
        return None
    spacing, persistence, rest = persist_match.groups()
    name_field_width = len(detected_name) + len(spacing)
    formatted_name = HEXACORE_NAME[:name_field_width].ljust(name_field_width)
    return prefix + formatted_name + persistence + rest + newline


def rewrite_text(text):
    rewritten_row = rewrite_table_row(text)
    if rewritten_row is not None:
        return rewritten_row
    return replace_known_names(text)


def pump_stream(stream, target):
    try:
        for chunk in iter(stream.readline, ""):
            if not chunk:
                break
            target.write(rewrite_text(chunk))
            target.flush()
    finally:
        stream.close()


def main():
    if os.environ.get("HEXACORE_NVIDIA_SMI_BYPASS") == "1":
        os.execv(REAL_NVIDIA_SMI, [REAL_NVIDIA_SMI, *sys.argv[1:]])

    if not os.path.exists(REAL_NVIDIA_SMI):
        print(f"missing real nvidia-smi binary: {REAL_NVIDIA_SMI}", file=sys.stderr)
        return 127

    proc = subprocess.Popen(
        [REAL_NVIDIA_SMI, *sys.argv[1:]],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    workers = [
        threading.Thread(target=pump_stream, args=(proc.stdout, sys.stdout), daemon=True),
        threading.Thread(target=pump_stream, args=(proc.stderr, sys.stderr), daemon=True),
    ]
    for worker in workers:
        worker.start()
    try:
        rc = proc.wait()
    except KeyboardInterrupt:
        proc.send_signal(signal.SIGINT)
        rc = proc.wait()
    for worker in workers:
        worker.join()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
"""

PRELOAD_INSTALL_PY = """
import os
import shutil

PRELOAD_PATH = "/etc/ld.so.preload"
BACKUP_PATH = "/etc/ld.so.preload.hexacore.bak"
LIB_PATH = "/usr/local/lib/libhexacore_userland_name_preload.so"

if os.path.exists(PRELOAD_PATH) and not os.path.exists(BACKUP_PATH):
    shutil.copy2(PRELOAD_PATH, BACKUP_PATH)

lines = []
if os.path.exists(PRELOAD_PATH):
    with open(PRELOAD_PATH, "r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if line and line != LIB_PATH:
                lines.append(line)

lines.append(LIB_PATH)
with open(PRELOAD_PATH, "w", encoding="utf-8") as handle:
    handle.write("\\n".join(lines) + "\\n")

print("Updated", PRELOAD_PATH, "with", LIB_PATH)
"""

def main():
    pci_fix_b64 = base64.b64encode(PY_FIX.encode()).decode()
    wrapper_b64 = base64.b64encode(NVIDIA_SMI_WRAPPER.encode()).decode()
    preload_src_b64 = base64.b64encode(_read_asset("hexacore_userland_name_preload.c").encode()).decode()
    preload_install_b64 = base64.b64encode(PRELOAD_INSTALL_PY.encode()).decode()
    preload_probe_src_b64 = base64.b64encode(_read_asset("hexacore_userland_name_probe.c").encode()).decode()

    print("=== Step 1: Diagnose pci.ids ===")
    out1 = connect_and_run_command(
        "grep -n '2321\\|2331\\|10de' /usr/share/hwdata/pci.ids 2>/dev/null | head -20 || "
        "grep -n '2321\\|2331\\|10de' /usr/share/misc/pci.ids 2>/dev/null | head -20"
    )
    print(out1 or "(no output)")

    print("\n=== Step 2: Apply fix ===")
    cmd2 = "echo '%s' | base64 -d | sudo python3" % pci_fix_b64
    out2 = connect_and_run_command(cmd2)
    print(out2 or "(no output)")

    print("\n=== Step 3: Install nvidia-smi HEXACORE wrapper ===")
    cmd3 = (
        "if [ -x /usr/bin/nvidia-smi ]; then "
        "sudo install -d -m 755 /usr/local/bin /usr/local/sbin && "
        "if [ ! -x /usr/bin/nvidia-smi.real ]; then "
        "sudo dpkg-divert --quiet --local --rename "
        "--divert /usr/bin/nvidia-smi.real --add /usr/bin/nvidia-smi; "
        "fi && "
        "echo '%s' | base64 -d | sudo tee /usr/bin/nvidia-smi >/dev/null && "
        "sudo chmod 755 /usr/bin/nvidia-smi && "
        "sudo ln -sf /usr/bin/nvidia-smi /usr/local/bin/nvidia-smi && "
        "sudo ln -sf /usr/bin/nvidia-smi /usr/local/sbin/nvidia-smi && "
        "echo 'Installed /usr/bin/nvidia-smi wrapper'; "
        "else echo 'SKIP: /usr/bin/nvidia-smi not present'; fi"
    ) % wrapper_b64
    out3 = connect_and_run_command(cmd3)
    print(out3 or "(no output)")

    print("\n=== Step 4: Build and test the user-space HEXACORE preload shim ===")
    cmd4 = (
        "sudo install -d -m 755 /usr/local/lib /usr/local/src/hexacore && "
        "echo '%s' | base64 -d | sudo tee /usr/local/src/hexacore/hexacore_userland_name_preload.c >/dev/null && "
        "echo '%s' | base64 -d | sudo tee /usr/local/src/hexacore/hexacore_userland_name_probe.c >/dev/null && "
        "sudo gcc -shared -fPIC -O2 -Wall -Wextra -std=c11 -D_GNU_SOURCE "
        "/usr/local/src/hexacore/hexacore_userland_name_preload.c -ldl "
        "-o /usr/local/lib/libhexacore_userland_name_preload.so && "
        "sudo gcc -O2 -Wall -Wextra -std=c11 "
        "/usr/local/src/hexacore/hexacore_userland_name_probe.c "
        "/lib/x86_64-linux-gnu/libcuda.so.1 /lib/x86_64-linux-gnu/libnvidia-ml.so.1 "
        "-o /usr/local/bin/hexacore_userland_name_probe && "
        "sudo chmod 755 /usr/local/lib/libhexacore_userland_name_preload.so && "
        "sudo chmod 755 /usr/local/bin/hexacore_userland_name_probe && "
        "LD_PRELOAD=/usr/local/lib/libhexacore_userland_name_preload.so "
        "/usr/local/bin/hexacore_userland_name_probe"
    ) % (preload_src_b64, preload_probe_src_b64)
    out4 = connect_and_run_command(cmd4)
    print(out4 or "(no output)")

    print("\n=== Step 5: Enable the preload shim globally for new user-space processes ===")
    out5 = connect_and_run_command(
        "echo '%s' | base64 -d | sudo python3 && "
        "(sudo systemctl restart ollama >/dev/null 2>&1 || true) && "
        "/usr/local/bin/hexacore_userland_name_probe"
        % preload_install_b64
    )
    print(out5 or "(no output)")

    print("\n=== Step 6: Verify lspci ===")
    out6 = connect_and_run_command("lspci | grep -i 'HEXACORE\\|NVIDIA'")
    print(out6 or "(no output)")

    print("\n=== Step 7: Verify nvidia-smi branding ===")
    out7 = connect_and_run_command(
        "command -v nvidia-smi; "
        "nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || true; "
        "nvidia-smi | sed -n '1,10p'; "
        "HEXACORE_NVIDIA_SMI_BYPASS=1 /usr/bin/nvidia-smi "
        "--query-gpu=name --format=csv,noheader 2>/dev/null || true"
    )
    print(out7 or "(no output)")

    if (
        out6
        and "HEXACORE" in out6
        and out7
        and "HEXACORE vH100 CAP" in out7
        and out5
        and "cuDeviceGetName_name=HEXACORE vH100 CAP" in out5
        and "nvmlDeviceGetName_name=HEXACORE vH100 CAP" in out5
    ):
        print(
            "\n*** SUCCESS: HEXACORE branding now appears in lspci, nvidia-smi, CUDA, and NVML user-space name queries ***"
        )
    elif out6 and "HEXACORE" in out6:
        print("\n*** lspci branding is active; check the CUDA/NVML and nvidia-smi verification output above ***")

    return 0

if __name__ == "__main__":
    sys.exit(main())
