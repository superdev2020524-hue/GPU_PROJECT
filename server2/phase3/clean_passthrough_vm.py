#!/usr/bin/env python3
"""Clean a Server 2 guest for pure PCI passthrough.

This helper removes leftover mediated-path guest overrides that shadow the real
NVIDIA driver inside a passthrough VM, then reapplies the `HEXACORE` lspci
branding for the real H100 PCI ID.

It is intentionally guest-only and keeps all logic inside the Server 2 registry.
"""

import base64
import os
import re
import subprocess
import sys


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from vm_config import REMOTE_PHASE3, VM_HOST, VM_USER  # noqa: E402


def log(message=""):
    print(message, flush=True)


def run_vm(command, timeout_sec=180):
    result = subprocess.run(
        [sys.executable, os.path.join(SCRIPT_DIR, "connect_vm.py"), command],
        capture_output=True,
        text=True,
        timeout=timeout_sec,
    )
    stdout = result.stdout or ""
    stderr = result.stderr or ""
    success = result.returncode == 0
    match = re.search(r"Remote command exit code:\s*(\d+)", stdout)
    if match:
        success = success and int(match.group(1)) == 0
    return success, stdout, stderr


def run_or_fail(step, command, timeout_sec=180):
    log(f"\n=== {step} ===")
    ok, out, err = run_vm(command, timeout_sec=timeout_sec)
    if out.strip():
        log(out.strip())
    if not ok:
        if err.strip():
            log(err.strip())
        raise RuntimeError(f"{step} failed")


REMOTE_CLEANUP_PY = r"""
import os
import shutil
import subprocess
import time


BACKUP_ROOT = "/root/server2_passthrough_cleanup_" + time.strftime("%Y%m%d_%H%M%S")
os.makedirs(BACKUP_ROOT, exist_ok=True)


def backup_path(path):
    if not os.path.lexists(path):
        return
    safe_name = path.lstrip("/").replace("/", "__") or "root"
    dst = os.path.join(BACKUP_ROOT, safe_name)
    if os.path.islink(path):
        with open(dst + ".symlink", "w", encoding="utf-8") as f:
            f.write(os.readlink(path))
        return
    if os.path.isdir(path):
        shutil.copytree(path, dst, symlinks=True)
        return
    shutil.copy2(path, dst)


def remove_path(path):
    if not os.path.lexists(path):
        return
    backup_path(path)
    if os.path.isdir(path) and not os.path.islink(path):
        shutil.rmtree(path)
    else:
        os.remove(path)


def replace_symlink(path, target):
    if os.path.lexists(path):
        backup_path(path)
        if os.path.isdir(path) and not os.path.islink(path):
            shutil.rmtree(path)
        else:
            os.remove(path)
    os.symlink(target, path)


def patch_pci_ids():
    target_ids = ("2321", "2331")
    for pci_ids in ("/usr/share/hwdata/pci.ids", "/usr/share/misc/pci.ids"):
        if not os.path.isfile(pci_ids):
            continue
        with open(pci_ids, encoding="utf-8") as f:
            lines = f.readlines()

        out = []
        added = False
        for line in lines:
            if line.startswith("\t") and any(dev_id in line for dev_id in target_ids):
                continue
            out.append(line)
            if not added and line.startswith("10de") and "NVIDIA" in line and not line.startswith("\t"):
                out.append("\t2321  HEXACORE vH100 CAP\n")
                out.append("\t2331  HEXACORE vH100 CAP\n")
                added = True

        backup_path(pci_ids)
        with open(pci_ids, "w", encoding="utf-8") as f:
            f.writelines(out)
        return pci_ids
    raise SystemExit("No pci.ids file found on guest")


for path in (
    "/etc/modprobe.d/blacklist-nvidia-real.conf",
    "/etc/profile.d/vgpu-cuda.sh",
    "/etc/ld.so.conf.d/vgpu-lib64.conf",
    "/etc/udev/rules.d/99-vgpu-nvidia.rules",
    "/etc/systemd/system/vgpu-devices.service",
    "/etc/systemd/system/ollama.service.d/vgpu.conf",
):
    remove_path(path)

ollama_dropin_dir = "/etc/systemd/system/ollama.service.d"
if os.path.isdir(ollama_dropin_dir) and not os.listdir(ollama_dropin_dir):
    os.rmdir(ollama_dropin_dir)

os.makedirs("/usr/lib64", exist_ok=True)
replace_symlink("/usr/lib64/libcuda.so.1", "/lib/x86_64-linux-gnu/libcuda.so.1")
replace_symlink("/usr/lib64/libcuda.so", "libcuda.so.1")
replace_symlink("/usr/lib64/libnvidia-ml.so.1", "/lib/x86_64-linux-gnu/libnvidia-ml.so.1")
replace_symlink("/usr/lib64/libnvidia-ml.so", "libnvidia-ml.so.1")

for stale in (
    "/usr/lib64/libcudart.so",
    "/usr/lib64/libcudart.so.12",
    "/usr/lib64/libvgpu-cuda.so",
    "/usr/lib64/libvgpu-cudart.so",
    "/usr/lib64/libvgpu-exec.so",
    "/usr/lib64/libvgpu-nvml.so",
    "/usr/lib64/libvgpu-syscall.so",
):
    if os.path.lexists(stale):
        backup_path(stale)
        os.remove(stale)

subprocess.run(["systemctl", "disable", "--now", "vgpu-devices.service"], check=False)
subprocess.run(["systemctl", "daemon-reload"], check=False)
subprocess.run(["ldconfig"], check=False)
subprocess.run(["update-initramfs", "-u"], check=False)

pci_ids_path = patch_pci_ids()

print("BACKUP_ROOT=" + BACKUP_ROOT)
print("PCI_IDS_PATH=" + pci_ids_path)
"""


def main():
    log(f"Target VM: {VM_USER}@{VM_HOST}")
    log(f"Remote phase3 path: {REMOTE_PHASE3}")
    log("Lane: Server 2 passthrough fast path")
    log("Active error: mixed guest mediated-path overrides shadow the real passthrough driver")

    run_or_fail(
        "Pre-check passthrough identity",
        (
            "lspci | grep -i '3D controller\\|HEXACORE\\|NVIDIA'; "
            "lspci -nn | grep -i '10de:2321\\|10de:2331'; "
            "lsmod | grep -E '^nvidia' || true; "
            "readlink -f /usr/lib64/libcuda.so.1 2>/dev/null || true; "
            "readlink -f /usr/lib64/libnvidia-ml.so.1 2>/dev/null || true"
        ),
        timeout_sec=120,
    )

    cleanup_b64 = base64.b64encode(REMOTE_CLEANUP_PY.encode("utf-8")).decode("ascii")
    run_or_fail(
        "Remove mediated-path guest overrides and restore real driver links",
        f"echo '{cleanup_b64}' | base64 -d | python3",
        timeout_sec=1200,
    )

    run_or_fail(
        "Restart Ollama on clean passthrough environment",
        "systemctl restart ollama && sleep 3 && systemctl is-active ollama",
        timeout_sec=120,
    )

    run_or_fail(
        "Verify HEXACORE lspci and NVIDIA driver health",
        (
            "lspci | grep -i 'HEXACORE\\|NVIDIA\\|3D controller\\|VGA'; "
            "echo '---'; "
            "lspci -nn | grep -i '10de:2321\\|10de:2331'; "
            "echo '---'; "
            "nvidia-smi; "
            "echo '---'; "
            "python3 -c \"import ctypes as C; "
            "cuda=C.CDLL('libcuda.so.1'); "
            "rc=cuda.cuInit(0); "
            "count=C.c_int(); "
            "rc2=cuda.cuDeviceGetCount(C.byref(count)); "
            "name=C.create_string_buffer(100); "
            "rc3=cuda.cuDeviceGetName(name, len(name), 0); "
            "decoded=name.value.decode(errors='replace'); "
            "print(f'cuInit={rc}'); "
            "print(f'cuDeviceGetCount={rc2} count={count.value}'); "
            "print(f'cuDeviceGetName={rc3} name={decoded}'); "
            "raise SystemExit(0 if rc == 0 and rc2 == 0 and count.value >= 1 else 1)\""
        ),
        timeout_sec=300,
    )

    ok, out, _ = run_vm("ollama list", timeout_sec=60)
    models = out.strip() if ok else ""
    if models and "NAME" in models:
        log("\n=== Optional Ollama inventory ===")
        log(models)
        model = None
        for candidate in ("qwen2.5:0.5b", "tinyllama:latest", "llama3.2:1b"):
            if candidate in models:
                model = candidate
                break
        if model:
            run_or_fail(
                f"Run short Ollama generate on {model}",
                (
                    "curl -s http://127.0.0.1:11434/api/generate "
                    f"-d '{{\"model\":\"{model}\",\"prompt\":\"Reply with exactly OK.\",\"stream\":false}}'; "
                    "echo '---'; "
                    "journalctl -u ollama --no-pager -n 80 | grep -E 'inference compute|offloaded|CUDA' | tail -20 || true"
                ),
                timeout_sec=300,
            )
        else:
            log("\nNo preferred test model is installed, so skipping the short Ollama generate.")
    else:
        log("\nNo Ollama model inventory was available, so skipping the short Ollama generate.")

    log("\nServer 2 passthrough guest cleanup completed.")


if __name__ == "__main__":
    main()
