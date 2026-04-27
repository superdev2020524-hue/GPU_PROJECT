#!/usr/bin/env python3
"""Bootstrap the Server 2 guest shim stack without touching Ollama.

This script is intentionally narrower than `guest-shim/install.sh`:
- it does not install or restart Ollama
- it does not write Ollama systemd drop-ins
- it focuses on the general CUDA/NVML/runtime path for Server 2
"""

import base64
import os
import pexpect
import re
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from vm_config import REMOTE_PHASE3, VM_HOST, VM_PASSWORD, VM_USER


CHUNK_SIZE = 40000


def log(message=""):
    print(message, flush=True)


def run_vm(cmd, timeout_sec=120):
    result = subprocess.run(
        [sys.executable, os.path.join(SCRIPT_DIR, "connect_vm.py"), cmd],
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


def send_file_scp(local_path, remote_path, timeout_sec=180):
    child = None
    try:
        child = pexpect.spawn(
            "scp",
            [
                "-o",
                "StrictHostKeyChecking=no",
                "-o",
                "ConnectTimeout=30",
                local_path,
                f"{VM_USER}@{VM_HOST}:{remote_path}",
            ],
            encoding="utf-8",
            timeout=timeout_sec,
        )
        while True:
            idx = child.expect(
                [
                    "Are you sure you want to continue connecting",
                    "[Pp]assword:",
                    pexpect.EOF,
                    pexpect.TIMEOUT,
                ]
            )
            if idx == 0:
                child.sendline("yes")
            elif idx == 1:
                child.sendline(VM_PASSWORD)
            elif idx == 2:
                child.close()
                return child.exitstatus == 0
            else:
                return False
    except Exception:
        return False
    finally:
        try:
            if child is not None and child.isalive():
                child.close(force=True)
        except Exception:
            pass


def send_file(local_path, remote_path, timeout_sec=180):
    if not os.path.isfile(local_path):
        log(f"Missing local file: {local_path}")
        return False

    if send_file_scp(local_path, remote_path, timeout_sec=timeout_sec):
        return True

    log(f"SCP transfer failed for {local_path}; falling back to base64 chunks")

    with open(local_path, "rb") as f:
        data = f.read()

    b64 = base64.b64encode(data).decode("ascii")

    def escape(chunk):
        return chunk.replace("'", "'\"'\"'")

    ok, _, _ = run_vm("rm -f /tmp/combined.b64", timeout_sec=30)
    if not ok:
        return False

    for i in range(0, len(b64), CHUNK_SIZE):
        chunk = b64[i : i + CHUNK_SIZE]
        ok, _, _ = run_vm(
            "echo -n '" + escape(chunk) + "' >> /tmp/combined.b64",
            timeout_sec=60,
        )
        if not ok:
            return False

    ok, _, _ = run_vm(
        f"base64 -d /tmp/combined.b64 > /tmp/out.tmp && mv /tmp/out.tmp {remote_path}",
        timeout_sec=30,
    )
    return ok


def run_or_fail(step, cmd, timeout_sec=120):
    log(f"\n=== {step} ===")
    ok, out, err = run_vm(cmd, timeout_sec=timeout_sec)
    if out.strip():
        log(out.strip())
    if not ok:
        if err.strip():
            log(err.strip())
        raise RuntimeError(f"{step} failed")


def main():
    log(f"Target VM: {VM_USER}@{VM_HOST}")
    log(f"Remote phase3 path: {REMOTE_PHASE3}")

    run_or_fail(
        "Ensure build tools",
        (
            "if command -v gcc >/dev/null 2>&1; then "
            "echo 'gcc already installed'; "
            "else "
            f"apt-get update -qq && apt-get install -y -qq gcc make libc6-dev; "
            "fi"
        ),
        timeout_sec=300,
    )

    run_or_fail(
        "Prepare directories",
        (
            f"mkdir -p {REMOTE_PHASE3}/guest-shim {REMOTE_PHASE3}/include "
            "/usr/lib64 /etc/udev/rules.d /etc/profile.d /etc/ld.so.conf.d /etc/modprobe.d"
        ),
        timeout_sec=60,
    )

    files = [
        ("guest-shim/libvgpu_cuda.c", "guest-shim/libvgpu_cuda.c"),
        ("guest-shim/libvgpu_nvml.c", "guest-shim/libvgpu_nvml.c"),
        ("guest-shim/libvgpu_cudart.c", "guest-shim/libvgpu_cudart.c"),
        ("guest-shim/cuda_transport.c", "guest-shim/cuda_transport.c"),
        ("guest-shim/cuda_transport.h", "guest-shim/cuda_transport.h"),
        ("guest-shim/gpu_properties.h", "guest-shim/gpu_properties.h"),
        ("guest-shim/libcudart.so.12.versionscript", "guest-shim/libcudart.so.12.versionscript"),
        ("include/cuda_protocol.h", "include/cuda_protocol.h"),
    ]

    log("\n=== Transfer guest shim sources ===")
    for rel_local, rel_remote in files:
        local_path = os.path.join(SCRIPT_DIR, rel_local)
        if not os.path.exists(local_path):
            if rel_local.endswith("libcudart.so.12.versionscript"):
                log(f"Optional file missing locally, skipping: {rel_local}")
                continue
            raise FileNotFoundError(local_path)
        remote_path = os.path.join(REMOTE_PHASE3, rel_remote)
        log(f"Sending {rel_local} -> {remote_path}")
        if not send_file(local_path, remote_path):
            raise RuntimeError(f"Failed to transfer {rel_local}")

    build_cmd = f"""
set -e
cd {REMOTE_PHASE3}
gcc -shared -fPIC -O2 -Wall -Wextra -std=c11 -D_GNU_SOURCE \
  -Iinclude -Iguest-shim \
  -Wl,-soname,libcuda.so.1 \
  -o guest-shim/libvgpu-cuda.so \
  guest-shim/libvgpu_cuda.c guest-shim/cuda_transport.c -lpthread -ldl
gcc -shared -fPIC -O2 -Wall -Wextra -std=c11 -D_GNU_SOURCE \
  -Iinclude -Iguest-shim \
  -Wl,-soname,libnvidia-ml.so.1 -Wl,--allow-shlib-undefined \
  -o guest-shim/libvgpu-nvml.so \
  guest-shim/libvgpu_nvml.c guest-shim/cuda_transport.c -lpthread -ldl
if [ -f guest-shim/libcudart.so.12.versionscript ]; then
  gcc -shared -fPIC -O2 -Wall -Wextra -std=c11 -D_GNU_SOURCE \
    -Iinclude -Iguest-shim \
    -Wl,--version-script=guest-shim/libcudart.so.12.versionscript \
    -Wl,-soname,libcudart.so.12 \
    -o guest-shim/libvgpu-cudart.so \
    guest-shim/libvgpu_cudart.c -ldl -lpthread
else
  gcc -shared -fPIC -O2 -Wall -Wextra -std=c11 -D_GNU_SOURCE \
    -Iinclude -Iguest-shim \
    -Wl,-soname,libcudart.so.12 \
    -o guest-shim/libvgpu-cudart.so \
    guest-shim/libvgpu_cudart.c -ldl -lpthread
fi
ls -la guest-shim/libvgpu-cuda.so guest-shim/libvgpu-nvml.so guest-shim/libvgpu-cudart.so
"""
    run_or_fail("Build guest shims", build_cmd, timeout_sec=600)

    install_cmd = f"""
set -e
cd {REMOTE_PHASE3}
install -m 755 guest-shim/libvgpu-cuda.so /usr/lib64/
install -m 755 guest-shim/libvgpu-nvml.so /usr/lib64/
install -m 755 guest-shim/libvgpu-cudart.so /usr/lib64/

for lib in libcuda.so.1 libcuda.so libnvidia-ml.so.1 libnvidia-ml.so libcudart.so.12 libcudart.so; do
  for d in /usr/lib64 /usr/lib/x86_64-linux-gnu /lib/x86_64-linux-gnu; do
    [ -d "$d" ] || continue
    if [ -f "$d/$lib" ] && [ ! -L "$d/$lib" ] && [ ! -f "$d/$lib.real" ]; then
      mv "$d/$lib" "$d/$lib.real" 2>/dev/null || true
    fi
  done
done

ln -sf /usr/lib64/libvgpu-cuda.so /usr/lib64/libcuda.so.1
ln -sf /usr/lib64/libvgpu-cuda.so /usr/lib64/libcuda.so
ln -sf /usr/lib64/libvgpu-nvml.so /usr/lib64/libnvidia-ml.so.1
ln -sf /usr/lib64/libvgpu-nvml.so /usr/lib64/libnvidia-ml.so
ln -sf /usr/lib64/libvgpu-cudart.so /usr/lib64/libcudart.so.12
ln -sf /usr/lib64/libvgpu-cudart.so /usr/lib64/libcudart.so

for d in /usr/lib/x86_64-linux-gnu /lib/x86_64-linux-gnu; do
  [ -d "$d" ] || continue
  cp -f /usr/lib64/libvgpu-cuda.so "$d/libcuda.so.1" 2>/dev/null || true
  cp -f /usr/lib64/libvgpu-nvml.so "$d/libnvidia-ml.so.1" 2>/dev/null || true
  cp -f /usr/lib64/libvgpu-cudart.so "$d/libcudart.so.12" 2>/dev/null || true
  ln -sf libcuda.so.1 "$d/libcuda.so" 2>/dev/null || true
  ln -sf libnvidia-ml.so.1 "$d/libnvidia-ml.so" 2>/dev/null || true
  ln -sf libcudart.so.12 "$d/libcudart.so" 2>/dev/null || true
done

cat > /etc/modprobe.d/blacklist-nvidia-real.conf <<'EOF'
blacklist nvidia
blacklist nvidia_drm
blacklist nvidia_modeset
blacklist nvidia_uvm
EOF

for dev in /dev/nvidia0 /dev/nvidiactl /dev/nvidia-uvm /dev/nvidia-uvm-tools; do
  case "$dev" in
    /dev/nvidia0) [ -e "$dev" ] || mknod "$dev" c 195 0 ;;
    /dev/nvidiactl) [ -e "$dev" ] || mknod "$dev" c 195 255 ;;
    /dev/nvidia-uvm) [ -e "$dev" ] || mknod "$dev" c 510 0 ;;
    /dev/nvidia-uvm-tools) [ -e "$dev" ] || mknod "$dev" c 510 1 ;;
  esac
  chmod 666 "$dev" 2>/dev/null || true
done

for d in /sys/bus/pci/devices/*; do
  v=$(cat "$d/vendor" 2>/dev/null || true)
  dev=$(cat "$d/device" 2>/dev/null || true)
  if [ "$v" = "0x10de" ] && [ "$dev" = "0x2331" ]; then
    chmod 0666 "$d/resource0" "$d/resource1" 2>/dev/null || true
  fi
done

cat > /etc/udev/rules.d/99-vgpu-nvidia.rules <<'EOF'
KERNEL=="nvidia[0-9]*", RUN+="/bin/chmod 666 /dev/%k"
KERNEL=="nvidiactl", RUN+="/bin/chmod 666 /dev/%k"
KERNEL=="nvidia-uvm", RUN+="/bin/chmod 666 /dev/%k"
KERNEL=="nvidia-uvm-tools", RUN+="/bin/chmod 666 /dev/%k"
SUBSYSTEM=="pci", ATTR{{vendor}}=="0x10de", ATTR{{device}}=="0x2331", RUN+="/bin/chmod 0666 /sys%p/resource0 /sys%p/resource1"
EOF

cat > /etc/profile.d/vgpu-cuda.sh <<'EOF'
export NVIDIA_VISIBLE_DEVICES=all
export NVIDIA_DRIVER_CAPABILITIES=compute,utility
if [ -d /usr/lib64 ]; then
  export LD_LIBRARY_PATH="${{LD_LIBRARY_PATH:+${{LD_LIBRARY_PATH}}:}}/usr/lib64"
fi
EOF

cat > /etc/ld.so.conf.d/vgpu-lib64.conf <<'EOF'
/usr/lib64
EOF
ldconfig 2>/dev/null || true

ls -la /usr/lib64/libvgpu-cuda.so /usr/lib64/libvgpu-nvml.so /usr/lib64/libvgpu-cudart.so
ls -la /usr/lib64/libcuda.so.1 /usr/lib64/libnvidia-ml.so.1 /usr/lib64/libcudart.so.12
"""
    run_or_fail("Install guest shims", install_cmd, timeout_sec=180)

    run_or_fail(
        "Check Secure Boot / lockdown state",
        (
            "printf 'SecureBoot: '; mokutil --sb-state 2>/dev/null || true; "
            "printf 'Lockdown: '; cat /sys/kernel/security/lockdown 2>/dev/null || true"
        ),
        timeout_sec=60,
    )

    verify_py = r"""
import ctypes
import glob
import mmap
import os
import sys

def check_bar0_access():
    matches = []
    for devdir in sorted(glob.glob("/sys/bus/pci/devices/*")):
        try:
            vendor = open(os.path.join(devdir, "vendor")).read().strip()
            device = open(os.path.join(devdir, "device")).read().strip()
        except OSError:
            continue
        if vendor == "0x10de" and device == "0x2331":
            matches.append(devdir)

    if not matches:
        print("BAR0 check: no 10de:2331 device found")
        raise SystemExit(1)

    bar0 = os.path.join(matches[0], "resource0")
    fd = os.open(bar0, os.O_RDWR | getattr(os, "O_SYNC", 0))
    try:
        mm = mmap.mmap(
            fd,
            4096,
            flags=mmap.MAP_SHARED,
            prot=mmap.PROT_READ | mmap.PROT_WRITE,
        )
    except OSError as exc:
        print(f"BAR0 mmap failed for {bar0}: {exc}")
        print("Likely blocker: guest Secure Boot / kernel lockdown is denying direct PCI BAR access.")
        raise SystemExit(1)
    else:
        print(f"BAR0 mmap ok: {bar0}")
        mm.close()
    finally:
        os.close(fd)

def check_cuda():
    cuda = ctypes.CDLL("libcuda.so.1")
    cuInit = cuda.cuInit
    cuInit.argtypes = [ctypes.c_uint]
    cuInit.restype = ctypes.c_int
    cuDeviceGetCount = cuda.cuDeviceGetCount
    cuDeviceGetCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
    cuDeviceGetCount.restype = ctypes.c_int
    cuDeviceGet = cuda.cuDeviceGet
    cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    cuDeviceGet.restype = ctypes.c_int
    cuDeviceGetName = cuda.cuDeviceGetName
    cuDeviceGetName.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
    cuDeviceGetName.restype = ctypes.c_int
    cuMemAlloc = cuda.cuMemAlloc_v2
    cuMemAlloc.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t]
    cuMemAlloc.restype = ctypes.c_int
    cuMemFree = cuda.cuMemFree_v2
    cuMemFree.argtypes = [ctypes.c_uint64]
    cuMemFree.restype = ctypes.c_int
    cuMemcpyHtoD = cuda.cuMemcpyHtoD_v2
    cuMemcpyHtoD.argtypes = [ctypes.c_uint64, ctypes.c_void_p, ctypes.c_size_t]
    cuMemcpyHtoD.restype = ctypes.c_int
    cuMemcpyDtoH = cuda.cuMemcpyDtoH_v2
    cuMemcpyDtoH.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_size_t]
    cuMemcpyDtoH.restype = ctypes.c_int

    rc = cuInit(0)
    print(f"cuInit rc={rc}")
    if rc != 0:
        raise SystemExit(1)

    count = ctypes.c_int()
    rc = cuDeviceGetCount(ctypes.byref(count))
    print(f"cuDeviceGetCount rc={rc} count={count.value}")
    if rc != 0 or count.value < 1:
        raise SystemExit(1)

    dev = ctypes.c_int()
    rc = cuDeviceGet(ctypes.byref(dev), 0)
    print(f"cuDeviceGet rc={rc} dev={dev.value}")
    if rc != 0:
        raise SystemExit(1)

    buf = ctypes.create_string_buffer(100)
    rc = cuDeviceGetName(buf, len(buf), dev.value)
    name = buf.value.decode("utf-8", "ignore")
    print(f"cuDeviceGetName rc={rc} name={name}")
    if rc != 0:
        raise SystemExit(1)

    ptr = ctypes.c_uint64()
    rc = cuMemAlloc(ctypes.byref(ptr), 4096)
    print(f"cuMemAlloc_v2 rc={rc} ptr={ptr.value}")
    if rc != 0 or ptr.value == 0:
        raise SystemExit(1)

    payload = bytes([1, 2, 3, 4, 5, 6, 7, 8])
    src = ctypes.create_string_buffer(payload)
    dst = ctypes.create_string_buffer(len(payload))

    rc = cuMemcpyHtoD(ptr.value, src, len(payload))
    print(f"cuMemcpyHtoD_v2 rc={rc}")
    if rc != 0:
        raise SystemExit(1)

    rc = cuMemcpyDtoH(dst, ptr.value, len(payload))
    print(f"cuMemcpyDtoH_v2 rc={rc}")
    if rc != 0:
        raise SystemExit(1)

    dst_bytes = bytes(dst.raw)
    print(f"roundtrip src={list(payload)}")
    print(f"roundtrip dst={list(dst_bytes)}")
    if dst_bytes != payload:
        raise SystemExit(1)

    rc = cuMemFree(ptr.value)
    print(f"cuMemFree_v2 rc={rc}")
    if rc != 0:
        raise SystemExit(1)

def check_nvml():
    nvml = ctypes.CDLL("libnvidia-ml.so.1")
    nvmlInit_v2 = nvml.nvmlInit_v2
    nvmlInit_v2.restype = ctypes.c_int
    nvmlDeviceGetCount_v2 = nvml.nvmlDeviceGetCount_v2
    nvmlDeviceGetCount_v2.argtypes = [ctypes.POINTER(ctypes.c_uint)]
    nvmlDeviceGetCount_v2.restype = ctypes.c_int
    nvmlDeviceGetHandleByIndex_v2 = nvml.nvmlDeviceGetHandleByIndex_v2
    nvmlDeviceGetHandleByIndex_v2.argtypes = [ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p)]
    nvmlDeviceGetHandleByIndex_v2.restype = ctypes.c_int
    nvmlDeviceGetName = nvml.nvmlDeviceGetName
    nvmlDeviceGetName.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint]
    nvmlDeviceGetName.restype = ctypes.c_int

    rc = nvmlInit_v2()
    print(f"nvmlInit_v2 rc={rc}")
    if rc != 0:
        raise SystemExit(1)

    count = ctypes.c_uint()
    rc = nvmlDeviceGetCount_v2(ctypes.byref(count))
    print(f"nvmlDeviceGetCount_v2 rc={rc} count={count.value}")
    if rc != 0 or count.value < 1:
        raise SystemExit(1)

    handle = ctypes.c_void_p()
    rc = nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(handle))
    print(f"nvmlDeviceGetHandleByIndex_v2 rc={rc} handle={handle.value}")
    if rc != 0:
        raise SystemExit(1)

    buf = ctypes.create_string_buffer(100)
    rc = nvmlDeviceGetName(handle, buf, len(buf))
    name = buf.value.decode("utf-8", "ignore")
    print(f"nvmlDeviceGetName rc={rc} name={name}")
    if rc != 0:
        raise SystemExit(1)

def check_cudart():
    cudart = ctypes.CDLL("libcudart.so.12")
    cudaGetDeviceCount = cudart.cudaGetDeviceCount
    cudaGetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
    cudaGetDeviceCount.restype = ctypes.c_int
    count = ctypes.c_int()
    rc = cudaGetDeviceCount(ctypes.byref(count))
    print(f"cudaGetDeviceCount rc={rc} count={count.value}")
    if rc != 0 or count.value < 1:
        raise SystemExit(1)

check_bar0_access()
check_cuda()
check_nvml()
check_cudart()
"""
    verify_b64 = base64.b64encode(verify_py.encode("utf-8")).decode("ascii")
    verify_cmd = (
        "echo '" + verify_b64 + "' | base64 -d > /tmp/vgpu_verify.py && "
        "LD_LIBRARY_PATH=/usr/lib64 python3 /tmp/vgpu_verify.py"
    )
    run_or_fail("Verify CUDA/NVML/runtime path", verify_cmd, timeout_sec=180)

    run_or_fail(
        "Final PCI check",
        "lspci | sed -n '/00:05/p'; ls -la /sys/bus/pci/devices/0000:00:05.0/resource0 /sys/bus/pci/devices/0000:00:05.0/resource1 2>/dev/null || true",
        timeout_sec=60,
    )

    log("\nServer 2 guest bootstrap completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
