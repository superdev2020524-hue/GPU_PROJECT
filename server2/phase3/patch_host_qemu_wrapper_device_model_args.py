#!/usr/bin/env python3
"""Patch qemu-wrapper to honor XenStore device-model-args.

Server 2 issue:
- `vgpu-admin register-vm` writes `platform:device-model-args`
- XenStore exposes `/local/domain/<domid>/platform/device-model-args`
- stock-style `/usr/lib64/xen/bin/qemu-wrapper` ignores that key
- QEMU starts without `-device vgpu-cuda,...`

This patch adds two behaviors:
1. read XenStore `device-model-args` and append it to `qemu_args`
2. reserve lower MMIO hole space when the XenStore args request a vGPU device
"""

import argparse
import datetime as _dt
from pathlib import Path
import sys


IMPORT_NEEDLE = "import subprocess\n"
IMPORT_ADD = "import subprocess\nimport shlex\n"

BLOCK_NEEDLE = (
    "    qemu_dm = '/usr/lib64/xen/bin/qemu-system-i386'\n"
    "    qemu_args = ['qemu-dm-%d' % domid]\n"
    "\n"
    "    mmio_start = HVM_BELOW_4G_MMIO_START\n"
    "    # vGPU now requires extra space in lower MMIO hole by default\n"
    "    if '-vgpu' in argv:\n"
    "        mmio_start -= HVM_BELOW_4G_MMIO_LENGTH\n"
)

BLOCK_REPLACE = (
    "    qemu_dm = '/usr/lib64/xen/bin/qemu-system-i386'\n"
    "    qemu_args = ['qemu-dm-%d' % domid]\n"
    "\n"
    "    device_model_args = xenstore_read(\n"
    "        \"/local/domain/%d/platform/device-model-args\" % domid\n"
    "    )\n"
    "    if device_model_args and isinstance(device_model_args, bytes):\n"
    "        device_model_args = device_model_args.decode('utf-8', 'ignore')\n"
    "    wants_vgpu = '-vgpu' in argv\n"
    "    if device_model_args:\n"
    "        wants_vgpu = wants_vgpu or ('vgpu' in device_model_args)\n"
    "\n"
    "    mmio_start = HVM_BELOW_4G_MMIO_START\n"
    "    # vGPU now requires extra space in lower MMIO hole by default\n"
    "    if wants_vgpu:\n"
    "        mmio_start -= HVM_BELOW_4G_MMIO_LENGTH\n"
)

ARGS_NEEDLE = "    qemu_args.extend(argv[2:])\n"
ARGS_REPLACE = (
    "    qemu_args.extend(argv[2:])\n"
    "    if device_model_args:\n"
    "        print(\"Appending XenStore device-model-args: %s\" % device_model_args)\n"
    "        qemu_args.extend(shlex.split(device_model_args))\n"
)

BUGGY_BLOCK = (
    "    device_model_args = xenstore_read(\n"
    "        \"/local/domain/%d/platform/device-model-args\" % domid\n"
    "    )\n"
    "    wants_vgpu = '-vgpu' in argv\n"
    "    if device_model_args:\n"
    "        wants_vgpu = wants_vgpu or ('vgpu' in device_model_args)\n"
)

SAFE_MARKER = "device_model_args.decode('utf-8', 'ignore')"


def patch_text(text):
    if SAFE_MARKER in text and "Appending XenStore device-model-args" in text:
        return text

    if BUGGY_BLOCK in text:
        return text.replace(BUGGY_BLOCK, BLOCK_REPLACE.split("    mmio_start =", 1)[0].rstrip() + "\n", 1)

    if IMPORT_NEEDLE not in text:
        raise RuntimeError("Could not find import anchor for subprocess")
    text = text.replace(IMPORT_NEEDLE, IMPORT_ADD, 1)

    if BLOCK_NEEDLE not in text:
        raise RuntimeError("Could not find MMIO / qemu_args anchor block")
    text = text.replace(BLOCK_NEEDLE, BLOCK_REPLACE, 1)

    if ARGS_NEEDLE not in text:
        raise RuntimeError("Could not find qemu_args.extend anchor")
    text = text.replace(ARGS_NEEDLE, ARGS_REPLACE, 1)

    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("wrapper_path", nargs="?", default="/usr/lib64/xen/bin/qemu-wrapper")
    args = parser.parse_args()

    wrapper = Path(args.wrapper_path)
    if not wrapper.exists():
        raise SystemExit(f"Wrapper not found: {wrapper}")

    original = wrapper.read_text()
    updated = patch_text(original)

    if updated == original:
        print(f"No changes needed: {wrapper}")
        return 0

    stamp = _dt.datetime.utcnow().strftime("%Y%m%d%H%M%S")
    backup = wrapper.with_name(f"{wrapper.name}.bak.{stamp}")
    backup.write_text(original)
    wrapper.write_text(updated)
    print(f"Patched {wrapper}")
    print(f"Backup saved to {backup}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
