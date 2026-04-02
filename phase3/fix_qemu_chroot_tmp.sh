#!/bin/sh
# Run on XCP-ng dom0 as root after a VM has started.
# qemu-dm runs chrooted with -runas; default /var/xen/qemu/root-<domid>/tmp is 0755,
# so the stub cannot create /tmp/vgpu_stub_pick.log etc. until tmp is world-writable+sticky.
#
# Usage:
#   ./fix_qemu_chroot_tmp.sh <vm-uuid>
#   VM_UUID=<uuid> ./fix_qemu_chroot_tmp.sh
set -e
UUID="${1:-${VM_UUID}}"
if [ -z "$UUID" ]; then
    echo "usage: $0 <vm-uuid>" >&2
    exit 1
fi
DOM=$(xe vm-param-get uuid="$UUID" param-name=dom-id)
T="/var/xen/qemu/root-${DOM}/tmp"
if [ ! -d "$T" ]; then
    echo "missing $T (VM not running or dom-id wrong?)" >&2
    exit 1
fi
chmod 1777 "$T"
ls -la "$T"
echo "OK chmod 1777 $T"
