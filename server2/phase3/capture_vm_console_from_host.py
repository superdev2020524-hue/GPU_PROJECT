#!/usr/bin/env python3
"""Capture a VM console frame from the Server 2 host VNC socket.

This helper runs locally, SSHes to the Server 2 host, asks a small remote
Python snippet to connect to the VM's unix-domain VNC socket, requests one raw
framebuffer update, saves it as a PPM on the host, SCPs that file back to the
local machine, and converts it to PNG for inspection.

The script stays inside the Server 2 registry so the capture/review workflow is
repeatable for future Secure Boot / MokManager / firmware console debugging.
"""

from __future__ import annotations

from pathlib import Path
import shlex
import sys
import tempfile
import textwrap

import pexpect

from vm_config import MEDIATOR_HOST, MEDIATOR_PASSWORD, MEDIATOR_USER


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DOMID = "13"


REMOTE_CAPTURE_TEMPLATE = r"""
import os
import socket
import struct
import sys

sock_path = "/var/run/xen/vnc-__DOMID__"
out_path = "/root/vnc-__DOMID__.png"

def recvn(sock, n):
    data = bytearray()
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            raise RuntimeError(f"short read: wanted {n}, got {len(data)}")
        data.extend(chunk)
    return bytes(data)

sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
sock.connect(sock_path)

banner = recvn(sock, 12)
sys.stdout.write(f"banner={banner.decode('ascii', 'ignore').strip()}\n")
sock.sendall(b"RFB 003.008\n")

nsec = recvn(sock, 1)[0]
if nsec == 0:
    reason_len = struct.unpack(">I", recvn(sock, 4))[0]
    reason = recvn(sock, reason_len).decode("utf-8", "ignore")
    raise RuntimeError(f"VNC server advertised failure: {reason}")

sec_types = recvn(sock, nsec)
if 1 not in sec_types:
    raise RuntimeError(f"VNC auth 'None' unavailable: {list(sec_types)}")
sock.sendall(b"\x01")

sec_result = struct.unpack(">I", recvn(sock, 4))[0]
if sec_result != 0:
    raise RuntimeError(f"VNC security result={sec_result}")

sock.sendall(b"\x01")  # shared connection

width, height = struct.unpack(">HH", recvn(sock, 4))
_server_pf = recvn(sock, 16)
name_len = struct.unpack(">I", recvn(sock, 4))[0]
name = recvn(sock, name_len).decode("utf-8", "ignore")
sys.stdout.write(f"size={width}x{height} name={name}\n")

# Ask for 32bpp little-endian true-color pixels: XRGB8888.
pixfmt = struct.pack(
    ">BBBBHHHBBBxxx",
    32,   # bits-per-pixel
    24,   # depth
    0,    # little endian
    1,    # true color
    255,  # red max
    255,  # green max
    255,  # blue max
    16,   # red shift
    8,    # green shift
    0,    # blue shift
)
sock.sendall(b"\x00\x00\x00\x00" + pixfmt)

# Raw only.
sock.sendall(struct.pack(">BBH", 2, 0, 1) + struct.pack(">i", 0))

# Full-screen refresh.
sock.sendall(struct.pack(">BBHHHH", 3, 0, 0, 0, width, height))

rgb = bytearray(width * height * 3)
seen_raw = False

while True:
    msg_type = recvn(sock, 1)[0]
    if msg_type != 0:
        raise RuntimeError(f"unexpected VNC message type={msg_type}")
    _pad = recvn(sock, 1)
    rects = struct.unpack(">H", recvn(sock, 2))[0]
    for _ in range(rects):
        x, y, w, h = struct.unpack(">HHHH", recvn(sock, 8))
        enc = struct.unpack(">i", recvn(sock, 4))[0]
        if enc == 0:
            seen_raw = True
            block = recvn(sock, w * h * 4)
            for row in range(h):
                for col in range(w):
                    src = (row * w + col) * 4
                    dst = ((y + row) * width + (x + col)) * 3
                    rgb[dst] = block[src + 1]
                    rgb[dst + 1] = block[src + 2]
                    rgb[dst + 2] = block[src + 3]
        elif enc == -223:
            # Desktop resize pseudo-encoding.
            width, height = w, h
            rgb = bytearray(width * height * 3)
        else:
            raise RuntimeError(f"unsupported VNC encoding={enc}")
    if seen_raw:
        break

with open(out_path, "wb") as f:
    import binascii
    import zlib

    def chunk(tag, data):
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", binascii.crc32(tag + data) & 0xFFFFFFFF)
        )

    header = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    scanlines = bytearray()
    row_bytes = width * 3
    for row in range(height):
        scanlines.append(0)
        start = row * row_bytes
        scanlines.extend(rgb[start:start + row_bytes])
    idat = zlib.compress(bytes(scanlines), level=6)
    f.write(header)
    f.write(chunk(b"IHDR", ihdr))
    f.write(chunk(b"IDAT", idat))
    f.write(chunk(b"IEND", b""))

sys.stdout.write(f"saved={out_path}\n")
sock.close()
"""


def build_ssh_command(*extra: str) -> str:
    return " ".join(
        [
            "ssh",
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "PreferredAuthentications=password",
            "-o",
            "PubkeyAuthentication=no",
            *extra,
            f"{MEDIATOR_USER}@{MEDIATOR_HOST}",
        ]
    )


def scp_to_host(local_path: Path, remote_path: Path) -> None:
    child = pexpect.spawn(
        "scp",
        [
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "PreferredAuthentications=password",
            "-o",
            "PubkeyAuthentication=no",
            str(local_path),
            f"{MEDIATOR_USER}@{MEDIATOR_HOST}:{remote_path}",
        ],
        encoding="utf-8",
        timeout=120,
    )
    try:
        idx = child.expect(["password:", "Password:", pexpect.EOF, pexpect.TIMEOUT], timeout=30)
        if idx in (0, 1):
            child.sendline(MEDIATOR_PASSWORD)
            child.expect([pexpect.EOF], timeout=120)
        child.close()
        if child.exitstatus != 0:
            raise RuntimeError(f"SCP to host failed with exit status {child.exitstatus}")
    finally:
        if child.isalive():
            child.close(force=True)


def run_remote_capture(domid: str) -> Path:
    remote_code = textwrap.dedent(
        REMOTE_CAPTURE_TEMPLATE.replace("__DOMID__", domid)
    ).strip()
    remote_script = Path(f"/root/vnc-capture-{domid}.py")
    with tempfile.TemporaryDirectory() as tmpdir:
        local_script = Path(tmpdir) / remote_script.name
        local_script.write_text(remote_code + "\n", encoding="utf-8")
        scp_to_host(local_script, remote_script)
    remote_cmd = f"python3 {shlex.quote(str(remote_script))}"

    child = pexpect.spawn(
        build_ssh_command(),
        encoding="utf-8",
        timeout=120,
    )
    try:
        idx = child.expect(["password:", "Password:", r"\$", "#", pexpect.EOF, pexpect.TIMEOUT])
        if idx in (0, 1):
            child.sendline(MEDIATOR_PASSWORD)
            child.expect([r"\$", "#"], timeout=45)

        child.sendline(remote_cmd)
        child.expect([r"\$", "#"], timeout=120)
        output = child.before or ""
        print(output.strip())
    finally:
        if child.isalive():
            child.sendline("exit")
            child.close(force=True)

    remote_png = Path(f"/root/vnc-{domid}.png")
    if f"saved={remote_png}" not in output:
        raise RuntimeError("Remote VNC capture did not report a saved image")
    return remote_png


def scp_from_host(remote_path: Path, local_path: Path) -> None:
    child = pexpect.spawn(
        "scp",
        [
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "PreferredAuthentications=password",
            "-o",
            "PubkeyAuthentication=no",
            f"{MEDIATOR_USER}@{MEDIATOR_HOST}:{remote_path}",
            str(local_path),
        ],
        encoding="utf-8",
        timeout=300,
    )
    try:
        idx = child.expect(["password:", "Password:", pexpect.EOF, pexpect.TIMEOUT], timeout=30)
        if idx in (0, 1):
            child.sendline(MEDIATOR_PASSWORD)
            child.expect([pexpect.EOF], timeout=300)
        child.close()
        if child.exitstatus != 0:
            raise RuntimeError(f"SCP failed with exit status {child.exitstatus}")
    finally:
        if child.isalive():
            child.close(force=True)


def main() -> int:
    domid = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DOMID
    local_dir = SCRIPT_DIR / "artifacts"
    local_dir.mkdir(parents=True, exist_ok=True)

    png_path = local_dir / f"vm-console-{domid}.png"

    remote_png = run_remote_capture(domid)
    scp_from_host(remote_png, png_path)

    print(f"PNG: {png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
