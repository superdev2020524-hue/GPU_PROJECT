#!/usr/bin/env python3
"""Milestone 07 malformed mediator socket probe.

This probe intentionally sends malformed one-shot messages to a mediator socket.
It avoids large payloads by default; oversized-payload behavior is recorded as a
source-audit finding unless explicitly enabled by a separate bounded test.
"""

import argparse
import json
import socket
import struct
import time
from pathlib import Path


VGPU_SOCKET_MAGIC = 0x56475055
VGPU_MSG_PING = 0x03
VGPU_MSG_CUDA_CALL = 0x10
VGPU_PRIORITY_LOW = 0
CUDA_MAX_INLINE_ARGS = 16
CUDA_CALL_INIT = 0x0001
CUDA_HEADER_SIZE = 88
VGPU_CUDA_SOCKET_MAX_PAYLOAD = 8 * 1024 * 1024


def header(magic, msg_type, vm_id, request_id, pool_id, priority, payload_len):
    return struct.pack(
        "<IIIIcBHI",
        magic,
        msg_type,
        vm_id,
        request_id,
        pool_id.encode("ascii"),
        priority,
        0,
        payload_len,
    )


def exchange(path, payload, read_reply=False, timeout=2.0):
    started = time.time()
    result = {
        "sent_bytes": len(payload),
        "reply_bytes": 0,
        "closed": False,
        "error": None,
        "elapsed_sec": None,
    }
    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        sock.connect(path)
        sock.sendall(payload)
        if read_reply:
            try:
                reply = sock.recv(4096)
                result["reply_bytes"] = len(reply)
                result["closed"] = len(reply) == 0
            except socket.timeout:
                result["error"] = "recv_timeout"
        else:
            try:
                reply = sock.recv(1)
                result["reply_bytes"] = len(reply)
                result["closed"] = len(reply) == 0
            except socket.timeout:
                result["error"] = "recv_timeout"
        sock.close()
    except Exception as exc:
        result["error"] = "%s: %s" % (exc.__class__.__name__, exc)
    result["elapsed_sec"] = round(time.time() - started, 3)
    return result


def cuda_header(magic, call_id, seq_num, vm_id, num_args, data_len, args=None):
    values = list(args or [])
    values = (values + [0] * CUDA_MAX_INLINE_ARGS)[:CUDA_MAX_INLINE_ARGS]
    return struct.pack(
        "<IIIIII16I",
        magic,
        call_id,
        seq_num,
        vm_id,
        num_args,
        data_len,
        *values,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket", required=True)
    parser.add_argument("--vm-id", type=int, default=6)
    parser.add_argument("--output", default="/tmp/m07_malformed_socket_probe.json")
    args = parser.parse_args()

    sock_path = Path(args.socket)
    cases = {}

    cases["invalid_magic"] = exchange(
        str(sock_path),
        header(0x12345678, VGPU_MSG_PING, args.vm_id, 7001, "B", VGPU_PRIORITY_LOW, 0),
    )

    cases["truncated_header"] = exchange(str(sock_path), b"VGPU")

    cases["unknown_msg_type"] = exchange(
        str(sock_path),
        header(VGPU_SOCKET_MAGIC, 0x99, args.vm_id, 7002, "B", VGPU_PRIORITY_LOW, 0),
    )

    cases["cuda_payload_too_short"] = exchange(
        str(sock_path),
        header(VGPU_SOCKET_MAGIC, VGPU_MSG_CUDA_CALL, args.vm_id, 7003, "B", VGPU_PRIORITY_LOW, 8)
        + b"\x00" * 8,
    )

    cases["oversized_declared_payload"] = exchange(
        str(sock_path),
        header(
            VGPU_SOCKET_MAGIC,
            VGPU_MSG_CUDA_CALL,
            args.vm_id,
            7005,
            "B",
            VGPU_PRIORITY_LOW,
            CUDA_HEADER_SIZE + VGPU_CUDA_SOCKET_MAX_PAYLOAD + 1,
        ),
    )

    cases["invalid_cuda_magic"] = exchange(
        str(sock_path),
        header(
            VGPU_SOCKET_MAGIC,
            VGPU_MSG_CUDA_CALL,
            args.vm_id,
            7006,
            "B",
            VGPU_PRIORITY_LOW,
            CUDA_HEADER_SIZE,
        )
        + cuda_header(0x87654321, CUDA_CALL_INIT, 1, args.vm_id, 0, 0),
    )

    cases["cuda_num_args_overflow"] = exchange(
        str(sock_path),
        header(
            VGPU_SOCKET_MAGIC,
            VGPU_MSG_CUDA_CALL,
            args.vm_id,
            7007,
            "B",
            VGPU_PRIORITY_LOW,
            CUDA_HEADER_SIZE,
        )
        + cuda_header(VGPU_SOCKET_MAGIC, CUDA_CALL_INIT, 2, args.vm_id, CUDA_MAX_INLINE_ARGS + 1, 0),
    )

    cases["cuda_data_len_mismatch"] = exchange(
        str(sock_path),
        header(
            VGPU_SOCKET_MAGIC,
            VGPU_MSG_CUDA_CALL,
            args.vm_id,
            7008,
            "B",
            VGPU_PRIORITY_LOW,
            CUDA_HEADER_SIZE + 4,
        )
        + cuda_header(VGPU_SOCKET_MAGIC, CUDA_CALL_INIT, 3, args.vm_id, 0, 8)
        + b"\x00" * 4,
    )

    # Control case: normal ping should receive a non-empty PONG response.
    cases["control_ping"] = exchange(
        str(sock_path),
        header(VGPU_SOCKET_MAGIC, VGPU_MSG_PING, args.vm_id, 7004, "B", VGPU_PRIORITY_LOW, 0),
        read_reply=True,
    )

    summary = {
        "socket": str(sock_path),
        "vm_id": args.vm_id,
        "cases": cases,
        "overall_pass": (
            cases["invalid_magic"]["closed"]
            and cases["truncated_header"]["closed"]
            and cases["unknown_msg_type"]["closed"]
            and cases["cuda_payload_too_short"]["closed"]
            and cases["oversized_declared_payload"]["closed"]
            and cases["invalid_cuda_magic"]["closed"]
            and cases["cuda_num_args_overflow"]["closed"]
            and cases["cuda_data_len_mismatch"]["closed"]
            and cases["control_ping"]["reply_bytes"] > 0
        ),
    }

    Path(args.output).write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
