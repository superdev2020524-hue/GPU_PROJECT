#!/usr/bin/env python3
"""Run a larger PTX vector-add workload on the Server 2 VM.

This is a repeatable application-style compute proof for the current guest shim
stack. It exercises:

1. module load
2. function lookup
3. device allocation
4. host-to-device copies
5. a many-thread kernel launch
6. device-to-host verification
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile

from vm_config import VM_HOST, VM_PASSWORD, VM_USER


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


REMOTE_TEST = r"""
import ctypes
import time


ELEMENTS = 65536
BLOCK_SIZE = 256
GRID_SIZE = ELEMENTS // BLOCK_SIZE

PTX = br'''
.version 6.4
.target sm_52
.address_size 64

.visible .entry vadd_many(
    .param .u64 a,
    .param .u64 b,
    .param .u64 c
)
{
    .reg .b32 %r<8>;
    .reg .b64 %rd<12>;

    mov.u32 %r1, %tid.x;
    mov.u32 %r2, %ctaid.x;
    mad.lo.s32 %r3, %r2, 256, %r1;
    mul.wide.u32 %rd4, %r3, 4;

    ld.param.u64 %rd1, [a];
    ld.param.u64 %rd2, [b];
    ld.param.u64 %rd3, [c];

    add.s64 %rd5, %rd1, %rd4;
    add.s64 %rd6, %rd2, %rd4;
    add.s64 %rd7, %rd3, %rd4;

    ld.global.u32 %r4, [%rd5];
    ld.global.u32 %r5, [%rd6];
    add.u32 %r6, %r4, %r5;
    st.global.u32 [%rd7], %r6;
    ret;
}
'''


def bind(func, argtypes, restype=ctypes.c_int):
    func.argtypes = argtypes
    func.restype = restype
    return func


def main():
    cuda = ctypes.CDLL("libcuda.so.1")

    cuInit = bind(cuda.cuInit, [ctypes.c_uint])
    cuDeviceGet = bind(cuda.cuDeviceGet, [ctypes.POINTER(ctypes.c_int), ctypes.c_int])
    cuDevicePrimaryCtxRetain = bind(
        cuda.cuDevicePrimaryCtxRetain,
        [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int],
    )
    cuCtxSetCurrent = bind(cuda.cuCtxSetCurrent, [ctypes.c_void_p])
    cuCtxSynchronize = bind(cuda.cuCtxSynchronize, [])
    cuModuleLoadData = bind(
        cuda.cuModuleLoadData,
        [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p],
    )
    cuModuleGetFunction = bind(
        cuda.cuModuleGetFunction,
        [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_char_p],
    )
    cuLaunchKernel = bind(
        cuda.cuLaunchKernel,
        [
            ctypes.c_void_p,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
        ],
    )
    cuMemAlloc = bind(
        cuda.cuMemAlloc_v2,
        [ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t],
    )
    cuMemFree = bind(cuda.cuMemFree_v2, [ctypes.c_uint64])
    cuMemcpyHtoD = bind(
        cuda.cuMemcpyHtoD_v2,
        [ctypes.c_uint64, ctypes.c_void_p, ctypes.c_size_t],
    )
    cuMemcpyDtoH = bind(
        cuda.cuMemcpyDtoH_v2,
        [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_size_t],
    )

    def chk(name, rc):
        print(f"{name} rc={rc}")
        if rc != 0:
            raise SystemExit(1)

    chk("cuInit", cuInit(0))

    dev = ctypes.c_int()
    chk("cuDeviceGet", cuDeviceGet(ctypes.byref(dev), 0))

    ctx = ctypes.c_void_p()
    chk("cuDevicePrimaryCtxRetain", cuDevicePrimaryCtxRetain(ctypes.byref(ctx), dev.value))
    chk("cuCtxSetCurrent", cuCtxSetCurrent(ctx))

    mod = ctypes.c_void_p()
    ptx_buf = ctypes.create_string_buffer(PTX)
    chk("cuModuleLoadData", cuModuleLoadData(ctypes.byref(mod), ctypes.cast(ptx_buf, ctypes.c_void_p)))

    fn = ctypes.c_void_p()
    chk("cuModuleGetFunction", cuModuleGetFunction(ctypes.byref(fn), mod, b"vadd_many"))

    a_host = (ctypes.c_uint32 * ELEMENTS)()
    b_host = (ctypes.c_uint32 * ELEMENTS)()
    c_host = (ctypes.c_uint32 * ELEMENTS)()
    for idx in range(ELEMENTS):
        a_host[idx] = idx
        b_host[idx] = 2 * idx + 1

    bytes_per_vector = ctypes.sizeof(a_host)
    print(f"elements={ELEMENTS} block={BLOCK_SIZE} grid={GRID_SIZE} bytes_per_vector={bytes_per_vector}")

    d_a = ctypes.c_uint64()
    d_b = ctypes.c_uint64()
    d_c = ctypes.c_uint64()
    chk("cuMemAlloc_v2(a)", cuMemAlloc(ctypes.byref(d_a), bytes_per_vector))
    chk("cuMemAlloc_v2(b)", cuMemAlloc(ctypes.byref(d_b), bytes_per_vector))
    chk("cuMemAlloc_v2(c)", cuMemAlloc(ctypes.byref(d_c), bytes_per_vector))

    try:
        chk("cuMemcpyHtoD_v2(a)", cuMemcpyHtoD(d_a.value, a_host, bytes_per_vector))
        chk("cuMemcpyHtoD_v2(b)", cuMemcpyHtoD(d_b.value, b_host, bytes_per_vector))

        arg_a = ctypes.c_uint64(d_a.value)
        arg_b = ctypes.c_uint64(d_b.value)
        arg_c = ctypes.c_uint64(d_c.value)
        kernel_params = (ctypes.c_void_p * 4)()
        kernel_params[0] = ctypes.cast(ctypes.byref(arg_a), ctypes.c_void_p)
        kernel_params[1] = ctypes.cast(ctypes.byref(arg_b), ctypes.c_void_p)
        kernel_params[2] = ctypes.cast(ctypes.byref(arg_c), ctypes.c_void_p)
        kernel_params[3] = None

        start = time.perf_counter()
        chk(
            "cuLaunchKernel",
            cuLaunchKernel(
                fn,
                GRID_SIZE,
                1,
                1,
                BLOCK_SIZE,
                1,
                1,
                0,
                None,
                kernel_params,
                None,
            ),
        )
        chk("cuCtxSynchronize", cuCtxSynchronize())
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        chk("cuMemcpyDtoH_v2(c)", cuMemcpyDtoH(c_host, d_c.value, bytes_per_vector))
    finally:
        if d_c.value:
            print(f"cuMemFree_v2(c) rc={cuMemFree(d_c.value)}")
        if d_b.value:
            print(f"cuMemFree_v2(b) rc={cuMemFree(d_b.value)}")
        if d_a.value:
            print(f"cuMemFree_v2(a) rc={cuMemFree(d_a.value)}")

    expected_sum = 0
    actual_sum = 0
    samples = [0, 1, 2, 12345, ELEMENTS - 3, ELEMENTS - 2, ELEMENTS - 1]
    for idx in range(ELEMENTS):
        expected = 3 * idx + 1
        actual = c_host[idx]
        expected_sum += expected
        actual_sum += actual
        if idx in samples:
            print(f"sample[{idx}] expected={expected} actual={actual}")
        if actual != expected:
            print(f"VECTOR_ADD_WORKLOAD_FAILED idx={idx} expected={expected} actual={actual}")
            raise SystemExit(1)

    print(f"checksum expected={expected_sum} actual={actual_sum}")
    print(f"kernel_elapsed_ms={elapsed_ms:.3f}")
    print("VECTOR_ADD_WORKLOAD_OK")


if __name__ == "__main__":
    main()
"""


def run_vm(cmd: str, timeout_sec: int = 240) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, os.path.join(SCRIPT_DIR, "connect_vm.py"), cmd],
        capture_output=True,
        text=True,
        timeout=timeout_sec,
    )


def scp_remote_test(local_path: str, remote_path: str) -> None:
    subprocess.run(
        [
            "sshpass",
            "-p",
            VM_PASSWORD,
            "scp",
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "ConnectTimeout=15",
            local_path,
            f"{VM_USER}@{VM_HOST}:{remote_path}",
        ],
        check=True,
        timeout=60,
    )


def main() -> int:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".py", delete=False) as tmp:
        tmp.write(REMOTE_TEST)
        local_tmp = tmp.name

    try:
        remote_path = "/tmp/vgpu_vector_add_workload.py"
        scp_remote_test(local_tmp, remote_path)
        result = run_vm(f"LD_LIBRARY_PATH=/usr/lib64 python3 {remote_path}", timeout_sec=360)
        sys.stdout.write(result.stdout or "")
        sys.stderr.write(result.stderr or "")
        return 0 if result.returncode == 0 else 1
    finally:
        try:
            os.unlink(local_tmp)
        except OSError:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
