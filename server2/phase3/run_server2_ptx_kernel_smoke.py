#!/usr/bin/env python3
"""Run a tiny PTX kernel-launch smoke test on the Server 2 VM.

This stays inside the Server 2 registry and uses `connect_vm.py` so the test can
be repeated on future VMs without manual shell work. The remote test:

1. Loads `libcuda.so.1` from the active guest shim path
2. Creates/sets a CUDA context
3. Uploads a tiny PTX kernel through `cuModuleLoadData`
4. Launches the kernel with `cuLaunchKernel`
5. Copies the result back and checks it
"""

from __future__ import annotations

import base64
import os
import subprocess
import sys


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


REMOTE_TEST = r"""
import ctypes
import sys


PTX = br'''
.version 6.4
.target sm_52
.address_size 64

.visible .entry vadd1(
    .param .u64 a,
    .param .u64 b,
    .param .u64 c
)
{
    .reg .b32 %r<4>;
    .reg .b64 %rd<6>;

    ld.param.u64 %rd1, [a];
    ld.param.u64 %rd2, [b];
    ld.param.u64 %rd3, [c];
    ld.global.u32 %r1, [%rd1];
    ld.global.u32 %r2, [%rd2];
    add.u32 %r3, %r1, %r2;
    st.global.u32 [%rd3], %r3;
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
    chk("cuModuleGetFunction", cuModuleGetFunction(ctypes.byref(fn), mod, b"vadd1"))

    a_host = (ctypes.c_uint32 * 1)(123)
    b_host = (ctypes.c_uint32 * 1)(456)
    c_host = (ctypes.c_uint32 * 1)(0)

    d_a = ctypes.c_uint64()
    d_b = ctypes.c_uint64()
    d_c = ctypes.c_uint64()
    chk("cuMemAlloc_v2(a)", cuMemAlloc(ctypes.byref(d_a), ctypes.sizeof(a_host)))
    chk("cuMemAlloc_v2(b)", cuMemAlloc(ctypes.byref(d_b), ctypes.sizeof(b_host)))
    chk("cuMemAlloc_v2(c)", cuMemAlloc(ctypes.byref(d_c), ctypes.sizeof(c_host)))

    try:
        chk("cuMemcpyHtoD_v2(a)", cuMemcpyHtoD(d_a.value, a_host, ctypes.sizeof(a_host)))
        chk("cuMemcpyHtoD_v2(b)", cuMemcpyHtoD(d_b.value, b_host, ctypes.sizeof(b_host)))

        arg_a = ctypes.c_uint64(d_a.value)
        arg_b = ctypes.c_uint64(d_b.value)
        arg_c = ctypes.c_uint64(d_c.value)
        kernel_params = (ctypes.c_void_p * 4)()
        kernel_params[0] = ctypes.cast(ctypes.byref(arg_a), ctypes.c_void_p)
        kernel_params[1] = ctypes.cast(ctypes.byref(arg_b), ctypes.c_void_p)
        kernel_params[2] = ctypes.cast(ctypes.byref(arg_c), ctypes.c_void_p)
        kernel_params[3] = None

        chk(
            "cuLaunchKernel",
            cuLaunchKernel(
                fn,
                1,
                1,
                1,
                1,
                1,
                1,
                0,
                None,
                kernel_params,
                None,
            ),
        )
        chk("cuCtxSynchronize", cuCtxSynchronize())
        chk("cuMemcpyDtoH_v2(c)", cuMemcpyDtoH(c_host, d_c.value, ctypes.sizeof(c_host)))
    finally:
        if d_c.value:
            print(f"cuMemFree_v2(c) rc={cuMemFree(d_c.value)}")
        if d_b.value:
            print(f"cuMemFree_v2(b) rc={cuMemFree(d_b.value)}")
        if d_a.value:
            print(f"cuMemFree_v2(a) rc={cuMemFree(d_a.value)}")

    print(f"result a={a_host[0]} b={b_host[0]} c={c_host[0]}")
    if c_host[0] != a_host[0] + b_host[0]:
        print("PTX_KERNEL_TEST_FAILED")
        raise SystemExit(1)

    print("PTX_KERNEL_TEST_OK")


if __name__ == "__main__":
    main()
"""


def run_vm(cmd: str, timeout_sec: int = 120) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, os.path.join(SCRIPT_DIR, "connect_vm.py"), cmd],
        capture_output=True,
        text=True,
        timeout=timeout_sec,
    )


def main() -> int:
    payload = base64.b64encode(REMOTE_TEST.encode("utf-8")).decode("ascii")
    remote_cmd = (
        "python3 -c \"import base64; "
        "open('/tmp/vgpu_ptx_smoke.py','wb').write(base64.b64decode('"
        + payload
        + "'))\""
        + " && LD_LIBRARY_PATH=/usr/lib64 python3 /tmp/vgpu_ptx_smoke.py"
    )
    result = run_vm(remote_cmd, timeout_sec=120)
    sys.stdout.write(result.stdout or "")
    sys.stderr.write(result.stderr or "")
    return 0 if result.returncode == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
