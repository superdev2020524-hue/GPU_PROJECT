# Work note: Deploy updated shim + enable coredumps (Mar 19)

## Done

1. **Deployed updated guest-shim** (`transfer_cuda_transport.py`)
   - `cuda_transport.c`: added `call_id_to_name()` for 0x0071 (cuEventCreateWithFlags), streams, events, modules; added `/tmp/vgpu_current_call.txt` overwrite before poll loop.
   - Build on VM: `BUILD_EXIT=0`; lib installed to `/opt/vgpu/lib/libvgpu-cuda.so.1`, ollama restarted.

2. **Enabled coredumps for ollama on VM**
   - `/etc/systemd/system/ollama.service.d/coredump.conf`: `[Service]` + `LimitCORE=infinity`
   - `systemctl daemon-reload` and `systemctl start ollama`; service reported `active`.

3. **Verification (short generate ~35s)**
   - `/tmp/vgpu_current_call.txt`: `call_id=0x0032 cuMemcpyHtoD_v2 seq=10 pid=163752` — current-call file working.
   - `/tmp/vgpu_call_sequence.log`: last lines show **cuGetGpuInfo**, **cuDevicePrimaryCtxRetain**, **cuCtxSetCurrent**, **cuMemAlloc_v2**, **cuMemcpyHtoD_v2** (readable names, no `?(call_id)`).

## Next

- On the **next** runner crash (exit 2): use `coredumpctl list` / `coredumpctl dump -o /tmp/core.ollama` on the VM, then `gdb /usr/local/bin/ollama.bin.new /tmp/core.ollama` and `bt full` to symbolicate rip and fix the failing code path.
- When **stuck** (long load): read `/tmp/vgpu_current_call.txt` on the VM to see which CUDA call is blocking.
