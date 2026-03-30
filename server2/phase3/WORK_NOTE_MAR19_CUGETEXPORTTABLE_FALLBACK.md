# Work note (Mar 19): cuGetExportTable UNKNOWN UUID fallback

## What changed

- In `guest-shim/libvgpu_cuda.c`, `cuGetExportTable()` previously returned
  `CUDA_ERROR_NOT_SUPPORTED` for unknown UUIDs.
- Added a safer compatibility fallback:
  1. Try forwarding unknown UUIDs to `RTLD_NEXT` `cuGetExportTable`.
  2. If not available/successful, return a small non-failing table
     (`g_context_wrapper`) instead of hard-failing.

## Why

- Journal showed:
  - `cuGetExportTable() UNKNOWN UUID ...`
  - `CUDA error: the requested functionality is not supported`
  immediately before runner crash/abort path.

## Deployment

- Deployed to VM via `transfer_libvgpu_cuda.py`.
- VM build/install/restart succeeded:
  - SHA256 verified
  - `BUILD_EXIT=0`
  - `libvgpu-cuda.so.1` replaced in `/opt/vgpu/lib`
  - `ollama` restarted

## Follow-up (continue)

- Added `CUDA_CALL_MEMSET_D8/D16/D32` names in `cuda_transport.c` so `vgpu_call_sequence.log` shows `cuMemsetD8_v2` instead of `?(call_id)` for `0x0035` etc.
- Long generate: use `curl -m 2400` with `CONNECT_VM_COMMAND_TIMEOUT_SEC=2500` so client does not abort load early.

## Verification after deploy

- The specific `UNKNOWN UUID` + `requested functionality is not supported` pair
  did **not** reappear in the post-deploy run window.
- Load path remains GPU-active (HtoD on host, `cuMemcpyHtoD_v2` in guest call sequence).
- Current observed blocker shifted to client timeout/load not finishing in short windows:
  - `client connection closed before server finished loading, aborting load`
  - `error loading llama server: timed out waiting for llama runner to start: context canceled`

