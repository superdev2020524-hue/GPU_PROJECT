# Baseline - Milestone 08 Server 2 Migration

## Preserved Server 1 Baseline

Server 1 is the protected engineering track. The latest Server 1 closure before
starting M08 is the refreshed M07 chain:

- M07 final Plan A:
  `/tmp/m07_final_after_tf_3param_planA.json` -> `overall_pass=True`.
- M07 final raw CUDA:
  `/tmp/m07_final_after_tf_3param_m01.json` -> `overall_pass=True`.
- M07 final PyTorch:
  `/tmp/m07_final_after_tf_3param_pytorch.json` -> `overall_pass=True`.
- M07 final CuPy:
  `/tmp/m07_final_after_tf_3param_cupy.json` -> `overall_pass=True`.
- M07 final TensorFlow:
  `/tmp/m07_current_preserve_tensorflow_after_3param_fix.json` ->
  `overall_pass=True`, `used_gpu_for_training=True`.
- M07 final malformed socket:
  `/tmp/m07_final_after_tf_3param_malformed.json` -> `overall_pass=True`.

Server 1 files must not be modified for Server 2 work. Server 2 work uses only
the `server2/` registry.

## Server 2 Known Historical Facts

From `server2/HOST2_PASSTHROUGH_FAST_PATH.md`:

- Host: `10.25.33.20`.
- Target VM: `Ubuntu-VM-1`.
- Target VM UUID: `5b9acc4b-d62b-6dc6-576f-82175e87fc2b`.
- Real GPU BDF: `0000:81:00.0`.
- Xen PCI record: `e3dfe1bb-1e88-655a-8031-06b22eea9433`.
- Historical verified passthrough result:
  - guest `lspci` showed `NVIDIA Corporation HEXACORE vH100 CAP`;
  - `nvidia-smi` worked and reported `NVIDIA H100 NVL`;
  - Ollama discovery reported `library=CUDA compute=9.0`;
  - a `qwen2.5:0.5b` generation returned `OK.`.

## Current Live Baseline Attempt

Current workstation route table:

- default via `192.168.119.2`;
- `10.10.20.0/24` via `wg0`;
- no working route to `10.25.33.0/24` observed.

Current live checks:

- SSH to `root@10.25.33.20`: `No route to host`.
- SSH to `root@10.25.33.21`: `No route to host`.
- `ping 10.25.33.20`: 100% packet loss.
- `ping 10.25.33.21`: 100% packet loss.

## Baseline Interpretation

M08 cannot currently verify or change Server 2 runtime state from this
workstation. The historical Server 2 passthrough path remains documented, but no
fresh live Server 2 conclusion may be made until reachability is restored or an
alternate access path is provided.
