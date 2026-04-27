# VM GPU Verification Checklist

Please run the commands below on the target VM and send back the full output for each section.

These checks are the acceptance criteria for GPU exposure, framework visibility, and basic GPU execution.

Run them as `root`. If you are using another account, prefix the commands with `sudo` where needed.

## 1. Basic device check

```bash
lspci | grep -i 'HEXACORE\|NVIDIA'
nvidia-smi
```

Expected result:

- `lspci` shows `HEXACORE vH100 CAP`
- `nvidia-smi` shows `HEXACORE vH100 CAP`

## 2. Ollama check

If `ollama` is part of your validation, run:

```bash
systemctl is-active ollama
ollama pull qwen2.5:0.5b
printf '%s' '{"model":"qwen2.5:0.5b","prompt":"Reply with OK only.","stream":false}' | curl -fsS http://127.0.0.1:11434/api/generate -d @-
ollama ps
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader
```

Expected result:

- `systemctl is-active ollama` returns `active`
- the generate call returns a JSON response
- `ollama ps` shows the model on `100% GPU`
- `nvidia-smi` shows an `ollama` process using GPU memory

## 3. TensorFlow check

If TensorFlow is part of your validation, activate your TensorFlow environment first, then run:

```bash
python3 - <<'PY'
import tensorflow as tf

print("VISIBLE_GPU_COUNT:", len(tf.config.list_physical_devices("GPU")))
print("VISIBLE_GPUS:", [g.name for g in tf.config.list_physical_devices("GPU")])

with tf.device("/GPU:0"):
    a = tf.random.normal([4096, 4096])
    b = tf.random.normal([4096, 4096])
    c = tf.matmul(a, b)
    _ = c.numpy()

print("ACTUAL_DEVICE:", c.device)
print("EXECUTION_BACKEND:", "GPU" if "GPU" in c.device.upper() else "CPU")
PY
```

Expected result:

- TensorFlow sees at least one GPU
- the framework logs or device output show `HEXACORE vH100 CAP`
- `EXECUTION_BACKEND` is `GPU`

## 4. PyTorch check

If PyTorch is part of your validation, activate your PyTorch environment first, then run:

```bash
python3 - <<'PY'
import torch

print("CUDA_AVAILABLE:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit(2)

print("DEVICE_NAME:", torch.cuda.get_device_name(0))
x = torch.randn((4096, 4096), device="cuda:0")
y = torch.matmul(x, x)
torch.cuda.synchronize()
print("RESULT_DEVICE:", y.device)
PY
```

Expected result:

- `CUDA_AVAILABLE` is `True`
- `DEVICE_NAME` is `HEXACORE vH100 CAP`
- `RESULT_DEVICE` is `cuda:0`

## 5. Important note

Do not use `/proc/driver/nvidia/gpus/.../information` as an acceptance check.

That file is produced by the kernel driver and can still show the physical NVIDIA board name. This is expected and does not indicate a GPU problem.

For customer acceptance, use:

- `lspci`
- `nvidia-smi`
- TensorFlow / PyTorch / Ollama outputs

## 6. If anything fails

Please send back:

- the exact command that failed
- the full terminal output
- the VM IP address
