#!/usr/bin/env python3
"""Run a small TensorFlow workload and report whether it executed on GPU or CPU."""

import glob
import os
import sys
import time


def _prepend_paths(env: dict[str, str], key: str, paths: list[str]) -> None:
    existing = [part for part in env.get(key, "").split(os.pathsep) if part]
    merged: list[str] = []
    for path in paths + existing:
        if path and path not in merged:
            merged.append(path)
    env[key] = os.pathsep.join(merged)


def _discover_cuda_wheel_paths() -> tuple[list[str], list[str], str]:
    pyver = f"python{sys.version_info.major}.{sys.version_info.minor}"
    base = os.path.join(sys.prefix, "lib", pyver, "site-packages", "nvidia")
    lib_dirs = sorted(glob.glob(os.path.join(base, "*", "lib")))
    bin_dirs = sorted(glob.glob(os.path.join(base, "*", "bin")))
    cuda_data_dir = os.path.join(base, "cuda_nvcc")
    return lib_dirs, bin_dirs, cuda_data_dir


def _maybe_reexec_with_cuda_paths() -> None:
    if os.environ.get("TF_DEVICE_CHECK_ENV_READY") == "1":
        return

    lib_dirs, bin_dirs, cuda_data_dir = _discover_cuda_wheel_paths()
    if not lib_dirs and not bin_dirs:
        return

    env = os.environ.copy()
    _prepend_paths(env, "LD_LIBRARY_PATH", lib_dirs)
    _prepend_paths(env, "PATH", bin_dirs)
    if os.path.isdir(cuda_data_dir):
        env.setdefault("XLA_FLAGS", f"--xla_gpu_cuda_data_dir={cuda_data_dir}")
    env["TF_DEVICE_CHECK_ENV_READY"] = "1"
    os.execvpe(sys.executable, [sys.executable, *sys.argv], env)


def main() -> int:
    _maybe_reexec_with_cuda_paths()

    try:
        import tensorflow as tf
    except Exception as exc:
        print("TENSORFLOW_IMPORT_OK: no")
        print(f"ERROR: {exc}")
        return 1

    lib_dirs, bin_dirs, cuda_data_dir = _discover_cuda_wheel_paths()
    print("TENSORFLOW_IMPORT_OK: yes")
    print(f"TENSORFLOW_VERSION: {tf.__version__}")
    print(f"BUILT_WITH_CUDA: {tf.test.is_built_with_cuda()}")
    print("CUDA_WHEEL_LIB_DIRS:", lib_dirs)
    print("CUDA_WHEEL_BIN_DIRS:", bin_dirs)
    print(f"CUDA_DATA_DIR: {cuda_data_dir}")

    gpus = tf.config.list_physical_devices("GPU")
    cpus = tf.config.list_physical_devices("CPU")
    print(f"VISIBLE_GPU_COUNT: {len(gpus)}")
    print(f"VISIBLE_CPU_COUNT: {len(cpus)}")
    print("VISIBLE_GPUS:", [gpu.name for gpu in gpus])

    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass

    requested_device = "/GPU:0" if gpus else "/CPU:0"
    matrix_size = int(os.environ.get("TF_CHECK_MATRIX_SIZE", "4096"))

    with tf.device(requested_device):
        lhs = tf.random.normal([matrix_size, matrix_size], dtype=tf.float32)
        rhs = tf.random.normal([matrix_size, matrix_size], dtype=tf.float32)
        start = time.time()
        result = tf.matmul(lhs, rhs)
        checksum = float(tf.reduce_sum(result).numpy())
        elapsed = time.time() - start

    actual_device = result.device or "UNKNOWN"
    execution_backend = "GPU" if "GPU" in actual_device.upper() else "CPU"

    print(f"REQUESTED_DEVICE: {requested_device}")
    print(f"ACTUAL_DEVICE: {actual_device}")
    print(f"EXECUTION_BACKEND: {execution_backend}")
    print(f"MATRIX_SIZE: {matrix_size}")
    print(f"ELAPSED_SECONDS: {elapsed:.3f}")
    print(f"CHECKSUM: {checksum}")

    return 0 if execution_backend == "GPU" else 2


if __name__ == "__main__":
    raise SystemExit(main())
