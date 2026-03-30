#!/bin/bash
# Ollama wrapper script for systemd
# Ensures LD_PRELOAD is set and propagated to all subprocesses

# Set shim libraries
# Do not preload libvgpu-exec / libvgpu-syscall with Ollama Go binary — causes
# exit 126 "Inappropriate ioctl for device" under systemd. CUDA + NVML shims suffice
# for discovery and mediated CUDA (see FATBIN_CUBLAS_CC_ANALYSIS_MAR21.md).
export LD_PRELOAD="/usr/lib64/libvgpu-cudart.so:/usr/lib64/libvgpu-cuda.so:/usr/lib64/libvgpu-nvml.so"
export LD_LIBRARY_PATH="/opt/vgpu/lib:/usr/local/lib/ollama/cuda_v12:/usr/local/lib/ollama:/usr/lib64"
export NVIDIA_VISIBLE_DEVICES=all
export OLLAMA_LLM_LIBRARY=cuda_v12
export OLLAMA_NUM_GPU=999

# Log wrapper execution
echo "[ollama-wrapper] Starting Ollama with shim injection (pid=$$, LD_PRELOAD=$LD_PRELOAD)" >&2

# Use the patched Phase3 binary (not /usr/local/bin/ollama → ollama.real chain).
exec /usr/local/bin/ollama.bin.new serve "$@"
