#!/bin/bash
# Ollama wrapper script for systemd
# Ensures LD_PRELOAD is set and propagated to all subprocesses

# Set shim libraries
export LD_PRELOAD="/usr/lib64/libvgpu-exec.so:/usr/lib64/libvgpu-syscall.so:/usr/lib64/libvgpu-cuda.so:/usr/lib64/libvgpu-nvml.so"
export LD_LIBRARY_PATH="/usr/local/lib/ollama/cuda_v12:/usr/local/lib/ollama:/usr/lib64"
export NVIDIA_VISIBLE_DEVICES=all
export OLLAMA_LLM_LIBRARY=cuda_v12
export OLLAMA_NUM_GPU=999

# Log wrapper execution
echo "[ollama-wrapper] Starting Ollama with shim injection (pid=$$, LD_PRELOAD=$LD_PRELOAD)" >&2

# Execute Ollama with all environment variables
exec /usr/local/bin/ollama serve "$@"
