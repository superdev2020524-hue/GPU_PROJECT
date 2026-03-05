#!/bin/bash
# Wrapper for /usr/local/bin/ollama so that when "ollama serve" spawns "ollama runner",
# the runner gets our vGPU shims (LD_PRELOAD/LD_LIBRARY_PATH). The runner often
# does not inherit the service environment; this ensures every ollama invocation has it.
export LD_PRELOAD="/opt/vgpu/lib/libnvidia-ml.so.1:/opt/vgpu/lib/libcuda.so.1:/opt/vgpu/lib/libcudart.so.12${LD_PRELOAD:+:$LD_PRELOAD}"
export LD_LIBRARY_PATH="/opt/vgpu/lib:/usr/local/lib/ollama/cuda_v12:/usr/local/lib/ollama:${LD_LIBRARY_PATH:-/usr/lib64}"
export OLLAMA_LLM_LIBRARY="${OLLAMA_LLM_LIBRARY:-cuda_v12}"
export OLLAMA_NUM_GPU="${OLLAMA_NUM_GPU:-999}"
exec /usr/local/bin/ollama.real "$@"
