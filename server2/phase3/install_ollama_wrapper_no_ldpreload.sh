#!/bin/bash
# Install ollama wrapper so runner gets LD_LIBRARY_PATH (no LD_PRELOAD).
# Per VM_TEST3_GPU_MODE_STATUS.md and GPU_MODE_DO_NOT_BREAK.md: runner must use
# real dlopen to load libggml-cuda; only LD_LIBRARY_PATH with /opt/vgpu/lib first.
set -e
if [ ! -f /usr/local/bin/ollama.real ]; then
  cp -a /usr/local/bin/ollama /usr/local/bin/ollama.real
fi
cat > /tmp/ollama_wrapper.sh << 'WRAP'
#!/bin/bash
export LD_LIBRARY_PATH="/opt/vgpu/lib:/usr/local/lib/ollama/cuda_v12:/usr/local/lib/ollama:${LD_LIBRARY_PATH:-}"
export OLLAMA_LIBRARY_PATH="/opt/vgpu/lib:/usr/local/lib/ollama/cuda_v12:/usr/local/lib/ollama"
export OLLAMA_LLM_LIBRARY="cuda_v12"
export OLLAMA_NUM_GPU="1"
exec /usr/local/bin/ollama.real "$@"
WRAP
chmod +x /tmp/ollama_wrapper.sh
cp /tmp/ollama_wrapper.sh /usr/local/bin/ollama
echo "Wrapper installed."
