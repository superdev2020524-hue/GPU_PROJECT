#!/usr/bin/env python3
"""Add file writes at entry of NewLlamaServer and right after StartRunner returns (Phase3). Run on VM."""
import sys
path = sys.argv[1] if len(sys.argv) > 1 else "/home/test-4/ollama/llm/server.go"
with open(path) as f:
    c = f.read()

if "phase3_newllama_entry.txt" in c:
    print("ALREADY")
    sys.exit(0)

# Entry: right after "func NewLlamaServer(...) (LlamaServer, error) {"
old_entry = """func NewLlamaServer(systemInfo ml.SystemInfo, gpus []ml.DeviceInfo, modelPath string, f *ggml.GGML, adapters, projectors []string, opts api.Options, numParallel int) (LlamaServer, error) {
	var llamaModel *llama.Model"""
new_entry = """func NewLlamaServer(systemInfo ml.SystemInfo, gpus []ml.DeviceInfo, modelPath string, f *ggml.GGML, adapters, projectors []string, opts api.Options, numParallel int) (LlamaServer, error) {
	if f, e := os.OpenFile("/tmp/phase3_newllama_entry.txt", os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644); e == nil { f.WriteString("entry\\n"); f.Close() }
	var llamaModel *llama.Model"""

# After StartRunner returns: after ")" of StartRunner(...,) and before "s := llmServer{"
old_after = """		ml.GetVisibleDevicesEnv(gpus, false),
	)

	s := llmServer{"""
new_after = """		ml.GetVisibleDevicesEnv(gpus, false),
	)

	if f, e := os.OpenFile("/tmp/phase3_newllama_entry.txt", os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644); e == nil { f.WriteString("after_start_runner\\n"); f.Close() }
	s := llmServer{"""

if old_entry not in c:
    print("NOT_FOUND_ENTRY")
    sys.exit(1)
if old_after not in c:
    print("NOT_FOUND_AFTER")
    sys.exit(1)
c = c.replace(old_entry, new_entry, 1)
c = c.replace(old_after, new_after, 1)
with open(path, "w") as f:
    f.write(c)
print("PATCHED")
