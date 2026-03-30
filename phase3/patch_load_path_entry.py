#!/usr/bin/env python3
"""Add file write at start of ollamaServer.Load and llamaServer.Load (Phase3). Run on VM."""
import sys
path = sys.argv[1] if len(sys.argv) > 1 else "/home/test-4/ollama/llm/server.go"
with open(path) as f:
    c = f.read()

if "phase3_load_path.txt" in c:
    print("ALREADY")
    sys.exit(0)

# ollamaServer.Load: after "requireFull bool) ([]ml.DeviceID, error) {" insert write before "var success bool"
old_ollama = """func (s *ollamaServer) Load(ctx context.Context, systemInfo ml.SystemInfo, gpus []ml.DeviceInfo, requireFull bool) ([]ml.DeviceID, error) {
	var success bool"""
new_ollama = """func (s *ollamaServer) Load(ctx context.Context, systemInfo ml.SystemInfo, gpus []ml.DeviceInfo, requireFull bool) ([]ml.DeviceID, error) {
	if f, e := os.OpenFile("/tmp/phase3_load_path.txt", os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644); e == nil { f.WriteString("ollama_load\\n"); f.Close() }
	var success bool"""

# llamaServer.Load: after "requireFull bool) ([]ml.DeviceID, error) {" insert write before "slog.Info"
old_llama = """func (s *llamaServer) Load(ctx context.Context, systemInfo ml.SystemInfo, systemGPUs []ml.DeviceInfo, requireFull bool) ([]ml.DeviceID, error) {
	slog.Info("loading model\""""
new_llama = """func (s *llamaServer) Load(ctx context.Context, systemInfo ml.SystemInfo, systemGPUs []ml.DeviceInfo, requireFull bool) ([]ml.DeviceID, error) {
	if f, e := os.OpenFile("/tmp/phase3_load_path.txt", os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644); e == nil { f.WriteString("llama_load\\n"); f.Close() }
	slog.Info("loading model\""""

if old_ollama not in c or old_llama not in c:
    print("NOT_FOUND")
    sys.exit(1)
c = c.replace(old_ollama, new_ollama, 1)
c = c.replace(old_llama, new_llama, 1)
with open(path, "w") as f:
    f.write(c)
print("PATCHED")
