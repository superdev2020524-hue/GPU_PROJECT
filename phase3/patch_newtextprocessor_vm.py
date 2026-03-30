#!/usr/bin/env python3
"""Patch llm/server.go: try NewTextProcessor when tok==nil to avoid LoadModelFromFile hang."""
import sys
path = "/home/test-4/ollama/llm/server.go"
with open(path) as f:
    s = f.read()
old = "\tif tok == nil {\n\t\tif lf2, le2 :="
new = """\t// Phase3: try Go-only tokenizer first to avoid C LoadModelFromFile hang in vGPU server
\tif tok == nil && len(projectors) == 0 {
\t\ttok, err = model.NewTextProcessor(modelPath)
\t\tif err != nil {
\t\t\tslog.Debug("Phase3 NewTextProcessor fallback failed, using LoadModelFromFile", "model", modelPath, "error", err)
\t\t}
\t}
\tif tok == nil {
\t\tif lf2, le2 :="""
if old not in s:
    print("OLD not found", file=sys.stderr)
    sys.exit(1)
s = s.replace(old, new, 1)
with open(path, "w") as f:
    f.write(s)
print("OK")
