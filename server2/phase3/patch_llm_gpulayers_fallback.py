#!/usr/bin/env python3
"""Patch llm/server.go: when gpus exist but gpuLayers is empty, force layer 0 onto first GPU (Phase3)."""
import sys
path = sys.argv[1] if len(sys.argv) > 1 else "/home/test-4/ollama/llm/server.go"
with open(path) as f:
    content = f.read()
# Insert before first s.loadRequest.GPULayers = gpuLayers (first Load path)
old = "\tif err := s.waitUntilRunnerLaunched(ctx); err != nil {\n\t\treturn nil, err\n\t}\n\n\ts.loadRequest.GPULayers = gpuLayers\n\tresp, err := s.initModel(ctx, s.loadRequest, LoadOperationCommit)"
new = "\tif err := s.waitUntilRunnerLaunched(ctx); err != nil {\n\t\treturn nil, err\n\t}\n\n\t// Phase3/vGPU: when gpus exist but createLayout returned empty, force one layer onto first GPU\n\tif len(gpus) > 0 && gpuLayers.Sum() == 0 {\n\t\tgpuLayers = ml.GPULayersList{{DeviceID: gpus[0].DeviceID, Layers: []int{0}}}\n\t}\n\n\ts.loadRequest.GPULayers = gpuLayers\n\tresp, err := s.initModel(ctx, s.loadRequest, LoadOperationCommit)"
if new in content:
    print("ALREADY_PATCHED")
    sys.exit(0)
if old not in content:
    print("NOT_FOUND")
    sys.exit(1)
content = content.replace(old, new, 1)
with open(path, "w") as f:
    f.write(content)
print("PATCHED")
