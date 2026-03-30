#!/usr/bin/env python3
"""Patch server/sched.go: when NumGPU==0 but getGpuFn returns GPUs, use GPU (Phase3)."""
import sys
path = sys.argv[1] if len(sys.argv) > 1 else "/home/test-4/ollama/server/sched.go"
with open(path) as f:
    c = f.read()

# Tabs as on VM (5 before //, 5 var, 4 if, 6 gpus=[], 4 } else, 5 Trace, 6 getGpuFn, 5 })
old = "\t\t\t\t\t// Get a refreshed GPU list\n\t\t\t\t\tvar gpus []ml.DeviceInfo\n\t\t\t\t\tif pending.opts.NumGPU == 0 {\n\t\t\t\t\t\tgpus = []ml.DeviceInfo{}\n\t\t\t\t\t} else {\n\t\t\t\t\t\tlogutil.Trace(\"refreshing GPU list\", \"model\", pending.model.ModelPath)\n\t\t\t\t\t\tgpus = s.getGpuFn(ctx, runnersSnapshot)\n\t\t\t\t\t}"

new = "\t\t\t\t\t// Get a refreshed GPU list\n\t\t\t\t\tvar gpus []ml.DeviceInfo\n\t\t\t\t\tlogutil.Trace(\"refreshing GPU list\", \"model\", pending.model.ModelPath)\n\t\t\t\t\tgpus = s.getGpuFn(ctx, runnersSnapshot)\n\t\t\t\t\tif pending.opts.NumGPU == 0 && len(gpus) > 0 {\n\t\t\t\t\t\t// Phase3/vGPU: API omitempty gives NumGPU=0; prefer GPU when available\n\t\t\t\t\t\tpending.opts.NumGPU = -1\n\t\t\t\t\t} else if pending.opts.NumGPU == 0 {\n\t\t\t\t\t\tgpus = []ml.DeviceInfo{}\n\t\t\t\t\t}"

if old not in c:
    print("NOT_FOUND")
    sys.exit(1)
c = c.replace(old, new, 1)
with open(path, "w") as f:
    f.write(c)
print("PATCHED_SCHED")
