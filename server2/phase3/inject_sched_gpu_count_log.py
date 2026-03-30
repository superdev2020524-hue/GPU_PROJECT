#!/usr/bin/env python3
"""Inject a log line in server/sched.go after getGpuFn so we can see gpu_count at load time (Phase3)."""
import sys
path = sys.argv[1] if len(sys.argv) > 1 else "/home/test-4/ollama/server/sched.go"
with open(path) as f:
    lines = f.readlines()

# Insert after the line containing "gpus = s.getGpuFn(ctx, runnersSnapshot)"
marker = "gpus = s.getGpuFn(ctx, runnersSnapshot)"
insert_line = '\t\t\t\t\tlogutil.Trace("Phase3 getGpuFn result", "gpu_count", len(gpus))\n'

new_lines = []
injected = False
for line in lines:
    new_lines.append(line)
    if not injected and marker in line and "getGpuFn" in line:
        new_lines.append(insert_line)
        injected = True

if not injected:
    print("NOT_FOUND")
    sys.exit(1)
with open(path, "w") as f:
    f.writelines(new_lines)
print("INJECTED")
