#!/usr/bin/env python3
"""
Inject runner-side log to /tmp/runner_load_gpulayers.txt in the Ollama runner's load handler.
Run on VM: python3 inject_runner_load_gpulayers_log.py /home/test-4/ollama/runner/runner.go
(or whatever path contains the load request log). Finds a line with "load" and "request" (e.g.
slog.Info("load", "request", req)) and inserts a block that appends gpulayers len/sum/op.
"""
import sys
import re

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 inject_runner_load_gpulayers_log.py <path/to/runner.go>")
        sys.exit(1)
    path = sys.argv[1]
    with open(path) as f:
        content = f.read()

    # Already patched?
    if "runner_load_gpulayers.txt" in content:
        print("ALREADY_PATCHED")
        return 0

    # Find line with load + request log (e.g. slog.Info("load", "request", req))
    lines = content.split("\n")
    insert_after = -1
    for i, line in enumerate(lines):
        if "load" in line and "request" in line and ("req" in line or "req)" in line):
            insert_after = i
            break
    if insert_after < 0:
        print("NOT_FOUND: no load/request log line in file")
        sys.exit(1)

    # Block to append after that line. Use req.GPULayers if present; Operation is load op (Fit=0, etc.)
    block = '''
	// Phase3: log GPULayers received by runner (LOAD_RUNNER_GPULAYERS_FIX, ERROR_TRACKING_STATUS)
	if f, err := os.OpenFile("/tmp/runner_load_gpulayers.txt", os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644); err == nil {
		gpulen, gpusum := 0, 0
		if req.GPULayers != nil {
			gpulen = len(req.GPULayers)
			for _, gl := range req.GPULayers {
				gpusum += len(gl.Layers)
			}
		}
		fmt.Fprintf(f, "gpulayers=%d sum=%d op=%d\\n", gpulen, gpusum, int(req.Operation))
		f.Close()
	}
'''
    new_lines = lines[: insert_after + 1] + block.strip().split("\n") + lines[insert_after + 1 :]
    out = "\n".join(new_lines)

    # Ensure "os" and "fmt" are imported (runner may already have them)
    if "os.OpenFile" in out and ' "os" ' not in out and '"os"' not in out:
        out = re.sub(r'(\bimport\s+\()', r'\1\n\t"os"', out, count=1)
    if "fmt.Fprintf" in out and ' "fmt" ' not in out and '"fmt"' not in out:
        out = re.sub(r'(\bimport\s+\()', r'\1\n\t"fmt"', out, count=1)

    with open(path, "w") as f:
        f.write(out)
    print("PATCHED")
    return 0

if __name__ == "__main__":
    sys.exit(main() or 0)
