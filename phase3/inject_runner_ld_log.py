#!/usr/bin/env python3
"""Insert debug log in server.go: log LD_LIBRARY_PATH that runner will get."""
import sys
path = sys.argv[1] if len(sys.argv) > 1 else "/home/test-4/ollama/llm/server.go"
with open(path) as f:
    content = f.read()
marker = "\t// Remove LD_PRELOAD from runner"
debug = '\tfor _, e := range cmd.Env {\n\t\tif strings.HasPrefix(e, "LD_LIBRARY_PATH=") {\n\t\t\tslog.Info("runner env LD_LIBRARY_PATH", "value", e)\n\t\t\tbreak\n\t\t}\n\t}\n\n'
if debug.strip() in content:
    print("ALREADY_HAS_DEBUG")
    sys.exit(0)
if marker not in content:
    print("MARKER_NOT_FOUND")
    sys.exit(1)
new_content = content.replace(marker, debug + marker, 1)
with open(path, "w") as f:
    f.write(new_content)
print("INSERTED_DEBUG")
