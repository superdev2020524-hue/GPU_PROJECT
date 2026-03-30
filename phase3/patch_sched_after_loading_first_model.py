#!/usr/bin/env python3
"""
Add a file write in server/sched.go immediately after the line that logs "loading first model".
Used to narrow where sched.load() blocks: between load() entry and llama.Load() (newServerFn).

Run on VM: python3 patch_sched_after_loading_first_model.py /home/test-4/ollama/server/sched.go
If the build fails with undefined "os", add "os" to the import block in sched.go.
Then rebuild ollama, install, restart, trigger generate. Check:
  - phase3_sched_load_entered.txt exists, phase3_sched_after_loading_first_model.txt missing
    -> block is before "loading first model" log.
  - both exist, phase3_before_llama_load.txt missing
    -> block is after that log, before llama.Load() (inside newServerFn/server creation).
"""
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "/home/test-4/ollama/server/sched.go"

with open(path) as f:
    lines = f.readlines()

if "phase3_sched_after_loading_first_model.txt" in "".join(lines):
    print("ALREADY_PATCHED")
    sys.exit(0)

# Insert right after the line that contains "loading first model" (e.g. logutil.Trace or slog.Info)
marker = "loading first model"
new_lines = []
injected = False
for line in lines:
    new_lines.append(line)
    if not injected and marker in line:
        # Use same leading whitespace as the log line (sched uses tabs)
        indent = len(line) - len(line.lstrip())
        prefix = line[:indent]
        insert_line = prefix + 'if f, e := os.OpenFile("/tmp/phase3_sched_after_loading_first_model.txt", os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644); e == nil { f.WriteString("1\\n"); f.Close() }\n'
        new_lines.append(insert_line)
        injected = True

if not injected:
    print("NOT_FOUND")
    sys.exit(1)

# If sched.go does not already import "os", the build will fail; add "os" to the import block manually if needed.
with open(path, "w") as f:
    f.writelines(new_lines)

print("PATCHED")
