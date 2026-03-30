#!/usr/bin/env python3
"""Fix GetRunner: add else branch so we enqueue when runner is nil or needs reload."""
path = "/home/test-4/ollama/server/sched.go"
with open(path) as f:
    lines = f.readlines()

# Find the block: after "req.useLoadedRunner(runner, s.finishedReqCh)" we have select{...};
# we need "} else {" between useLoadedRunner and select.
new_lines = []
i = 0
while i < len(lines):
    line = lines[i]
    new_lines.append(line)
    # If this is the useLoadedRunner line (with only that statement in the if body so far),
    # next we need "} else {" before the select. So we're looking for the line that is
    # "\t\tselect {" right after "\t\treq.useLoadedRunner(runner, s.finishedReqCh)"
    if "req.useLoadedRunner(runner, s.finishedReqCh)" in line and i + 1 < len(lines):
        next_line = lines[i + 1]
        if "select {" in next_line and "} else {" not in "".join(lines[i - 2 : i + 2]):
            # Insert "} else {" with same indent as the if's body (one more tab than select)
            new_lines.append("\t} else {\n")
            i += 1
            continue
    i += 1

with open(path, "w") as f:
    f.writelines(new_lines)
print("Done")
