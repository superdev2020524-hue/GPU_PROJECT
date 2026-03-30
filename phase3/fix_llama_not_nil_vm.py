#!/usr/bin/env python3
path = "/home/test-4/ollama/server/sched.go"
with open(path) as f:
    lines = f.readlines()
# Find line with llama_not_nil that doesn't end with \n"); w.Close()
out = []
i = 0
while i < len(lines):
    line = lines[i]
    if "llama_not_nil" in line and "); w.Close()" not in line and i + 1 < len(lines):
        next_line = lines[i + 1]
        if '"); w.Close()' in next_line or '"); w.Close()' in next_line:
            # Merge: line ends with WriteString("llama_not_nil, next is "); w.Close() }
            # Want: WriteString("llama_not_nil\n"); w.Close() }
            new_line = line.rstrip()
            if 'llama_not_nil' in new_line and not new_line.endswith('"); w.Close() }'):
                new_line = new_line + '\\n"); w.Close() }\n'
                out.append(new_line)
                i += 2  # skip next line
                continue
    out.append(line)
    i += 1
with open(path, 'w') as f:
    f.writelines(out)
print("OK")
