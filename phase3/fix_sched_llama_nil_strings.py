#!/usr/bin/env python3
"""Fix broken multi-line WriteString in sched.go to single line."""
import sys
path = sys.argv[1]
with open(path) as f:
    content = f.read()
# Fix: "llama_nil\n");  on next line -> "llama_nil\\n"); on same line
content = content.replace(
    'w.WriteString("llama_nil\n"); w.Close() }',
    'w.WriteString("llama_nil\\n"); w.Close() }'
)
# Fix the split version: line ending with llama_nil and next line "); w.Close()
content = content.replace(
    'e == nil { w.WriteString("llama_nil\n"); w.Close() }',
    'e == nil { w.WriteString("llama_nil\\n"); w.Close() }'
)
# Multiline: ... "llama_nil  then newline  "); w.Close()
import re
# Fix actual broken split: line ends with WriteString("llama_nil, next line is "); w.Close()
content = re.sub(
    r'(w\.WriteString\("llama_nil)\n\s*("\); w\.Close\(\) \})',
    r'\1\\n\2',
    content
)
content = re.sub(
    r'(w\.WriteString\("llama_not_nil)\n\s*("\); w\.Close\(\) \})',
    r'\1\\n\2',
    content
)
with open(path, 'w') as f:
    f.write(content)
print("OK")
