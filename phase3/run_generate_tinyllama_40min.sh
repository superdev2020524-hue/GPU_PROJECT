#!/bin/bash
# Generate with tinyllama and 40-minute client timeout.
#
# From host (phase3 dir): CONNECT_VM_COMMAND_TIMEOUT_SEC=2500 python3 connect_vm.py "curl -s -X POST http://127.0.0.1:11434/api/generate -d '{\"model\":\"tinyllama\",\"prompt\":\"Hi\",\"stream\":false}' -m 2400 -o /tmp/gen_tinyllama_40m.json; echo RC=\$?; ls -la /tmp/gen_tinyllama_40m.json; head -c 400 /tmp/gen_tinyllama_40m.json"
#
# Or run this script on the VM: bash run_generate_tinyllama_40min.sh

CLIENT_TIMEOUT=2400   # 40 minutes in seconds
OUTPUT_FILE="${OUTPUT_FILE:-/tmp/gen_tinyllama_40m.json}"

curl -s -X POST http://127.0.0.1:11434/api/generate \
  -d '{"model":"tinyllama","prompt":"Hi","stream":false}' \
  -m "$CLIENT_TIMEOUT" \
  -o "$OUTPUT_FILE"
echo "RC=$?"
ls -la "$OUTPUT_FILE" 2>&1
head -c 500 "$OUTPUT_FILE" 2>/dev/null
