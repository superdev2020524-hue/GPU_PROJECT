#!/bin/sh
# Run on VM via: connect_vm.py "bash -s" < run_vm_quick_generate.sh
# Or paste the body into one connect_vm line.
: > /tmp/vgpu_fatbinary_fingerprint.log
printf '%s\n' '{"model":"tinyllama","prompt":"ok","stream":false,"options":{"num_predict":4}}' > /tmp/gen_req.json
nohup curl -sS -m 7200 -o /tmp/phase3_gen_out.json -w "\nhttp_code=%{http_code}\n" \
  -H "Content-Type: application/json" -d @/tmp/gen_req.json \
  http://127.0.0.1:11434/api/generate >> /tmp/phase3_gen_curl.log 2>&1 &
echo "STARTED curl pid=$!"
