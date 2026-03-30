#!/usr/bin/env bash
set -euo pipefail

echo "__VM_PROBE_BEGIN__"
systemctl is-active ollama
hostname
whoami
pwd

curl -s -m 90 -X POST http://127.0.0.1:11434/api/generate \
  -H 'Content-Type: application/json' \
  -d '{"model":"tinyllama","prompt":"Hi","stream":false,"options":{"num_predict":4}}' \
  -o /tmp/cursor_stage1_probe.json \
  -w 'HTTP=%{http_code}\n'

echo "__VM_PROBE_END__"
