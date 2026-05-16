#!/usr/bin/env bash
# Resident March 29 Test-4: PHASE3_NO_HTTP_TIMEOUT_STRATEGY.md §7 (CPU prime) then strict Test-4.
# Do NOT run archival Test-4 (omit num_gpu) alone — can HTTP 000 @ curl -m 185 (ERROR_TRACKING_STATUS.md 2026-05-16).
# Use run_mar29_section8_chain.sh after systemctl restart (full §8).
set -uo pipefail

LOGTAG="m29_res_$(date +%Y%m%d_%H%M%S)"
CPU_JSON="/tmp/${LOGTAG}_cpu.json"
GPU_JSON="/tmp/${LOGTAG}_gpu.json"
echo "[$LOGTAG] §7 CPU prime (curl -m 120)"
curl -sS -m 120 -o "$CPU_JSON" -w "HTTPCPU:%{http_code} CPUtime:%{time_total}\n" \
  http://127.0.0.1:11434/api/generate \
  -d '{"model":"tinyllama:latest","prompt":"Hello","stream":false,"options":{"num_gpu":0,"num_predict":16,"temperature":0.3,"top_p":0.85}}'
head -c 720 "$CPU_JSON"
echo

echo "[$LOGTAG] strict Test-4 omit num_gpu (curl -m 185)"
curl -sS -m 185 -o "$GPU_JSON" -w "HTTPT4:%{http_code} T4time:%{time_total}\n" \
  http://127.0.0.1:11434/api/generate \
  -d '{"model":"tinyllama:latest","prompt":"Hello","stream":false,"options":{"num_predict":16,"temperature":0.3,"top_p":0.85}}'
head -c 720 "$GPU_JSON"
echo

rm -f "$CPU_JSON" "$GPU_JSON"

echo "[$LOGTAG] done"
