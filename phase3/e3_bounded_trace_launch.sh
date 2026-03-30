#!/usr/bin/env bash
# Run ON THE VM: bounded POST /api/generate + journal follow (not 4h).
# Workstation: scp + ssh, or: python3 connect_vm.py "bash -s" < e3_bounded_trace_launch.sh
set -euo pipefail
TS="$(date +%Y%m%d_%H%M%S)"
BASE="/tmp/e3_bounded_${TS}"
mkdir -p "$BASE"
echo "$BASE" > /tmp/e3_trace_latest.txt
{
  echo "phase3_e3_bounded_trace=$TS"
  echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "curl_client_timeout_sec=${E3_CURL_TIMEOUT_SEC:-7200}"
} | tee "$BASE/meta.txt"

REQ='{"model":"tinyllama:latest","prompt":"Say hello in one sentence.","stream":false,"options":{"num_predict":32}}'
printf '%s\n' "$REQ" > "$BASE/generate_request.json"

journalctl -u ollama -f --no-pager >>"$BASE/journal_follow.log" 2>>"$BASE/journal_follow.err" &
echo $! >"$BASE/journal_follow.pid"

CURL_SEC="${E3_CURL_TIMEOUT_SEC:-7200}"
curl -sS -m "$CURL_SEC" \
  -w '\nHTTP_CODE:%{http_code}\nTIME_TOTAL:%{time_total}\nSIZE_DOWNLOAD:%{size_download}\n' \
  -X POST http://127.0.0.1:11434/api/generate \
  -H 'Content-Type: application/json' \
  -d @"$BASE/generate_request.json" \
  -o "$BASE/curl_generate.out.json" \
  -D "$BASE/curl_generate.response_headers.txt" \
  >>"$BASE/curl_generate.stdout.log" 2>>"$BASE/curl_generate.stderr.log" &
echo $! >"$BASE/curl.pid"

echo "E3_TRACE_BASE=$BASE"
echo "journal_follow_pid=$(cat "$BASE/journal_follow.pid")"
echo "curl_pid=$(cat "$BASE/curl.pid")"
echo "Artifacts: $BASE/ (journal_follow.log, curl_*.json, meta.txt)"
