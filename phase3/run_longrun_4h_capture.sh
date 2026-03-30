#!/usr/bin/env bash
# Run ON THE VM after mediator + ollama reset. Single 4h POST /api/generate with full capture.
# Ref: LONG_RUN_4H_LOG_PATHS.md — client -m 14400 matches OLLAMA_LOAD_TIMEOUT=4h.
set -euo pipefail

TS="${PHASE3_LONGRUN_TS:-$(date +%Y%m%d_%H%M%S)}"
BASE="/tmp/phase3_longrun_${TS}"
mkdir -p "$BASE"
chmod 755 "$BASE" 2>/dev/null || true

{
  echo "phase3_longrun_session=$TS"
  echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "hostname=$(hostname)"
  echo "BASE=$BASE"
} | tee "$BASE/session_meta.txt"

{
  echo "=== systemctl status ollama ==="
  systemctl --no-pager status ollama 2>&1 || true
  echo "=== curl /api/tags ==="
  curl -sS -m 15 http://127.0.0.1:11434/api/tags 2>&1 || true
  echo "=== journalctl ollama (last 200 lines, boot) ==="
  journalctl -u ollama -b --no-pager -n 200 2>&1 || true
} > "$BASE/preflight.txt" 2>&1

{
  echo "=== Error-trace baseline (boot journal, E1–E5 keywords) — before long curl ==="
  echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  journalctl -u ollama -b --no-pager 2>/dev/null | grep -Ei 'mmq|mmq_x|3884|GGML_ABORT|ggml_abort|SIGABRT|fatal error|llama_init|401312|INVALID|runner|sched\.go.*error|exit status|context canceled|CUDA_ERROR|illegal address|rc=700' | tail -120 || true
} > "$BASE/checkpoint_vm_error_trace.txt" 2>&1

REQ_JSON='{"model":"tinyllama:latest","prompt":"Say hello in one sentence.","stream":false,"options":{"num_predict":32}}'
printf '%s\n' "$REQ_JSON" > "$BASE/generate_request.json"

pkill -f 'curl.*127\.0\.0\.1:11434/api/generate' 2>/dev/null || true
sleep 2

# Inner runner: journal follow + 4h curl; stop follow when curl exits
cat > "$BASE/run_inner.sh" <<EOF
#!/usr/bin/env bash
set -u
BASE="$BASE"
exec >>"\$BASE/nohup_inner.log" 2>&1
set -x
echo "inner_start=\$(date -Is) pid=\$\$"
journalctl -u ollama -f --no-pager >> "\$BASE/journal_ollama_follow.log" 2>>"\$BASE/journal_ollama_follow.err" &
JPID=\$!
echo "\$JPID" > "\$BASE/journal_follow.pid"
set +e
curl -sS -m 14400 \\
  -w '\\nHTTP_CODE:%{http_code}\\nTIME_TOTAL:%{time_total}\\nSIZE_DOWNLOAD:%{size_download}\\n' \\
  -X POST http://127.0.0.1:11434/api/generate \\
  -H 'Content-Type: application/json' \\
  -d @"\$BASE/generate_request.json" \\
  -o "\$BASE/curl_generate.out.json" \\
  -D "\$BASE/curl_generate.response_headers.txt" \\
  >>"\$BASE/curl_generate.stdout.log" 2>>"\$BASE/curl_generate.stderr.log"
CURL_RC=\$?
set -e
{
  echo "curl_exit_code=\$CURL_RC"
  echo "curl_finished=\$(date -Is)"
  ls -la "\$BASE/curl_generate.out.json" 2>&1 || true
  wc -c "\$BASE/curl_generate.out.json" 2>&1 || true
  head -c 1200 "\$BASE/curl_generate.out.json" 2>&1 || true
} >> "\$BASE/curl_summary.txt"
kill \$JPID 2>/dev/null || true
echo "inner_done=\$(date -Is)"
EOF
chmod +x "$BASE/run_inner.sh"

nohup "$BASE/run_inner.sh" >> "$BASE/nohup_wrapper.log" 2>&1 &
echo $! > "$BASE/nohup_wrapper.pid"

{
  echo "All artifacts under: $BASE"
  echo "inner script: $BASE/run_inner.sh"
  echo "nohup_wrapper_pid=$(cat "$BASE/nohup_wrapper.pid")"
  echo ""
  echo "Host (dom0) after run — mediator slice:"
  echo "  ssh root@<dom0> \"grep -E '401312|700|INVALID|GEMM|module-load|HtoD|cuda-executor' /tmp/mediator.log | tail -800\""
  echo "  ssh root@<dom0> \"ls -la /tmp/fail401312.bin\""
  echo "Error-trace (VM, pre-curl): $BASE/checkpoint_vm_error_trace.txt"
  echo "Workstation: collect_host_longrun_slice.sh + LONGRUN_SESSION_<TS>.md (error trace sections)"
} | tee "$BASE/README.txt"

echo "STARTED_BASE=$BASE"
