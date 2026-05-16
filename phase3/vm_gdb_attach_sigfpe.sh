#!/usr/bin/env bash
# VM-6 E3: attach GDB to the llama runner (excludes discovery 'runner --ollama-engine')
# as soon as it appears during /api/generate; catch SIGFPE; print backtraces.
# Requires: sudo (CAP_SYS_PTRACE), gdb; ollama active; see CRASH_SYMBOLICATION_AND_COREDUMPS.md §4.
set -euo pipefail

OUT_LOG="${1:-/tmp/gdb_attach_sigfpe.log}"
GEN_JSON="${2:-/tmp/gdb_attach_gen.json}"
: >"$OUT_LOG"

is_engine_runner() {
  local p="$1"
  local cmd
  cmd="$(tr '\0' ' ' <"/proc/$p/cmdline" 2>/dev/null || echo '')"
  [[ "$cmd" == *ollama-engine* ]]
}

poll_llama_runner_pid() {
  local p cmd
  for p in $(pgrep -f 'ollama\.bin runner' 2>/dev/null || true); do
    if is_engine_runner "$p"; then
      continue
    fi
    echo "$p"
    return 0
  done
  return 1
}

curl -sS -o "$GEN_JSON" -w "curl_http:%{http_code}\n" -m 180 \
  -H 'Content-Type: application/json' \
  -d '{"model":"tinyllama:latest","prompt":"gdb_attach","stream":false,"options":{"num_predict":2}}' \
  http://127.0.0.1:11434/api/generate &
curl_pid=$!

attached=
while kill -0 "$curl_pid" 2>/dev/null; do
  if rp=$(poll_llama_runner_pid); then
    cmd=$(tr '\0' ' ' <"/proc/$rp/cmdline" 2>/dev/null || echo '')
    {
      echo "=== attaching gdb to pid=$rp cmd=$cmd ==="
      sudo gdb -q -batch \
        -ex 'set pagination off' \
        -ex 'catch signal SIGFPE' \
        -ex 'continue' \
        -ex 'printf "\n=== faulting thread $pc / rip ===\n"' \
        -ex 'p/x $pc' \
        -ex 'x/16i $pc' \
        -ex 'info registers rip rsp rbp rax rbx rcx rdx rsi rdi r8 r9 r10 r11' \
        -ex 'thread apply all bt full' \
        -ex 'detach' \
        -ex 'quit' \
        -p "$rp" 2>&1 || true
    } | tee -a "$OUT_LOG"
    attached=1
    break
  fi
  sleep 0.1
done

wait "$curl_pid" || true
echo "=== curl finished ===" | tee -a "$OUT_LOG"
head -c 500 "$GEN_JSON" 2>/dev/null | tee -a "$OUT_LOG" || true
echo "" | tee -a "$OUT_LOG"

if [[ -z "${attached:-}" ]]; then
  echo "=== WARN: never attached (no non-engine runner seen before curl ended) ===" | tee -a "$OUT_LOG"
fi
