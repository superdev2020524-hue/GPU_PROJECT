#!/usr/bin/env bash
# VM-6: full cold chain per PHASE3_NO_HTTP_TIMEOUT_STRATEGY.md §8 (see vm_async_preload.sh trailer).
# Run on the guest: bash run_mar29_section8_chain.sh  OR via connect_vm.py.
set -uo pipefail

LOGTAG="m29_s8_$(date +%Y%m%d_%H%M%S)"
echo "[$LOGTAG] sudo systemctl restart ollama"
sudo systemctl restart ollama

echo "[$LOGTAG] wait for /api/tags"
ok=0
for i in $(seq 1 40); do
  if curl -sf -m 8 http://127.0.0.1:11434/api/tags >/dev/null; then
    echo "[$LOGTAG] tags_ok iter=$i"
    ok=1
    break
  fi
  sleep 2
done
if [ "$ok" != 1 ]; then
  echo "[$LOGTAG] FATAL tags_timeout"
  exit 2
fi

echo "[$LOGTAG] §1 nohup preload (keep_alive:-1)"
nohup curl -sS -X POST http://127.0.0.1:11434/api/generate \
  -d '{"model":"tinyllama:latest","keep_alive":-1}' \
  -o "/tmp/${LOGTAG}_pl.out" \
  2>>"/tmp/${LOGTAG}_pl.err" &
echo "[$LOGTAG] PL_PID=$!"

echo "[$LOGTAG] poll GET /api/ps until tinyllama (60s x max 30)"
w=0
hit=0
while [ "$w" -lt 30 ]; do
  if curl -sS -m 20 http://127.0.0.1:11434/api/ps | grep -q tinyllama; then
    echo "[$LOGTAG] PS_HIT w=$w"
    hit=1
    break
  fi
  w=$((w + 1))
  echo "[$LOGTAG] PS_POLL w=$w"
  sleep 60
done
if [ "$hit" != 1 ]; then
  echo "[$LOGTAG] FATAL PS_TIMEOUT"
  exit 3
fi

echo "[$LOGTAG] §7 CPU prime (curl -m 120)"
curl -sS -w "\nHTTPCPU:%{http_code} CPUtime:%{time_total}\n" -m 120 \
  http://127.0.0.1:11434/api/generate \
  -d '{"model":"tinyllama:latest","prompt":"Hello","stream":false,"options":{"num_gpu":0,"num_predict":16,"temperature":0.3,"top_p":0.85}}' \
  | head -c 720
echo

echo "[$LOGTAG] §8 strict Test-4 omit num_gpu (curl -m 185)"
curl -sS -w "\nHTTPT4:%{http_code} T4time:%{time_total}\n" -m 185 \
  http://127.0.0.1:11434/api/generate \
  -d '{"model":"tinyllama:latest","prompt":"Hello","stream":false,"options":{"num_predict":16,"temperature":0.3,"top_p":0.85}}' \
  | head -c 720
echo

echo "[$LOGTAG] GET /api/ps"
curl -sS -m 20 http://127.0.0.1:11434/api/ps | head -c 1200
echo

echo "[$LOGTAG] bounded journal (30 min): inference / load path"
journalctl -u ollama -S '30 min ago' --no-pager \
  | grep -E 'inference compute|load_tensors:|GPU model buffer|CUDA model buffer|library=CUDA' \
  | tail -25 || true

echo "[$LOGTAG] done"
