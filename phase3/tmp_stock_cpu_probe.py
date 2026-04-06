import json
import os
import signal
import subprocess
import time
import urllib.request

PORT = "11522"
BASE = f"http://127.0.0.1:{PORT}"

env = {
    "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
    "HOME": "/tmp/ollama-stock-cpu-home",
    "OLLAMA_HOST": f"127.0.0.1:{PORT}",
    "OLLAMA_MODELS": "/usr/share/ollama/.ollama/models",
    "OLLAMA_LLM_LIBRARY": "cpu",
    "OLLAMA_NUM_GPU": "0",
    "CUDA_VISIBLE_DEVICES": "",
    "LD_LIBRARY_PATH": "/tmp/ollama_stock_cpu_v0190/lib/ollama",
}

os.makedirs("/tmp/ollama-stock-cpu-home", exist_ok=True)
log = open("/tmp/ollama_stock_cpu_probe.log", "w")
p = subprocess.Popen(
    ["/tmp/ollama_stock_cpu_v0190/bin/ollama", "serve"],
    env=env,
    stdout=log,
    stderr=log,
    preexec_fn=os.setsid,
)

try:
    ready = False
    for _ in range(90):
        try:
            urllib.request.urlopen(BASE + "/api/tags", timeout=2).read()
            ready = True
            break
        except Exception:
            time.sleep(1)
    print("STOCK_CPU_READY", ready)
    if not ready:
        raise SystemExit(0)

    tests = [("tinyllama:latest", "2+2=?"), ("qwen2.5:0.5b", "2+2=?")]
    for model, prompt in tests:
        print("MODEL", model)
        for raw in (False, True):
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "raw": raw,
                "options": {
                    "temperature": 0,
                    "top_k": 1,
                    "top_p": 1,
                    "num_predict": 1,
                    "num_gpu": 0,
                },
            }
            req = urllib.request.Request(
                BASE + "/api/generate",
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
            )
            t = time.time()
            try:
                out = json.loads(urllib.request.urlopen(req, timeout=1800).read().decode())
                print(
                    " RAW",
                    raw,
                    "SEC",
                    round(time.time() - t, 2),
                    "LOAD",
                    round((out.get("load_duration") or 0) / 1e9, 3),
                    "RESP",
                    repr(out.get("response")),
                    "PEVAL",
                    out.get("prompt_eval_count"),
                    "EVAL",
                    out.get("eval_count"),
                )
            except Exception as e:
                print(" RAW", raw, "ERR", type(e).__name__, e, "SEC", round(time.time() - t, 2))
finally:
    try:
        os.killpg(os.getpgid(p.pid), signal.SIGTERM)
    except Exception:
        pass
    time.sleep(1)
