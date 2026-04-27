#!/usr/bin/env python3
"""
Phase 1 Plan C client-facing gate.

This gate validates the real one-shot `ollama run` path for a dedicated client
model. It force-cleans resident models before and after the gate so Plan A and
Plan B can be verified serially on the same service without cross-model
contamination.
"""

import argparse
import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Tuple


def run_request(method: str, url: str, payload: Dict[str, Any], timeout_sec: float) -> Tuple[int, str, float]:
    t0 = time.time()
    body = None
    headers = {}
    if method == "POST":
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            return resp.getcode(), resp.read().decode("utf-8", errors="replace"), time.time() - t0
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", errors="replace"), time.time() - t0
    except Exception as e:
        return 0, f"{type(e).__name__}: {e}", time.time() - t0


def api_get(url: str, timeout_sec: float) -> Tuple[int, str, float]:
    return run_request("GET", url, {}, timeout_sec)


def api_post(url: str, payload: Dict[str, Any], timeout_sec: float) -> Tuple[int, str, float]:
    return run_request("POST", url, payload, timeout_sec)


def parse_response_text(raw_body: str) -> str:
    try:
        data = json.loads(raw_body)
        resp = data.get("response", "")
        if isinstance(resp, str):
            return resp.strip()
        return str(resp).strip()
    except Exception:
        return raw_body.strip()


def parse_ps_models(raw_body: str) -> List[str]:
    try:
        parsed = json.loads(raw_body)
    except Exception:
        return []
    models = []
    for item in parsed.get("models", []):
        name = item.get("model")
        if isinstance(name, str) and name:
            models.append(name)
    return models


def model_in_ps(ps_body: str, model_name: str) -> bool:
    return model_name in parse_ps_models(ps_body)


def unload_model(gen_url: str, model_name: str, timeout_sec: float) -> Dict[str, Any]:
    payload = {
        "model": model_name,
        "prompt": "",
        "stream": False,
        "keep_alive": 0,
    }
    code, body, wall = api_post(gen_url, payload, timeout_sec)
    return {
        "model": model_name,
        "http_code": code,
        "wall_sec": round(wall, 3),
        "response_preview": parse_response_text(body)[:120],
    }


def run_cli(model_name: str, prompt: str, base_url: str, timeout_sec: float) -> Dict[str, Any]:
    env = os.environ.copy()
    env["OLLAMA_HOST"] = base_url
    env["TERM"] = "dumb"
    t0 = time.time()
    try:
        proc = subprocess.run(
            ["ollama", "run", model_name, prompt],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            env=env,
            check=False,
        )
        return {
            "rc": proc.returncode,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
            "wall_sec": round(time.time() - t0, 3),
        }
    except subprocess.TimeoutExpired as e:
        return {
            "rc": 124,
            "stdout": (e.stdout or "").strip(),
            "stderr": (e.stderr or "").strip(),
            "wall_sec": round(time.time() - t0, 3),
        }


def wait_for_model_absent(ps_url: str, model_name: str, attempts: int = 6, sleep_sec: float = 2.0) -> Tuple[int, str, bool]:
    last_code = 0
    last_body = ""
    for idx in range(attempts):
        last_code, last_body, _ = api_get(ps_url, 20.0)
        if last_code == 200 and not model_in_ps(last_body, model_name):
            return last_code, last_body, True
        if idx + 1 < attempts:
            time.sleep(sleep_sec)
    return last_code, last_body, False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--model", default="qwen2.5:3b")
    parser.add_argument("--timeout-sec", type=float, default=300.0)
    parser.add_argument("--output", default="/tmp/phase1_plan_c_client_gate_report.json")
    args = parser.parse_args()

    gen_url = f"{args.base_url}/api/generate"
    ps_url = f"{args.base_url}/api/ps"

    report: Dict[str, Any] = {
        "lane": "Plan C",
        "gate_version": "approved-2026-04-07-client-facing-qwen3b",
        "model": args.model,
        "base_url": args.base_url,
        "verification_mode": "ollama_run_cli",
        "started_at": int(time.time()),
        "cases": {},
    }

    ps_code, ps_body, _ = api_get(ps_url, 20.0)
    resident_before = parse_ps_models(ps_body) if ps_code == 200 else []
    report["preflight"] = {
        "ps_http_code": ps_code,
        "resident_before": resident_before,
    }

    forced_unloads = []
    for model_name in resident_before:
        forced_unloads.append(unload_model(gen_url, model_name, args.timeout_sec))
    if forced_unloads:
        ps_code, ps_body, _ = api_get(ps_url, 20.0)
    report["preflight"]["forced_unloads"] = forced_unloads
    report["preflight"]["resident_after_clean"] = parse_ps_models(ps_body) if ps_code == 200 else []

    cases = [
        ("C1_small_arithmetic_cli_style", "What is 4 + 8? Reply with digits only.", "12"),
        ("C2_large_arithmetic_cli_style", "What is 444 + 8? Reply with digits only.", "452"),
        ("C3_second_large_arithmetic_cli_style", "What is 444 + 18? Reply with digits only.", "462"),
        ("C4_reference_arithmetic_cli_style", "What is 37 + 58? Reply with digits only.", "95"),
    ]

    summary: Dict[str, bool] = {}
    for case_id, prompt, expected in cases:
        print(f"[plan-c] start {case_id}", flush=True)
        cli = run_cli(args.model, prompt, args.base_url, args.timeout_sec)
        ps_code, ps_body, _ = api_get(ps_url, 20.0)
        resident_after = model_in_ps(ps_body, args.model)
        case_pass = cli["rc"] == 0 and cli["stdout"] == expected and ps_code == 200 and resident_after
        summary[case_id] = case_pass
        report["cases"][case_id] = {
            "prompt": prompt,
            "expected": expected,
            "rc": cli["rc"],
            "wall_sec": cli["wall_sec"],
            "stdout_preview": cli["stdout"][:120],
            "stderr_preview": cli["stderr"][:240],
            "ps_http_code": ps_code,
            "resident_after": resident_after,
            "pass": case_pass,
        }
        print(f"[plan-c] done {case_id} rc={cli['rc']} wall={cli['wall_sec']:.3f}s pass={case_pass}", flush=True)

    print("[plan-c] start C5_force_unload", flush=True)
    cleanup = unload_model(gen_url, args.model, args.timeout_sec)
    ps_code, ps_body, absent = wait_for_model_absent(ps_url, args.model)
    c5_pass = cleanup["http_code"] == 200 and ps_code == 200 and absent
    summary["C5_force_unload"] = c5_pass
    report["cases"]["C5_force_unload"] = {
        **cleanup,
        "ps_http_code": ps_code,
        "resident_after": model_in_ps(ps_body, args.model),
        "pass": c5_pass,
    }
    print(f"[plan-c] done C5_force_unload http={cleanup['http_code']} wall={cleanup['wall_sec']:.3f}s pass={c5_pass}", flush=True)

    report["summary"] = summary
    report["overall_pass"] = all(summary.values())
    report["finished_at"] = int(time.time())

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)
        f.write("\n")

    print(json.dumps(report, indent=2))
    return 0 if report["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
