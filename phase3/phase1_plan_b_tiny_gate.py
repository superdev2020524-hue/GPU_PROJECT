#!/usr/bin/env python3
"""
Approved Phase 1 Plan B Tiny gate.

Binding cases:
- B1 cold residency pin
- B2 warm arithmetic structured JSON
- B3 warm JSON strict
- B4 force unload
"""

import argparse
import json
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Tuple


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


def parse_response_json(raw_body: str) -> Dict[str, Any]:
    try:
        data = json.loads(raw_body)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


def parse_response_text(raw_body: str) -> str:
    data = parse_response_json(raw_body)
    resp = data.get("response", "")
    if isinstance(resp, str):
        return resp
    return str(resp)


def model_in_ps(ps_body: str, model_name: str) -> bool:
    try:
        parsed = json.loads(ps_body)
    except Exception:
        return False
    for model in parsed.get("models", []):
        if model.get("model") == model_name:
            return True
    return False


def strict_json_ok(text: str) -> Tuple[bool, Dict[str, Any]]:
    decoder = json.JSONDecoder()
    trimmed = text.lstrip()
    try:
        obj, idx = decoder.raw_decode(trimmed)
        trailing = trimmed[idx:].strip()
    except Exception:
        return False, {"parsed": None, "trailing": None}
    ok = isinstance(obj, dict) and obj.get("ok") is True and obj.get("n") == 7 and trailing == ""
    return ok, {"parsed": obj, "trailing": trailing}


def strict_sum_json_ok(text: str) -> Tuple[bool, Dict[str, Any]]:
    decoder = json.JSONDecoder()
    trimmed = text.lstrip()
    try:
        obj, idx = decoder.raw_decode(trimmed)
        trailing = trimmed[idx:].strip()
    except Exception:
        return False, {"parsed": None, "trailing": None}
    ok = isinstance(obj, dict) and obj.get("sum") == 95 and trailing == ""
    return ok, {"parsed": obj, "trailing": trailing}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--model", default="tinyllama:latest")
    parser.add_argument("--timeout-sec", type=float, default=180.0)
    parser.add_argument("--output", default="/tmp/phase1_plan_b_tiny_gate_report.json")
    args = parser.parse_args()

    gen_url = f"{args.base_url}/api/generate"
    ps_url = f"{args.base_url}/api/ps"
    code_fence = "`" * 3

    report: Dict[str, Any] = {
        "lane": "Plan B",
        "gate_version": "approved-2026-04-07-revised-b2-json",
        "model": args.model,
        "base_url": args.base_url,
        "started_at": int(time.time()),
        "cases": {},
    }

    ps_code, ps_body, _ = api_get(ps_url, 20.0)
    report["preflight"] = {
        "ps_http_code": ps_code,
        "resident_before": model_in_ps(ps_body, args.model),
        "ps_preview": ps_body[:240],
    }

    if report["preflight"]["resident_before"]:
        unload_payload = {
            "model": args.model,
            "prompt": "",
            "stream": False,
            "keep_alive": 0,
            "options": {"temperature": 0, "seed": 1, "num_predict": 1},
        }
        code, body, wall = api_post(gen_url, unload_payload, args.timeout_sec)
        ps_code, ps_body, _ = api_get(ps_url, 20.0)
        report["preflight"]["forced_cold_reset"] = {
            "http_code": code,
            "wall_sec": round(wall, 3),
            "resident_after": model_in_ps(ps_body, args.model),
            "response_preview": parse_response_text(body)[:120],
        }

    # B1 cold residency pin
    b1_payload = {
        "model": args.model,
        "prompt": "warm pin",
        "stream": False,
        "keep_alive": -1,
        "options": {"temperature": 0, "seed": 1, "num_predict": 2},
    }
    code, body, wall = api_post(gen_url, b1_payload, args.timeout_sec)
    ps_code, ps_body, _ = api_get(ps_url, 20.0)
    b1_pass = code == 200 and ps_code == 200 and model_in_ps(ps_body, args.model)
    report["cases"]["B1_cold_residency_pin"] = {
        "http_code": code,
        "wall_sec": round(wall, 3),
        "response_preview": parse_response_text(body)[:120],
        "ps_http_code": ps_code,
        "resident_after": model_in_ps(ps_body, args.model),
        "pass": b1_pass,
    }

    # B2 warm arithmetic structured JSON
    b2_payload = {
        "model": args.model,
        "prompt": 'Compute 37 + 58. Respond with JSON only: {"sum":95}',
        "stream": False,
        "keep_alive": -1,
        "options": {
            "temperature": 0,
            "seed": 1234,
            "num_predict": 64,
            "stop": [code_fence, "\n\n", "In this example"],
        },
    }
    code, body, wall = api_post(gen_url, b2_payload, args.timeout_sec)
    text = parse_response_text(body)
    parsed = parse_response_json(body)
    strict_ok, sum_detail = strict_sum_json_ok(text)
    ps_code, ps_body, _ = api_get(ps_url, 20.0)
    b2_pass = (
        code == 200
        and strict_ok
        and ps_code == 200
        and model_in_ps(ps_body, args.model)
    )
    report["cases"]["B2_warm_arithmetic_strict"] = {
        "http_code": code,
        "wall_sec": round(wall, 3),
        "response_preview": text[:160],
        "done_reason": parsed.get("done_reason"),
        "load_duration_ns": parsed.get("load_duration"),
        "parsed_prefix": sum_detail["parsed"],
        "trailing_preview": (sum_detail["trailing"] or "")[:120] if sum_detail["trailing"] is not None else None,
        "ps_http_code": ps_code,
        "resident_after": model_in_ps(ps_body, args.model),
        "pass": b2_pass,
    }

    # B3 warm JSON strict
    b3_payload = {
        "model": args.model,
        "prompt": 'Respond with JSON only: {"ok":true,"n":7}',
        "stream": False,
        "keep_alive": -1,
        "options": {
            "temperature": 0,
            "seed": 1234,
            "num_predict": 64,
            "stop": [code_fence, "\n\n", "In this example"],
        },
    }
    code, body, wall = api_post(gen_url, b3_payload, args.timeout_sec)
    text = parse_response_text(body)
    parsed = parse_response_json(body)
    strict_ok, json_detail = strict_json_ok(text)
    ps_code, ps_body, _ = api_get(ps_url, 20.0)
    b3_pass = (
        code == 200
        and strict_ok
        and ps_code == 200
        and model_in_ps(ps_body, args.model)
    )
    report["cases"]["B3_warm_json_strict"] = {
        "http_code": code,
        "wall_sec": round(wall, 3),
        "response_preview": text[:200],
        "done_reason": parsed.get("done_reason"),
        "load_duration_ns": parsed.get("load_duration"),
        "parsed_prefix": json_detail["parsed"],
        "trailing_preview": (json_detail["trailing"] or "")[:120] if json_detail["trailing"] is not None else None,
        "ps_http_code": ps_code,
        "resident_after": model_in_ps(ps_body, args.model),
        "pass": b3_pass,
    }

    # B4 force unload
    b4_payload = {
        "model": args.model,
        "prompt": "",
        "stream": False,
        "keep_alive": 0,
        "options": {"temperature": 0, "seed": 1, "num_predict": 1},
    }
    code, body, wall = api_post(gen_url, b4_payload, args.timeout_sec)
    ps_code, ps_body, _ = api_get(ps_url, 20.0)
    b4_pass = code == 200 and ps_code == 200 and not model_in_ps(ps_body, args.model)
    report["cases"]["B4_force_unload"] = {
        "http_code": code,
        "wall_sec": round(wall, 3),
        "response_preview": parse_response_text(body)[:120],
        "ps_http_code": ps_code,
        "resident_after": model_in_ps(ps_body, args.model),
        "pass": b4_pass,
    }

    report["summary"] = {
        "B1_pass": b1_pass,
        "B2_pass": b2_pass,
        "B3_pass": b3_pass,
        "B4_pass": b4_pass,
    }
    report["overall_pass"] = all(report["summary"].values())
    report["finished_at"] = int(time.time())

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)
        f.write("\n")

    print(json.dumps(report, indent=2))
    return 0 if report["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
