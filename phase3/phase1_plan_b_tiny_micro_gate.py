#!/usr/bin/env python3
"""
Plan B Tiny micro-gate.

Purpose:
- prove Tiny is still on the repaired GPU-backed path,
- distinguish deeper semantic/corruption failure from strict output-shape failure,
- avoid conflating stop-condition or token-budget mismatch with transport breakage.
"""

import argparse
import json
import re
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
            text = resp.read().decode("utf-8", errors="replace")
            return resp.getcode(), text, time.time() - t0
    except urllib.error.HTTPError as e:
        text = e.read().decode("utf-8", errors="replace")
        return e.code, text, time.time() - t0
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


def parse_response_json(raw_body: str) -> Dict[str, Any]:
    try:
        data = json.loads(raw_body)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


def model_in_ps(ps_body: str, model_name: str) -> bool:
    try:
        parsed = json.loads(ps_body)
    except Exception:
        return False
    models = parsed.get("models", [])
    for model in models:
        if model.get("model") == model_name:
            return True
    return False


def eval_arithmetic(text: str) -> Dict[str, Any]:
    semantic_pass = re.search(r"\b95\b", text) is not None
    strict_pass = re.fullmatch(r"\s*95\s*", text) is not None
    return {
        "semantic_pass": semantic_pass,
        "strict_pass": strict_pass,
    }


def eval_json_shape(text: str) -> Dict[str, Any]:
    decoder = json.JSONDecoder()
    trimmed = text.lstrip()
    semantic_pass = False
    strict_pass = False
    parsed_prefix = None
    trailing = None
    try:
        obj, idx = decoder.raw_decode(trimmed)
        parsed_prefix = obj
        trailing = trimmed[idx:].strip()
        if isinstance(obj, dict) and obj.get("ok") is True and obj.get("n") == 7:
            semantic_pass = True
            strict_pass = trailing == ""
    except Exception:
        semantic_pass = False
        strict_pass = False
    return {
        "semantic_pass": semantic_pass,
        "strict_pass": strict_pass,
        "parsed_prefix": parsed_prefix,
        "trailing_preview": (trailing or "")[:160],
    }


def summarize_case(case_id: str, raw_body: str, wall_sec: float) -> Dict[str, Any]:
    parsed = parse_response_json(raw_body)
    text = parse_response_text(raw_body)
    summary: Dict[str, Any] = {
        "id": case_id,
        "response_preview": text[:240],
        "wall_sec": round(wall_sec, 3),
        "done_reason": parsed.get("done_reason"),
        "load_duration_ns": parsed.get("load_duration"),
        "eval_duration_ns": parsed.get("eval_duration"),
    }
    if case_id == "arithmetic":
        summary.update(eval_arithmetic(text))
    elif case_id == "json":
        summary.update(eval_json_shape(text))
    return summary


def classify(report: Dict[str, Any]) -> Dict[str, Any]:
    arithmetic = report["cases"]["arithmetic"]
    json_case = report["cases"]["json"]

    any_semantic = arithmetic.get("semantic_pass") or json_case.get("semantic_pass")
    all_strict = arithmetic.get("strict_pass") and json_case.get("strict_pass")

    if all_strict:
        verdict = "pass"
        reason = "Tiny satisfies both semantic and strict output-shape checks."
    elif any_semantic:
        verdict = "strict_shape_failure"
        reason = (
            "At least one Tiny response is semantically correct or prefix-correct, "
            "but strict deterministic output-shape compliance still fails."
        )
    else:
        verdict = "semantic_failure"
        reason = "Tiny does not yet satisfy even the looser semantic checks."

    if arithmetic.get("done_reason") == "length" and not arithmetic.get("semantic_pass"):
        reason += " Arithmetic remains sensitive to the current token budget."

    return {
        "verdict": verdict,
        "reason": reason,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--model", default="tinyllama:latest")
    parser.add_argument("--timeout-sec", type=float, default=180.0)
    parser.add_argument(
        "--output",
        default="/tmp/phase1_plan_b_tiny_micro_gate_report.json",
    )
    args = parser.parse_args()

    gen_url = f"{args.base_url}/api/generate"
    ps_url = f"{args.base_url}/api/ps"

    report: Dict[str, Any] = {
        "lane": "Plan B",
        "model": args.model,
        "base_url": args.base_url,
        "started_at": int(time.time()),
        "preflight": {},
        "cases": {},
        "classification": {},
    }

    ps_code, ps_body, _ = api_get(ps_url, 20.0)
    report["preflight"]["ps_http_code"] = ps_code
    report["preflight"]["resident_before"] = model_in_ps(ps_body, args.model)
    report["preflight"]["ps_preview"] = ps_body[:240]

    if not report["preflight"]["resident_before"]:
        pin_payload = {
            "model": args.model,
            "prompt": "warm pin",
            "stream": False,
            "keep_alive": -1,
            "options": {"temperature": 0, "seed": 1, "num_predict": 2},
        }
        pin_code, pin_body, pin_wall = api_post(gen_url, pin_payload, args.timeout_sec)
        ps2_code, ps2_body, _ = api_get(ps_url, 20.0)
        report["preflight"]["warm_pin"] = {
            "http_code": pin_code,
            "wall_sec": round(pin_wall, 3),
            "resident_after": model_in_ps(ps2_body, args.model),
            "response_preview": parse_response_text(pin_body)[:160],
        }

    arithmetic_payload = {
        "model": args.model,
        "prompt": "What is 37 + 58? Reply with digits only.",
        "stream": False,
        "keep_alive": -1,
        "options": {"temperature": 0, "seed": 1234, "num_predict": 16},
    }
    code, body, wall = api_post(gen_url, arithmetic_payload, args.timeout_sec)
    report["cases"]["arithmetic"] = {
        "http_code": code,
        **summarize_case("arithmetic", body, wall),
    }

    json_payload = {
        "model": args.model,
        "prompt": 'Respond with JSON only: {"ok":true,"n":7}',
        "stream": False,
        "keep_alive": -1,
        "options": {"temperature": 0, "seed": 1234, "num_predict": 32},
    }
    code, body, wall = api_post(gen_url, json_payload, args.timeout_sec)
    report["cases"]["json"] = {
        "http_code": code,
        **summarize_case("json", body, wall),
    }

    ps3_code, ps3_body, _ = api_get(ps_url, 20.0)
    report["postflight"] = {
        "ps_http_code": ps3_code,
        "resident_after": model_in_ps(ps3_body, args.model),
        "ps_preview": ps3_body[:240],
    }

    report["classification"] = classify(report)
    report["finished_at"] = int(time.time())

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)
        f.write("\n")

    print(json.dumps(report, indent=2))
    return 0 if report["classification"]["verdict"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
