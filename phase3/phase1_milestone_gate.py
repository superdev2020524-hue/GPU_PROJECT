#!/usr/bin/env python3
"""
Phase 1 pre-integration gate runner.

Runs deterministic accuracy/speed/residency checks against Ollama API before
integration-level debugging. Intended to run on the VM or any host that can
reach the VM API endpoint.
"""

import argparse
import json
import multiprocessing
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Tuple


def _request_worker(method: str, url: str, payload: Dict[str, Any], timeout_sec: float, queue: multiprocessing.Queue) -> None:
    body = None
    headers = {}
    if method == "POST":
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            text = resp.read().decode("utf-8", errors="replace")
            queue.put((resp.getcode(), text))
    except urllib.error.HTTPError as e:
        text = e.read().decode("utf-8", errors="replace")
        queue.put((e.code, text))
    except urllib.error.URLError as e:
        queue.put((0, f"URLERROR: {e}"))
    except TimeoutError:
        queue.put((0, "SOCKET_TIMEOUT"))
    except Exception as e:  # pragma: no cover - defensive capture for field debugging
        queue.put((0, f"EXCEPTION: {e}"))


def run_request(method: str, url: str, payload: Dict[str, Any], timeout_sec: float) -> Tuple[int, str, float]:
    t0 = time.time()
    queue: multiprocessing.Queue = multiprocessing.Queue()
    proc = multiprocessing.Process(
        target=_request_worker,
        args=(method, url, payload, timeout_sec, queue),
    )
    proc.start()
    proc.join(timeout_sec)
    if proc.is_alive():
        proc.terminate()
        proc.join(5)
        return 0, "WALL_TIMEOUT", time.time() - t0
    if queue.empty():
        return 0, "NO_RESULT", time.time() - t0
    code, body = queue.get()
    return code, body, time.time() - t0


def api_post(url: str, payload: Dict[str, Any], timeout_sec: float) -> Tuple[int, str, float]:
    return run_request("POST", url, payload, timeout_sec)


def api_get(url: str, timeout_sec: float) -> Tuple[int, str]:
    code, body, _ = run_request("GET", url, {}, timeout_sec)
    return code, body


def model_in_ps(ps_json: Dict[str, Any], model_name: str) -> bool:
    models = ps_json.get("models", [])
    for m in models:
        name = m.get("model", "")
        if name == model_name:
            return True
    return False


def parse_response_text(raw_body: str) -> str:
    try:
        data = json.loads(raw_body)
        resp = data.get("response", "")
        if isinstance(resp, str):
            return resp.strip()
        return str(resp).strip()
    except Exception:
        return raw_body.strip()


def eval_expect(expect: Dict[str, Any], actual_text: str) -> Tuple[bool, str]:
    etype = expect.get("type")
    if etype == "contains":
        wanted = str(expect.get("value", ""))
        ok = wanted in actual_text
        return ok, f"contains('{wanted}')"
    if etype == "regex":
        pat = str(expect.get("value", ""))
        ok = re.search(pat, actual_text) is not None
        return ok, f"regex('{pat}')"
    if etype == "json_contains":
        wanted = expect.get("value", {})
        try:
            parsed = json.loads(actual_text)
        except Exception:
            return False, "json_contains(parse_failed)"
        for k, v in wanted.items():
            if parsed.get(k) != v:
                return False, f"json_contains(mismatch:{k})"
        return True, "json_contains"
    return False, f"unsupported_expect_type:{etype}"


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suite",
        default=str(script_dir / "phase1_milestone_test_suite.json"),
        help="Path to JSON suite file.",
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:11434",
        help="Ollama base URL.",
    )
    parser.add_argument(
        "--timeout-sec",
        type=float,
        default=600.0,
        help="Per-request timeout in seconds.",
    )
    parser.add_argument(
        "--output",
        default="/tmp/phase1_milestone_gate_report.json",
        help="Output report JSON path.",
    )
    args = parser.parse_args()

    with open(args.suite, "r", encoding="utf-8") as f:
        suite = json.load(f)

    model = suite.get("model", "tinyllama:latest")
    gen_url = f"{args.base_url}/api/generate"
    ps_url = f"{args.base_url}/api/ps"

    report: Dict[str, Any] = {
        "suite_version": suite.get("version", "unknown"),
        "model": model,
        "base_url": args.base_url,
        "started_at": int(time.time()),
        "accuracy": [],
        "speed": {},
        "residency": {},
        "overall_pass": False,
    }

    accuracy_pass = True
    for case in suite.get("accuracy_cases", []):
        print(f"[accuracy] start {case['id']}", flush=True)
        payload = dict(case["request"])
        payload["model"] = model
        code, body, wall = api_post(gen_url, payload, args.timeout_sec)
        text = parse_response_text(body)
        ok_http = code == 200
        ok_expect, expect_desc = eval_expect(case["expect"], text)
        case_ok = ok_http and ok_expect
        if not case_ok:
            accuracy_pass = False
        report["accuracy"].append(
            {
                "id": case["id"],
                "http_code": code,
                "wall_sec": round(wall, 3),
                "expect": expect_desc,
                "pass": case_ok,
                "response_preview": text[:220],
            }
        )
        print(f"[accuracy] done {case['id']} http={code} wall={wall:.3f}s pass={case_ok}", flush=True)

    speed = suite.get("speed_cases", {})
    speed_pass = True
    for key in ("cold", "warm"):
        print(f"[speed] start {key}", flush=True)
        s = speed.get(key, {})
        payload = dict(s.get("request", {}))
        payload["model"] = model
        code, body, wall = api_post(gen_url, payload, args.timeout_sec)
        max_wall = float(s.get("max_wall_sec", 999999.0))
        case_ok = code == 200 and wall <= max_wall
        if not case_ok:
            speed_pass = False
        load_duration_ns = None
        try:
            parsed = json.loads(body)
            load_duration_ns = parsed.get("load_duration")
        except Exception:
            pass
        report["speed"][key] = {
            "http_code": code,
            "wall_sec": round(wall, 3),
            "max_wall_sec": max_wall,
            "load_duration_ns": load_duration_ns,
            "pass": case_ok,
        }
        print(f"[speed] done {key} http={code} wall={wall:.3f}s pass={case_ok}", flush=True)

    residency_pass = True
    res = suite.get("residency_cases", {})

    keep_case = res.get("keep_loaded", {})
    print("[residency] start keep_loaded", flush=True)
    keep_payload = dict(keep_case.get("request", {}))
    keep_payload["model"] = model
    keep_code, keep_body, keep_wall = api_post(gen_url, keep_payload, args.timeout_sec)
    ps_code, ps_body = api_get(ps_url, args.timeout_sec)
    keep_seen = False
    try:
        keep_seen = model_in_ps(json.loads(ps_body), model)
    except Exception:
        keep_seen = False
    keep_expect = bool(keep_case.get("expect_model_in_ps", True))
    keep_ok = keep_code == 200 and ps_code == 200 and keep_seen == keep_expect
    if not keep_ok:
        residency_pass = False

    report["residency"]["keep_loaded"] = {
        "http_code": keep_code,
        "wall_sec": round(keep_wall, 3),
        "ps_http_code": ps_code,
        "model_seen_in_ps": keep_seen,
        "expected_model_in_ps": keep_expect,
        "pass": keep_ok,
    }
    print(f"[residency] done keep_loaded http={keep_code} wall={keep_wall:.3f}s pass={keep_ok}", flush=True)

    unload_case = res.get("force_unload", {})
    print("[residency] start force_unload", flush=True)
    unload_payload = dict(unload_case.get("request", {}))
    unload_payload["model"] = model
    unload_code, unload_body, unload_wall = api_post(gen_url, unload_payload, args.timeout_sec)
    ps2_code, ps2_body = api_get(ps_url, args.timeout_sec)
    unload_seen = True
    try:
        unload_seen = model_in_ps(json.loads(ps2_body), model)
    except Exception:
        unload_seen = True
    unload_expect = bool(unload_case.get("expect_model_in_ps", False))
    unload_ok = unload_code == 200 and ps2_code == 200 and unload_seen == unload_expect
    if not unload_ok:
        residency_pass = False

    report["residency"]["force_unload"] = {
        "http_code": unload_code,
        "wall_sec": round(unload_wall, 3),
        "ps_http_code": ps2_code,
        "model_seen_in_ps": unload_seen,
        "expected_model_in_ps": unload_expect,
        "pass": unload_ok,
    }
    print(f"[residency] done force_unload http={unload_code} wall={unload_wall:.3f}s pass={unload_ok}", flush=True)

    report["summary"] = {
        "accuracy_pass": accuracy_pass,
        "speed_pass": speed_pass,
        "residency_pass": residency_pass,
    }
    report["overall_pass"] = accuracy_pass and speed_pass and residency_pass
    report["finished_at"] = int(time.time())

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)
        f.write("\n")

    print(json.dumps(report["summary"], indent=2))
    print(f"overall_pass={report['overall_pass']}")
    print(f"report={args.output}")

    return 0 if report["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
