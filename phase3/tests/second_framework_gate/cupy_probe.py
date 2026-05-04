#!/usr/bin/env python3
"""Milestone 05 CuPy second-framework probe."""

from __future__ import annotations

import json
import traceback
from typing import Any, Callable


def case(cases: dict[str, dict[str, Any]], name: str, fn: Callable[[], Any]) -> None:
    try:
        detail = fn()
        cases[name] = {"pass": True, "detail": detail}
    except Exception as exc:  # pragma: no cover - field diagnostics
        cases[name] = {
            "pass": False,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=8),
        }


def main() -> int:
    cases: dict[str, dict[str, Any]] = {}

    try:
        import cupy as cp
    except Exception as exc:
        report = {
            "overall_pass": False,
            "cases": {
                "import_cupy": {
                    "pass": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            },
        }
        print(json.dumps(report, indent=2, sort_keys=True))
        return 1

    case(cases, "import_cupy", lambda: {"version": cp.__version__})
    case(cases, "device_count", lambda: int(cp.cuda.runtime.getDeviceCount()))

    def device_name() -> str:
        props = cp.cuda.runtime.getDeviceProperties(0)
        name = props.get("name") if isinstance(props, dict) else getattr(props, "name", "")
        if isinstance(name, bytes):
            return name.decode("utf-8", errors="replace")
        return str(name)

    case(cases, "device_name", device_name)

    def transfer() -> list[list[float]]:
        host = [[float(i * 4 + j) for j in range(4)] for i in range(4)]
        arr = cp.asarray(host, dtype=cp.float32)
        got = cp.asnumpy(arr).tolist()
        if got != host:
            raise AssertionError({"expected": host, "got": got})
        return got

    case(cases, "tensor_htod_dtoh", transfer)

    def elementwise_add() -> list[list[float]]:
        a = cp.arange(16, dtype=cp.float32).reshape(4, 4)
        got = cp.asnumpy(a + cp.float32(2.0)).tolist()
        expected = [[float(i * 4 + j + 2) for j in range(4)] for i in range(4)]
        if got != expected:
            raise AssertionError({"expected": expected, "got": got})
        return got

    case(cases, "elementwise_add", elementwise_add)

    def matmul() -> list[list[float]]:
        a = cp.arange(16, dtype=cp.float32).reshape(4, 4)
        b_host = [[2.0 if i == j else 0.0 for j in range(4)] for i in range(4)]
        b = cp.asarray(b_host, dtype=cp.float32)
        got = cp.asnumpy(a @ b).tolist()
        expected = [[float((i * 4 + j) * 2) for j in range(4)] for i in range(4)]
        if got != expected:
            raise AssertionError({"expected": expected, "got": got})
        return got

    case(cases, "matmul", matmul)

    def repeated_warm_execution() -> list[list[list[float]]]:
        results = []
        for _ in range(5):
            a = cp.arange(16, dtype=cp.float32).reshape(4, 4)
            b = a + cp.float32(2.0)
            results.append(cp.asnumpy(b).tolist())
        expected = [[float(i * 4 + j + 2) for j in range(4)] for i in range(4)]
        if any(result != expected for result in results):
            raise AssertionError({"expected": expected, "got": results})
        return results

    case(cases, "repeated_warm_execution", repeated_warm_execution)

    overall = all(item.get("pass") for item in cases.values())
    report = {"overall_pass": overall, "cases": cases}
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
