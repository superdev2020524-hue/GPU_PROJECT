#!/usr/bin/env python3
"""Run two CuPy framework probes concurrently for Milestone 06."""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any


PYTHON = Path("/mnt/m04-pytorch/venv/bin/python")
PROBE = Path("/mnt/m04-pytorch/cupy_probe.py")
OUT_DIR = Path("/tmp/m06_two_process_cupy")
CHILD_TIMEOUT_SEC = 300


def parse_report(text: str) -> dict[str, Any] | None:
    start = text.find("{")
    if start < 0:
        return None
    try:
        return json.loads(text[start:])
    except json.JSONDecodeError:
        return None


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = "/opt/vgpu/lib:" + env.get("LD_LIBRARY_PATH", "")

    children: list[dict[str, Any]] = []
    for idx in range(2):
        out_path = OUT_DIR / f"child_{idx + 1}.json"
        out_file = out_path.open("w")
        started = time.time()
        proc = subprocess.Popen(
            ["timeout", f"{CHILD_TIMEOUT_SEC}s", str(PYTHON), str(PROBE)],
            stdout=out_file,
            stderr=subprocess.STDOUT,
            env=env,
        )
        children.append(
            {
                "index": idx + 1,
                "proc": proc,
                "out_file": out_file,
                "out_path": out_path,
                "started": started,
            }
        )

    reports: list[dict[str, Any]] = []
    for child in children:
        proc: subprocess.Popen[bytes] = child["proc"]
        rc = proc.wait(timeout=CHILD_TIMEOUT_SEC + 15)
        elapsed = time.time() - child["started"]
        child["out_file"].close()
        text = child["out_path"].read_text(errors="replace")
        parsed = parse_report(text)
        ok = rc == 0 and bool(parsed and parsed.get("overall_pass") is True)
        reports.append(
            {
                "index": child["index"],
                "returncode": rc,
                "elapsed_sec": round(elapsed, 3),
                "output_path": str(child["out_path"]),
                "json_ok": parsed is not None,
                "overall_pass": bool(parsed and parsed.get("overall_pass") is True),
                "pass": ok,
                "cases": parsed.get("cases", {}) if parsed else {},
                "tail": text[-1000:],
            }
        )

    summary = {
        "overall_pass": all(item["pass"] for item in reports),
        "children": reports,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
