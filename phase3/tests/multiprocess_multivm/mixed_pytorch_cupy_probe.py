#!/usr/bin/env python3
"""Run PyTorch and CuPy framework probes concurrently for Milestone 06."""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any


PYTHON = Path("/mnt/m04-pytorch/venv/bin/python")
PROBES = [
    ("pytorch", Path("/mnt/m04-pytorch/pytorch_probe.py"), 420),
    ("cupy", Path("/mnt/m04-pytorch/cupy_probe.py"), 300),
]
OUT_DIR = Path("/tmp/m06_mixed_pytorch_cupy")


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
    for name, script, timeout_sec in PROBES:
        out_path = OUT_DIR / f"{name}.json"
        out_file = out_path.open("w")
        proc = subprocess.Popen(
            ["timeout", f"{timeout_sec}s", str(PYTHON), str(script)],
            stdout=out_file,
            stderr=subprocess.STDOUT,
            env=env,
        )
        children.append(
            {
                "name": name,
                "proc": proc,
                "out_file": out_file,
                "out_path": out_path,
                "started": time.time(),
                "timeout_sec": timeout_sec,
            }
        )

    reports: list[dict[str, Any]] = []
    for child in children:
        proc: subprocess.Popen[bytes] = child["proc"]
        rc = proc.wait(timeout=child["timeout_sec"] + 30)
        elapsed = time.time() - child["started"]
        child["out_file"].close()
        text = child["out_path"].read_text(errors="replace")
        parsed = parse_report(text)
        ok = rc == 0 and bool(parsed and parsed.get("overall_pass") is True)
        reports.append(
            {
                "name": child["name"],
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
