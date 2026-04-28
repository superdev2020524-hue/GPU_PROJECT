#!/usr/bin/env python3
"""Bounded Milestone 04 PyTorch probe."""

import json
import sys
import traceback


def main() -> int:
    result = {
        "overall_pass": False,
        "cases": {},
        "error": None,
    }

    try:
        import torch

        result["torch"] = torch.__version__
        result["torch_cuda"] = torch.version.cuda

        def record(name, ok, detail=None):
            result["cases"][name] = {"pass": bool(ok), "detail": detail}
            if not ok:
                raise RuntimeError(f"{name} failed: {detail}")

        record("cuda_available", torch.cuda.is_available())
        record("device_count", torch.cuda.device_count() >= 1, torch.cuda.device_count())
        result["device_name"] = torch.cuda.get_device_name(0)

        device = torch.device("cuda:0")

        cpu = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        gpu = cpu.to(device)
        roundtrip = gpu.cpu()
        record("tensor_htod_dtoh", torch.equal(cpu, roundtrip), roundtrip.tolist())

        elem = (gpu + 2.0).cpu()
        record("elementwise_add", torch.equal(elem, cpu + 2.0), elem.tolist())

        a = torch.arange(16, dtype=torch.float32).reshape(4, 4).to(device)
        b = (torch.eye(4, dtype=torch.float32) * 2.0).to(device)
        matmul = (a @ b).cpu()
        record("matmul", torch.equal(matmul, (torch.arange(16, dtype=torch.float32).reshape(4, 4) * 2.0)), matmul.tolist())

        torch.manual_seed(7)
        model = torch.nn.Sequential(
            torch.nn.Linear(4, 8),
            torch.nn.Linear(8, 2),
        ).to(device)
        x = torch.ones(3, 4, dtype=torch.float32).to(device)
        y = model(x).detach().cpu()
        finite = bool(torch.isfinite(y).all().item()) and y.shape == (3, 2)
        record("small_nn_inference", finite, {"shape": list(y.shape), "sum": float(y.sum().item())})

        warm = []
        for i in range(5):
            z = (a @ b).detach().cpu()
            warm.append(z)
        expected = [torch.arange(16, dtype=torch.float32).reshape(4, 4) * 2.0 for _ in range(5)]
        record(
            "repeated_warm_execution",
            all(torch.equal(x, y) for x, y in zip(warm, expected)),
            {"got": [x.tolist() for x in warm], "expected": [y.tolist() for y in expected]},
        )

        torch.cuda.synchronize()
        result["overall_pass"] = all(case["pass"] for case in result["cases"].values())
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback_tail"] = traceback.format_exc().splitlines()[-20:]

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
