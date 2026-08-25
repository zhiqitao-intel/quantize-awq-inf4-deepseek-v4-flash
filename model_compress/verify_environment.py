#!/usr/bin/env python3
"""Verify that compressor installation did not replace Intel XPU Torch."""

from __future__ import annotations

import argparse
import json
import os

import torch


def identity() -> dict:
    xpu = getattr(torch, "xpu", None)
    available = bool(xpu and xpu.is_available())
    return {
        "torch_version": torch.__version__,
        "torch_path": os.path.realpath(torch.__file__),
        "torch_cuda_version": torch.version.cuda,
        "xpu_api_present": xpu is not None,
        "xpu_available": available,
        "xpu_device_count": xpu.device_count() if available else 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-xpu", action="store_true")
    parser.add_argument("--expect-version")
    parser.add_argument("--expect-path")
    args = parser.parse_args()
    data = identity()
    print(json.dumps(data, indent=2, sort_keys=True))

    failures = []
    if args.expect_version and data["torch_version"] != args.expect_version:
        failures.append("Torch version changed")
    if args.expect_path and data["torch_path"] != os.path.realpath(args.expect_path):
        failures.append("Torch import path changed")
    if not data["xpu_api_present"]:
        failures.append("torch.xpu is absent")
    if args.require_xpu and not data["xpu_available"]:
        failures.append("no Intel XPU is available")
    if failures:
        raise SystemExit("environment verification failed: " + "; ".join(failures))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
