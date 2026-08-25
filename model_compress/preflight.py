#!/usr/bin/env python3
"""Inspect a local Hugging Face checkpoint before an AWQ experiment."""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
from collections import Counter
from pathlib import Path


FLOAT_DTYPES = {"F64", "F32", "F16", "BF16"}
REDUCED_FLOAT_DTYPES = {"F8_E4M3", "F8_E5M2"}
INTEGER_DTYPES = {"I64", "I32", "I16", "I8", "U64", "U32", "U16", "U8"}
PACKED_SUFFIXES = (
    ".weight_packed",
    ".qweight",
    ".qzeros",
    ".scales",
)


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def safetensors_header(path: Path) -> dict:
    with path.open("rb") as handle:
        raw = handle.read(8)
        if len(raw) != 8:
            raise ValueError("missing safetensors header length")
        size = struct.unpack("<Q", raw)[0]
        if size > 256 * 1024 * 1024:
            raise ValueError(f"unreasonable safetensors header size: {size}")
        payload = handle.read(size)
    return json.loads(payload)


def inspect_local_model(model: Path) -> dict:
    config_path = model / "config.json"
    if not config_path.is_file():
        raise SystemExit(f"config.json not found under {model}")

    config = read_json(config_path)
    architectures = config.get("architectures") or []
    quant = config.get("quantization_config")
    declared_dtype = config.get("torch_dtype") or config.get("dtype")
    dtype_counts: Counter[str] = Counter()
    packed_names: list[str] = []
    tensor_count = 0

    shards = sorted(model.glob("*.safetensors"))
    for shard in shards:
        try:
            header = safetensors_header(shard)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise SystemExit(f"cannot inspect {shard.name}: {exc}") from exc
        for name, metadata in header.items():
            if name == "__metadata__":
                continue
            tensor_count += 1
            dtype_counts[str(metadata.get("dtype", "UNKNOWN"))] += 1
            if name.endswith(PACKED_SUFFIXES):
                packed_names.append(name)

    dtypes = set(dtype_counts)
    indicators: list[str] = []
    if quant:
        indicators.append("config.quantization_config")
    if packed_names:
        indicators.append("packed tensor names")
    if dtypes & REDUCED_FLOAT_DTYPES:
        indicators.append("FP8 tensors")
    if dtypes & INTEGER_DTYPES:
        indicators.append("integer tensors")

    compressed = bool(indicators)
    if not shards:
        classification = "metadata-only"
    elif compressed:
        classification = "compressed-or-quantized"
    elif dtypes and dtypes <= FLOAT_DTYPES:
        classification = "floating-point"
    else:
        classification = "unknown"

    return {
        "model_path": str(model.resolve()),
        "architectures": architectures,
        "model_type": config.get("model_type"),
        "declared_dtype": declared_dtype,
        "tensor_dtypes": dict(sorted(dtype_counts.items())),
        "tensor_count": tensor_count,
        "safetensors_shards": len(shards),
        "classification": classification,
        "requantization_indicators": indicators,
        "packed_tensor_examples": packed_names[:10],
        "has_remote_code": any(model.glob("modeling_*.py")),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="local Hugging Face snapshot")
    parser.add_argument("--allow-requantize", action="store_true")
    parser.add_argument("--json-output")
    args = parser.parse_args()

    model = Path(os.path.expanduser(args.model))
    if not model.is_dir():
        raise SystemExit(f"model directory not found: {model}")

    report = inspect_local_model(model)
    report["requantization_allowed"] = bool(args.allow_requantize)
    report["status"] = "inspect-only"

    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.json_output:
        output = Path(args.json_output)
        output.write_text(rendered + "\n", encoding="utf-8")

    if report["classification"] == "compressed-or-quantized":
        print(
            "\nSOURCE IS ALREADY COMPRESSED OR QUANTIZED. AWQ would be a "
            "requantization experiment.",
            file=sys.stderr,
        )
        if not args.allow_requantize:
            print(
                "Refusing by default. Re-run with --allow-requantize only after "
                "selecting a loader that materializes ordinary floating tensors.",
                file=sys.stderr,
            )
            return 3
        print(
            "Planning is allowed, but this does not validate a materialization "
            "adapter or authorize a full run.",
            file=sys.stderr,
        )
    elif report["classification"] in {"unknown", "metadata-only"}:
        print(
            "\nSource format is not fully established; a recipe must resolve it "
            "before quantization.",
            file=sys.stderr,
        )
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
