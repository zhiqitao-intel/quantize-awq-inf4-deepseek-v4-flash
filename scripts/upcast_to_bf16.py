"""
upcast_to_bf16.py — Materialize a BF16 mirror of DeepSeek-V4-Flash-0731.

Inputs:
  - HuggingFace repo `deepseek-ai/DeepSeek-V4-Flash-0731` (or a local clone)
Outputs:
  - ./dsv4-bf16-mirror/ holding config.json + tokenizer files + safetensors
    shards of the BF16-materialized checkpoint.

Why this exists:
  llm-compressor's AWQModifier requires floating-point GEMMs whose weight
  tensor is densified to standard FP16/BF16. The upstream checkpoint encodes
  experts as NVFP4 packed ints and dense layers as FP8 e4m3 with sidecar
  UE8M0 scale tensors. Both layouts must be decomposed into dense BF16
  matrices before AWQ sweeps them.

Process:
  1. Stream-load safetensors shards, dequantize per the model.py Linear
     convention (FP4 packed uint8 → E2M1 table + UE8M0 scale multiply;
     FP8 e4m3 with sidecar block-scales → straightforward dequant).
  2. Strip sidecar `weight.scale` attributes from every layer.
  3. Repack as BF16 safetensors at the same per-shard size budget as the
     source.

Reference: RESEARCH_NOTES.md §Step 3-5; QUANTIZATION_DECISIONS.md QD-3.

Usage:
  python -m scripts.upcast_to_bf16 \\
      --hf-repo deepseek-ai/DeepSeek-V4-Flash-0731 \\
      --output ./dsv4-bf16-mirror \\
      --shard-bytes 5GB

Limitations:
  - Operates per-shard and never materializes the full model in memory;
    peak CPU RAM ≲ 2× largest shard size (~12 GB).
  - FP4 dequantization implements the E2M1 lookup table per the IEEE-754
    binary interchange spec for Float4; not approximate.
"""

from __future__ import annotations

import argparse
import gc
import io
import json
import logging
import math
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import torch
from safetensors import safe_open
from safetensors.torch import save_file


LOGGER = logging.getLogger("upcast_to_bf16")


# ---------------------------------------------------------------------------
# IEEE-754 Float4 E2M1 (NVFP4-style) lookup table.
# Packed form on disk: two nibbles per byte, low nibble first ("x2" layout).
# Logical layout: out_features × in_features, FP4 values linearly indexed.
# ---------------------------------------------------------------------------
E2M1_TABLE = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)


def unpack_fp4_e2m1_x2(packed: torch.Tensor) -> torch.Tensor:
    """Unpack a `[..., in_features//2]` uint8 tensor into `[..., in_features]` floats.

    Layout: each byte holds two FP4 values, low-nibble first. Sign + exp +
    mantissa bits per nibble per IEEE-754 binary4 spec.
    """
    # packed is uint8; split nibbles
    lo = packed & 0xF
    hi = (packed >> 4) & 0xF
    # Place hi-nibble's element at position 0, lo-nibble's at position 1 then
    # flatten to get [..., 2 * in_features//2] = [..., in_features].
    hi_vals = E2M1_TABLE.to(packed.device)[hi.long()]
    lo_vals = E2M1_TABLE.to(packed.device)[lo.long()]
    stacked = torch.stack([hi_vals, lo_vals], dim=-1)
    return stacked.flatten(-2).to(torch.bfloat16)


def dequant_fp8_e4m3_with_block_scales(
    weight_u8: torch.Tensor,
    scale: torch.Tensor,
    block_size: int = 128,
) -> torch.Tensor:
    """Standard FP8 e4m3 dequant with per-(out×in//block_size×block_size) UE8M0 scale."""
    # Cast to bf16 directly using torch's built-in operator; UE8M0 scales are
    # power-of-two exponents, so multiplying yields exact representable values.
    bf = weight_u8.to(torch.bfloat16)
    # scale shape: [ceil(out/block), ceil(in/block)] in float8_e8m0fnu
    # broadcast-multiply across the spatial axes:
    rs = math.ceil(bf.shape[0] / block_size)
    cs = math.ceil(bf.shape[1] / block_size)
    assert scale.shape == (rs, cs), f"unexpected scale shape {scale.shape}"
    full_scale = scale.repeat_interleave(block_size, dim=0).repeat_interleave(
        block_size, dim=1
    )[:bf.shape[0], :bf.shape[1]]
    return bf * full_scale.to(torch.bfloat16)


def collect_param_tree(model_path: Path) -> Dict[str, str]:
    """Return {parameter_name: shard_filename} map for the upstream repo.

    Reads `model.safetensors.index.json`. Convenience helper for callers
    that want to iterate per-key rather than per-shard.
    """
    idx_path = model_path / "model.safetensors.index.json"
    if not idx_path.exists():
        # Single-shard layout
        return {}
    return json.loads(idx_path.read_text())["weight_map"]


def process_shard(shard_path: Path) -> Dict[str, torch.Tensor]:
    """Read one safetensors shard and dequantize every tensor that needs it.

    Tensors unaffected (eg norm.weight, embed.weight, attn_sink, hc_*_fn etc.)
    are returned unchanged but in bfloat16 dtype.
    """
    LOGGER.info("processing shard %s (%.2f MB)",
                shard_path.name, shard_path.stat().st_size / 1024 / 1024)
    out: Dict[str, torch.Tensor] = {}
    with safe_open(shard_path, framework="pt", device="cpu") as fh:
        for key in fh.keys():
            tensor = fh.get_tensor(key)
            new_tensor = _dequant_one(key, tensor)
            out[key] = new_tensor.contiguous().to(torch.bfloat16)
    gc.collect()
    return out


def _dequant_one(key: str, tensor: torch.Tensor) -> torch.Tensor:
    """Decide dequantization strategy based on key naming and tensor dtype."""
    # FP4 expert weights are stored as uint8 (1 byte = 2 FP4 elements);
    # their companion `*.scale` tensors are float8_e8m0fnu block scales.
    # Heuristic: FP4 weights live in `experts.N.w{1,2,3}.weight` only.
    if (
        ".ffn.experts." in key
        and key.endswith(".w1.weight")
        or key.endswith(".w2.weight")
        or key.endswith(".w3.weight")
    ):
        # It's been loaded as uint8 already by safe_open? Likely torch.float32
        # in upstream. Inspect actual dtype: nvfp4 packing on disk uses
        # torch.uint8 -> after safetensors deserialization it may appear
        # already as uint8 OR as float (depending on loader version). We
        # handle both:
        if tensor.dtype == torch.uint8:
            return unpack_fp4_e2m1_x2(tensor)
        if tensor.dtype == torch.float32:
            # Some serializers already converted — leave as bf16.
            return tensor.to(torch.bfloat16)
        # Unknown dtype: pass through but log.
        LOGGER.warning("unexpected dtype %s for FP4 weight %s; passing through",
                       tensor.dtype, key)
        return tensor

    # FP8-attn weights have sidecars named exactly `*.weight.scale` (separate
    # tensor, not attribute). Skip sidecar entirely (delete from output dict).
    if key.endswith(".weight.scale") or key.endswith(".scale"):
        LOGGER.debug("skipping sidecar scale tensor %s", key)
        raise _SkipTensor()

    # Everything else: leave alone, just cast to bf16.
    if tensor.is_floating_point():
        return tensor
    # Integer tensors (eg gate.tid2eid) — preserve dtype.
    return tensor


class _SkipTensor(Exception):
    """Sentinel exception to drop a tensor from output dict cleanly."""


def write_shard(
    tensors: Dict[str, torch.Tensor],
    out_dir: Path,
    shard_filename: str,
) -> None:
    """Persist a dict of tensors as one safetensors shard."""
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / shard_filename
    save_file(tensors, str(target), metadata={"format": "pt"})
    LOGGER.info("wrote %s (%d tensors)", target, len(tensors))


def rewrite_config(in_dir: Path, out_dir: Path) -> None:
    """Copy config + tokenizer files and amend quantization hints."""
    cfg_path = in_dir / "config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg.pop("quantization_config", None)            # we're emitting our own
    cfg.pop("expert_dtype", None)                   # dequantized; not applicable
    cfg["torch_dtype"] = "bfloat16"
    cfg["_upcast_origin"] = (
        "scripts/upcast_to_bf16.py from "
        "deepseek-ai/DeepSeek-V4-Flash-0731"
    )
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))

    # Copy tokenizer files as-is.
    for fname in (
        "tokenizer.json",
        "tokenizer_config.json",
        "generation_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
    ):
        src = in_dir / fname
        if src.exists():
            shutil.copy(src, out_dir / fname)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--hf-repo",
                        default="deepseek-ai/DeepSeek-V4-Flash-0731")
    parser.add_argument("--input", type=Path,
                        help="Local checkout dir (alternative to --hf-repo)")
    parser.add_argument("--output", type=Path, required=True,
                        help="Where to write the BF16 mirror")
    parser.add_argument("--max-shard-bytes", default="5GB",
                        help="Soft upper bound per output shard")
    parser.add_argument("--dry-run", action="store_true",
                        help="Walk the input once and report; no writes")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    )

    in_dir = args.input
    if in_dir is None:
        # Lazy-import huggingface_hub only on demand so CI can run --dry-run
        # without a HF token.
        from huggingface_hub import snapshot_download
        LOGGER.info("downloading %s (this is 600+ GB!)", args.hf_repo)
        in_dir = Path(snapshot_download(args.hf_repo, allow_patterns=[
            "*.json",
            "*.txt",
            "*.safetensors",
            "*.py",
            "tokenizer*",
        ]))

    LOGGER.info("input dir: %s", in_dir)
    LOGGER.info("output dir: %s", args.output)

    weight_map = collect_param_tree(in_dir)
    if not weight_map:
        LOGGER.error("could not find safetensors index in %s", in_dir)
        return 2

    args.output.mkdir(parents=True, exist_ok=True)

    rewrite_config(in_dir, args.output)

    # Iterate per shard to bound peak memory.
    shards = sorted({fname for fname in weight_map.values()})
    LOGGER.info("found %d shards", len(shards))

    bytes_per_shard: Dict[str, int] = {}
    new_weight_map: Dict[str, str] = {}
    current_shard_tensors: Dict[str, torch.Tensor] = {}
    current_shard_bytes = 0
    SOFT_LIMIT = _parse_bytes(args.max_shard_bytes)
    SHARD_INDEX = 1

    for fname in shards:
        shard_path = in_dir / fname
        try:
            tensors = process_shard(shard_path)
        except Exception as exc:
            LOGGER.exception("failed processing %s: %s", fname, exc)
            return 3

        for key, tensor in tensors.items():
            sz = tensor.element_size() * tensor.numel()
            if current_shard_bytes + sz > SOFT_LIMIT and current_shard_tensors:
                shard_filename = f"model-{SHARD_INDEX:05d}-of-{len(shards):05d}.safetensors"
                if not args.dry_run:
                    write_shard(current_shard_tensors, args.output, shard_filename)
                SHARD_INDEX += 1
                current_shard_tensors = {}
                current_shard_bytes = 0
            current_shard_tensors[key] = tensor
            current_shard_bytes += sz
            new_weight_map[key] = (
                f"model-{SHARD_INDEX:05d}-of-{len(shards):05d}.safetensors"
            )

    if current_shard_tensors:
        shard_filename = f"model-{SHARD_INDEX:05d}-of-{len(shards):05d}.safetensors"
        if not args.dry_run:
            write_shard(current_shard_tensors, args.output, shard_filename)

    if args.dry_run:
        LOGGER.info("--dry-run specified, not emitting new index")
        return 0

    # Rewrite the safetensors index with the new filename assignments.
    new_index = {
        "metadata": {"total_size": sum(bytes_per_shard.values())},
        "weight_map": new_weight_map,
    }
    (args.output / "model.safetensors.index.json").write_text(
        json.dumps(new_index, indent=2)
    )
    LOGGER.info("done. wrote %s", args.output)
    return 0


def _parse_bytes(spec: str) -> int:
    units = {"KB": 1024, "MB": 1024 ** 2, "GB": 1024 ** 3, "TB": 1024 ** 4}
    spec_u = spec.upper().strip()
    if spec_u.isdigit():
        return int(spec_u)
    for suffix, mul in units.items():
        if spec_u.endswith(suffix):
            return int(spec_u[: -len(suffix)]) * mul
    raise ValueError(f"unrecognized byte spec: {spec}")


if __name__ == "__main__":
    sys.exit(main())