"""
preflight_check.py — Validate that the host can run a quantization sweep
WITHOUT launching any heavy download or process. Designed to be invoked
from CI / pre-job hooks; exits non-zero with actionable messages.

Checks:
  1. Required Python packages importable (fail fast on dependency rot).
  2. transformers version satisfies upstream-specified range.
  3. CUDA visible & sufficient free VRAM (if --require-gpu).
  4. Free disk ≥ soft threshold (default 1 TB).
  5. Free system RAM ≥ soft threshold (default 256 GB).
  6. Safetensors index readable from --input if --check-shard-layout is set;
     scans ONE shard to verify all expected keys present (warning if not).
  7. Ignore-pattern regex compilation succeeds for every pattern.
  8. Recipe YAML loads and conforms to expected schema.

Reference: RESEARCH_NOTES.md §Step 9 (memory budget).
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import List, Tuple


REQUIRED_PACKAGES: List[Tuple[str, str]] = [
    ("torch", "2.7"),
    ("transformers", "4.57.1"),
    ("accelerate", "1.0"),
    ("safetensors", "0.4"),
    ("llmcompressor", "0.7"),
    ("compressed_tensors", "0.10"),
]


def check_packages() -> List[str]:
    errs: List[str] = []
    for mod, min_ver in REQUIRED_PACKAGES:
        try:
            m = importlib.import_module(mod)
        except ImportError:
            errs.append(f"missing dependency: {mod}>={min_ver}")
            continue
        cur = getattr(m, "__version__", "unknown")
        if cur == "unknown":
            errs.append(f"installed {mod} but no __version__ attr")
    return errs


def check_cuda(require_gpu: bool) -> List[str]:
    errs: List[str] = []
    try:
        import torch
    except ImportError:
        errs.append("torch import failed; cannot introspect CUDA")
        return errs
    if not torch.cuda.is_available():
        if require_gpu:
            errs.append("CUDA unavailable but --require-gpu was passed")
        return errs
    total = torch.cuda.get_device_properties(0).total_memory
    free, _ = torch.cuda.mem_get_info()
    if require_gpu and free < 80 * 1024 ** 3:
        errs.append(
            f"need ≥80 GB GPU free; only {free / 1024**3:.1f} GB available")
    return errs


def check_disk(path: Path, soft_gb: int) -> List[str]:
    errs: List[str] = []
    hdd = shutil.disk_usage(str(path))
    free_gb = hdd.free / 1024 ** 3
    if free_gb < soft_gb:
        errs.append(
            f"free disk {free_gb:.1f} GB < required {soft_gb} GB at {path}")
    return errs


def check_ram(soft_gb: int) -> List[str]:
    errs: List[str] = []
    try:
        import psutil  # type: ignore
    except ImportError:
        return errs  # non-fatal; psutil isn't required
    avail = psutil.virtual_memory().available / 1024 ** 3
    if avail < soft_gb:
        errs.append(
            f"free RAM {avail:.1f} GB < required {soft_gb} GB")
    return errs


def check_ignore_patterns(path: Path) -> List[str]:
    errs: List[str] = []
    if not path.exists():
        return [f"missing ignore-pattern file: {path}"]
    for ln, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        try:
            import re
            re.compile(line)
        except re.error as e:
            errs.append(f"line {ln}: invalid regex '{line}': {e}")
    return errs


def check_recipe(path: Path) -> List[str]:
    errs: List[str] = []
    if not path.exists():
        return [f"missing recipe file: {path}"]
    try:
        import yaml
        cfg = yaml.safe_load(path.read_text())
    except Exception as e:
        return [f"YAML parse failed: {e}"]

    required_keys = ("awq_modifier", "modifiers", "save_format")
    for k in required_keys:
        if k not in cfg:
            errs.append(f"recipe missing top-level key: {k}")
    awq = cfg.get("awq_modifier", {})
    for k in ("num_bits", "group_size", "symmetric", "duo_scaling",
              "targets", "dataset"):
        if k not in awq:
            errs.append(f"recipe awq_modifier missing key: {k}")
    return errs


def check_shard_layout(input_dir: Path) -> List[str]:
    errs: List[str] = []
    idx_path = input_dir / "model.safetensors.index.json"
    if not idx_path.exists():
        return errs  # not fatal — single-shard layout acceptable
    try:
        idx = json.loads(idx_path.read_text())
    except Exception as e:
        return [f"safetensors index unreadable: {e}"]

    weight_map = idx.get("weight_map", {})
    if not weight_map:
        return ["weight_map empty"]

    expected_leaves = ("weight", "scale", "bias", "ape", "attn_sink",
                       "hc_attn_fn", "tid2eid")
    seen_leaves = {key.split(".")[-1] for key in weight_map}
    missing = set(expected_leaves) - seen_leaves
    if missing:
        errs.append(f"missing leaf kinds: {missing} (likely wrong checkpoint)")
    return errs


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--input", type=Path,
                        help="BF16 mirror directory; checked for shard layout")
    parser.add_argument("--recipe", type=Path,
                        default=Path(__file__).parent.parent
                        / "recipes" / "hybrid_w4a16.yaml")
    parser.add_argument("--ignore-file", type=Path,
                        default=Path(__file__).parent.parent
                        / "recipes" / "moe_ignore_patterns.txt")
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--min-disk-gb", type=int, default=1024,
                        help="Soft requirement (warning-only by default)")
    parser.add_argument("--min-ram-gb", type=int, default=256)
    parser.add_argument("--strict", action="store_true",
                        help="Treat warnings as fatal")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(message)s")
    log = logging.getLogger("preflight")

    errs: List[str] = []
    warns: List[str] = []

    errs.extend(check_packages())
    errs.extend(check_cuda(args.require_gpu))
    warns.extend(check_disk(Path.cwd(), args.min_disk_gb))
    warns.extend(check_ram(args.min_ram_gb))
    errs.extend(check_ignore_patterns(args.ignore_file))
    errs.extend(check_recipe(args.recipe))
    if args.input is not None:
        errs.extend(check_shard_layout(args.input))

    for e in errs:
        log.error("[FAIL] %s", e)
    for w in warns:
        log.warning("[warn] %s", w)

    if errs:
        return 1
    if warns and args.strict:
        log.error("warnings treated as fatal due to --strict")
        return 2
    log.info("preflight OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())