"""
quantize_llmcompressor.py — Drive an AWQ-W4A16 quantization sweep.

Reads:
  - A BF16 mirror produced by `scripts/upcast_to_bf16.py`
  - A calibration dataset (256 default, configurable)
  - The hybrid recipe at `recipes/hybrid_w4a16.yaml`
  - Ignore patterns from `recipes/moe_ignore_patterns.txt`

Produces:
  - A new checkpoint under --output conforming to the
    `compressed-tensors` v0.10 int-quantized format consumable by vLLM/SGLang.

Example:
  python -m scripts.quantize_llmcompressor \\
      --bf16-input ./dsv4-bf16-mirror \\
      --output ./dsv4-w4a16 \\
      --dataset ./calib/wiki-sonnet-gsm8k \\
      --seed 0xb0bacafe

Reference:
  - RESEARCH_NOTES.md
  - QUANTIZATION_DECISIONS.md
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Iterable, List, Optional

import torch


LOGGER = logging.getLogger("quantize")

DEFAULT_RECIPE = Path(__file__).parent.parent / "recipes" / "hybrid_w4a16.yaml"
DEFAULT_IGNORE_FILE = (
    Path(__file__).parent.parent / "recipes" / "moe_ignore_patterns.txt"
)


# ----------------------------------------------------------------------------
# Helper utilities
# ----------------------------------------------------------------------------

def set_global_seed(seed: int) -> None:
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_ignore_patterns(path: Path) -> List[str]:
    """Read a newline-delimited pattern file; ignore blanks/comments."""
    if not path.exists():
        raise FileNotFoundError(path)
    patterns = []
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        patterns.append(stripped)
    LOGGER.info("loaded %d ignore patterns from %s", len(patterns), path)
    return patterns


def attach_ignore_set(module: torch.nn.Module, patterns: Iterable[str]) -> None:
    """Mark parameters matching any pattern as `requires_grad=False`.

    Defensive — llm-compressor respect its own `ignore=` list, but this
    guards against regressions in third-party modifier implementations.
    """
    pats = tuple(patterns)
    matched = 0
    total = 0
    for name, param in module.named_parameters(recurse=True):
        total += 1
        if any(p in name for p in pats):
            param.requires_grad = False
            matched += 1
    LOGGER.info("attached ignore set: %d/%d params frozen", matched, total)


def load_yaml(path: Path) -> dict:
    import yaml
    return yaml.safe_load(path.read_text())


# ----------------------------------------------------------------------------
# Calibration loader wrapper
# ----------------------------------------------------------------------------

def build_calibration_loader(dataset_path: Path, max_seq_length: int,
                             num_samples: int, seed: int):
    """Return a list of tokenized examples suitable for
    `llmcompressor.transformers.oneshot(...dataset=...)`.

    Three input forms accepted:
    - HuggingFace dataset (arrow) directory with `train` split
    - Pre-tokenized `.pt`/`.bin` file shaped [N, max_seq_length]
    - Directory of raw text files (one .txt per sequence)
    """
    if (dataset_path / "dataset_info.json").exists():
        # Lazy import to keep --help cheap.
        from datasets import load_from_disk
        ds = load_from_disk(str(dataset_path))["train"]
        LOGGER.info("loaded HF dataset from %s with %d examples",
                    dataset_path, len(ds))
        return ds.select(range(min(num_samples, len(ds))))

    bin_files = sorted(dataset_path.glob("tokens_*.pt"))
    if bin_files:
        chunks = []
        for bf in bin_files:
            chunks.extend(torch.load(bf, map_location="cpu"))
        random.Random(seed).shuffle(chunks)
        LOGGER.info("loaded %d pre-tokenized chunks from %s",
                    len(chunks), dataset_path)
        return [{"input_ids": c[:max_seq_length]} for c in chunks[:num_samples]]

    txt_files = sorted(dataset_path.glob("*.txt"))[:num_samples]
    if not txt_files:
        raise FileNotFoundError(
            f"no recognized dataset form found under {dataset_path}; "
            "expected dataset_info.json (HF arrow), tokens_*.pt, or *.txt")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        os.environ.get("UPSTREAM_TOKENIZER",
                       "deepseek-ai/DeepSeek-V4-Flash-0731"),
        trust_remote_code=False,
    )
    examples = []
    for tf in txt_files:
        ids = tok(tf.read_text(), return_tensors="pt",
                  truncation=True, max_length=max_seq_length)["input_ids"][0]
        examples.append({"input_ids": ids})
    LOGGER.info("built %d text-derived examples", len(examples))
    return examples


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])

    parser.add_argument("--bf16-input", type=Path, required=True,
                        help="BF16 mirror produced by scripts/upcast_to_bf16.py")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output directory for INT4 checkpoint")
    parser.add_argument("--dataset", type=Path, required=True,
                        help="Calibration dataset directory")
    parser.add_argument("--recipe", type=Path, default=DEFAULT_RECIPE)
    parser.add_argument("--ignore-file", type=Path, default=DEFAULT_IGNORE_FILE)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--num-samples", type=int, default=256)
    parser.add_argument("--seed", type=lambda s: int(s, 0), default=0xb0bacafe)
    parser.add_argument("--group-size", type=int, default=128,
                        help="Override recipe group_size")
    parser.add_argument("--num-bits", type=int, default=4,
                        help="Override recipe num_bits")
    parser.add_argument("--symmetric", action="store_true",
                        help="Override recipe to symmetric quantization")
    parser.add_argument("--no-duo-scaling", dest="duo_scaling",
                        action="store_false",
                        help="Disable duo-scaling (faster, lower quality)")
    parser.add_argument("--quantize-embedding-and-head",
                        action="store_true",
                        help="OPTIONAL: also quantize embed_tokens + lm_head")
    parser.add_argument("--device-map", default="auto",
                        help='Passed to from_pretrained (try "cuda" '
                             'or "cpu" or "auto")')
    parser.add_argument("--deterministic-flash-attn",
                        action="store_true",
                        help="Force reproducible FlashAttn reductions")
    parser.add_argument("--save-final-config-json", type=Path,
                        default=None,
                        help="Dump final merged recipe to this JSON "
                             "before kicking off")

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    )
    set_global_seed(args.seed)
    if args.deterministic_flash_attn:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    recipe_cfg = load_yaml(args.recipe)
    awq_cfg = recipe_cfg["awq_modifier"]
    ignore_patterns = load_ignore_patterns(args.ignore_file)

    # Optional: hide lm_head / embed when user requests quantization there too.
    if not args.quantize_embedding_and_head:
        ignore_patterns.extend([
            "model.embed.weight",
            "model.head.weight",
        ])

    # Resolve env substitution in recipe.
    awq_cfg["dataset"]["source"] = str(args.dataset)
    awq_cfg["dataset"]["max_seq_length"] = args.max_seq_length
    awq_cfg["dataset"]["num_samples"] = args.num_samples
    awq_cfg["num_bits"] = args.num_bits
    awq_cfg["group_size"] = args.group_size
    awq_cfg["symmetric"] = bool(args.symmetric)
    awq_cfg["duo_scaling"] = bool(args.duo_scaling)
    awq_cfg.setdefault("ignore", [])
    awq_cfg["ignore"].extend(ignore_patterns)

    if args.save_final_config_json:
        args.save_final_config_json.parent.mkdir(parents=True, exist_ok=True)
        args.save_final_config_json.write_text(json.dumps(awq_cfg, indent=2))
        LOGGER.info("final merged recipe dumped to %s",
                    args.save_final_config_json)

    # ----------------------------------------------------------------------
    # Construct AWQModifier
    # ----------------------------------------------------------------------
    from llmcompressor.modifiers.awq import AWQModifier
    from llmcompressor.transformers import oneshot
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    LOGGER.info("loading model from %s (device-map=%s)",
                args.bf16_input, args.device_map)
    model_kwargs = dict(
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    if args.device_map == "auto":
        model_kwargs["device_map"] = "auto"
        model_kwargs["low_cpu_mem_usage"] = True
    else:
        model_kwargs["device_map"] = args.device_map
    model = AutoModelForCausalLM.from_pretrained(args.bf16_input, **model_kwargs)
    attach_ignore_set(model, ignore_patterns)

    tokenizer = AutoTokenizer.from_pretrained(
        args.bf16_input, trust_remote_code=True,
    )

    calibration = build_calibration_loader(
        args.dataset,
        max_seq_length=args.max_seq_length,
        num_samples=args.num_samples,
        seed=args.seed,
    )
    LOGGER.info("calibration set: %d examples", len(calibration))

    # Construct the AWQModifier explicitly so we can pass `ignore=` filtered
    # via full-qualified names (matching substring semantics per
    # IGNORE_PATTERNS_DERIVATION.md).
    modifier = AWQModifier(
        targets=awq_cfg["targets"],
        num_bits=awq_cfg["num_bits"],
        group_size=awq_cfg["group_size"],
        symmetric=awq_cfg["symmetric"],
        duo_scaling=awq_cfg["duo_scaling"],
        max_chunk_size=awq_cfg["max_chunk_size"],
        alpha_step=awq_cfg["alpha_step"],
        ignore=awq_cfg["ignore"],
    )

    args.output.mkdir(parents=True, exist_ok=True)
    LOGGER.info("starting oneshot AWQ sweep; output=%s", args.output)

    oneshot(
        model=model,
        dataset=calibration,
        recipe=[modifier],
        output_dir=str(args.output),
        max_seq_length=args.max_seq_length,
        num_calibration_samples=args.num_samples,
        save_compressed=True,
    )

    # Persist tokenizer alongside the quantized checkpoint.
    tokenizer.save_pretrained(str(args.output))
    LOGGER.info("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())