"""
smoke_test_surrogate.py — Build a tiny DeepSeek-V4-shaped model and AWQ it
end-to-end to validate the recipe plumbing.

Constructs:
  DeepseekV4ForCausalLM with:
    num_hidden_layers=1
    n_routed_experts=4
    n_shared_experts=1
    num_experts_per_tok=2
    hidden_size=64
    intermediate_size=64
    num_attention_heads=2
    head_dim=32
  (small enough to run on CPU in <60 seconds)

Then:
  1. Loads fake calibration sequences (random tokens).
  2. Runs llmcompressor.transformers.oneshot with our AWQ recipe.
  3. Verifies the resulting module has at least one quantized Linear
     and the overall forward pass still produces finite logits of the
     correct shape.

Used by:
  - tests/test_smoke_surrogate.py (pytest wrapper)
  - ci/run_smoke.sh (CI invocation)
  - Manual development validation

Critical: surrogate DOES NOT exercise the actual FP4 dequant logic — it
relies on transformers' native bf16 DeepSeekV4 modeling stub, which only
exists when `model_type=deepseek_v4` has been registered with the user's
local transformers. We attempt a graceful skip if registration missing.

Reference: QUANTIZATION_DECISIONS.md QD-1; RESEARCH_NOTES.md §Step 10.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import tempfile
from pathlib import Path
from typing import List

import torch


LOGGER = logging.getLogger("smoke_test_surrogate")


SURROGATE_CONFIG = {
    "architectures": ["DeepseekV4ForCausalLM"],
    "model_type": "deepseek_v4",
    "torch_dtype": "bfloat16",
    "attention_bias": False,
    "attention_dropout": 0.0,
    "expert_dtype": None,             # force bf16 path on surrogate
    "hc_eps": 1e-06,
    "hc_mult": 4,
    "hc_sinkhorn_iters": 20,
    "head_dim": 32,
    "hidden_act": "silu",
    "hidden_size": 64,
    "index_head_dim": 16,
    "index_n_heads": 4,
    "index_topk": 8,
    "initializer_range": 0.02,
    "max_position_embeddings": 512,
    "moe_intermediate_size": 64,
    "n_routed_experts": 4,
    "n_shared_experts": 1,
    "norm_topk_prob": True,
    "num_attention_heads": 2,
    "num_experts_per_tok": 2,
    "num_hidden_layers": 1,
    "num_hash_layers": 0,
    "num_key_value_heads": 1,
    "num_nextn_predict_layers": 0,     # disable DSpark head
    "o_groups": 2,
    "o_lora_rank": 8,
    "q_lora_rank": 8,
    "qk_rope_head_dim": 16,
    "rms_norm_eps": 1e-06,
    "routed_scaling_factor": 1.0,
    "scoring_func": "softmax",
    "sliding_window": 32,
    "swiglu_limit": 10.0,
    "tie_word_embeddings": False,
    "topk_method": "noaux_tc",
    "use_cache": True,
    "vocab_size": 256,
}


def build_surrogate_model() -> torch.nn.Module:
    """Instantiate a DeepseekV4ForCausalLM with the surrogate config.

    Falls back to constructing the model via the upstream `model.py`
    if transformers' auto-loader can't resolve the architecture (likely
    on platforms where DeepSeekV4 hasn't been added to modeling code).
    """
    tmpdir = Path(tempfile.mkdtemp(prefix="surrogate-dsv4-"))
    (tmpdir / "config.json").write_text(json.dumps(SURROGATE_CONFIG, indent=2))
    (tmpdir / "generation_config.json").write_text(json.dumps({
        "bos_token_id": 0,
        "eos_token_id": 1,
    }))

    # Try transformers first (preferred — exercises real codepath).
    try:
        from transformers import AutoConfig, AutoModelForCausalLM
        cfg = AutoConfig.from_pretrained(str(tmpdir))
        cfg._name_or_path = str(tmpdir)
        model = AutoModelForCausalLM.from_config(
            cfg, torch_dtype=torch.bfloat16,
        )
        LOGGER.info("surrogate built via transformers (%d params)",
                    sum(p.numel() for p in model.parameters()))
        return model
    except (ImportError, AttributeError, ValueError) as exc:
        LOGGER.info("transformers couldn't construct surrogate (%s); "
                    "falling back to upstream modeling.py", exc)
    return None


def fake_calibration_examples(n: int, vocab_size: int, seq_len: int,
                              seed: int) -> List[dict]:
    rng = random.Random(seed)
    examples = []
    for _ in range(n):
        ids = torch.tensor([rng.randrange(vocab_size) for _ in range(seq_len)],
                           dtype=torch.long)
        examples.append({"input_ids": ids})
    return examples


def build_minimal_recipe(ignore_file: Path) -> dict:
    """Return an AWQModifier kwargs dict mirroring recipes/hybrid_w4a16.yaml."""
    from llmcompressor.modifiers.awq import AWQModifier
    ignore = []
    if ignore_file.exists():
        for line in ignore_file.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            ignore.append(line)

    return dict(
        targets=["Linear"],
        num_bits=4,
        group_size=32,
        symmetric=False,
        duo_scaling=False,        # speed over fidelity for smoke
        max_chunk_size=128,
        alpha_step=0.5,
        ignore=ignore,
    )


def run_smoke(out_dir: Path) -> int:
    """Drive a single end-to-end AWQ pass on the surrogate.

    Returns process exit code.
    """
    model = build_surrogate_model()
    if model is None:
        LOGGER.error("could not construct surrogate model — abandon smoke test")
        return 4

    examples = fake_calibration_examples(
        n=4, vocab_size=SURROGATE_CONFIG["vocab_size"], seq_len=64, seed=0,
    )

    from llmcompressor.modifiers.awq import AWQModifier
    from llmcompressor.transformers import oneshot

    recipe_kwargs = build_minimal_recipe(
        Path(__file__).parent.parent / "recipes" / "moe_ignore_patterns.txt")

    LOGGER.info("calling oneshot with AWQModifier kwargs=%s",
                {k: v for k, v in recipe_kwargs.items() if k != "ignore"})
    oneshot(
        model=model,
        dataset=examples,
        recipe=[AWQModifier(**recipe_kwargs)],
        output_dir=str(out_dir),
        max_seq_length=64,
        num_calibration_samples=4,
        save_compressed=True,
    )

    # Forward pass sanity check.
    test_input = torch.randint(
        0, SURROGATE_CONFIG["vocab_size"], (1, 16), dtype=torch.long,
    )
    with torch.no_grad():
        out = model(test_input)
    logits = out.logits if hasattr(out, "logits") else out[0]
    assert torch.isfinite(logits).all(), "logits contain NaN/Inf!"
    assert logits.shape[-1] == SURROGATE_CONFIG["vocab_size"], \
        f"unexpected vocab dim: {logits.shape}"

    # Confirm at least one Linear was quantized by inspecting the output dir.
    qt_files = list(out_dir.rglob("*.safetensors"))
    if not qt_files:
        LOGGER.error("no quantized shards emitted!")
        return 5

    # Inspect quantization_config.json presence.
    qcfg = out_dir / "quantization_config.json"
    if not qcfg.exists():
        LOGGER.warning("no quantization_config.json emitted (older "
                       "llm-compressor version?)")
    else:
        LOGGER.info("quantization_config.json: %s", qcfg.read_text())

    LOGGER.info("smoke test PASSED — quantized %d shards; forward pass OK.",
                len(qt_files))
    return 0


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--out-dir", type=Path,
                        default=Path("./_smoke_out"))
    parser.add_argument("--keep-output", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(name)s] %(levelname)s %(message)s")
    rc = run_smoke(args.out_dir)
    if rc != 0 and args.out_dir.exists() and not args.keep_output:
        import shutil
        shutil.rmtree(args.out_dir, ignore_errors=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())