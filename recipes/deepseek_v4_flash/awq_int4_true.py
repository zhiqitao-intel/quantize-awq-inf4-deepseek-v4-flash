#!/usr/bin/env python3
"""True AWQ (activation-aware scaling) for DeepSeek-V4-Flash-0731.

Differs from awq_int4_cpu.py by adding AWQModifier with a custom mapping
registered for DeepseekV4ForCausalLM. The stock mappings expect Llama-style
names (q_proj/k_proj/v_proj/gate_proj/up_proj); V4 uses:

  attention:  q_a_proj, q_a_norm, q_b_proj, kv_proj, kv_norm,
              o_a_proj, o_b_proj
  compressor: gate_proj, kv_proj (on self_attn.compressor and
              self_attn.compressor.indexer)
  MoE:        ffn.experts.{j}.{gate,up,down}_proj + shared_experts.* +
              mlp.gate (router)

Mappings below pair each norm with the Linears whose inputs it scales.
Ratios follow the same convention as DeepseekV3ForCausalLM's registered
mapping in llm-compressor.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import shutil
import struct
import sys

import torch
import llmcompressor
import compressed_tensors
from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
from compressed_tensors.offload import disable_onloading
from datasets import concatenate_datasets, load_dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.transform.awq import AWQModifier
from llmcompressor.modifiers.transform.awq.mappings import (
    AWQ_MAPPING_REGISTRY,
    AWQMapping,
)
from llmcompressor.modifiers.quantization import QuantizationModifier
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# Register DeepSeek-V4 AWQ mappings BEFORE constructing modifiers.
# Pairs: (smooth_norm_regex, [balance_linear_regexes]).
# ---------------------------------------------------------------------------

def register_deepseek_v4_awq_mappings():
    if "DeepseekV4ForCausalLM" in AWQ_MAPPING_REGISTRY:
        return  # already registered

    AWQ_MAPPING_REGISTRY["DeepseekV4ForCausalLM"] = [
        # input_layernorm feeds q_a_proj and kv_proj (MLA down-projections)
        AWQMapping(
            smooth_layer="re:.*input_layernorm$",
            balance_layers=["re:.*self_attn\\.q_a_proj$", "re:.*self_attn\\.kv_proj$"],
        ),
        # q_a_norm feeds q_b_proj (q up-projection)
        AWQMapping(
            smooth_layer="re:.*self_attn\\.q_a_norm$",
            balance_layers=["re:.*self_attn\\.q_b_proj$"],
        ),
        # kv_norm feeds o_a_proj (post-attention spread)
        AWQMapping(
            smooth_layer="re:.*self_attn\\.kv_norm$",
            balance_layers=["re:.*self_attn\\.o_a_proj$"],
        ),
        # post_attention_layernorm feeds router gate + first expert projections
        AWQMapping(
            smooth_layer="re:.*post_attention_layernorm$",
            balance_layers=[
                "re:.*mlp\\.gate$",
                "re:.*mlp\\.experts\\.\\d+\\.gate_proj$",
                "re:.*mlp\\.experts\\.\\d+\\.up_proj$",
                "re:.*mlp\\.shared_experts\\.gate_proj$",
                "re:.*mlp\\.shared_experts\\.up_proj$",
            ],
        ),
        # w3 -> w2 within each expert (SwiGLU pairing)
        AWQMapping(
            smooth_layer="re:.*mlp\\.experts\\.\\d+\\.up_proj$",
            balance_layers=["re:.*mlp\\.experts\\.\\d+\\.down_proj$"],
        ),
        AWQMapping(
            smooth_layer="re:.*mlp\\.shared_experts\\.up_proj$",
            balance_layers=["re:.*mlp\\.shared_experts\\.down_proj$"],
        ),
    ]
    print("registered custom AWQ mappings for DeepseekV4ForCausalLM")


# ---------------------------------------------------------------------------
# Reuse helpers from awq_int4_cpu
# ---------------------------------------------------------------------------

# Import the working pieces rather than duplicating them
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from awq_int4_cpu import (  # noqa: E402
    disable_pin_memory_without_cuda,
    get_calib_dataset,
    data_collator,
    upcast_source,
)


def patch_linear_dtype_mismatch():
    """AWQ's scale computation runs in float32; model weights are bfloat16.

    During calibration forward passes through quantized Linears,
    compressed-tensors' quantized_forward dispatches to whatever primitive
    the module's own forward uses. Two paths matter in DeepSeek-V4:

      1. standard nn.Linear → F.linear(input, weight)
      2. DeepseekV4GroupedLinear (o_a_proj) → torch.bmm(x, w).transpose(0,1)

    When AWQ wraps weight with a float32 scale tensor, activations arrive
    as float32 while the underlying weight stays bfloat16, raising
    "expected m1 and m2 to have the same dtype". Patch both primitives to
    promote the narrower operand via torch.promote_types — bf16 → fp32 is
    exact so no numeric change.
    """
    import torch.nn.functional as F

    # --- F.linear ---
    original_linear = F.linear

    def _auto_cast_linear(input, weight, bias=None):
        if input.dtype != weight.dtype:
            target = torch.promote_types(input.dtype, weight.dtype)
            return original_linear(
                input.to(target),
                weight.to(target),
                None if bias is None else bias.to(target),
            )
        return original_linear(input, weight, bias)

    F.linear = _auto_cast_linear

    # --- torch.bmm (used by DeepseekV4GroupedLinear) ---
    original_bmm = torch.bmm

    def _auto_cast_bmm(a, b, *, out=None):
        if a.dtype != b.dtype:
            target = torch.promote_types(a.dtype, b.dtype)
            return original_bmm(a.to(target), b.to(target), out=out)
        return original_bmm(a, b, out=out)

    torch.bmm = _auto_cast_bmm

    # --- torch.mm (defensive: same class of bug on 2D matmul path) ---
    original_mm = torch.mm

    def _auto_cast_mm(a, b, *, out=None):
        if a.dtype != b.dtype:
            target = torch.promote_types(a.dtype, b.dtype)
            return original_mm(a.to(target), b.to(target), out=out)
        return original_mm(a, b, out=out)

    torch.mm = _auto_cast_mm

    print("patched F.linear / torch.bmm / torch.mm to auto-cast mixed dtypes")


CODE_DATASET = "codeparrot/self-instruct-starcoder"
CODE_SPLIT = "curated"
CHAT_DATASET = "HuggingFaceH4/ultrachat_200k"
CHAT_SPLIT = "train_sft"

ROUTED_EXPERT = re.compile(r"^model\.layers\.\d+\.ffn\.experts\.\d+\.(?:gate|up|down)_proj$")
DSPARK_KEY = re.compile(r"^(model\.)?(layers\.(40|41|42)\.)?dspark\.|^mtp\.")


def collect_ignore_list(model):
    """Everything that must stay at source precision."""
    PRESERVE = re.compile(
        r"\.(?:"
        r"hc_attn_fn|hc_attn_base|hc_attn_scale|"
        r"hc_ffn_fn|hc_ffn_base|hc_ffn_scale|"
        r"hc_head_fn|hc_head_base|hc_head_scale|"
        r"attn_sink|tid2eid|"
        r"compressor\.ape|compressor\.norm\.weight|indexer\.compressor\.ape|"
        r"indexer\.compressor\.norm\.weight|"
        r"scorer\.weights_proj|"          # indexer scoring head — routing-critical
        r"q_a_norm\.weight|kv_norm\.weight|"
        r"input_layernorm\.weight|post_attention_layernorm\.weight|"
        r"norm\.weight"
        r")$"
    )
    ignore = []
    for name, _ in model.named_modules():
        if PRESERVE.search(name) or name.endswith(".gate"):
            ignore.append(name)
    ignore.extend(["lm_head", "model.embed_tokens"])
    seen, out = set(), []
    for x in ignore:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return sorted(out)


def drop_container_module_ignores(output):
    path = os.path.join(output, "config.json")
    config = json.load(open(path))
    quant = config.get("quantization_config")
    if not quant or "ignore" not in quant:
        return 0
    container = re.compile(
        r"\.(linear_attn|self_attn|mlp|attn|experts|ffn|visual|shared_experts|compressor|indexer)$"
    )
    before = len(quant["ignore"])
    quant["ignore"] = [x for x in quant["ignore"] if not container.search(x)]
    dropped = before - len(quant["ignore"])
    if dropped:
        json.dump(config, open(path, "w"), indent=2)
        print(f"dropped {dropped} container-module ignore entries")
    return dropped


def graft_dspark_weights(source, output):
    from safetensors import safe_open
    from safetensors.torch import save_file
    dspark = {}
    for shard in sorted(glob.glob(os.path.join(source, "*.safetensors"))):
        with safe_open(shard, framework="pt") as f:
            for key in f.keys():
                if DSPARK_KEY.match(key) or ".markov_head." in key \
                   or ".confidence_head." in key or ".main_proj." in key \
                   or ".hc_head_" in key or key.startswith("mtp."):
                    dspark[key] = f.get_tensor(key)
    if not dspark:
        print("no DSpark/MTP tensors found; skipping graft")
        return 0
    shard_name = "model-dspark.safetensors"
    save_file(dspark, os.path.join(output, shard_name), metadata={"format": "pt"})
    index_path = os.path.join(output, "model.safetensors.index.json")
    index = json.load(open(index_path))
    index["weight_map"].update({k: shard_name for k in dspark})
    index["metadata"]["total_size"] = index["metadata"].get("total_size", 0) + sum(
        t.numel() * t.element_size() for t in dspark.values()
    )
    json.dump(index, open(index_path, "w"), indent=2)
    print(f"grafted {len(dspark)} DSpark/MTP tensors")
    return len(dspark)


def verify_output(output):
    index_path = os.path.join(output, "model.safetensors.index.json")
    idx = json.load(open(index_path))
    packed = [k for k in idx["weight_map"] if k.endswith(".weight_packed")]
    routed = [k for k in packed if ROUTED_EXPERT.match(k[: -len(".weight_packed")])]
    attn = [k for k in packed if "self_attn" in k]
    print(f"verified: {len(packed)} packed total, "
          f"{len(routed)} routed-expert, {len(attn)} attention")
    if len(packed) == 0:
        raise SystemExit("no packed tensors emitted")
    return {"packed": len(packed), "routed": len(routed), "attention": len(attn)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="original snapshot OR bf16 mirror")
    ap.add_argument("--output", required=True)
    ap.add_argument("--work-dir", default=None)
    ap.add_argument("--samples", type=int, default=32)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--group-size", type=int, default=32)
    ap.add_argument("--code-fraction", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-grid", type=int, default=5)
    ap.add_argument("--duo-scaling", action="store_true",
                    help="enable duo scaling (2x slower, better quality)")
    ap.add_argument("--skip-upcast", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    disable_pin_memory_without_cuda()
    patch_linear_dtype_mismatch()
    register_deepseek_v4_awq_mappings()

    work_dir = args.work_dir or (args.output.rstrip("/") + ".work")
    os.makedirs(work_dir, exist_ok=True)

    if args.skip_upcast:
        mirror_dir = args.model
    else:
        marker = os.path.join(work_dir, "bf16-mirror", "model.safetensors.index.json")
        if os.path.exists(marker):
            mirror_dir = os.path.join(work_dir, "bf16-mirror")
            print(f"reusing mirror at {mirror_dir}")
        else:
            mirror_dir = upcast_source(args.model, work_dir)

    tokenizer = AutoTokenizer.from_pretrained(mirror_dir, trust_remote_code=True)
    calib = get_calib_dataset(tokenizer, args.samples, args.seed, args.code_fraction)

    print(f"loading model from {mirror_dir}...")
    model = AutoModelForCausalLM.from_pretrained(
        mirror_dir, dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True
    )

    ignore = collect_ignore_list(model)
    print(f"ignore list: {len(ignore)} entries")

    scheme = QuantizationScheme(
        targets=["Linear"],
        weights=QuantizationArgs(
            num_bits=4, type="int", symmetric=False,
            group_size=args.group_size, strategy="group",
            observer="mse", zp_dtype=torch.int8,
        ),
    )

    recipe = [
        AWQModifier(
            duo_scaling=args.duo_scaling,
            n_grid=args.n_grid,
            # Let AWQModifier use our registered mappings automatically
        ),
        QuantizationModifier(
            config_groups={"group_0": scheme},
            targets=["Linear"],
            ignore=ignore,
        ),
    ]

    oneshot(
        model=model,
        dataset=calib,
        recipe=recipe,
        max_seq_length=args.seq_len,
        num_calibration_samples=args.samples,
        data_collator=data_collator,
        pipeline="independent",
        propagate_error=False,
        log_dir=None,
    )

    with disable_onloading():
        model.save_pretrained(args.output, save_compressed=True)
        tokenizer.save_pretrained(args.output)

    drop_container_module_ignores(args.output)
    graft_dspark_weights(args.model, args.output)
    report = verify_output(args.output)

    print(f"\nwrote {args.output}")
    cfg_out = json.load(open(os.path.join(args.output, "config.json")))
    quant = cfg_out.get("quantization_config", {})
    print(json.dumps(quant.get("config_groups", {}), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())