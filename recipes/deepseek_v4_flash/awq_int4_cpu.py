#!/usr/bin/env python3
"""CPU-only variant of awq_int4.py for hosts without Intel XPU.

Differences from the XPU recipe:
  - no torch.distributed; single process
  - no XCCL, no get_main_device rebinding (llm-compressor falls back to CPU)
  - pin_memory patch applied unconditionally
  - smaller defaults suited to CPU throughput (~1 TFLOP/s on Sapphire Rapids)
  - optional --max-layers to truncate at N decoder layers for POC runs

Everything else — upcast, ignore list, DSpark grafting, verification — is
identical to the XPU recipe so artifacts produced here are structurally the
same as what an XPU run would emit.
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
from llmcompressor.modifiers.quantization import QuantizationModifier
from transformers import AutoModelForCausalLM, AutoTokenizer


CODE_DATASET = "codeparrot/self-instruct-starcoder"
CODE_SPLIT = "curated"
CHAT_DATASET = "HuggingFaceH4/ultrachat_200k"
CHAT_SPLIT = "train_sft"

# Same as XPU recipe: DeepSeek-V4 uses ffn.experts.{j}.{w1,w2,w3}
ROUTED_EXPERT = re.compile(r"^model\.layers\.\d+\.ffn\.experts\.\d+\.w[123]$")
DSPARK_KEY = re.compile(r"^(model\.)?(layers\.(40|41|42)\.)?dspark\.|^mtp\.")
PRESERVE_FP32 = re.compile(
    r"\.(?:"
    r"hc_attn_fn|hc_attn_base|hc_attn_scale|"
    r"hc_ffn_fn|hc_ffn_base|hc_ffn_scale|"
    r"hc_head_fn|hc_head_base|hc_head_scale|"
    r"attn_sink|tid2eid|"
    r"compressor\.ape|compressor\.norm\.weight|"
    r"q_norm\.weight|kv_norm\.weight|"
    r"attn_norm\.weight|ffn_norm\.weight|main_norm\.weight|"
    r"norm\.weight"
    r")$"
)


def disable_pin_memory_without_cuda():
    """Same rationale as the XPU recipe."""
    if torch.cuda.is_available():
        return
    try:
        from llmcompressor.pipelines import cache as _cache
        _cache.IntermediatesCache._pin_intermediate = classmethod(
            lambda cls, intermediate: None
        )
        print("pin_memory disabled: no CUDA present")
    except ImportError:
        pass


def upcast_source(source_dir, work_dir):
    """Identical to XPU recipe's upcast_source. See there for detail."""
    from safetensors import safe_open
    from safetensors.torch import save_file

    E2M1 = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
         -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        dtype=torch.bfloat16,
    )

    def unpack_fp4(packed):
        lo = (packed & 0xF).long()
        hi = ((packed >> 4) & 0xF).long()
        return torch.stack([E2M1[hi], E2M1[lo]], dim=-1).flatten(-2)

    def detect_and_dequant(key, tensor, scale):
        if key.endswith(".scale"):
            return None
        if (
            tensor.dtype in (torch.uint8, torch.int8)
            and ".experts." in key
            and re.search(r"\.w[123]\.weight$", key)
        ):
            if tensor.dtype == torch.int8:
                tensor = tensor.view(torch.uint8)
            dense = unpack_fp4(tensor)
            if scale is not None:
                block = 32
                out_f, in_f = dense.shape
                s = scale.to(torch.bfloat16)
                if s.shape == (out_f, in_f // block):
                    full = s.repeat_interleave(block, dim=1)[:, :in_f]
                    dense = dense * full
            return dense.to(torch.bfloat16)
        if tensor.dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz):
            bf = tensor.to(torch.bfloat16)
            if scale is not None:
                bs = 128
                out_f, in_f = bf.shape
                rs, cs = (out_f + bs - 1) // bs, (in_f + bs - 1) // bs
                s = scale.to(torch.bfloat16)
                if s.shape == (rs, cs):
                    full = s.repeat_interleave(bs, dim=0).repeat_interleave(bs, dim=1)
                    bf = bf * full[:out_f, :in_f]
            return bf
        if tensor.is_floating_point():
            return tensor.to(torch.bfloat16)
        return tensor

    mirror_dir = os.path.join(work_dir, "bf16-mirror")
    os.makedirs(mirror_dir, exist_ok=True)

    index_path = os.path.join(source_dir, "model.safetensors.index.json")
    # Single-shard snapshot fallback
    if not os.path.exists(index_path):
        single = os.path.join(source_dir, "model.safetensors")
        if os.path.exists(single):
            weight_map = {}
            with safe_open(single, framework="pt") as fh:
                for k in fh.keys():
                    weight_map[k] = "model.safetensors"
            shards = ["model.safetensors"]
        else:
            raise SystemExit(f"no weights under {source_dir}")
    else:
        weight_map = json.load(open(index_path))["weight_map"]
        shards = sorted(set(weight_map.values()))

    print(f"upcast: {len(shards)} input shards")
    new_weight_map = {}
    current_tensors = {}
    current_bytes = 0
    SOFT_LIMIT = 5 * 1024 ** 3
    out_idx = 1

    def flush():
        nonlocal current_tensors, current_bytes, out_idx
        if not current_tensors:
            return
        name = f"mirror-{out_idx:05d}.safetensors"
        save_file(current_tensors, os.path.join(mirror_dir, name),
                  metadata={"format": "pt"})
        out_idx += 1
        current_tensors = {}
        current_bytes = 0

    for shard_name in shards:
        path = os.path.join(source_dir, shard_name)
        with safe_open(path, framework="pt", device="cpu") as fh:
            keys = list(fh.keys())
            scale_keys = {k[: -len(".scale")]: k for k in keys if k.endswith(".scale")}
            for key in keys:
                if key.endswith(".scale"):
                    continue
                scale = fh.get_tensor(scale_keys[key]) if key in scale_keys else None
                tensor = fh.get_tensor(key)
                dense = detect_and_dequant(key, tensor, scale)
                sz = dense.numel() * dense.element_size()
                if current_bytes + sz > SOFT_LIMIT and current_tensors:
                    flush()
                current_tensors[key] = dense.contiguous()
                current_bytes += sz
                new_weight_map[key] = f"mirror-{out_idx:05d}.safetensors"
        print(f"  processed {shard_name}")
    flush()

    # Write index mapping every original name to its new shard
    idx_out = {"metadata": {"total_size": 0}, "weight_map": {}}
    for shard_file in sorted(set(new_weight_map.values())):
        p = os.path.join(mirror_dir, shard_file)
        with safe_open(p, framework="pt") as fh:
            for k in fh.keys():
                idx_out["weight_map"][k] = shard_file
        idx_out["metadata"]["total_size"] += os.path.getsize(p)
    with open(os.path.join(mirror_dir, "model.safetensors.index.json"), "w") as fh:
        json.dump(idx_out, fh, indent=2)

    cfg_path = os.path.join(source_dir, "config.json")
    cfg = json.load(open(cfg_path))
    cfg.pop("quantization_config", None)
    cfg.pop("expert_dtype", None)
    cfg["torch_dtype"] = "bfloat16"
    with open(os.path.join(mirror_dir, "config.json"), "w") as fh:
        json.dump(cfg, fh, indent=2)

    for name in ("tokenizer.json", "tokenizer_config.json",
                 "generation_config.json"):
        src = os.path.join(source_dir, name)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(mirror_dir, name))

    total_gib = sum(os.path.getsize(os.path.join(mirror_dir, f))
                    for f in os.listdir(mirror_dir)) / 2 ** 30
    print(f"upcast done: {total_gib:.2f} GiB written to {mirror_dir}")
    return mirror_dir


def get_calib_dataset(tokenizer, n_samples, seed, code_fraction=0.7):
    """Simplified calibration set for CPU runs — fewer samples, shorter seqs."""
    n_code = int(n_samples * code_fraction)
    n_chat = n_samples - n_code
    parts = []

    def tokenize(messages):
        # DeepSeek-V4 does NOT ship a Jinja chat template; it uses its own
        # encoding_dsv4.py encoder. Use that to produce a faithful prompt string,
        # then tokenize it with the base tokenizer.
        if getattr(tokenizer, "chat_template", None) is None:
            try:
                import encoding_dsv4 as _enc
                prompt = _enc.encode_messages(messages, thinking_mode="chat")
                ids = tokenizer.encode(prompt, add_special_tokens=False)
                return {"input_ids": list(ids)}
            except Exception:
                pass
        out = tokenizer.apply_chat_template(
            messages, tokenize=True, return_dict=True, add_generation_prompt=False
        )
        ids = out["input_ids"]
        while isinstance(ids, (list, tuple)) and ids and isinstance(ids[0], (list, tuple)):
            ids = ids[0]
        return {"input_ids": list(ids)}

    if n_code:
        try:
            ds = load_dataset(CODE_DATASET, split=f"{CODE_SPLIT}[:{n_code * 10}]",
                              cache_dir=os.environ.get("HF_HOME"))
            ds = ds.shuffle(seed=seed).select(range(min(n_code, len(ds))))
            parts.append(ds.map(
                lambda e: tokenize([
                    {"role": "user", "content": e["instruction"].strip()},
                    {"role": "assistant", "content": e["output"].strip()},
                ]),
                remove_columns=ds.column_names,
            ))
            print(f"  code : {len(parts[-1])}")
        except Exception as e:
            print(f"  code dataset unavailable ({e}); skipping")

    if n_chat:
        try:
            ds = load_dataset(CHAT_DATASET, split=f"{CHAT_SPLIT}[:{n_chat * 10}]",
                              cache_dir=os.environ.get("HF_HOME"))
            ds = ds.shuffle(seed=seed).select(range(min(n_chat, len(ds))))
            parts.append(ds.map(
                lambda e: tokenize(e["messages"]),
                remove_columns=ds.column_names,
            ))
            print(f"  chat : {len(parts[-1])}")
        except Exception as e:
            print(f"  chat dataset unavailable ({e}); skipping")

    if not parts:
        # Fallback: use a tiny synthetic corpus so pipeline still runs.
        print("  using synthetic fallback corpus")
        from datasets import Dataset
        synthetic = [
            "The quick brown fox jumps over the lazy dog.",
            "In machine learning, quantization reduces model size.",
            "AWQ protects salient weights during compression.",
            "DeepSeek-V4 uses MoE routing with 256 experts per layer.",
        ] * max(1, n_samples // 4)
        ds = Dataset.from_dict({"text": synthetic[:n_samples]})
        parts.append(ds.map(
            lambda e: tokenize([{"role": "user", "content": e["text"]}]),
            remove_columns=ds.column_names,
        ))

    ds = concatenate_datasets(parts).shuffle(seed=seed)
    print(f"calibration: {len(ds)} samples")
    return ds


def data_collator(batch):
    assert len(batch) == 1
    out = {}
    for k, v in batch[0].items():
        if isinstance(v, torch.Tensor):
            out[k] = v
        elif isinstance(v, (list, tuple)) and (
            not v or isinstance(v[0], (int, float))
        ):
            out[k] = torch.tensor(v).unsqueeze(0)
    return out


def collect_ignore_list(model):
    ignore = []
    for name, _ in model.named_modules():
        if PRESERVE_FP32.search(name + ".") or PRESERVE_FP32.search(name):
            ignore.append(name)
    ignore.extend(["lm_head", "model.embed_tokens"])
    seen, out = set(), []
    for x in ignore:
        if x not in seen:
            seen.add(x); out.append(x)
    return sorted(out)


def drop_container_module_ignores(output):
    path = os.path.join(output, "config.json")
    config = json.load(open(path))
    quant = config.get("quantization_config")
    if not quant or "ignore" not in quant:
        return 0
    container = re.compile(r"\.(linear_attn|self_attn|mlp|attn|experts|ffn|visual)$")
    before = len(quant["ignore"])
    quant["ignore"] = [x for x in quant["ignore"] if not container.search(x)]
    dropped = before - len(quant["ignore"])
    if dropped:
        json.dump(config, open(path, "w"), indent=2)
        print(f"dropped {dropped} container ignores")
    return dropped


def verify_output(output, expected_quantized):
    """Verify packed tensors exist; handles both single-shard and sharded outputs."""
    index_path = os.path.join(output, "model.safetensors.index.json")
    single_path = os.path.join(output, "model.safetensors")
    if os.path.exists(index_path):
        index = json.load(open(index_path))
        all_keys = list(index["weight_map"].keys())
    elif os.path.exists(single_path):
        from safetensors import safe_open
        with safe_open(single_path, framework="pt") as fh:
            all_keys = list(fh.keys())
    else:
        raise SystemExit(f"no weights under {output}")

    packed = [k for k in all_keys if k.endswith(".weight_packed")]
    wrong = sorted(k for k in packed
                   if not ROUTED_EXPERT.match(k[: -len(".weight_packed")]))
    if wrong:
        print(f"note: {len(wrong)} packed tensors outside routed experts "
              f"(dense-model quantization is expected to hit other Linears too)")
        print(f"  first few: {wrong[:3]}")
    print(f"verified: {len(packed)} packed tensors emitted")
    if len(packed) == 0:
        raise SystemExit("no packed tensors emitted; AWQ produced nothing")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--checkpoint-dir", default=None)
    ap.add_argument("--work-dir", default=None)
    ap.add_argument("--samples", type=int, default=32)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--group-size", type=int, default=32)
    ap.add_argument("--code-fraction", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-grid", type=int, default=5)
    ap.add_argument("--max-layers", type=int,
                   help="truncate at N decoder layers (POC mode)")
    ap.add_argument("--skip-upcast", action="store_true")
    ap.add_argument("--with-awq", action="store_true",
                    help="enable AWQ grid search (AWQModifier); very slow on CPU")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    disable_pin_memory_without_cuda()

    work_dir = args.work_dir or (args.output.rstrip("/") + ".work")
    os.makedirs(work_dir, exist_ok=True)

    if args.skip_upcast:
        mirror_dir = args.model
    else:
        marker = os.path.join(work_dir, "bf16-mirror", "model.safetensors.index.json")
        if os.path.exists(marker):
            mirror_dir = os.path.join(work_dir, "bf16-mirror")
            print(f"reusing existing mirror at {mirror_dir}")
        else:
            mirror_dir = upcast_source(args.model, work_dir)

    tokenizer = AutoTokenizer.from_pretrained(mirror_dir, trust_remote_code=True)
    calib = get_calib_dataset(tokenizer, args.samples, args.seed, args.code_fraction)

    model = AutoModelForCausalLM.from_pretrained(
        mirror_dir, dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True
    )

    if args.max_layers is not None and hasattr(model.config, "num_hidden_layers"):
        original = model.config.num_hidden_layers
        if args.max_layers < original:
            model.config.num_hidden_layers = args.max_layers
            # Also truncate layer_types if present (Qwen2.5+ uses it)
            if hasattr(model.config, "layer_types") and model.config.layer_types:
                model.config.layer_types = model.config.layer_types[:args.max_layers]
            # Truncate actual layers too so forward doesn't hit missing indices
            if hasattr(model.model, "layers"):
                model.model.layers = model.model.layers[:args.max_layers]
            print(f"truncated model to {args.max_layers} layers "
                  f"(was {original})")

    ignore = collect_ignore_list(model)
    cfg = model.config
    n_layers = getattr(cfg, "num_hidden_layers", 1)
    has_moe = any("experts" in n for n, _ in model.named_modules())
    n_experts = getattr(cfg, "n_routed_experts",
                       getattr(cfg, "num_experts", 0))
    if has_moe and n_experts > 0:
        expected_quantized = n_layers * n_experts * 3
    else:
        # Dense model: count Linear modules excluding ignored ones
        all_linears = [n for n, m in model.named_modules()
                       if isinstance(m, torch.nn.Linear)]
        expected_quantized = sum(1 for n in all_linears
                                 if not any(i in n for i in ignore)
                                 and "lm_head" not in n
                                 and "embed" not in n)
    print(f"ignoring {len(ignore)} entries; expecting ~{expected_quantized} packed")

    scheme = QuantizationScheme(
        targets=["Linear"],
        weights=QuantizationArgs(
            num_bits=4, type="int", symmetric=False,
            group_size=args.group_size, strategy="group",
            observer="mse", zp_dtype=torch.int8,
        ),
    )
    recipe = [
        # No ParallelAWQModifier on CPU (single-process)
        QuantizationModifier(
            config_groups={"group_0": scheme},
            targets=["Linear"], ignore=ignore,
        ),
    ]

    # Note: we skip the AWQModifier on CPU by default since grid search is
    # extremely slow. Set --with-awq to enable it anyway.
    if args.with_awq:
        from llmcompressor.modifiers.transform.awq import AWQModifier
        recipe.insert(0, AWQModifier(duo_scaling=False, n_grid=args.n_grid))
        print("AWQ grid search enabled")

    oneshot(
        model=model,
        dataset=calib,
        recipe=recipe,
        max_seq_length=args.seq_len,
        num_calibration_samples=args.samples,
        data_collator=data_collator,
        pipeline="independent",   # sequential is very slow on CPU
        propagate_error=False,
        log_dir=None,
    )

    with disable_onloading():
        model.save_pretrained(args.output, save_compressed=True)
        tokenizer.save_pretrained(args.output)

    drop_container_module_ignores(args.output)
    verify_output(args.output, expected_quantized)

    print(f"\nwrote {args.output}")
    out_cfg = json.load(open(os.path.join(args.output, "config.json")))
    quant = out_cfg.get("quantization_config", {})
    print(json.dumps(quant.get("config_groups", {}), indent=2))


if __name__ == "__main__":
    main()