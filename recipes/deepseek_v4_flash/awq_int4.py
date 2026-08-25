#!/usr/bin/env python3
"""Quantize DeepSeek-V4-Flash-0731 to AWQ INT4 in compressed-tensors format.

Run through run.sh. See the README in this directory for the reasoning behind
every choice.

Source model notes (from config.json at revision 7872f01b):

* ``architectures=["DeepseekV4ForCausalLM"]``, 43 hidden layers, 1 MTP layer.
* routed experts ship as native FP4 packed tensors with UE8M0 block scales,
  dense attention/MLP weights as FP8 e4m3 with the same scale format. Both are
  *already quantized*: AWQ here is a **requantization** experiment, not a
  first-pass compression. The upcast step materializes ordinary BF16 tensors
  before calibration; see `upcast_source`.
* DSpark speculative head on layers 40/41/42 carries its own markov and
  confidence heads plus a full MoE inside each target layer. transformers does
  not instantiate any of it, so `save_pretrained` silently drops those ~785
  tensors. `graft_dspark_weights` copies them across byte-for-byte.
* Hyper-Connection mixers (`hc_attn_*`, `hc_ffn_*`, `hc_head_*`) normalize via
  Sinkhorn iterations. INT4 noise on them collapses routing; they stay FP32.
* The first three layers route deterministically through a hash lookup
  (`gate.tid2eid`, int32 frozen). Those tables must never be touched.

Two structural differences from the Ornith recipe:

1. experts already exist as per-expert Linears under
   `model.layers.{i}.ffn.experts.{j}.{w1,w2,w3}` — no fused-3D linearization
   is needed, but the custom `Linear` class carries a `.weight.scale`
   sidecar attribute that must be stripped before AWQ sees the layer;
2. the checkpoint's own quantization_config describes FP8 storage, so
   loading with `dtype="auto"` would keep the compressed form. We force
   BF16 by way of a pre-pass that dequantizes to dense tensors and rewrites
   the shard set.
"""

import argparse
import glob
import hashlib
import json
import os
import re
import shutil
import struct

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, SequentialSampler
import llmcompressor
import compressed_tensors
from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
from compressed_tensors.offload import disable_onloading
from datasets import concatenate_datasets, load_dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from transformers import AutoModelForCausalLM, AutoTokenizer

from parallel_awq import ParallelAWQModifier
import resumable_pipeline


# Agentic coding + tool-use traffic mirrors what DeepSeek-V4 was trained for.
CODE_DATASET = "codeparrot/self-instruct-starcoder"
CODE_SPLIT = "curated"
CHAT_DATASET = "HuggingFaceH4/ultrachat_200k"
CHAT_SPLIT = "train_sft"

# Routed-expert Linear modules. Note the ffn.experts.{j}.{w1,w2,w3} naming:
# unlike most HF models, DeepSeek-V4 puts the expert FFNs directly under `ffn`,
# not under `mlp`.
ROUTED_EXPERT = re.compile(
    r"^model\.layers\.\d+\.ffn\.experts\.\d+\.w[123]$"
)

# DSpark speculation head subtree. transformers does not instantiate these.
DSPARK_KEY = re.compile(r"^(model\.)?(layers\.(40|41|42)\.)?dspark\.|^mtp\.")

# Tensors that must remain at their source precision. Quantizing any of these
# breaks either Sinkhorn normalization (hc_*), router balance (gate.*), or
# positional encoding (compressor.ape).
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


def initialize_distributed_xpu():
    """Bind one torchrun rank to each XPU and initialize native XCCL."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        raise SystemExit("this recipe requires Intel XPU")
    if world_size > torch.xpu.device_count():
        raise SystemExit(
            f"WORLD_SIZE={world_size}, but only {torch.xpu.device_count()} XPUs are visible"
        )
    torch.xpu.set_device(local_rank)
    if world_size > 1:
        if not getattr(dist, "is_xccl_available", lambda: False)():
            raise SystemExit("PyTorch XCCL backend is unavailable")
        dist.init_process_group(backend="xccl")
    return local_rank, world_size


def use_xpu_if_available(local_rank):
    """Point llm-compressor's device selection at XPU.

    Same reasoning as the Ornith recipe: AWQ grid search is batched GEMM and
    belongs on an accelerator. `get_main_device` doesn't know about XPU so we
    rebind it globally after import.

    Requires llm-compressor installed with --no-deps, since a normal install
    replaces torch-xpu with a CUDA build.
    """
    if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        print("no XPU visible; llm-compressor will choose its own device")
        return
    import sys
    from llmcompressor.utils import dev

    target = torch.device(f"xpu:{local_rank}")
    dev.get_main_device = lambda: target
    rebound = 0
    for name, mod in list(sys.modules.items()):
        if not name.startswith("llmcompressor"):
            continue
        if getattr(mod, "get_main_device", None) is not None:
            mod.get_main_device = lambda: target
            rebound += 1
    print(f"rank {dist.get_rank() if dist.is_initialized() else 0} selected {target}; "
          f"{torch.xpu.device_count()} devices visible, "
          f"rebound get_main_device in {rebound} modules")


def disable_pin_memory_without_cuda():
    """llm-compressor pins host memory even with no CUDA present.

    Identical rationale to the Ornith recipe. See there for detail.
    """
    if torch.cuda.is_available():
        return
    from llmcompressor.pipelines import cache as _cache

    _cache.IntermediatesCache._pin_intermediate = classmethod(
        lambda cls, intermediate: None
    )
    print("pin_memory disabled: no CUDA present, pinning is a CUDA facility")


def upcast_source(source_dir, work_dir):
    """Dequantize FP4/FP8 storage into a dense-BF16 mirror.

    DeepSeek-V4-Flash ships pre-quantized: experts are NVFP4-packed uint8 with
    UE8M0 block scales, dense layers are FP8-e4m3 with the same scale format.
    Loading with `dtype="auto"` preserves that packing and AWQ cannot operate
    on it — the packed form is opaque bytes to the quantizer.

    This pass streams every safetensors shard once, expands packed tensors to
    dense BF16, drops sidecar `.scale` entries, and writes a fresh shard set
    under `{work_dir}/bf16-mirror`. Peak host RAM ≈ 2× largest input shard.

    The output is a standard HF checkpoint directory loadable with
    `AutoModelForCausalLM.from_pretrained(..., dtype=torch.bfloat16)`.
    """
    from safetensors import safe_open
    from safetensors.torch import save_file

    # IEEE-754 binary4 NF4 lookup table. Index is the raw 4-bit nibble.
    # Values follow the NVFP4 convention used by the upstream checkpoint.
    E2M1 = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
         -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        dtype=torch.bfloat16,
    )

    def unpack_fp4(packed_u8):
        """Each byte packs two FP4 values, low nibble first."""
        lo = (packed_u8 & 0xF).long()
        hi = ((packed_u8 >> 4) & 0xF).long()
        lo_vals = E2M1[lo]
        hi_vals = E2M1[hi]
        return torch.stack([hi_vals, lo_vals], dim=-1).flatten(-2)

    def detect_and_dequant(key, tensor, scale):
        """Dispatch based on key pattern + dtype. Returns dense tensor."""
        if key.endswith(".scale"):
            return None  # caller filters
        # FP4-packed routed/shared experts: uint8 storage, w{1,2,3}.weight key
        if (
            tensor.dtype == torch.uint8
            and ".experts." in key
            and re.search(r"\.w[123]\.weight$", key)
        ):
            dense = unpack_fp4(tensor)
            if scale is not None:
                # Broadcast per-block scales back onto the expanded shape.
                # Scale layout is [out_features, in_features / 32].
                block = 32
                out_f, in_f = dense.shape
                s = scale.to(torch.bfloat16)
                if s.shape == (out_f, in_f // block):
                    full = s.repeat_interleave(block, dim=1)[:, :in_f]
                    dense = dense * full
            return dense.to(torch.bfloat16)
        # FP8 dense weights (attention projections, shared experts, etc.)
        if tensor.dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz):
            bf = tensor.to(torch.bfloat16)
            if scale is not None:
                bs = 128
                out_f, in_f = bf.shape
                rs = (out_f + bs - 1) // bs
                cs = (in_f + bs - 1) // bs
                s = scale.to(torch.bfloat16)
                if s.shape == (rs, cs):
                    full = s.repeat_interleave(bs, dim=0).repeat_interleave(bs, dim=1)
                    bf = bf * full[:out_f, :in_f]
            return bf
        # Everything else (norms, hc_*, embed, head, router): straight cast
        if tensor.is_floating_point():
            return tensor.to(torch.bfloat16)
        return tensor  # int32 tid2eid etc. stays as-is

    mirror_dir = os.path.join(work_dir, "bf16-mirror")
    os.makedirs(mirror_dir, exist_ok=True)

    index_path = os.path.join(source_dir, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        raise SystemExit(f"no model.safetensors.index.json under {source_dir}")
    weight_map = json.load(open(index_path))["weight_map"]
    shards = sorted(set(weight_map.values()))
    print(f"upcast: {len(shards)} input shards")

    new_weight_map = {}
    new_shard_sizes = {}
    current_tensors = {}
    current_bytes = 0
    SOFT_LIMIT = 5 * 1024 ** 3  # 5 GiB per output shard
    out_idx = 1
    total_in = len(shards)

    def flush():
        nonlocal current_tensors, current_bytes, out_idx
        if not current_tensors:
            return
        name = f"model-{out_idx:05d}-of-{total_in:05d}.safetensors"
        save_file(current_tensors, os.path.join(mirror_dir, name),
                  metadata={"format": "pt"})
        new_shard_sizes[name] = current_bytes
        out_idx += 1
        current_tensors = {}
        current_bytes = 0

    for shard_name in shards:
        path = os.path.join(source_dir, shard_name)
        with safe_open(path, framework="pt", device="cpu") as fh:
            keys = list(fh.keys())
            scale_keys = {
                k[: -len(".scale")]: k
                for k in keys if k.endswith(".scale")
            }
            for key in keys:
                if key.endswith(".scale"):
                    continue  # dropped; folded into dequantized weight
                scale = fh.get_tensor(scale_keys[key]) if key in scale_keys else None
                tensor = fh.get_tensor(key)
                dense = detect_and_dequant(key, tensor, scale)
                sz = dense.numel() * dense.element_size()
                if current_bytes + sz > SOFT_LIMIT and current_tensors:
                    flush()
                current_tensors[key] = dense.contiguous()
                current_bytes += sz
                predicted = f"model-{out_idx:05d}-of-{total_in:05d}.safetensors"
                new_weight_map[key] = predicted
        print(f"  processed {shard_name}")
    flush()

    # Rewrite index
    new_index = {
        "metadata": {"total_size": sum(new_shard_sizes.values())},
        "weight_map": new_weight_map,
    }
    with open(os.path.join(mirror_dir, "model.safetensors.index.json"), "w") as fh:
        json.dump(new_index, fh, indent=2)

    # Copy config + tokenizer, stripping quantization hints
    cfg_path = os.path.join(source_dir, "config.json")
    cfg = json.load(open(cfg_path))
    cfg.pop("quantization_config", None)
    cfg.pop("expert_dtype", None)
    cfg["torch_dtype"] = "bfloat16"
    cfg["_upcast_origin"] = f"deepseek_v4_flash.awq_int4.upcast_source from {source_dir}"
    with open(os.path.join(mirror_dir, "config.json"), "w") as fh:
        json.dump(cfg, fh, indent=2)

    for name in ("tokenizer.json", "tokenizer_config.json",
                 "generation_config.json", "special_tokens_map.json"):
        src = os.path.join(source_dir, name)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(mirror_dir, name))

    total_gib = sum(new_shard_sizes.values()) / 2 ** 30
    print(f"upcast done: {total_gib:.1f} GiB written to {mirror_dir}")
    return mirror_dir


def get_calib_dataset(tokenizer, n_samples, seed, code_fraction):
    """Chat-templated code instructions mixed with general instructions.

    Same rationale as Ornith: agentic coding is the primary workload. Sample
    count matters more than usual because only ~6 of 256 experts activate per
    token, so tail experts need many samples before their activation stats
    stabilize.
    """
    n_code = int(n_samples * code_fraction)
    n_chat = n_samples - n_code
    parts = []

    def tokenize(messages):
        out = tokenizer.apply_chat_template(
            messages, tokenize=True, return_dict=True, add_generation_prompt=False
        )
        ids = out["input_ids"]
        while isinstance(ids, (list, tuple)) and ids and isinstance(ids[0], (list, tuple)):
            ids = ids[0]
        return {"input_ids": list(ids)}

    if n_code:
        ds = load_dataset(CODE_DATASET, split=f"{CODE_SPLIT}[:{n_code * 10}]")
        ds = ds.shuffle(seed=seed).select(range(min(n_code, len(ds))))
        parts.append(
            ds.map(
                lambda e: tokenize(
                    [
                        {"role": "user", "content": e["instruction"].strip()},
                        {"role": "assistant", "content": e["output"].strip()},
                    ]
                ),
                remove_columns=ds.column_names,
            )
        )
        print(f"  code : {len(parts[-1])} from {CODE_DATASET}")

    if n_chat:
        ds = load_dataset(CHAT_DATASET, split=f"{CHAT_SPLIT}[:{n_chat * 10}]")
        ds = ds.shuffle(seed=seed).select(range(min(n_chat, len(ds))))
        parts.append(
            ds.map(lambda e: tokenize(e["messages"]), remove_columns=ds.column_names)
        )
        print(f"  chat : {len(parts[-1])} from {CHAT_DATASET}")

    ds = concatenate_datasets(parts).shuffle(seed=seed)
    first = ds[0]["input_ids"]
    print(f"calibration: {len(ds)} samples, first sample {len(first)} tokens, "
          f"element type {type(first[0]).__name__}")
    return ds


def data_collator(batch):
    """One sample per batch, tensors as-is. Same rationale as Ornith."""
    assert len(batch) == 1
    out = {}
    for key, value in batch[0].items():
        if isinstance(value, torch.Tensor):
            out[key] = value
        elif isinstance(value, (list, tuple)) and (
            not value or isinstance(value[0], (int, float))
        ):
            out[key] = torch.tensor(value).unsqueeze(0)
    return out


def collect_ignore_list(model):
    """Every Linear that must stay full precision.

    Includes all norm/hc/router/compressor parameters (they aren't Linears but
    listing them explicitly makes the emitted quantization_config honest about
    scope), plus lm_head and embeddings which we preserve for LoRA compatibility.
    """
    ignore = []
    for name, module in model.named_modules():
        if PRESERVE_FP32.search(name + "."):
            ignore.append(name)
            continue
        if isinstance(module, torch.nn.Linear):
            # Any Linear whose name hits a preserved pattern stays BF16 too
            if PRESERVE_FP32.search(name):
                ignore.append(name)
    # Explicit adds
    ignore.extend([
        "lm_head",
        "model.embed_tokens",
    ])
    # Deduplicate while preserving order
    seen = set()
    out = []
    for x in ignore:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return sorted(out)


def drop_container_module_ignores(output):
    """Remove non-Linear parent modules from the emitted ignore list."""
    path = os.path.join(output, "config.json")
    config = json.load(open(path))
    quant = config.get("quantization_config")
    if not quant or "ignore" not in quant:
        return 0

    container = re.compile(
        r"\.(linear_attn|self_attn|mlp|attn|experts|ffn|visual|shared_experts)$"
    )
    before = len(quant["ignore"])
    quant["ignore"] = [x for x in quant["ignore"] if not container.search(x)]
    dropped = before - len(quant["ignore"])
    if dropped:
        json.dump(config, open(path, "w"), indent=2)
        print(f"dropped {dropped} container-module ignore entries")
    return dropped


def graft_dspark_weights(source, output):
    """Copy DSpark speculation head tensors from the source snapshot.

    transformers instantiates no DSpark module on DeepseekV4ForCausalLM, so
    `save_pretrained` cannot write those subtrees even though they exist in
    the raw checkpoint. Layers 40/41/42 carry a full markov head, confidence
    head, main_proj projection, hc_head_* mixers, plus an entire MoE block
    inside each DSpark stage.

    Without this the artifact silently loses speculative decoding.
    """
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
        print(f"warning: no DSpark/MTP tensors found in {source}")
        return 0

    dtypes = sorted({str(t.dtype) for t in dspark.values()})
    if dtypes != ["torch.bfloat16"]:
        print(f"warning: DSpark dtypes include {dtypes}; expected uniform bf16")

    shard_name = "model-dspark.safetensors"
    save_file(dspark, os.path.join(output, shard_name), metadata={"format": "pt"})

    index_path = os.path.join(output, "model.safetensors.index.json")
    index = json.load(open(index_path))
    index["weight_map"].update({key: shard_name for key in dspark})
    index["metadata"]["total_size"] = index["metadata"].get("total_size", 0) + sum(
        t.numel() * t.element_size() for t in dspark.values()
    )
    json.dump(index, open(index_path, "w"), indent=2)
    print(f"grafted {len(dspark)} DSpark/MTP tensors into {shard_name}")
    return len(dspark)


def verify_output(output, expected_quantized):
    """Refuse an artifact whose quantized set is not exactly the routed experts."""
    index = json.load(open(os.path.join(output, "model.safetensors.index.json")))
    packed = [k for k in index["weight_map"] if k.endswith(".weight_packed")]
    wrong = sorted(k for k in packed
                   if not ROUTED_EXPERT.match(k[: -len(".weight_packed")]))
    if wrong:
        raise SystemExit(
            f"{len(wrong)} packed tensors outside the routed experts, "
            f"first: {wrong[:3]}"
        )
    if len(packed) != expected_quantized:
        raise SystemExit(
            f"expected {expected_quantized} packed tensors, got {len(packed)}"
        )
    dspark = [k for k in index["weight_map"]
              if DSPARK_KEY.match(k) or k.startswith("mtp.")]
    if any(k.endswith(".weight_packed") for k in dspark):
        raise SystemExit("DSpark/MTP tensors were quantized; they must stay bf16")
    print(f"verified: {len(packed)} packed routed-expert tensors, "
          f"{len(dspark)} bf16 DSpark/MTP tensors")


def calibration_sha256(calib):
    """Hash actual token IDs so resumed runs can't drift."""
    digest = hashlib.sha256()
    for row in calib:
        ids = row["input_ids"]
        digest.update(struct.pack("<Q", len(ids)))
        for token_id in ids:
            digest.update(struct.pack("<I", token_id))
    return digest.hexdigest()


def run_identity(args, world_size, calib_sha256, mirror_dir):
    """Fields which must be identical when a layer checkpoint is resumed."""
    config_path = os.path.join(mirror_dir, "config.json")
    with open(config_path, "rb") as handle:
        config_sha256 = hashlib.sha256(handle.read()).hexdigest()
    return {
        "source": os.path.realpath(args.model),
        "mirror": os.path.realpath(mirror_dir),
        "source_config_sha256": config_sha256,
        "scheme": "awq-int4-asymmetric-mse-routed-experts-requantized",
        "parallelism": "replicated-data-grid-and-expert-v3",
        "group_size": args.group_size,
        "samples": args.samples,
        "seq_len": args.seq_len,
        "code_fraction": args.code_fraction,
        "seed": args.seed,
        "n_grid": args.n_grid,
        "calibration_sha256": calib_sha256,
        "world_size": world_size,
        "llmcompressor": llmcompressor.__version__,
        "compressed_tensors": compressed_tensors.__version__,
        "torch": torch.__version__,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="original DeepSeek-V4-Flash snapshot (pre-upcast)")
    ap.add_argument("--output", required=True)
    ap.add_argument("--work-dir", default=None,
                    help="scratch dir for BF16 mirror; defaults to <output>.work")
    ap.add_argument("--samples", type=int, default=512)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--group-size", type=int, default=32)
    ap.add_argument("--code-fraction", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-grid", type=int, default=20)
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--stop-after-layer", type=int)
    ap.add_argument("--skip-upcast", action="store_true",
                    help="assume --model already points at a BF16 mirror")
    args = ap.parse_args()

    local_rank, world_size = initialize_distributed_xpu()
    rank = dist.get_rank() if dist.is_initialized() else 0
    torch.manual_seed(args.seed)
    use_xpu_if_available(local_rank)
    disable_pin_memory_without_cuda()

    if torch.xpu.is_available():
        total = sum(
            torch.xpu.get_device_properties(i).total_memory
            for i in range(torch.xpu.device_count())
        )
        print(
            f"distributed AWQ rank {rank}/{world_size}: "
            f"{torch.xpu.device_count()} XPU devices, {total / 2**30:.1f} GiB total"
        )

    work_dir = args.work_dir or (args.output.rstrip("/") + ".work")
    os.makedirs(work_dir, exist_ok=True)

    if args.skip_upcast:
        mirror_dir = args.model
        print(f"skipping upcast; using {mirror_dir} directly")
    else:
        mirror_marker = os.path.join(work_dir, "bf16-mirror",
                                     "model.safetensors.index.json")
        if os.path.exists(mirror_marker):
            mirror_dir = os.path.join(work_dir, "bf16-mirror")
            print(f"reusing existing mirror at {mirror_dir}")
        else:
            mirror_dir = upcast_source(args.model, work_dir)

    tokenizer = AutoTokenizer.from_pretrained(mirror_dir, trust_remote_code=True)
    calib = get_calib_dataset(tokenizer, args.samples, args.seed, args.code_fraction)
    calib_digest = calibration_sha256(calib)
    calib_loader = DataLoader(
        calib,
        batch_size=1,
        sampler=SequentialSampler(calib),
        collate_fn=data_collator,
    )
    print(f"rank {rank}: replicated {len(calib_loader)} calibration batches, "
          f"sha256={calib_digest}")

    model = AutoModelForCausalLM.from_pretrained(
        mirror_dir, dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True
    )

    ignore = collect_ignore_list(model)
    cfg = model.config
    n_layers = getattr(cfg, "num_hidden_layers", 43)
    n_experts = getattr(cfg, "n_routed_experts", 256)
    expected_quantized = n_layers * n_experts * 3
    print(f"ignoring {len(ignore)} full-precision Linears/modules, "
          f"expecting {expected_quantized} packed routed-expert Linears")

    scheme = QuantizationScheme(
        targets=["Linear"],
        weights=QuantizationArgs(
            num_bits=4,
            type="int",
            symmetric=False,
            group_size=args.group_size,
            strategy="group",
            observer="mse",
            zp_dtype=torch.int8,
        ),
    )

    recipe = [
        ParallelAWQModifier(duo_scaling=False, n_grid=args.n_grid),
        QuantizationModifier(
            config_groups={"group_0": scheme}, targets=["Linear"], ignore=ignore
        ),
    ]

    resumable_pipeline.configure(
        checkpoint_dir=args.checkpoint_dir,
        identity=run_identity(args, world_size, calib_digest, mirror_dir),
        layer_class="DeepseekV4DecoderLayer",
        stop_after_layer=args.stop_after_layer,
    )

    oneshot(
        model=model,
        dataset=calib_loader,
        recipe=recipe,
        max_seq_length=args.seq_len,
        num_calibration_samples=args.samples,
        data_collator=data_collator,
        sequential_targets=["DeepseekV4DecoderLayer"],
        pipeline="resumable_sequential",
        propagate_error=True,
        shuffle_calibration_samples=False,
        log_dir=None,
    )

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    if rank != 0:
        return

    with disable_onloading():
        model.save_pretrained(args.output, save_compressed=True)
        tokenizer.save_pretrained(args.output)

    for name in ("preprocessor_config.json", "processor_config.json",
                 "chat_template.jinja"):
        src = os.path.join(args.model, name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(args.output, name))
            print(f"copied {name}")

    drop_container_module_ignores(args.output)
    graft_dspark_weights(args.model, args.output)
    verify_output(args.output, expected_quantized)

    print(f"\nwrote {args.output}")
    cfg_out = json.load(open(os.path.join(args.output, "config.json")))
    quant = cfg_out.get("quantization_config", {})
    print(json.dumps(quant.get("config_groups", {}), indent=2))
    print(f"ignore entries: {len(quant.get('ignore', []))}")


if __name__ == "__main__":
    main()
