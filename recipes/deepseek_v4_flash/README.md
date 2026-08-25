# DeepSeek-V4-Flash-0731 recipe

Quantizes `deepseek-ai/DeepSeek-V4-Flash-0731` (304 B MoE) to AWQ INT4 in
compressed-tensors format, targeting vLLM / SGLang serving.

This is the second verified model-specific recipe. It is a **requantization**
experiment: the source checkpoint ships as native FP4 packed experts plus FP8
dense weights with UE8M0 block scales. The first pipeline stage materializes a
dense BF16 mirror; the standard AWQ sweep then runs against that mirror.

## Source contract

From `config.json` at revision `7872f01b`:

| Field | Value | Consequence |
|---|---|---|
| `architectures` | `["DeepseekV4ForCausalLM"]` | custom arch class, not registered upstream in llm-compressor's mapping table |
| `model_type` | `"deepseek_v4"` | same |
| `torch_dtype` | `bfloat16` | outer container dtype |
| `expert_dtype` | `"fp4"` | routed + shared expert FFNs stored as NVFP4-packed uint8 |
| `quantization_config` | fp8/e4m3/ue8m0, block 128×128 | attention + dense MLP projections are FP8 |
| `n_routed_experts` | 256 | per-layer MoE size |
| `num_experts_per_tok` | 6 | sparse routing |
| `num_hidden_layers` | 43 (+1 MTP layer via `num_nextn_predict_layers`) | decoder depth |
| `num_hash_layers` | 3 | first 3 layers route deterministically via `gate.tid2eid` (int32 frozen lookup — never quantize) |
| `hc_mult` | 4 | Hyper-Connections residual width |
| `dspark_target_layer_ids` | `[40, 41, 42]` | DSpark speculative head layers |
| `max_position_embeddings` | 1048576 | YaRN ×16 million-token context |

Total on-disk footprint: ~167 GiB across 48 safetensors shards.

## What this recipe does differently from Ornith

### 1. Requantization pre-pass (`upcast_source`)

The upstream Linear subclass carries a `.weight.scale` sidecar attribute and
the underlying tensor is packed bytes, not floats. Loading with
`dtype="auto"` preserves that packing and AWQ cannot operate on it.

`upcast_source` streams each shard once:

* unpacks NVFP4 nibble pairs into dense values via the IEEE-754 binary4 E2M1
  table, multiplies by the broadcast UE8M0 block scales;
* dequantizes FP8-e4m3 dense weights by multiplying through their sidecar scales;
* drops every `.scale` key from the output;
* writes fresh ~5 GiB shards under `$WORK/bf16-mirror`.

Peak host RAM ≈ 2× largest input shard (~12 GiB). Total mirror size ~600 GiB.
The mirror is reused if it already exists so interrupted runs don't pay the
cost twice.

### 2. No fused-MoE linearization needed

Unlike Ornith 1.5 which stores experts as fused 3D parameters
(`mlp.experts.gate_up_proj`), DeepSeek-V4 already exposes them as per-expert
Linears under `model.layers.{i}.ffn.experts.{j}.{w1,w2,w3}`. The AWQ
`targets=["Linear"]` matcher finds all of them directly.

Note the path prefix: DeepSeek-V4 puts the MoE under `ffn`, not `mlp`. The
routed-expert regex in `awq_int4.py` reflects that.

### 3. Hyper-Connection parameters stay FP32

`hc_attn_{fn,base,scale}`, `hc_ffn_{fn,base,scale}`, `hc_head_{fn,base,scale}`
normalize via Sinkhorn iterations over the 4-wide residual stream. INT4 noise
on any of these collapses routing within a few layers. They are added to the
ignore list unconditionally.

### 4. Hash-routing tables are data, not weights

`model.layers.{0,1,2}.gate.tid2eid` is an int32 lookup of shape
[vocab_size × num_experts_per_tok] used for deterministic routing on the
first three layers. Quantizing it would corrupt token→expert assignment for
those layers. Explicitly ignored.

### 5. DSpark head grafting

transformers does not instantiate DSpark modules on `DeepseekV4ForCausalLM`,
so `save_pretrained` silently drops those subtrees even though they exist in
the source snapshot. Layers 40/41/42 carry markov heads, confidence heads,
`main_proj` projections, `hc_head_*` mixers, plus an entire MoE inside each
DSpark stage. Roughly 785 tensors / ~1.7 GiB total.

`graft_dspark_weights` copies them byte-for-byte into the output after the
main save. Without this step the artifact loses speculative decoding, and the
loss is invisible until someone enables DSpark and the weights aren't there.

### 6. Calibration mix

Agentic coding is DeepSeek-V4's primary workload, so code instructions
dominate the calibration set at 70% by default (`--code-fraction`). Chat
instructions fill the remainder. Both are chat-templated before tokenization.

Sample count matters more here than for a dense model because only 6 of 256
experts activate per token — tail experts need many samples to accumulate
useful activation statistics.

## Running

Dry run first (small sample count, short sequences):

```bash
MODEL=/path/to/deepseek-v4-flash-snapshot \
WORK=$PWD/.work \
GPUS=1 SAMPLES=8 SEQ_LEN=64 N_GRID=2 \
bash recipes/deepseek_v4_flash/run.sh
```

Full run (defaults tuned for a single B300 node with 8 accelerators):

```bash
MODEL=/path/to/deepseek-v4-flash-snapshot \
WORK=$PWD/.work \
GPUS=8 \
bash recipes/deepseek_v4_flash/run.sh
```

Environment requirements:

* `/dev/dri` visible for Intel XPU passthrough
* `render` group membership for non-root GPU access
* Docker with `--ipc=host --shm-size=64g` for XCCL collective scratch space
* ≥600 GiB free under `$WORK` for the BF16 mirror
* ≥200 GiB free under `$OUTPUT` for the packed artifact

Approximate wall-clock on 8× B300: upcast ~45 min, AWQ sweep ~6 h,
packing + verification ~30 min.

## Output verification

The runner refuses to emit an artifact unless:

1. every `.weight_packed` name matches the routed-expert regex exactly;
2. the count equals `num_hidden_layers × n_routed_experts × 3`;
3. no DSpark/MTP tensor was quantized;
4. no container-module ignore entry survived post-processing.

Any mismatch aborts with an explicit message rather than producing garbage.

## Known limitations

* Only the routed experts are quantized. Dense attention/MLP projections
  remain BF16 in the output. Extending coverage requires re-deriving the
  ignore list against the serving stack's kernel support matrix.
* Group size defaults to 32 (matching the reference AWQ artifacts for other
  MoE models). Lowering to 16 improves quality at ~2× scale-storage cost.
* No FP8→INT4 kernel benchmark was run; throughput gains vs the source FP8
  checkpoint have not been measured on real traffic.
