# DeepSeek-V4-Flash-0731 — Architecture Reconstruction

Source material:
- `https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/raw/main/config.json`
- `https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/raw/main/inference/model.py`
- `https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/raw/main/inference/kernel.py`
- `https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/raw/main/inference/convert.py`
- `https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/raw/main/model.safetensors.index.json`

This doc reconstructs the architecture purely from those files plus the technical
report linked at `arxiv.org/abs/2606.19348`. Nothing here assumes info beyond
those sources.

## High-level overview

DeepSeek-V4-Flash-0731 is a **304 B-parameter** sparse Mixture-of-Experts
decoder-only Transformer (43 hidden layers + 1 multi-token-prediction head +
1 MQA-style draft head built on layers 40–42) trained primarily for
agentic / tool-use workloads. It ships already heavily quantized:

| Surface | Format | Block-wise scaling |
|---|---|---|
| Outer `torch_dtype` | bfloat16 | — |
| Activations (online at inference) | FP8 e4m3 + UE8M0 scale | 128-element blocks |
| Attention & expert weights (saved) | **FP4 e2m1** + UE8M0 scale | 32-element K-block scales |
| Compressor/HC/Embed/Sinkhorn-related | FP32 with manual casting | n/a |

Total weights breakdown from HF API:
```
BF16        :   1.48 B params  (residuals, surface state)
I64         :   2.33 M params  (probably dsparse index/lookup padding)
F32         :  37.74 M params  (HC coefficients, RMSNorm)
F8_E4M3     :   6.30 B params  (rotary/transition buffers)
I8          : 296.35 B params  (raw byte count of experts)
Total       : 304.18 B params
```

The "I8" figure of 296 B is the byte-count of the FP4 packed expert tensors
(2 FP4 nibbles per byte), divided by 2 nibbles-and-pack-conventionally. Convert
back to logical FP4 parameter count: ~592 B — but functionally, each expert
stores only `dim × inter_dim × 3 (gate/up/down) × 2 bytes/element-equivalent
after packing`, totaling 256 experts × 43 layers × 3 matrices × (4096 × 2048)
× 0.5 byte-per-element-effective ≈ 145 GB at FP4, consistent with reported
304 B total param count.

## Module graph (from `inference/model.py`)

```
Transformer
├── ParallelEmbedding              [vocab=129280 → dim=4096, bf16]
├── ModuleList (n_layers=43)
│   └── Block(i)
│       ├── Attention(i)           [MLA, includes Compressor(i) optionally Indexer(i)]
│       ├── MoE(i)
│       │   ├── Gate(i)            [router + hash routing on first 3 layers]
│       │   ├── ModuleList
│       │   │   └── Expert(j)      [j in 0..n_routed_experts-1; j=0..255]
│       │   │       ├── Linear(w1) [dim → inter_dim=2048, FP4]
│       │   │       ├── Linear(w2) [inter_dim → dim,       FP4]
│       │   │       └── Linear(w3) [dim → inter_dim=2048, FP4]
│       │   └── SharedExperts      [Expert(dim=4096, inter_dim=2048, BF16)]
│       ├── Attn-Norms              [RMSNorm(fp32)]
│       └── HyperConnection mixer  [sinkhorn-normalized fp32 mix]
├── RMSNorm(fp32)                  [head input]
└── ParallelHead                   [dim → vocab, fp32 logits]

Optionally (when dspark_block_size > 0):
└── ModuleList (n_mtp_layers=1, sometimes counted as 3 due to DSpark stages)
    └── DSparkBlock(stage_id)
        ├── DSparkAttention(stage_id)   [extends Attention, uses main_x from target layers]
        ├── DSparkMarkovHead            [Embedding + ParallelHead]
        └── DSparkConfidenceHead        [Linear(fp32)]
```

## Linear subclass — the central abstraction

Everything you'd want to AWQ lives inside this class:

```python
class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=False, dtype=None):
        if dtype == torch.float4_e2m1fn_x2:
            self.weight = nn.Parameter(torch.empty(out, in//2, dtype=float4))
            self.weight.scale = self.scale = nn.Parameter(
                torch.empty(out, in//fp4_block_size, dtype=float8_e8m0fnu)
            )
        elif dtype == torch.float8_e4m3fn:
            self.weight = nn.Parameter(torch.empty(out, in, dtype=float8))
            self.weight.scale = self.scale = nn.Parameter(
                torch.empty((out+bs-1)//bs, (in+bs-1)//bs, dtype=float8_e8m0fnu)
            )
        else:  # bf16 or fp32
            self.weight = nn.Parameter(torch.empty(out, in, dtype=dtype))
            self.scale = None
```

Implications:
- Subclass is `nn.Linear`-compatible → `llm-compressor.AWQModifier(targets=[...])` finds it via isinstance().
- The scale is bound as `weight.scale` (attribute) AND mirrored as `self.scale`
  (parameter). Both names must be registered as `ignore=` to prevent
  `awq-modifier` from trying to "quantize" the scale tensor.
- Upcasting FP4/FP8 → BF16 requires removing the `weight.scale` attr BEFORE
  AWQ runs (otherwise activation capture skips it incorrectly).

## Module-by-module parameter inventory

Module → typical parameter count → dtype → quantization decision

| Module | Param | Shape | Dtype | Decision |
|---|---|---|---|---|
| `embed.weight` | 129280 × 4096 | vocab-projected | bf16 | **SKIP** (often expensive; preserving makes quality recoverable for downstream LoRA) |
| `norm.weight` | 4096 | scalar vector | fp32 | **SKIP** (RT-negligible, breaks numerics) |
| `head.weight` | 129280 × 4096 | vocab-projected | fp32 | **SKIP** (downstream tuning often retunes head; also tied to tie_word_embeddings=False) |
| `attn.wq_a` | 4096 × 1024 | MLP-down-proj | bf16 | **AWQ** |
| `attn.q_norm.weight` | 1024 | RMSNorm | fp32 | **SKIP** |
| `attn.wq_b` | 1024 × 32768 | q heads × head_dim | bf16 | **AWQ** (per-tensor) |
| `attn.wkv` | 4096 × 512 | kv latent | bf16 | **AWQ** |
| `attn.kv_norm.weight` | 512 | RMSNorm | fp32 | **SKIP** |
| `attn.wo_a` | head_dim × n_groups × o_lora_rank | grouped low-rank O | bf16 | **AWQ** |
| `attn.wo_b` | n_groups × o_lora_rank × dim | gather-back-O | bf16 | **AWQ** |
| `attn.attn_sink` | n_local_heads | vector | fp32 | **SKIP** |
| `compressor.ape` | compress_ratio × (coff·head_dim) | position-bias | fp32 | **SKIP** |
| `compressor.wkv` | dim → coff·head_dim | fp32 | **SKIP** (precision-sensitive Sinkhorn input) |
| `compressor.wgate` | dim → coff·head_dim | fp32 | **SKIP** |
| `indexer.wq_b` | q_lora_rank → n_index·head_dim | fp32 | **SKIP** |
| `indexer.weights_proj` | dim → n_index | bf16 | **AWQ** |
| `indexer.compressor.*` | mirrors main compressor | | **SKIP** (cascading KC) |
| `ffn.attn_norm` / `ffn.ffn_norm` | RMSNorm(fp32) | | **SKIP** |
| `moe.gate.weight` | n_routed_experts × dim | bf16 | **SKIP** (router is precision-critical; ~1.0 MB — free) |
| `moe.gate.bias` | n_routed_experts | fp32 | **SKIP** |
| `moe.gate.tid2eid` | vocab × n_active | int32 frozen | **SKIP** |
| `expert.{w1,w2,w3}` × 256 × 43 | 4096×2048 / 2048×4096 / 4096×2048 | FP4 packed | **AWQ** (after upcast to bf16) |
| `shared_experts.{w1,w2,w3}` × 43 | same shape | bf16 | **AWQ** |
| `hc_attn_fn / hc_attn_base / hc_attn_scale` × 43 | (2+hc_mult)*hc_mult × hc_dim, hc_mult, 3 | fp32 sinkhorn params | **SKIP** |
| `hc_ffn_fn / hc_ffn_base / hc_ffn_scale` × 43 | same | fp32 | **SKIP** |
| `hc_head_fn / hc_head_base / hc_head_scale` × 1 | hc_mult × hc_dim, hc_mult, 1 | fp32 | **SKIP** |
| `mtp.{attn,ffn,...}` × 1 stage | mirrors a Block but cheaper | mostly bf16 | **AWQ where Linear/bf16** |
| `mtp.markov_head.{markov_w1,markov_w2}` | vocab × rank, rank × vocab | bf16 (parallel embed+head aliases) | **SKIP** (tiny; risking draft fidelity) |
| `mtp.confidence_head.proj` | dim+rank → 1 | fp32 | **SKIP** |
| `mtp.hc_head_*` | fp32 | | **SKIP** |

(Documented independently in [IGNORE_PATTERNS_DERIVATION.md](./IGNORE_PATTERNS_DERIVATION.md).)

## Layer-wise dispatch

Layers 0..2 use `Gate(hash=True)` (deterministic routing from `tid2eid`).
Layers 3..42 use `Gate(hash=False)` (sigmoid/softmax-score routing).

Layers whose `compress_ratios[i] == 4` carry an `Indexer` submodule (those
with `compress_ratio == 128` carry a non-rotated `Compressor` only;
those with `compress_ratio == 0` carry neither).

Layers 40, 41, 42 (`dspark_target_layer_ids`) feed the DSparkMarkovChain
through their `main_x` projection in the DSpark stage.

## Numerical-precision caveats

1. **`weight.scale` is mirrored.** Assigning to either `weight.scale` or
   `self.scale` mutates both. Casting the model to BF16 must delete both.
2. **HC mix matrices need Sinkhorn.** Even tiny perturbations destabilize
   Sinkhorn iterations and cascade through 4 residual copies → catastrophic.
   Anything HC-related is unconditionally ignored.
3. **Hash-routing int32 lookup** (`tid2eid`) is treated as data, not weights.
   `llm-compressor.AWQModifier` with `targets=[nn.Linear]` should already skip
   it; reinforced by `ignore=` pattern.
4. **Markov head reuses embed & head** (`markov_w1 = embed`, `markov_w2 = head`
   aliased pointers). Saves a few GB; would still be skipped anyway.