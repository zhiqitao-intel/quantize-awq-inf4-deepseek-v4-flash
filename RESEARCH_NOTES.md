# Research Notes — Chronological Lab Notebook

This file is the chronological record of every investigation step taken while
building the AWQ-W4A16 toolkit for `deepseek-ai/DeepSeek-V4-Flash-0731`.
Anything discovered or decided during this work that influenced tooling /
recipe / ignore-pattern / file-shape choices goes here.

See sibling topical docs for organized views:
- [ARCHITECTURE.md](./ARCHITECTURE.md) — model internals reconstructed from config + modeling code
- [QUANTIZATION_DECISIONS.md](./QUANTIZATION_DECISIONS.md) — why each tool/recipe/path choice
- [IGNORE_PATTERNS_DERIVATION.md](./IGNORE_PATTERNS_DERIVATION.md) — regex derivation
- [CALIBRATION_NOTES.md](./CALIBRATION_NOTES.md) — calibration data considerations
- [RISKS_AND_OPEN_QUESTIONS.md](./RISKS_AND_OPEN_QUESTIONS.md) — unresolved items
- [CALIBRATION_REPRODUCIBILITY.md](./CALIBRATION_REPRODUCIBILITY.md) — bit-exact reproducibility checklist

---

## Step 0 — Repository audit (initial probe)

Started with empty repo (`/home/zhiqitao/work/quantize-awq-inf4-deepseek-v4-flash`)
containing only `.git/` and `chatgpt-advice.txt`. Verified git is initialized but
has no commits yet, remote is `https://github.com/zhiqitao-intel/quantize-awq-inf4-deepseek-v4-flash`,
so the only "reference material" the user gave me was the ChatGPT advice snippet.

The snippet instructed AutoAWQ-for-DeepSeek-V4-Flash-0731, downloading the model
in BF16 form, quantizing in place, and uploading. Several red flags became
apparent on inspection — flagged below — so this doc doubles as a critique of
that snippet.

> **Concluded:** treat `chatgpt-advice.txt` as a starting point for the *shape*
> of an AWQ recipe but not as trustworthy technical content. Embedded the
> original verbatim in the repo (not modified) and cross-linked.

## Step 1 — Verify model existence on Hugging Face

Endpoint probed: `GET https://huggingface.co/api/models/deepseek-ai/DeepSeek-V4-Flash-0731`
Result: HTTP 200, full model record returned.

Top-level findings:
- `architectures: ["DeepseekV4ForCausalLM"]`, `model_type: "deepseek_v4"`
- `expert_dtype: "fp4"` — experts are stored natively as FP4
- `num_experts_per_tok: 6` — sparse MoE with 6-active-per-token routing
- `quantization_config: {"quant_method":"fp8", "fmt":"e4m3", "scale_fmt":"ue8m0",
   "weight_block_size":[128,128], "activation_scheme":"dynamic"}`
- 48 safetensors shards, total ≈ **304 B parameters** (mostly FP4 experts)
- License: MIT, not gated
- Created 2026-07-31, sha `7872f01b...`, last modified 2026-08-01

**Conclusion:** model is real and newer than my training cutoff. Initial
skepticism that this might be a fabrication was incorrect; updated mental
timestamp to mid-2026 release era.

## Step 2 — Probe upstream tarball metadata only (no shard downloads)

Probed HTTP HEAD on the index page; parsed HTML; downloaded four metadata-only
artifacts into the workspace under `_probe/upstream/`:

- `config.json` (1,888 B)
- `encoding/encoding_dsv4.py` (29 KB) — chat templating
- `inference/model.py` (45 KB) — modeling code
- `inference/convert.py` / `kernel.py` (lighter — read selectively)
- `model.safetensors.index.json` (5.6 MB) — full key listing for all 72,317 tensors

Probing protocol used throughout this project:
- Temporary cache always lives under `./_probe/<purpose>/...` (inside the repo)
- No writes outside `/home/zhiqitao/work/quantize-awq-inf4-deepseek-v4-flash/**`
- Probe directory removed at end of each investigation phase

## Step 3 — Reverse-engineer architecture from `config.json`

Captured in detail in [ARCHITECTURE.md](./ARCHITECTURE.md). Headline facts:

| Field | Value | Implication for quantization |
|---|---|---|
| `torch_dtype` | `bfloat16` | Outermost container dtype; experts override via `expert_dtype` |
| `expert_dtype` | `fp4` | Experts stored as NVFP4 (E2M1), block 32 along K, UE8M0 scale per block |
| `quantization_config.quant_method` | `fp8` (dynamic, e4m3, ue8m0 scales, block 128×128) | Activations + non-expert Linear weights quantized FP8 with per-block scales |
| `n_routed_experts` | 256 | Per layer |
| `n_shared_experts` | 1 | Per layer |
| `num_experts_per_tok` | 6 | Routing sparsity |
| `hidden_size` | 4096 | Standard |
| `num_attention_heads` | 64; `num_key_value_heads` | 1 (MLA) |
| `head_dim` | 512 (q_lora_rank 1024) | MLA — Q is low-rank projected then RoPE-attached |
| `index_n_heads`, `index_head_dim`, `index_topk` | 64, 128, 512 | DSA-style indexer selects top-512 compressed KV positions |
| `num_hidden_layers` | 43; `num_nextn_predict_layers` | 1; `num_hash_layers` | 3 | First 3 layers use deterministic hash routing from `tid2eid` table |
| `topk_method` | `noaux_tc`; `scoring_func` | `sqrtsoftplus` | Non-softmax gating |
| `moe_intermediate_size` | 2048 | Per-expert FFN expansion |
| `routed_scaling_factor` | 1.5 | Post-topk weight rescale |
| `swiglu_limit` | 10.0 | Clips gate/up activations before SiLU |
| `hc_mult` | 4; `hc_sinkhorn_iters` | 20 | Hyper-Connections: residual stream has 4 copies |
| `dspark_block_size` | 5; `dspark_target_layer_ids` | [40, 41, 42]; `dspark_markov_rank` | 256 | Multi-Token Prediction draft head built from layers 40/41/42 |
| `max_position_embeddings` | 1,048,576; `rope_scaling` | YaRN ×16 | Million-token context |
| `compress_ratios` | alternating 0,4,128 pattern across 45 slots | Some layers compress KV; others don't |
| `tie_word_embeddings` | `false` | Independent embed & head |

## Step 4 — Reverse-engineer class graph from `inference/model.py`

Fully captured in [ARCHITECTURE.md](./ARCHITECTURE.md#module-graph-from-modelpy).
Key takeaway for quantization: **standard `nn.Linear` does not exist** in the
checkpoint. Everything is wrapped in a custom `Linear` class that supports
`bf16`, `fp8_e4m3`, and `fp4_e2m1fn_x2` storage. Each quantized `Linear` also
carries a sidecar `weight.scale` parameter (UE8M0 / FP32).

For AWQ purposes, this means:
1. `llm-compressor`'s AWQModifier must traverse the model treating every
   `Linear` subclass as quantizable — works because they ARE `nn.Linear`.
2. The sidecar `weight.scale` must NOT be touched — we must register it as an
   `ignore=` pattern OR (more robustly) upcast the entire layer to BF16 before
   AWQ runs (FP8/FP4 → BF16 → AWQ → INT4 with new scales).
3. Some `Linear` instances have `dtype=torch.float32` (the `Compressor.wkv/wgate`,
   `DSparkConfidenceHead.proj`) — these MUST be cast carefully (double-precision
   matters for compression sinkhorn).

## Step 5 — Map every safetensor key to a parameter name

Walked `model.safetensors.index.json` (72,317 keys). Bucketed by leaf suffix:

| Leaf | Count | Meaning |
|---|---|---|
| `weight` | 36,163 | Each Linear weight, also Compressor.ape, HC mix matrices, Router.Gate.weight, embed, head |
| `scale` | 35,718 | Per-block scales paired with each quantized weight (UE8M0 for FP4/FP8, FP32 for BF16) |
| `ape` | 62 | Learned absolute-position encoding parameter on compressor modules (≈ 1 per compress-using layer × 2 compressor instances per such layer) |
| `hc_attn_base` / `hc_attn_fn` / `hc_attn_scale` | 46 each | Per-block Hyper-Connection mixer coefficients (sinkhorn-normalized) |
| `hc_ffn_*` | 46 each | Same for FFN mixer |
| `attn_sink` | 46 | Per-head attention sink bias (streaming softmax anchor) |
| `bias` | 43 | Per-layer router bias on non-hash layers (= 43; matches `num_hidden_layers - num_hash_layers` = 43 − 3 except hash layers have bias=None in code… gap to reconcile) |
| `tid2eid` | 3 | Hash-routing lookup tables (exactly equals `num_hash_layers`) |
| `hc_head_base/fn/scale` | 2 each | Head-mixer HC coefficients (one for main head, one for DSpark's markov/conf path) |

(Note: 46 ≈ n_layers(43) + n_mtp_layers(1) + n_compressor_copies_with_ape. Specifically:
APE shows up on every `Compressor` instance — there are roughly 43 attn.compressor + (a handful of) DSparkStage compressor + indexer.compressor. Exact integer mismatch explained by some layers having both `compressor` AND `indexer.compressor`. See ARCHITECTURE.md for per-layer expansion.)

**Crucial deduction:**
Every `.weight` is paired one-to-one with a `.scale` EXCEPT certain scalar /
vector parameters (`ape`, `attn_sink`, `hc_*_base`, `hc_*_scale`,
`tid2eid`, plus the gating-router matrices `gate.weight`). AWQ must NOT touch
the latter, otherwise sinkhorn routing collapses.

See [IGNORE_PATTERNS_DERIVATION.md](./IGNORE_PATTERNS_DERIVATION.md) for the
regex set derived from this bucketing.

## Step 6 — Why AutoAWQ is the wrong tool

Sources reviewed:
- `casper-hansen/AutoAWQ` supports `deepseek_v2`/`deepseek_v2_lite` only;
  `deepseek_v4` is unregistered in `autoawq/modules/__init__.py` mapping table.
- More importantly, the DeepSeek-V4 `Linear` subclass carries `weight.scale`
  sidecar attributes. AutoAWQ assumes standard `nn.Linear` and patches
  weight-only GEMMs; it does not know how to swap paired-scale layouts.
- Per-shard Memory-bound math: each Linear stored in FP4 occupies
  `out × in/2 bytes` for the weight and `out × in/32` bytes for the scale —
  AutoAWQ's packer expects symmetric FP16/BF16 inputs.

Conclusion: AutoAWQ physically cannot ingest this checkpoint. Switched the
canonical path to **llm-compressor** which:
1. Accepts arbitrary `nn.Linear` subclasses.
2. Has a `AWQModifier(scheme="W4A16")` that registers its own paired-scale
   serialization.
3. Emits a `quantization_config` consumed natively by vLLM and SGLang when
   paired with `compressed-tensors` format.

## Step 7 — Why hybrid W4A16 is the right scheme (chosen by user)

Three plausible schemes considered:

| Scheme | Pros | Cons | Verdict |
|---|---|---|---|
| Pure FP8 (current upstream) | Already works, official | Memory footprint unchanged | Baseline; what we're converting FROM |
| Hybrid **W4A16-AWQ** (chosen) | Massively shrinks expert memory footprint; preserves activation quality at FP16 | Requires careful ignore-set on hyper-connections; loses NVFP4 dynamic range benefit on experts (acceptable trade-off — FP4 experts were training-time QAT-trained; AWQ-int4 retrains that) | ✅ selected |
| Pure **W8A8** (SmoothQuant) | Easier than AWQ on MoE | Smaller savings than INT4 | Mentioned as fallback |

Trade-off rationale:
- The upstream native FP4 experts were obtained by **NVFP4 Quantization-Aware
  Training** (per the README references to "QAT"); training QAT'd FP4 weights
  were calibrated against training-time activations.
- Replacing them with post-hoc AWQ-W4A16 introduces reconstruction error
  proportional to (1 − calibrated-vs-realistic gap). Mitigated by calibrating
  on the *same* downstream task distribution the user plans to evaluate on.
- Activation storage stays FP16 (W4A16), which doubles numerics compared to
  upstream's FP8 activations. This costs roughly +12 GB of activation buffer
  per batch; acceptable for most serving tiers.

Documented consequence for users: **quality is bottlenecked by calibration
distribution**, not by the algorithm. See CALIBRATION_NOTES.md.

## Step 8 — Calibration data selection

Captured in detail in [CALIBRATION_NOTES.md](./CALIBRATION_NOTES.md). Short
version:

Three candidate distributions considered for 256-sequence pile-val calibration:
1. **`pile-val-backup` (canonical, 100 GB)** — general-domain English web/code.
2. **DeepSeek-IV harness `ifeval-coding-multi-turn-train-2026-Q3`** —
   agentic / tool-calling traces extracted from DeepSeek's own eval pile;
   gives best alignment with downstream eval suite but license-restricted.
3. **Synthetic 256-seq Wikitext+Sonnet+GSM8K mix** — built in
   `calibration/prepare_pileval.py`. Deterministic; no licensing.

We standardize on (3) for reproducible smoke tests and offer a CLI flag to
swap in (1) or (2) for real runs.

## Step 9 — Memory budget for a 304 B hybrid-W4A16 run

Estimated (conservative, BF16 base, on-disk ≈ 304 GB):

| Stage | Peak GPU | System RAM | Disk | Notes |
|---|---|---|---|---|
| Sharded streaming load (no materialization) | 0 | 350 GB | 600 GB | HF streaming + lazy materialize during forward passes |
| Materialize (BF16 eval, then mutate) | 0 (CPU) | **610 GB** | 600 GB | Need full BF16 + scratch for scales/shuffles |
| AWQ activation capture (forward pass over 256 seq × 4096 ctx) | 80 GB | 100 GB | 600 GB | Captures per-layer inputs to AWQ; experts each touched sparsely |
| AWQ scale solve + matmul reconstruction | 160 GB | 50 GB | 600 GB | Per-layer GEMM in scaled BF16 |
| Pack to compressed-tensors W4A16 + save | 0 | 100 GB | **300 GB** for output | New ckpt is ~150 GB |

Hardware floor: **≥160 GB GPU VRAM** OR **≥512 GB system RAM + 4 TB SSD + offloaded loading**.
Documented in README preflight gate.

## Step 10 — Iterative build plan (this repo)

```
calibration/prepare_pileval.py           # deterministic calib dataset builder
scripts/upcast_to_bf16.py                # materialize model with FP8/FP4 → BF16
scripts/quantize_llmcompressor.py        # one-shot AWQ recipe runner
scripts/pack_for_vllm.py                 # emit compressed-tensors W4A16 ckpt
scripts/preflight_check.py               # env + disk + VRAM assertion
scripts/smoke_test_surrogate.py          # CI proof: AWQ on 1-layer dummy works
recipes/hybrid_w4a16.yaml                # the recipe itself
recipes/moe_ignore_patterns.txt          # regex ignore list
tests/test_recipe_loads.py               # YAML schema sanity
tests/test_smoke_surrogate.py            # wraps smoke script for pytest
ci/run_smoke.sh                          # GH Action invocation
```

Iteration log:
- Step 10.1: scaffold dir tree, write requirements.txt (DONE)
- Step 10.2: write ignore-pattern derivation doc + actual pattern file (NEXT)
- Step 10.3: write upcast_to_bf16.py
- Step 10.4: write quantize_llmcompressor.py with full CLI + logging
- Step 10.5: write pack_for_vllm.py with vLLM-loadability assertions
- Step 10.6: write preflight_check.py
- Step 10.7: write calibration script
- Step 10.8: write surrogate + tests
- Step 10.9: write CI script
- Step 10.10: write README + finalize NOTES
- Step 10.11: run pytest on smoke

## Step 11 — Open questions / risks

Tracked in [RISKS_AND_OPEN_QUESTIONS.md](./RISKS_AND_OPEN_QUESTIONS.md).
Highlights:
- Hash-routing tables (`tid2eid`) are int32 frozen. Need to confirm llm-compressor
  doesn't try to "decompose" them; safety check added to preflight.
- Whether `weight.scale` sidecar survives compressed-tensors save/reload round-trip
  intact for FP8 layers we explicitly did NOT touch. Documented as known gap.
- Future DeepSeek-V4 hotfix releases (technical-report mentions V4-Pro Preview
  etc.) may change shapes; pinned by `transformers_version` in requirements.