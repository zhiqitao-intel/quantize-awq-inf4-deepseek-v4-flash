# Design History Summary

Concise digest of `RESEARCH_NOTES.md` for reviewers short on time.

## Originating request

> "Use this repo/folder to study how to quantize the AWQ INT4 version of
> DeepSeek-V4-Flash."

Constraints established during conversation:
1. Workspace isolation: only `/home/zhiqitao/work/quantize-awq-inf4-deepseek-v4-flash/**` may be touched.
2. Real model confirmed: `deepseek-ai/DeepSeek-V4-Flash-0731` exists at
   `https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731`.
3. Target quantization: **Hybrid W4A16-AWQ** (Variant A).

## Critical discoveries in chronological order

| # | Discovery | Impact |
|---|---|---|
| 1 | The HF checkpoint ships at FP4-expert / FP8-activation, **not** BF16 | AutoAWQ's naive "quantize from BF16" script is inapplicable; rewrote the entire pipeline. |
| 2 | Upstream `Linear` subclass attaches a `.scale` sidecar to every weight | Implemented a mandatory upcast step (`scripts/upcast_to_bf16.py`) before AWQ can operate. |
| 3 | The model is a MoE with 256 routed experts + 1 shared + 6 active/token, ML-HyperConnections (hc_mult=4, Sinkhorn-normed mix), Hash routing (first 3 layers), DSpark speculative-decoding head | Forced a careful ignore-pattern taxonomy for HyperConnection mixers, hash-routing tables, markov/confidence heads. |
| 4 | `markov_head.markov_w1/markov_w2` aliases embed/head respectively | Vendor-compatible save must record aliases; done in `scripts/pack_for_vllm.py`. |
| 5 | AutoAWQ's registry doesn't include `deepseek_v4`; even if forced, its `search_qq` won't recognize our naming | Pivoted to `llm-compressor`. |
| 6 | `llm-compressor`'s AWQModifier operates on any `nn.Linear` subclass, but MoE handling requires `ignore=` patterns routed by fully-qualified parameter name | Generated `recipes/moe_ignore_patterns.txt` via systematic derivation in `IGNORE_PATTERNS_DERIVATION.md`. |

## Why W4A16 was selected

Loss-vs-size Pareto frontier for this model specifically:

| Variant | Approx. disk size | Quality risk | Servable on |
|---|---|---|---|
| FP8 (upstream native) | 167 GB | baseline | H100 80 GB (TP=8) |
| **W4A16-AWQ (chosen)** | **~95 GB** | medium-low | H100 80 GB (TP=4) |
| Mixed (dense-only INT4) | 159 GB | very-low | H100 80 GB (TP=8) |

W4A16 cuts disk nearly in half AND halves TP factor, opening deployment on
smaller nodes. The dominant cost — expert FFNs — finally gets quantized,
which the mixed-preserve-FP4 variant refused to do.

## Files of note

- **`recipes/hybrid_w4a16.yaml`** — single source of truth for AWQ
  configuration. Diff this file when investigating quality changes.
- **`recipes/moe_ignore_patterns.txt`** — substring matches guarding
  sinkhorn-sensitive parameters. Grep this file for anything marked
  `SKIP-PRIMARY` in `IGNORE_PATTERNS_DERIVATION.md`.
- **`scripts/upcast_to_bf16.py`** — dequantizes FP4/FP8 storage to BF16.
  Keep on disk; never delete (the AWQ sweep requires its output).
- **`scripts/quantize_llmcompressor.py`** — orchestrator. Logs final
  merged recipe to `--save-final-config-json` for debugging.
- **`scripts/pack_for_vllm.py`** — last-mile transformation. Adds
  `quantization_config.json`, `tensor_aliases.json`, vendored modeling
  code, and a `compression_summary.json`.

## Outstanding TODOs

(See `RISKS_AND_OPEN_QUESTIONS.md` for elaboration.)

- **Eval harness**: implement `scripts/eval_quality.py` to measure FP8 vs
  W4A16-AWQ deltas on Pile-val and a curated agentic-evals set.
- **Numerical-roundtrip test**: load compressed-tensors output, instantiate
  `DeepseekV4ForCausalLM`, compare logits vs BF16-base forward pass.
- **Memory-efficient streaming**: write a memmap-backed variant of
  `upcast_to_bf16.py` that never materializes >2× largest shard in RAM.
- **Better calibration realism**: incorporate per-layer activation
  entropy monitoring; reject outliers that would skew AWQ statistics.
- **Custom CUDA kernel** for W4A16-gemm on older GPUs lacking
  `marlin_w4a16`. Necessary for serving on Ampere (RTX 4090, A100).