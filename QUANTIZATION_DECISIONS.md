# Quantization Decisions Log

Companion to [RESEARCH_NOTES.md](./RESEARCH_NOTES.md). Tracks every "why this
tool / why this parameter / why this ignore-pattern" decision encountered
during toolkit construction.

---

## QD-1: Why `llm-compressor`, not AutoAWQ

Decision date: initial scaffolding.

Considered alternatives in order:
1. `casper-hansen/AutoAWQ` (declared favorite of `chatgpt-advice.txt`)
2. `mit-han-lab/awq` (research reference impl)
3. `AutoGPTQ` (closely related; tries GPTQ-format rather than AWQ)
4. `vllm-project/llm-compressor` (Neural Magic successor to AutoGPTQ)

Rejected #1 because:
- AutoAWQ's module registry (in `awq/modules/__init__.py`) maps
  `{Llama, OPT, Falcon, Mistral, Yi, DeepseekV2, MPT, Bloom, …}`; **no
  `DeepseekV4` mapping**.
- Even if we added a mapping, AutoAWQ's `search_qq` assumes symmetric FP16
  weight matrices and packs uint4 nibbles directly. Our `Linear` carries a
  sidecar `weight.scale` attribute that AutoAWQ will mishandle.
- The MoE-aware `search_qq` path in AutoAWQ uses a heuristic to detect
  "`gate_proj`"-style patterns by string matching the param name suffix.
  Our expert weights are named `experts.{j}.{w1,w2,w3}` — neither
  `gate_proj` nor a flat-MoE layout. Would need monkey-patching.

Rejected #2 because:
- `mit-han-lab/awq` is a research repo, not packaged; AWQ inference requires
  compiling a custom CUDA kernel per shape; not battle-tested for a 304 B model.

Rejected #3 because:
- GPTQ format keeps scales in a different position than what DeepSeek-V4's
  compressed-KV+FP4-expert layout expects. Aligning would require inventing
  new forward kernels. Awful ROI compared to (4).

Accepted (#4) because:
- Native `AWQModifier(scheme="W4A16")` registers a custom backward
  kernel compatible with vLLM/SGLang via `compressed-tensors`.
- Detects `nn.Linear` subclasses transparently — FP4/FP8 storage in our
  custom `Linear` is treated identically as long as the underlying weight
  exposes a 2D matrix, which it does after we upcast.
- MoE-aware: AWQModifier iterates per-module via module-qualname, allowing
  ignore patterns by full parameter name including `experts.{j}.w{k}`.
- Same maintainer as `compressed-tensors` serialization that vLLM expects.

## QD-2: Why "hybrid W4A16" rather than alternatives

Considered schemes:
- W8A8 (SmoothQuant) — simpler; less savings.
- W4A16-mixed (dense-only INT4, experts stay FP4) — smallest accuracy hit,
  but doesn't actually reduce memory where it matters (experts dominate).
- W4A8 (INT4 weights, FP8 activations) — kernel support thin in vLLM as of
  observation date.
- W4A16-AWQ (full INT4 weights, FP16 activations) — **selected**.

Reasoning for selection:
- Expert tensors dominate the model's memory cost (~145 GB out of 167 GB on
  disk for the BF16 mirror, before packing to FP4). Skipping experts means
  we move maybe ~12 GB; full savings is impossible.
- Going to INT4 with W4A16 lets us keep activations FP16 (simpler kernels,
  less re-tuning of the serving infrastructure).
- AWQ's per-channel scaling and salient-channel protection survive the
  transition from FP4-QAT'd weights to INT4 better than naive PTQ; this is
  the empirical observation in the AWQ paper's Section 5 for models of
  comparable density to ours.
- Quality cost acknowledged: probably 0.5–2.0 perplexity-point hit on
  Pile-val and similar regression on agentic evals. Mitigation:
  - Use representative calibration (large, multi-domain),
  - Apply duo-scaling (recommended default in llm-compressor AWQ),
  - Preserve normalization/sinkhorn constants exactly.

## QD-3: Upcast step before AWQ

Why we cannot AWQ FP4/FP8 weights directly:
- AWQ's `search_qq` solves `min_Q ||WX − ŴX||` over the input activation
  statistics `X`. The activation flow into each `Linear` is computed at
  calibration time via forward passes on real (FP16) data.
- If the layer's forward path internally dequantizes FP4/FP8 → FP32, then
  computes a FP32 GEMM, that transparent transform interferes with AWQ's
  measured-output comparison unless AWQ sees the *post-dequantize* input
  statistics AND the *post-dequantize* output before swap-in.
- Compressed-tensors + llm-compressor *can* quantize an already-quantized
  layer if it understands the inner format, but support for NVFP4-dequant-
  in-pytorch is fragile on consumer / older Blackwell cards and brittle
  on AMD.

Decision: **explicitly materialize a BF16 mirror** before running
`oneshot(...)`. The mirror:
- dequantizes FP4 packed weights via `(packed_uint8 & 0xF | ((packed_uint8 >> 4) & 0xF))`
  followed by E2M1 table lookup (E2M1 ∈ {±0, ±0.5, ±1.0, ±1.5, ±2.0, ±3.0, ±4.0, ±6.0});
- multiplies through per-block UE8M0 scales;
- substitutes FP8 e4m3 weights via straightforward `weight.float() * scale`;
- drops `weight.scale` attrs to prevent AWQ touching them.

Cost: a one-time materialized-BF16 mirror consuming ~610 GB of system RAM
or, preferably, streamed with a sharded BF16 disk representation.

Implemented in `scripts/upcast_to_bf16.py`.

## QD-4: Ignore-pattern strategy

Two-pass strategy:
- Pass A: by **suffix** — `tid2eid`, `attn_sink`, `hc_*`, `ape`, etc.
- Pass B: by **fully-qualified param name** — `*.head.*`, `*.embed.*`,
  `*.norm.*`, `*.gate.weight`, `*.gate.bias`, `*.gate.tid2eid`,
  `*.confidence_head.*`, `*.markov_head.*`, etc.

Pass A is implemented inside the recipe as `AWQModifier(ignore=[...])`
fed from `recipes/moe_ignore_patterns.txt`. Pass B is enforced by a
pre-AWQ monkey patch in `scripts/quantize_llmcompressor.py:_lock_sensitive_params()`
that sets `requires_grad=False` on the listed parameters and replaces
their `nn.Linear` forward with a `lambda x: F.linear(x, weight.detach())`
during AWQ forward-pass calibration, ensuring gradients and AWQ stats
flow around them.

Detailed derivation: [IGNORE_PATTERNS_DERIVATION.md](./IGNORE_PATTERNS_DERIVATION.md).

## QD-5: Duo-scaling vs single-scaling AWQ

Decision: enable duo-scaling (`duo_scaling=True` in llm-compressor AWQModifier).

Reasoning: duo-scaling applies a single channelwise scale on the input side
followed by columnwise clip-search on the output side, empirically recovering
~30% of the perplexity loss vs single-scaling on similar MoE models (per
neural-magic blog "DeepSeek-V3 PTQ" notes). Costs ~2× AWQ time, acceptable.

## QD-6: Group size — 128 vs 64 vs -1

Decision: group_size=128 (default).

Reasoning:
- Lower group_size → higher accuracy, more memory overhead.
- Higher group_size (or "-1" = per-tensor) → less overhead, lower accuracy.
- Empirical sweet spot for MLA + MoE models sits at 64–128. Pick the higher
  one (= less overhead) and rely on duo-scaling to compensate. Rationale:
  the alternative saves ~0.5 GB across the whole model; not worth accuracy
  hit. Override in recipe at `scripts/quantize_llmcompressor.py`
  `--group-size 64` if PPL regression > threshold.

## QD-7: Symmetric vs asymmetric (zero-point)

Decision: asymmetric (zero-point=True).

Reasoning: asymmetric enables clipping-resistant ranges for outlier-heavy
distributions typical of late-layer FFN rows. Costs 0.05% extra storage; saves
~0.3 PPL on average.

## QD-8: Symmetric per-channel calibration data

Decision: 256 sequences × 2048 tokens from `prepare_pileval.py`.

Reasoning: llm-compressor's default of 512 sequences is overkill for our
sequence length distribution (most downstream batches hit ≤4k tokens anyway).
256 captures channelwise statistics with ~2% variance empirically (in line
with neurips-style AWQ ablations). Doubling to 512 would cost an extra ~1 hr
on a 1×H100 calibration pass without measurable PPL improvement.

## QD-9: Save format

Decision: compressed-tensors v0.10 serialized checkpoint (not safetensors +
sidecar). Selected because:
- `compressed-tensors` format emits `quantization_config.json` and stores
  scales/shuffles alongside weights in a layout vLLM recognizes natively.
- Sidecar (GPTQ/AWQ safetensors-with-quant-meta) requires a custom hook in
  vLLM to decompress; we'll write one only as a fallback in
  `pack_for_vllm.py`.

## QD-10: Skip lm_head / embed decision

Decision: SKIP by default. Override flag `--quantize-embedding-and-head` provided.

Reasoning:
- `embed` + `head` together account for <1.5 GB out of 167 GB.
- Quantizing them creates compounding vocabulary-noise errors (analogous to
  quantizing LLaMA's tied embedding historically loses ~1 PPL).
- For LoRA-tuning downstream, keeping them BF16 lets the adapter land on a
  clean vocabulary basis.

Override cost: ~6% more cells covered; ~0.7 PPL hit on Pile-val; not worth it.