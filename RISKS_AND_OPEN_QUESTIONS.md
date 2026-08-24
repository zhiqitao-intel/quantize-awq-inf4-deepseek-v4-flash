# Risks and Open Questions

Living document tracking things I'm not 100% sure about, broken into two
buckets: (A) blockers that would invalidate the toolkit's correctness, and
(B) quality / performance issues that we'd address in subsequent revisions.

## Bucket A — Correctness risks

### A-1: Does `compressed-tensors` preserve FP8/FP4 scales untouched?

After upcasting to BF16, no `.scale` attrs should remain on quantizable layers.
But the `compress_rope_theta` / `compress_ratios` buffers and `idx` lookup
buffers may carry auxiliary scale tables. Need empirical round-trip test:
- Load quantized output → instantiate Transformer from new checkpoint →
- Run forward pass → compare logits to baseline.

Covered in `tests/test_roundtrip_inference.py` (planned; not written yet).

### A-2: MTP head weight sharing

`markov_head.markov_w1` aliases `embed.weight`; `markov_head.markov_w2`
aliases `head.weight`. If `compressed-tensors` serialization treats aliases
as distinct storage (rather than pointer reuse), we'd accidentally inflate
disk usage by 1 GB and potentially introduce two slightly different copies
under quantization.

Workaround: explicitly detach aliases before save. Implementation planned
in `scripts/pack_for_vllm.py:_detach_aliases()`.

### A-3: HC fr Parameter dtype mismatch

`hc_*_fn` declared fp32 in `model.py` but stored as bf16 in the checkpoint
(see comment in upstream model.py line 301: "wkv and wgate in the checkpoint
is stored in bf16, while the parameter here is stored in fp32 for convenient").

Upcasting a fp32-declared-but-bf16-loaded param back to fp32 should be a no-op
numerically. But if `safetensors` ships strict dtype metadata mismatched
against the Python type hint, `from_pretrained(strict=True)` may refuse to
load. Adding `strict=False` in `scripts/upcast_to_bf16.py`.

### A-4: `attn_sink` interpretation as weight vs buffer

`attn_sink` is registered as `nn.Parameter` but functionally acts like a
buffer (added inside attention softmax denoms; never gradient-updated
post-init). Should still be caught by `ignore=`; documenting for posterity.

### A-5: vLLM kernel availability for compiled FP4 dtypes on consumer GPUs

If the target inference host lacks `deep_gemm_mega_moe` or `flashinfer_mxfp4`
kernels (e.g., running on RTX 4090s instead of H100/B200), the W4A16 output
won't load. Fallback path:
- Emit AWQ-format int4 weights via a custom unpacker.
- Document kernel-support matrix in README.

### A-6: Awq modifier's MoE coverage assumption

`llm-compressor`'s AWQModifier was originally designed for dense models and
later extended for MoE. Its MoE extension lists experts under
`model.decoder.layers.{i}.mlp.experts.{j}.*` by hard-coded path-string match
in some versions. Our experts live under
`model.layers.{i}.ffn.experts.{j}.*`. Need to verify the active
llm-compressor version picks our path up correctly; covered by smoke test.

### A-7: `act_quant` invocation in checkpoint roundtrip

The DeepSeek-V4 forward path invokes an internal `act_quant` (FP8 sim) inside
several places. If the BF16-mirror-loaded model bypasses that, downstream
attention output magnitudes may shift, breaking the relative importance
ratios AWQ captured. Test: compare BF16-base logit distribution (with and
without FP8 sim) on identical inputs.

### A-8: `dspark_block_size=5` configurability

The DSpark draft head relies on `dspark_target_layer_ids=[40, 41, 42]` and
expects `forward_spec` to be called alongside `forward`. Our `smoke_test_surrogate`
runs only `forward`; DSpark is untested. Acceptable trade-off because the
DSpark stage lives in fp32+bf16 space (not FP4), so quantization shouldn't
affect its numerics meaningfully.

## Bucket B — Quality risks

### B-1: Hash-routing tier-1 degradation

The first 3 layers use deterministic hash routing. Even small perturbation
in router weights could shift assignment (though hash mode bypasses `bias`).
Should be inert; flagging because it has bitten previous experiments.

### B-2: Million-token context loss

YaRN-×16 scaling diverges outside its training zone. AWQ cannot bring back
fine-grained positional precision lost in INT4. Expect degraded long-tail
performance on 1 M-token contexts vs the FP8 baseline.

Mitigation: mention in README that production serves should benchmark
`needle-in-haystack` retrieval at multiple context lengths.

### B-3: MoE expert imbalance after quantization

256 experts with 6 active per token relies on balanced utilization to
spread compute. AWQ's columnwise clip optimization could amplify imbalance
on already-unbalanced experts. Monitor: per-expert activation count during
serving, alert if any expert becomes >10× underrepresented vs median.

### B-4: Generation parity baseline

We lack a quantitative parity baseline for FP8→INT4 transition. Recommend:
- On a 1000-sample held-out reasoning benchmark, compare FP8 baseline vs
  W4A16-AWQ; expect ≤1.5% accuracy delta, ≤0.8 PPL delta on Pile-val.

Currently no benchmark harness in this repo (beyond the smoke test).
Adding a `scripts/eval_quality.py` skeleton planned for follow-up.

### B-5: Cross-attention-free assumption

Model has no encoder, so cross-attention quantization isn't a concern.
Documenting only because reviewers will ask.

## Open questions for the model team

If we wanted to send a question upstream to DeepSeek (their service email is
publicly listed in the model card), reasonable questions are:

1. Are the FP4 expert weights calibration-frozen, or are they
   "live-tunable" during continued pretraining? Affects whether further
   finetuning post-AWQ is advisable.
2. Does the model tolerate mixed-precision inputs (some layers BF16,
   others INT4)? Currently assumed yes but unconfirmed.
3. Are the `ape` compressor tables meant to be altered at finetune time,
   or absolute constants? Affects finetuning compatibility post-AWQ.

## Bug / limitation reporting

If during use of this toolkit you encounter a failure not covered above,
open an issue in this repo tagged `awq-v4-research` and include:
- exact command line invoking `scripts/quantize_llmcompressor.py`,
- output of `scripts/preflight_check.py`,
- SHA256 of the offending safetensors shard (download `.sha256` from HF).

We'll triage against this doc and either add a new bucket-A entry or a
bucket-B entry depending on severity.