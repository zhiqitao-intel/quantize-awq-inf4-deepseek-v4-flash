# Calibration Data Notes

## Background

Activation-aware Weight Quantization (AWQ) learns per-channel scaling
coefficients by observing the *input activation distribution* to each Linear
layer during forward passes on a calibration dataset. The chosen dataset must
be statistically representative of the workload the user intends to serve,
otherwise the AWQ-learned scales optimize for the wrong distribution.

## Candidate datasets considered

| Dataset | Source | License | Domain fit | Practical concerns |
|---|---|---|---|---|
| **`pile-val-backup`** | monology/pile-uncopyrighted | CC-BY-SA / ODC-By | General English web/code — closest analogue to pretraining mixture | 100 GB raw; need to shard/select. De facto AWQ standard. |
| **`wikitext-103-v1`** | HuggingFace `wikitext` | CC-BY-SA | English narrative | Too narrow for code/agentic |
| **DeepSeek-IV `ifeval-coding-multi-turn-train-2026-Q3`** | DeepSeek private eval pile | Restricted | Best fit for the model's intended downstream workload (code-agent eval) | Distribution-restricted; not usable as a public artefact |
| **Synthetic Sonnet + Wikitext + GSM8K mix** | Built in `calibration/prepare_pileval.py` | Apache-2.0 (own composition) | Broad coverage including code/math; deterministic seed | Less statistically powerful than pile-val |
| **`cosmopedia`** | HuggingFace `HuggingFaceTB/cosmopedia` | Apache-2.0 | Synthetic textbooks; diverse domains | Reasonable backup |

Recommendation (documented in `scripts/quantize_llmcompressor.py` CLI default):
**synthetic Sonnet/Wikitext/GSM8K mix** for reproducibility; allow swap to
**pile-val-backup** via `--dataset pile-val` flag.

## Sequence-length budget

AWQ calibration accumulates per-layer activation statistics. Statistically
useful coverage of a 4096-wide linear requires approximately 512 unique
activation columns visited, equivalently `≥256 × 2048 = 524,288 tokens`
across reasonably diverse contexts.

Our default:
- 256 sequences
- 2048 token sequence length
- Sampled with fixed seed 0xb0ba_cafe for full determinism
- Total ≈ 524k tokens, ≈1 hour of forward passes on a single H100

Override flags:
- `--num-calibration-samples 512` doubles token count (recommended for
  final-run deployments).
- `--max-seq-length 4096` extends to longer contexts at +30 min cost.

## Composition rationale for synthetic mix

Equal weighting across:
- ⅓ Sonnet — English narrative, varied sentiment (covers attention patterns
  common in chat)
- ⅓ Wikitext-103 — encyclopedia narrative (covers factual/structured text)
- ⅓ GSM8K + MATH-train concatenated and shuffled — mathematical reasoning
  (covers long-form chain-of-thought, activates routing-heavy paths)

Determinism is enforced by:
1. Seeding the `random.Random` with `0xb0bacafe`
2. Sorting source documents lexicographically before sampling
3. Setting PyTorch / numpy / tensorflow seeds inside the loader

Verified bit-exact reproducibility in `tests/test_calibration_determinism.py`.

## Pre-processing transformations

Each sequence undergoes:
1. Truncation / padding to `max_seq_length` via the model tokenizer.
2. No BOS/EOS augmentation — the calibration sees the raw distribution.
3. Drop overly repetitive sequences (entropy < 1.5 bits/char) to avoid
   starving AWQ of signal in extreme distribution tails.

Implementation: `calibration/prepare_pileval.py:build_calibration_dataset`.

## Sampling efficiency tricks

Three optimizations baked into the runner:
1. **Lazy `dataset.map(...)`** with `writer_batch_size=64` and
   `num_proc=max(1, os.cpu_count() // 2)` to keep multiprocess mapping honest.
2. **`trust_remote_code=False` for the dataset** so the runner doesn't try to
   load PyArrow with arbitrary remote code.
3. **Activation caching to disk** between AWQ capture stages — prevents
   recomputing forward passes if multiple AWQ sweeps are tried (e.g.,
   comparing group_size=128 vs 64). Cache keyed by `(dataset, max_seq, nsamples, hash(model_config))`.

## Failure modes observed during research

While preparing this doc the following misbehaviors were noted (none of these
made it into the final pipeline but they're traps to avoid):

- **Dataset contamination**: accidentally calibrating on test-set splits leaks
  data into the saved scales. Always source from `*-train` or `*-validation`
  partitions explicitly.
- **Tokenizer drift**: changing tokenizer version between calibration and
  serving causes subtle mismatch. Pin tokenizer revision.
- **Stale activation cache**: changing `num_attention_heads` without bumping
  the cache hash key causes silent stale-cache hits. Cache key includes
  `model.config._commit_hash` precisely to prevent this.
- **Cold-start warmup**: skipping the first 5 sequences with warmup keys
  contaminated the captured statistics with shape-detection transient spikes.
  Fixed by adding `--skip-warmup-samples 5` (default ON).