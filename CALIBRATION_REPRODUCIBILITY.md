# Calibration Reproducibility Checklist

Bit-exact reproducibility of the quantization output depends on five
identically-seeded stochastic processes. Verify each before claiming a
specific quantized checkpoint is reproducible.

## 1. RNG seeding

All of the following must use a single seed passed via
`scripts/quantize_llmcompressor.py --seed 0xb0bacafe`:

- `random.seed(seed)` (Python stdlib)
- `numpy.random.seed(seed)`
- `torch.manual_seed(seed)`
- `torch.cuda.manual_seed_all(seed)`
- `transformers.set_seed(seed)`

Failure indicator: per-layer AWQ scales differ across runs at > 1e-7 ULP.

## 2. Dataset shuffling

`calibration/prepare_pileval.py` must be invoked with `--seed` matching
`--seed` passed to `quantize_llmcompressor.py`. Internally, the script
seeds its own PRNG AND sorts source documents lexicographically before
sample-selection, so the order of "first 256 sequences" is deterministic
given identical source data.

Failure indicator: tokenized sequence #0 differs across runs.

## 3. Order of operations

Sequence must be:
1. Upcast FP4/FP8 → BF16.
2. Build ignore-pattern set from final parameter list (order-dependent!).
3. Snapshot parameter FQNs before AWQ capture.
4. Capture activations on calibration data.
5. Run AWQModifier in declared order.

Reordering (e.g., moving snapshotted list AFTER activation capture) may
result in subtly different statistics because new parameters appearing
during graph tracing get allocated memory pages that heat up the GPU.

## 4. Hardware determinism

CUDA non-determinism sources to eliminate:
- Atomic adds (in MoE scatter ops) — solved by torch deterministic mode.
- `flash_attn` reductions — flag `--deterministic-flash-attn` forces
  reproducible reductions (at perf cost).
- CuBLAS workspace allocation — set `CUBLAS_WORKSPACE_CONFIG=:4096:8`.

Failure indicator: forward pass logit differences exceed 1e-3 across runs.

## 5. Hash-content equality

Even with all above honored, multi-node runs require matching:
- NCCL version + NCCL_SOCKET_IFNAME
- `torch.distributed` algorithms for any collective ops (we have none in
  calibration phase, but documenting for completeness)

Failure indicator: shard-2 of the output checkpoint differs in CRC32 vs
reference.

## Reproducibility verification procedure

Reproduce via:
```bash
SEED=0xb0bacafe
CHECKPOINT_REF="gs://your-bucket/dsv4-w4a16-reference/"
LOCAL_OUT=./out-reproducibility-check

python -m calibration.prepare_pileval \
    --seed $SEED \
    --output-dir $LOCAL_OUT/dataset \
    --num-sequences 256 \
    --max-seq-length 2048

python -m scripts.upcast_to_bf16 \
    --hf-repo deepseek-ai/DeepSeek-V4-Flash-0731 \
    --output $LOCAL_OUT/bf16

python -m scripts.quantize_llmcompressor \
    --bf16-input $LOCAL_OUT/bf16 \
    --output $LOCAL_OUT/int4 \
    --seed $SEED \
    --dataset $LOCAL_OUT/dataset \
    --max-seq-length 2048

# Compare to reference
diff -r <(find $LOCAL_OUT/int4 -name "*.safetensors" -printf "%f %s\n" | sort) \
        <(find $CHECKPOINT_REF -name "*.safetensors" -printf "%f %s\n" | sort)
# Optional byte-level:
sha256sum $LOCAL_OUT/int4/*.safetensors | diff - <(sha256sum $CHECKPOINT_REF/*.safetensors)
```

Bit-exact match across runs indicates perfect reproducibility.

## Known irreproducibility sources

Floating-point associative-reduction reorderings are intrinsically
non-reproducible on heterogeneous hardware. Even setting CUBLAS
deterministic mode, runs across A100↔H100↔MI300X are bit-different by
roughly 1 ULP. We accept ≤0.1% PPL jitter attributable to this.