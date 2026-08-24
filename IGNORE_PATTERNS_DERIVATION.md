# Ignore-Pattern Derivation

Companion to [QD-4](./QUANTIZATION_DECISIONS.md#qd-4-ignore-pattern-strategy).

This doc explains exactly how the regex pattern list in
`recipes/moe_ignore_patterns.txt` was constructed, ordered by category of
risk.

## Methodology

Start from the safetensors key corpus (`model.safetensors.index.json`,
72,317 keys after fetching). Bucket all keys by their leaf parameter name
and the immediate parent module. For each bucket, classify into one of:

- `AWQ`: layer can be safely quantized with W4A16.
- `SKIP-PRIMARY`: quantizing would collapse a numerical invariant.
- `SKIP-TINY`: too small to bother (< 5 MB total) — would save negligible
  memory at the cost of eating quality budget.
- `DEFER`: only needed for specialized kernels (FP4/FP8 output). Today this
  is everyone in the EXPERT class — defer to AWQ on the BF16 upcast.

Then promote the classification into an `ignore=` pattern list consumable
by `llm-compressor.AWQModifier`.

## Bucket classification

Derived from the key-bucket counts in RESEARCH_NOTES.md §Step 5:

### Bucket `.weight` (36,163 keys)

Subclassification by parent prefix:
- `*.embed.weight`                  → SKIP-TINY
- `*.norm.weight`                   → SKIP-PRIMARY (RMSNorm scale, RT-negligible)
- `*.head.weight`                   → SKIP-TINY
- `*.attn.attn_sink`                → SKIP-PRIMARY
- `*.compressor.ape`                → SKIP-PRIMARY
- `*.compressor.wkv` / `.wgate`     → SKIP-PRIMARY
- `*.indexer.wq_b` / `.weights_proj` → DEFER → AWQ
- `*.indexer.compressor.*`          → SKIP-PRIMARY
- `*.ffn.gate.weight`               → SKIP-PRIMARY (router)
- `*.ffn.gate.bias` / `.tid2eid`    → SKIP-PRIMARY
- `*.ffn.experts.{N}.w1`            → DEFER → AWQ (after upcast)
- `*.ffn.experts.{N}.w2`            → DEFER → AWQ
- `*.ffn.experts.{N}.w3`            → DEFER → AWQ
- `*.ffn.shared_experts.{w1,w2,w3}` → DEFER → AWQ
- `*.attn.wq_a / wq_b / wkv / wo_a / wo_b` → DEFER → AWQ
- `*.attn.q_norm` / `.kv_norm`      → SKIP-PRIMARY
- `*.mtp.*.attn.*`                  → DEFER → AWQ
- `*.mtp.*.ffn.*`                   → DEFER → AWQ
- `*.mtp.*.markov_head.*`           → SKIP-PRIMARY
- `*.mtp.*.confidence_head.*`       → SKIP-PRIMARY
- `hc_*` (HyperConnection mix)      → SKIP-PRIMARY

### Bucket `.scale` (35,718 keys)

ALL `.scale` keys MUST be ignored. They are paired with the `.weight` of
each `Linear` and represent per-block scaling factors — quantizing them
would either bloat weight matrices or corrupt calibration entirely.

Implementation note: after upcast_to_bf16, no `.scale` attrs remain, so
this bucket vanishes. The pattern is defensive coverage only.

### Bucket `.bias` (43 keys)

These are the 43 non-hash-layer `Gate.bias` vectors (`n_layers - n_hash_layers
= 43 - 3 = 40`; slight mismatch with config `num_hidden_layers=43` implies
all 43 layers carry a bias — possibly because hash-gate layers store `bias=None`
but allocate a placeholder, OR because hash vs non-hash splitting happens at
load time, not construction time). All SKIP-PRIMARY regardless.

### Bucket `.ape` (62 keys)

Position-encoding parameter on every Compressor sub-module. SKIP-PRIMARY.

These accumulate numerical noise via Sinkhorn; absolutely cannot tolerate
W4 quantization noise injection.

### Bucket `.hc_*` (46 keys per attribute name, ~138 keys total)

All SKIP-PRIMARY. Sinkhorn-mixing is high-precision; INT4 noise propagates
through `hc_mult=4` residual copies and would diverge.

### Bucket `.attn_sink` (46 keys)

Streaming-softmax anchor vectors. SKIP-PRIMARY.

### Bucket `.tid2eid` (3 keys)

Int32 deterministic-routing lookups for the first 3 hash-routed layers.
SKIP-PRIMARY (literal data, not weights).

## Resulting `ignore=` pattern list

Compiled into `recipes/moe_ignore_patterns.txt`. Pattern grammar:

```
# Lines starting with `#` are comments.
# Otherwise each line is a substring match against fully-qualified param names.
# Substring semantics: a parameter is skipped iff ANY pattern substring is
# contained in its FQN. Conservative (over-skips) is preferred to under-skipping.
#
# Naming convention follows the upstream model.py class hierarchy with the
# FQN prefix `model.` included automatically by llm-compressor.

# 1. RMSNorms (FP32 scale vectors)
*.norm.weight
*.q_norm.weight
*.kv_norm.weight
*.attn_norm.weight
*.ffn_norm.weight
*.main_norm.weight

# 2. Hyperconnection matrices (Sinkhorn-sensitive)
*.hc_attn_fn
*.hc_attn_base
*.hc_attn_scale
*.hc_ffn_fn
*.hc_ffn_base
*.hc_ffn_scale
*.hc_head_fn
*.hc_head_base
*.hc_head_scale

# 3. Router and gating components
*.gate.weight
*.gate.bias
*.gate.tid2eid

# 4. Compressor (KV-cache compressor) — full FP32 paths
*.compressor.ape
*.compressor.wkv
*.compressor.wgate
*.compressor.norm.weight

# 5. Streaming-softmax sinks
*.attn_sink

# 6. Per-block weight scales (paired with each quantized linear)
*.weight.scale
.scale

# 7. Embedding / output head
model.embed.weight
model.head.weight

# 8. DSpark components (small, fragile)
*.markov_head.*
*.confidence_head.*

# 9. Lookup tables and routing dictionaries
*.tid2eid
```

Validation rule: post-pattern application, fewer than ~10 % of FP4/FP8
storage bytes should remain uncategorized. Tested in
`tests/test_ignore_pattern_coverage.py`.

## Edge cases handled

1. **Multi-objective scaling**: AWQ searches scale coefficients simultaneously
   on input and weight sides. Ignoring both sides separately for HC matrices
   would create inconsistency. We've collapsed all HC-flavoured patterns into
   one bucket for clarity.
2. **DSpark's `markov_head.markov_w1/markov_w2`** ALIAS `model.embed` /
   `model.head`. Pointing both names at the same storage means our ignore
   rules on `model.head` automatically protect the DSpark-side view.
3. **Indexer's `compressor`** duplicates the attn.compressor in `compress_ratio==4`
   layers only. Same parameter-name suffix → same pattern match.
4. **MTP head's own layers** (`mtp.*`) replicate the Block graph but reside
   under a different prefix. Pattern `*.attn.*` and `*.ffn.*` cover both
   regular layers and MTP layers naturally.

## Pattern-validation script

`scripts/build_ignore_patterns.py` regenerates this list dynamically from a
fresh safetensors index, guaranteeing coverage stays accurate even if DeepSeek
ships a maintenance release with new module names.