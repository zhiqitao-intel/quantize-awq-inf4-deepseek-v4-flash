# DeepSeek-V4-Flash-0731 — AWQ INT4 Quantization Toolkit

Quantize `deepseek-ai/DeepSeek-V4-Flash-0731` (304 B MoE, native FP4 experts
+ FP8 activations + DSpark speculative head) into a **W4A16**-AWQ checkpoint
loadable by vLLM (`deep_gemm_mega_moe` backend) and SGLang (`flashinfer_mxfp4`
backend).

## Quick start (Intel XPU node)

```bash
MODEL=/path/to/deepseek-v4-flash-snapshot \
WORK=$PWD/.work \
GPUS=8 \
bash recipes/deepseek_v4_flash/run.sh
```

See `recipes/deepseek_v4_flash/README.md` for the reasoning behind every step.

## Status

✅ **Model-specific recipe** at `recipes/deepseek_v4_flash/` — modeled on the proven
   Intel XPU Ornith 1.5 recipe from `~/work/model-compression`.
✅ Requantization pre-pass (`upcast_source`) verified against synthetic FP4/FP8 fixtures.
✅ Preflight correctly identifies the source as compressed-or-quantized.
✅ DSpark head grafting, container-module ignore cleanup, packed-tensor verification.
⚠️  Full 304 B run requires an Intel XPU node (≥200 GB accelerator VRAM total).
⚠️  Eval harness (FP8 baseline vs W4A16-AWQ deltas) pending; see
    `RISKS_AND_OPEN_QUESTIONS.md`.

## What lives here

```
.
├── README.md                         (this file)
├── NOTES.md                          (design-history summary)
├── RESEARCH_NOTES.md                 (chronological lab notebook)
├── ARCHITECTURE.md                   (model reverse-engineering)
├── QUANTIZATION_DECISIONS.md         (every tool/param choice)
├── IGNORE_PATTERNS_DERIVATION.md     (regex derivation)
├── CALIBRATION_NOTES.md              (calibration data design)
├── CALIBRATION_REPRODUCIBILITY.md    (bit-exact repro procedure)
├── RISKS_AND_OPEN_QUESTIONS.md       (known unknowns)
├── chatgpt-advice.txt                (verbatim original advice for context)
│
├── env/
│   ├── requirements.txt              (pinned deps)
│   └── Dockerfile                    (NGC PyTorch base image)
├── recipes/
│   ├── deepseek_v4_flash/              (⭐ model-specific Intel XPU recipe)
│   │   ├── awq_int4.py                 (main runner)
│   │   ├── run.sh                      (containerized entrypoint)
│   │   ├── parallel_awq.py             (multi-XPU AWQ grid partitioner)
│   │   ├── resumable_pipeline.py       (layer-checkpointed sequential pipeline)
│   │   ├── README.md                   (reasoning per design choice)
│   │   └── requirements.lock           (pinned deps for --no-deps install)
│   ├── hybrid_w4a16.yaml               (generic AWQ-W4A16 recipe)
│   └── moe_ignore_patterns.txt         (regex ignore list)
├── model_compress/                     (preflight + environment verification)
│   ├── preflight.py                    (inspect source before loading tensors)
│   └── verify_environment.py           (confirm torch+xpu intact post-install)
├── scripts/
│   ├── upcast_to_bf16.py             (FP4/FP8 → BF16 mirror)
│   ├── quantize_llmcompressor.py     (driver: oneshot AWQ sweep)
│   ├── pack_for_vllm.py              (post-process for vLLM/SGLang)
│   ├── preflight_check.py            (env + disk + VRAM assertion)
│   └── smoke_test_surrogate.py       (CI-grade end-to-end micro-test)
├── calibration/
│   └── prepare_pileval.py            (256-seq deterministic mix builder)
├── tests/
│   └── test_pipeline.py              (pytest suite)
├── ci/
│   └── run_smoke.sh                  (GitHub Actions invocation)
└── .github/
    └── workflows/smoke.yml           (push-triggered smoke CI)
```

## Five-step overview

1. **Preflight.** Validate env, GPU, disk, RAM, recipe, ignore-patterns.
   ```bash
   python -m scripts.preflight_check --input /data/bf16-mirror --strict
   ```
2. **Build BF16 mirror.** Streams through FP4-expert / FP8-dense storage
   and emits a densified BF16 checkpoint that AWQ can read.
   ```bash
   python -m scripts.upcast_to_bf16 \
       --hf-repo deepseek-ai/DeepSeek-V4-Flash-0731 \
       --output /data/dsv4-bf16-mirror
   ```
3. **Build calibration set.**
   ```bash
   python -m calibration.prepare_pileval \
       --output /data/dsv4-calib --num-sequences 256 \
       --seed 0xb0bacafe
   ```
4. **Run the AWQ sweep.**
   ```bash
   python -m scripts.quantize_llmcompressor \
       --bf16-input /data/dsv4-bf16-mirror \
       --output /data/dsv4-w4a16 \
       --dataset /data/dsv4-calib
   ```
5. **Pack for vLLM/SGLang.**
   ```bash
   python -m scripts.pack_for_vllm \
       --input /data/dsv4-w4a16 \
       --output /data/dsv4-w4a16-vllm
   ```

## Running the surrogate smoke (no GPU, ~60 s)

Confirms the AWQ plumbing works end-to-end against a 1-layer fake
DeepSeek-V4 model:

```bash
RUN_SMOKE=1 python -m pytest -q tests/test_pipeline.py::test_smoke_surrogate_full
```

## Memory budget (304 B full sweep)

Per `RESEARCH_NOTES.md` §Step 9:
| Stage | GPU | RAM | Disk |
|---|---|---|---|
| BF16 upcast materialization | 0 | 610 GB | 600 GB |
| AWQ activation capture | 80 GB | 100 GB | 600 GB |
| AWQ solve + repack | 160 GB | 50 GB | 300 GB (output) |

Recommended host: **single node with ≥1×NVIDIA B200/MI300X-class GPU
(or 8×H100 80 GB NVLink) and ≥2 TB local NVMe.**

## Key design choices (summary; see `QUANTIZATION_DECISIONS.md`)

- **Tool: `llm-compressor`**, not AutoAWQ (custom Linear with FP8/FP4 sidecar
  scale attribute breaks AutoAWQ's packer; AutoAWQ has no `deepseek_v4`
  architecture registry).
- **Scheme: W4A16** (INT4 weights, FP16 activations) — hybrid variant A from
  the planning discussion.
- **Upcast step first:** dequant FP4/FP8 to BF16 before AWQ so the search
  operates on densities it can analyze.
- **Ignore patterns:** RMSNorm weights, hyper-connection sinkhorn matrices,
  router weights & biases, hash-routing tables, compressor fp32 paths,
  attn_sinks, embeddings, lm_head, DSpark components. Derived in
  `IGNORE_PATTERNS_DERIVATION.md`.
- **Duo-scaling:** enabled (`α-step=0.1`), group size 128, asymmetric
  zero-points.
- **Save format:** `compressed-tensors` v0.10 with `int-quantized` schema,
  vendored modeling code for `--trust-remote-code` portability.

## Limitations & cautions

1. The check-point model is **trademark-trademarked** for downstream training
   use under the upstream MIT license; AWQ redistribution requires preserving
   the LICENSE file in your derivative.
2. Hash-routing (`tid2eid`) tables MUST be preserved — they're int32 frozen
   data, not weights. The recipe's ignore list already covers them; do not
   override.
3. The DSparkMarkovChain shares `markov_w1/markov_w2` with embed/head;
   re-loading must treat these as aliases. `pack_for_vllm.py` emits a
   `tensor_aliases.json` record for vLLM awareness.
4. **Pretrained evaluation deltas are unmeasured.** Until someone runs an
   eval harness against FP8-baseline-vs-this-W4A16 checkpoint on the same
   prompts, treat quality claims as theoretical. Plan to add an eval script
   post-run.

## License

- Upstream model: MIT (DeepSeek).
- This toolkit: MIT (see `LICENSE`).
- `chatgpt-advice.txt`: kept unmodified as historical record; do not treat
  as authoritative (many of its claims are inaccurate; see
  `RESEARCH_NOTES.md` Step 0).