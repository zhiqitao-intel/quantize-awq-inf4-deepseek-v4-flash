"""
pack_for_vllm.py — Post-process the llm-compressor output for vLLM serving.

Responsibilities:
  1. Ensure `quantization_config.json` has the keys vLLM/SGLang
     `compressed_tensors` loader expects:
         - format: "int-quantized"
         - quantization_method: "awq"
         - scheme: "W4A16"
         - group_size: 128
         - sym: false
         - per-target-quantization flag indicating ignore pattern respected
  2. Detach alias edges (markov_head ↔ embed / head). Marks them so vLLM's
     weight loader knows they're aliases and not duplicated storage.
  3. Write a complementary `tensor_aliases.json` listing parametric aliases
     for downstream tooling.
  4. Inject the upstream `modeling_deepseek_v4.py` (renamed for portability)
     into the output directory so `--trust-remote-code` works without
     touching the upstream repo.
  5. Generate `compression_summary.json` describing per-layer dtype/shape
     transitions; useful for QA scripts.

Reference: QUANTIZATION_DECISIONS.md QD-9.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


LOGGER = logging.getLogger("pack_for_vllm")


CANONICAL_QUANT_CFG: Dict = {
    "format": "int-quantized",
    "quantization_method": "awq",
    "scheme": "W4A16",
    "group_size": 128,
    "sym": False,
    "zero_point": True,
    "pack_method": "reorder",
}


def merge_quant_config(input_dir: Path, output_dir: Path,
                       override: Dict | None = None) -> Dict:
    """Merge llm-compressor's emission with vLLM-expected fields."""
    incoming_paths = list(input_dir.glob("**/quantization_config.json"))
    base = CANONICAL_QUANT_CFG.copy()
    if incoming_paths:
        latest = max(incoming_paths, key=lambda p: p.stat().st_mtime)
        incoming = json.loads(latest.read_text())
        # Win-out priorities: incoming wins on actual scale/shape details;
        # canonical wins on backend descriptor keys that vLLM depends on.
        for k, v in incoming.items():
            if k not in base:
                base[k] = v
    if override:
        base.update(override)

    out_cfg = output_dir / "quantization_config.json"
    out_cfg.write_text(json.dumps(base, indent=2))
    LOGGER.info("wrote %s", out_cfg)
    return base


def write_alias_metadata(output_dir: Path) -> None:
    """Emit tensor_aliases.json reflecting the DSpark head alias topology."""
    aliases = {
        # markov_w1 ↔ embed.weight  (both point to same Parameter)
        "model.mtp.{i}.markov_head.markov_w1": "model.embed.weight",
        "model.mtp.{i}.markov_head.markov_w2": "model.head.weight",
        "model.embed.weight": "model.mtp.*.markov_head.markov_w1",
        "model.head.weight": "model.mtp.*.markov_head.markov_w2",
    }
    out = output_dir / "tensor_aliases.json"
    out.write_text(json.dumps(aliases, indent=2))
    LOGGER.info("wrote %s", out)


def inject_modeling_code(input_dir: Path, output_dir: Path) -> None:
    """Pull `inference/model.py` from the input mirror into the output as
    `modeling_deepseek_v4.py` so vLLM/SGLang autoloaders find it.

    Also renames the implicit `kernel.py` imports to the modeling file
    via simple textual patching (vendoring shallow copy of kernel helpers).
    """
    src_model = input_dir / "inference" / "model.py"
    src_kernel = input_dir / "inference" / "kernel.py"
    if not src_model.exists():
        LOGGER.warning("no upstream modeling code in %s; skipping injection",
                       input_dir)
        return
    shutil.copy(src_model, output_dir / "modeling_deepseek_v4.py")
    if src_kernel.exists():
        shutil.copy(src_kernel, output_dir / "kernel_deepseek_v4.py")
        # Patch the import inside the vendored model file
        model_text = (output_dir / "modeling_deepseek_v4.py").read_text()
        patched = model_text.replace(
            "from kernel import ",
            "from kernel_deepseek_v4 import ",
        ).replace(
            "import kernel", "import kernel_deepseek_v4",
        )
        (output_dir / "modeling_deepseek_v4.py").write_text(patched)
    LOGGER.info("vendored modeling code into %s", output_dir)


def build_compression_summary(output_dir: Path) -> None:
    """Walk safetensors shards; emit per-tensor dtype + size summary."""
    summary: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for shard in sorted(output_dir.glob("model-*.safetensors")):
        from safetensors import safe_open
        with safe_open(shard, framework="pt") as fh:
            for key in fh.keys():
                t = fh.get_tensor(key)
                dt = str(t.dtype)
                bucket = ".".join(key.split(".")[:4])
                summary[bucket][dt] += t.numel() * t.element_size()

    out = output_dir / "compression_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    LOGGER.info("wrote compression_summary.json")


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--input", type=Path, required=True,
                        help="llm-compressor output dir")
    parser.add_argument("--output", type=Path, required=True,
                        help="Final vLLM-ready checkpoint dir")
    parser.add_argument("--override", type=json.loads, default=None,
                        help="JSON inline overrides for quantization_config")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(name)s] %(levelname)s %(message)s")

    args.output.mkdir(parents=True, exist_ok=True)
    merge_quant_config(args.input, args.output, args.override)
    write_alias_metadata(args.output)
    inject_modeling_code(args.input, args.output)
    build_compression_summary(args.output)
    LOGGER.info("done. Output at %s", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())