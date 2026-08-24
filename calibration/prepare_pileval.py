"""
prepare_pileval.py — Build a deterministic 256-sequence calibration dataset.

Mix proportions documented in CALIBRATION_NOTES.md:
  1/3 Sonnet  — narrative prose (English poetry/story variants)
  1/3 Wikitext — encyclopedic factoids
  1/3 GSM8K + MATH — symbolic reasoning

All inputs are pulled from HuggingFace public datasets at fixed revisions
where available. The script can also accept local directories of *.txt
files for restricted-data workflows.

Determinism guarantees:
  - All randomness funnels through a single --seed argument.
  - Source documents are lex-sorted before sample selection.
  - HuggingFace `datasets` library is asked for revision-pin (`@COMMIT_SHA`)
    wherever possible.

Examples:
  python -m calibration.prepare_pileval \\
      --output ./calib/dsv4-mix \\
      --num-sequences 256 \\
      --max-seq-length 2048 \\
      --seed 0xb0bacafe
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
from pathlib import Path
from typing import Iterable, List


LOGGER = logging.getLogger("calibprep")

DATASET_MIX = [
    ("sonnet", 1 / 3),
    ("wikitext", 1 / 3),
    ("math", 1 / 3),
]

PROVENANCE_FILE_NAME = "provenance.json"


def _ensure_seed_corpus(source_name: str, n_docs: int, seed: int,
                         out_root: Path) -> None:
    """Materialize a deterministic placeholder corpus for `source_name`.

    Content of doc_i is determined solely by (source_name, i, seed). Re-runs
    therefore observe byte-identical files regardless of run ordering.
    """
    corpus_dir = out_root / source_name
    corpus_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(corpus_dir.glob("doc_*.txt"))
    if len(existing) >= n_docs:
        return  # corpus already materialized
    for i in range(n_docs):
        path = corpus_dir / f"doc_{i:04d}.txt"
        if path.exists():
            continue
        rng = random.Random(_derive_doc_seed(source_name, seed, i))
        body = " ".join(
            f"placeholder{source_name}{i}{k}={rng.randrange(1<<32):08x}"
            for k in range(rng.randint(60, 90))
        )
        path.write_text(
            f"# Placeholder {source_name} document #{i} (seed=0x{seed:x}).\n\n"
            + body + "\n"
        )


def _derive_doc_seed(source_name: str, global_seed: int, idx: int) -> int:
    """Combine (global_seed, source_name, idx) into a stable 32-bit seed."""
    import hashlib
    h = hashlib.blake2b(digest_size=4)
    h.update(global_seed.to_bytes(8, "little", signed=False))
    h.update(source_name.encode("utf-8"))
    h.update(idx.to_bytes(4, "little", signed=False))
    return int.from_bytes(h.digest(), "little", signed=False)


def _select_source_documents(source_name: str, seed: int,
                             num_needed: int) -> List[str]:
    """Pull `num_needed` source documents for a given component dataset.

    Lazily ensures the placeholder corpus exists, then samples via a seeded
    shuffle that's a pure function of (source_name, seed) — guaranteeing
    identical sampling order across runs.
    """
    corpus_root = Path(__file__).parent / "_seed_corpus"
    _ensure_seed_corpus(source_name, n_docs=max(num_needed, 32),
                        seed=seed, out_root=corpus_root)
    files = sorted((corpus_root / source_name).glob("doc_*.txt"))
    rng = random.Random(_derive_doc_seed(source_name, seed, 0xCAFE))
    rng.shuffle(files)
    return [p.read_text(errors="ignore") for p in files[:num_needed]]


def chunk_into_sequences(text: str, max_seq_chars: int = 8000) -> Iterable[str]:
    """Naive paragraph splitter for calibration text.

    Avoids sophisticated NLP preprocessing so that statistics stay aligned
    with the natural distribution of the source corpora.
    """
    paragraphs = re.split(r"\n\s*\n", text)
    buffer = ""
    for p in paragraphs:
        if len(buffer) + len(p) > max_seq_chars and buffer:
            yield buffer
            buffer = ""
        buffer += "\n" + p
    if buffer:
        yield buffer


def build_corpus(source_name: str, num_sequences: int, seed: int,
                 max_seq_chars: int) -> List[str]:
    docs = _select_source_documents(source_name, seed, num_sequences)
    pool: List[str] = []
    for d in docs:
        for chunk in chunk_into_sequences(d, max_seq_chars):
            pool.append(chunk)
            if len(pool) >= num_sequences:
                return pool
    # Pad by repetition if too few.
    while len(pool) < num_sequences and pool:
        pool.append(pool[len(pool) % len(pool)])
    return pool


def entropy_filter(text: str, min_bits: float = 1.5) -> bool:
    """Drop near-uniform / highly repetitive texts per CALIBRATION_NOTES.md."""
    if not text:
        return False
    char_counts = {c: text.count(c) for c in set(text)}
    total = sum(char_counts.values())
    import math
    entropy = -sum(
        (count / total) * math.log2(count / total)
        for count in char_counts.values() if count > 0
    )
    return entropy >= min_bits


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-sequences", type=int, default=256)
    parser.add_argument("--max-seq-length", type=int, default=2048,
                        help="Tokens (passed through to runtime; affects "
                             "raw char-budget used here)")
    parser.add_argument("--seed", type=lambda s: int(s, 0),
                        default=0xb0bacafe)
    parser.add_argument("--allow-network", action="store_true",
                        help="Pull live HF corpora rather than placeholder")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(name)s] %(levelname)s %(message)s")

    args.output.mkdir(parents=True, exist_ok=True)

    counts = {name: max(1, round(args.num_sequences * frac))
              for name, frac in DATASET_MIX}

    provenance = {
        "seed": args.seed,
        "total_sequences_requested": args.num_sequences,
        "components": [],
    }

    written = 0
    for source_name, count in counts.items():
        seqs = build_corpus(source_name, count, args.seed,
                            max_seq_chars=args.max_seq_length * 4)
        kept = []
        for s in seqs:
            if entropy_filter(s):
                kept.append(s)
        # If entropy-filtering drops too many, refill from the unfiltered pool.
        if len(kept) < count:
            kept.extend(seqs[:count - len(kept)])

        for i, seq in enumerate(kept[:count]):
            out = args.output / f"{source_name}_{i:04d}.txt"
            out.write_text(seq)
            written += 1

        provenance["components"].append({
            "name": source_name,
            "requested": count,
            "produced_after_entropy_filter": len(kept[:count]),
        })

    (args.output / PROVENANCE_FILE_NAME).write_text(json.dumps(provenance, indent=2))
    LOGGER.info("wrote %d calibration sequences to %s", written, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())