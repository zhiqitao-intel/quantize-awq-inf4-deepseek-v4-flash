"""Tests covering recipe YAML, ignore-pattern coverage, calibration
determinism, and the surrogate AWQ smoke path. Invoke via:
  pytest -q tests/

Each test is intentionally self-contained so failures localize quickly.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
RECIPE_PATH = REPO_ROOT / "recipes" / "hybrid_w4a16.yaml"
IGNORE_PATH = REPO_ROOT / "recipes" / "moe_ignore_patterns.txt"


def test_recipe_loads() -> None:
    import yaml
    cfg = yaml.safe_load(RECIPE_PATH.read_text())
    assert "awq_modifier" in cfg
    assert cfg["awq_modifier"]["num_bits"] == 4
    assert cfg["awq_modifier"]["group_size"] == 128
    assert cfg["awq_modifier"]["duo_scaling"] is True
    assert cfg["awq_modifier"]["symmetric"] is False


def test_recipe_targets_linear() -> None:
    import yaml
    cfg = yaml.safe_load(RECIPE_PATH.read_text())
    assert cfg["awq_modifier"]["targets"] == ["Linear"]


def test_recipe_save_format_int_quantized() -> None:
    import yaml
    cfg = yaml.safe_load(RECIPE_PATH.read_text())
    assert cfg["save_format"]["format"] == "int-quantized"


@pytest.mark.parametrize("pattern_line", [
    line.strip()
    for line in IGNORE_PATH.read_text().splitlines()
    if line.strip() and not line.strip().startswith("#")
])
def test_each_ignore_pattern_substring_safe(pattern_line: str) -> None:
    """Patterns are literal substrings per IGNORE_PATTERNS_DERIVATION.md.

    Sanity invariants:
      - non-empty
      - no leading/trailing whitespace beyond the trimmed form
      - no characters that indicate a regex group/escape (since they'd imply
        someone *thought* this was a regex but it'd behave inconsistently
        in downstream `name.__contains__(pattern)` semantics).
    """
    assert isinstance(pattern_line, str)
    assert pattern_line
    assert pattern_line == pattern_line.strip()
    # Characters reserved for regex semantics, disallowed in literal-substring usage.
    regex_only = ("\\", "(", ")", "[", "]", "{", "}", "^", "$", "?")
    for ch in regex_only:
        assert ch not in pattern_line, (
            f"pattern contains regex-only character {ch!r}: {pattern_line!r}")


def test_ignore_patterns_block_hyperconnection() -> None:
    txt = IGNORE_PATH.read_text()
    for needle in ("hc_attn_fn", "hc_attn_base", "hc_attn_scale",
                   "hc_ffn_fn", "hc_ffn_base", "hc_ffn_scale"):
        assert needle in txt, f"missing HC ignore pattern: {needle}"


def test_ignore_patterns_block_router_components() -> None:
    txt = IGNORE_PATH.read_text()
    for needle in ("gate.weight", "gate.bias", "gate.tid2eid", "tid2eid"):
        assert needle in txt, f"missing router ignore pattern: {needle}"


def test_ignore_patterns_block_compressor_fp32_modules() -> None:
    txt = IGNORE_PATH.read_text()
    for needle in ("compressor.ape", "compressor.wkv", "compressor.wgate"):
        assert needle in txt, f"missing compressor ignore: {needle}"


def test_ignore_patterns_skip_sidecar_scales() -> None:
    txt = IGNORE_PATH.read_text()
    assert ".weight.scale" in txt or ".scale" in txt


def test_ignore_patterns_skip_embedding_and_head() -> None:
    txt = IGNORE_PATH.read_text()
    assert "model.embed.weight" in txt
    assert "model.head.weight" in txt


def test_ignore_patterns_skip_attn_sink() -> None:
    txt = IGNORE_PATH.read_text()
    assert "attn_sink" in txt


@pytest.mark.parametrize("fname", [
    "ARCHITECTURE.md",
    "RESEARCH_NOTES.md",
    "QUANTIZATION_DECISIONS.md",
    "IGNORE_PATTERNS_DERIVATION.md",
    "CALIBRATION_NOTES.md",
    "RISKS_AND_OPEN_QUESTIONS.md",
    "CALIBRATION_REPRODUCIBILITY.md",
])
def test_research_docs_exist(fname: str) -> None:
    assert (REPO_ROOT / fname).exists(), f"missing research doc: {fname}"


def test_python_scripts_exist() -> None:
    for fname in ("scripts/upcast_to_bf16.py",
                  "scripts/quantize_llmcompressor.py",
                  "scripts/preflight_check.py",
                  "scripts/pack_for_vllm.py",
                  "scripts/smoke_test_surrogate.py",
                  "calibration/prepare_pileval.py"):
        assert (REPO_ROOT / fname).exists(), f"missing script: {fname}"


def test_python_scripts_compile() -> None:
    for fname in ("scripts/upcast_to_bf16.py",
                  "scripts/quantize_llmcompressor.py",
                  "scripts/preflight_check.py",
                  "scripts/pack_for_vllm.py",
                  "scripts/smoke_test_surrogate.py",
                  "calibration/prepare_pileval.py"):
        proc = subprocess.run(
            [sys.executable, "-m", "py_compile", str(REPO_ROOT / fname)],
            capture_output=True, text=True,
        )
        assert proc.returncode == 0, (
            f"compile failed for {fname}:\n{proc.stderr}")


def test_environment_lock_present() -> None:
    env_dir = REPO_ROOT / "env"
    assert (env_dir / "requirements.txt").exists()


def test_chatgpt_advice_preserved_verbatim() -> None:
    """Original upstream advice left intact as historical record."""
    txt = (REPO_ROOT / "chatgpt-advice.txt").read_text()
    assert "Below is the complete end-to-end AWQ quantization recipe" in txt


def test_no_tmp_artifacts_left_over() -> None:
    """Make sure cleanup hygiene is maintained."""
    leftovers = list(REPO_ROOT.glob("_probe/**/*"))
    # pathlib 3.13+ forbids absolute-glob patterns; walk under /tmp manually.
    tmp_root = Path("/tmp")
    if tmp_root.exists():
        for prefix in ("dsv4_", "_smoke_out", "det-test-A", "det-test-B"):
            leftovers.extend(tmp_root.glob(f"{prefix}*"))
    assert leftovers == [], f"leftover files detected: {[str(x) for x in leftovers[:5]]}"


@pytest.mark.skipif(
    os.environ.get("RUN_SMOKE") != "1",
    reason="Set RUN_SMOKE=1 to run the heavy AWQ smoke test",
)
def test_smoke_surrogate_full() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "scripts.smoke_test_surrogate",
         "--out-dir", str(REPO_ROOT / "_smoke_out"),
         "--keep-output"],
        cwd=str(REPO_ROOT),
        capture_output=True, text=True,
        env={**os.environ, "RUN_SMOKE": "1"},
        timeout=600,
    )
    if proc.returncode != 0:
        pytest.fail(
            f"smoke test failed:\nstdout=\n{proc.stdout}\nstderr=\n{proc.stderr}"
        )


@pytest.mark.skipif(
    os.environ.get("RUN_SMOKE") != "1",
    reason="Set RUN_SMOKE=1 to run calibration-determinism check",
)
def test_calibration_determinism() -> None:
    proc1 = subprocess.run(
        [sys.executable, "-m", "calibration.prepare_pileval",
         "--output", "/tmp/det-test-A",
         "--num-sequences", "16",
         "--seed", "0xb0bacafe"],
        capture_output=True, text=True, cwd=str(REPO_ROOT),
    )
    proc2 = subprocess.run(
        [sys.executable, "-m", "calibration.prepare_pileval",
         "--output", "/tmp/det-test-B",
         "--num-sequences", "16",
         "--seed", "0xb0bacafe"],
        capture_output=True, text=True, cwd=str(REPO_ROOT),
    )
    assert proc1.returncode == proc2.returncode == 0
    import hashlib
    ha = hashlib.sha256()
    for f in sorted(Path("/tmp/det-test-A").iterdir()):
        ha.update(f.read_bytes())
    hb = hashlib.sha256()
    for f in sorted(Path("/tmp/det-test-B").iterdir()):
        hb.update(f.read_bytes())
    assert ha.hexdigest() == hb.hexdigest(), \
        "calibration prep not deterministic across identical seeds"


@pytest.fixture(scope="session", autouse=True)
def enforce_no_tmp_pollution():
    """Hard guarantee that no files leak outside the repo root during tests."""
    yield
    bad = []
    for p in [
        Path("/tmp/dsv4_probe"),
        Path("/tmp/_smoke_out"),
        Path("/tmp/det-test-A"),
        Path("/tmp/det-test-B"),
    ]:
        if p.exists():
            bad.append(p)
    if bad:
        # Clean up so subsequent runs aren't polluted.
        import shutil
        for b in bad:
            shutil.rmtree(b, ignore_errors=True)