"""Garante que core/{text_norm,text_blocks,adapter_scale,reference_prep}.py == repo de treino."""
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_shared_modules_in_sync():
    proc = subprocess.run([sys.executable, str(ROOT / "scripts" / "sync_shared.py"), "--check"],
                          capture_output=True, text=True)
    if "repo de treino nao encontrado" in (proc.stdout + proc.stderr):
        pytest.skip("repo de treino nao encontrado (defina PTBR_TRAINING_REPO)")
    assert proc.returncode == 0, proc.stdout
