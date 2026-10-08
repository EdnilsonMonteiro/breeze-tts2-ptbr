"""Stubs de torch/gradio/common_breeze para testar a LOGICA da UI sem GPU nem Gradio."""
import itertools
import shutil
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "core", ROOT / "ui", ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# Pasta de trabalho dos testes: `.pytest_tmp/` na raiz do repo (ignorada pelo git).
# Usamos `mkdir` simples em vez de `tempfile.mkdtemp`/`tmp_path` porque pastas criadas
# com modo 0o700 podem ficar inacessiveis sob sandbox (WinError 5) — aqui e sempre
# gravavel, e o conteudo e apagado ao fim de cada teste.
WORKDIR_ROOT = ROOT / ".pytest_tmp"
_counter = itertools.count()


@pytest.fixture
def workdir():
    """Pasta vazia e isolada por teste (equivalente ao `tmp_path`, mas gravavel em sandbox)."""
    d = WORKDIR_ROOT / f"t{next(_counter)}"
    d.mkdir(parents=True, exist_ok=True)
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


class _GrError(Exception):
    pass


def _install_stubs(tmp_root: Path) -> None:
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    sys.modules.setdefault("torch", torch)

    gr = MagicMock()
    gr.Error = _GrError
    gr.Warning = MagicMock()
    sys.modules.setdefault("gradio", gr)

    # Adapters LoRA: pasta do repo (principal), runs de treino e o legado fora do repo.
    repo_adapters = tmp_root / "adapters"
    runs = tmp_root / "training" / "runs"
    hf = tmp_root / "hf"

    cb = types.ModuleType("common_breeze")
    cb.OUT_DIR = tmp_root / "out"
    cb.ARTIFACTS = tmp_root / "artifacts"
    cb.TRAINING = tmp_root / "training"
    cb.REPO_ADAPTERS_DIR = repo_adapters
    cb.ADAPTERS_DIR = repo_adapters
    cb.TRAINING_RUNS_DIR = runs
    cb.HF_ADAPTERS_DIR = hf
    cb.ADAPTER_REPO = "EdnilsonMonts/Breeze-tts-2-brazillian-lora"
    cb.ADAPTER_REPO_PAGE = "https://huggingface.co/EdnilsonMonts/Breeze-tts-2-brazillian-lora"
    cb.ADAPTER_REPO_URL = f"{cb.ADAPTER_REPO_PAGE}/tree/main"
    # Espelha core/paths.py + core/common_breeze.py: pasta do repo primeiro, runs de
    # treino depois e o legado `hf/<nome>` por ultimo.
    cb.adapter_search_dirs = lambda: [repo_adapters, runs, hf]
    cb.adapter_sources = lambda: [
        ("hf", d) if d == hf else (None, d) for d in cb.adapter_search_dirs()
    ]
    cb.adapter_dir_for = lambda repo_id=None: repo_adapters / (
        (repo_id or cb.ADAPTER_REPO).split("/")[-1]
    )
    cb.adapter_installed = lambda repo_id=None, dest=None: None   # "nao baixado" por padrao
    cb.prepare_adapter = lambda path: str(path)
    cb.download_adapter = MagicMock()
    cb.speaker_embed = MagicMock()
    sys.modules["common_breeze"] = cb


_tmp = WORKDIR_ROOT / "_stubs"
_tmp.mkdir(parents=True, exist_ok=True)
_install_stubs(_tmp)
