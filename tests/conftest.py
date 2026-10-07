"""Stubs de torch/gradio/common_breeze para testar a LOGICA da UI sem GPU nem Gradio."""
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "core", ROOT / "ui"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


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

    cb = types.ModuleType("common_breeze")
    cb.OUT_DIR = tmp_root / "out"
    cb.ARTIFACTS = tmp_root / "artifacts"
    cb.ADAPTERS_DIR = tmp_root / "adapters"
    cb.TRAINING = tmp_root / "training"
    cb.HF_ADAPTERS_DIR = tmp_root / "hf"
    cb.speaker_embed = MagicMock()
    sys.modules["common_breeze"] = cb


_tmp = Path(__import__("tempfile").mkdtemp(prefix="ui_tests_"))
_install_stubs(_tmp)
