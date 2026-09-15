"""paths.py — resolucao de caminhos do repo de inferencia/UI (dirigida por ambiente).

O engine (breeze-tts) entra como submódulo pinado em `breeze-tts/`; os pesos e
os adapters LoRA ficam FORA do git (ver .env.example).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# core/paths.py -> parents[1] = raiz deste repo
REPO = Path(__file__).resolve().parents[1]


def _load_dotenv() -> None:
    env = REPO / ".env"
    if not env.is_file():
        return
    for raw in env.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


_load_dotenv()


def _p(env: str, default: Path) -> Path:
    val = os.environ.get(env)
    return Path(val).expanduser().resolve() if val else default


# Engine: submódulo oficial (ou aponte com BREEZE_TTS_REPO).
BREEZE_REPO = _p("BREEZE_TTS_REPO", REPO / "breeze-tts")

# Artefatos fora do git.
ARTIFACTS = _p("PTBR_ARTIFACTS", REPO / "artifacts")
CKPT = _p("BREEZE_CKPT", ARTIFACTS / "models" / "Breeze-TTS-2")
TRAINING = _p("BREEZE_TRAINING_DIR", ARTIFACTS / "training")

# Onde a UI/CLI procura adapters LoRA: <ADAPTERS_DIR>/<run>/checkpoints/<ckpt>.
ADAPTERS_DIR = _p("PTBR_ADAPTERS_DIR", TRAINING / "runs")
OUT_DIR = _p("PTBR_OUT_DIR", TRAINING / "ui_out")

# Adapters baixados automaticamente do Hugging Face (uma pasta por repo).
HF_ADAPTERS_DIR = _p("PTBR_HF_ADAPTERS_DIR", ARTIFACTS / "adapters")

BREEZE_PY = os.environ.get("BREEZE_PY") or sys.executable

# ------------------------------------------------------------------ downloads
# Repos do Hugging Face usados no primeiro uso (auto-download).
# Troque PTBR_ADAPTER_REPO pelo seu repo quando publicar o adapter.
BASE_MODEL_REPO = os.environ.get("BREEZE_BASE_MODEL_REPO", "BreezeBlue/Breeze-TTS-2")
ADAPTER_REPO = os.environ.get("PTBR_ADAPTER_REPO", "EdnilsonMonteiro/Breeze-TTS-2-lora-ptbr")
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
