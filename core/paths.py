"""paths.py — resolucao de caminhos do repo de inferencia/UI (dirigida por ambiente).

O engine (breeze-tts) entra como submódulo pinado em `breeze-tts/`. Nenhum peso vai
para o git: o modelo base fica em `PTBR_ARTIFACTS` (fora da arvore, por padrao) e os
adapters LoRA ficam em `adapters/` NA RAIZ deste repo, cujo conteudo e ignorado pelo
git (ver `adapters/.gitignore`) — assim o checkpoint pode morar dentro do projeto.
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

# --------------------------------------------------------------- adapters LoRA
# Pasta DENTRO da raiz do repo onde ficam os adapters LoRA (ex.:
# adapters/Breeze-tts-2-brazillian-lora/adapter_config.json). O conteudo dela e
# ignorado pelo git (adapters/.gitignore), entao baixar/copiar pesos para ca nao
# suja o repositorio.
REPO_ADAPTERS_DIR = _p("PTBR_REPO_ADAPTERS_DIR", REPO / "adapters")

# Onde a UI/CLI procuram adapters, NESTA ordem:
#   1) REPO_ADAPTERS_DIR   -> <raiz>/adapters/<nome>/adapter_config.json   (principal,
#                             tambem o destino do download feito pela UI)
#   2) ADAPTERS_DIR        -> pasta alternativa escolhida em PTBR_ADAPTERS_DIR
#   3) TRAINING_RUNS_DIR   -> <runs>/<run>/checkpoints/<ckpt>/             (treino)
#   4) HF_ADAPTERS_DIR     -> adapters baixados por versoes anteriores      (legado)
ADAPTERS_DIR = _p("PTBR_ADAPTERS_DIR", REPO_ADAPTERS_DIR)
TRAINING_RUNS_DIR = _p("PTBR_TRAINING_RUNS_DIR", TRAINING / "runs")
HF_ADAPTERS_DIR = _p("PTBR_HF_ADAPTERS_DIR", ARTIFACTS / "adapters")
OUT_DIR = _p("PTBR_OUT_DIR", TRAINING / "ui_out")


def adapter_search_dirs() -> list[Path]:
    """Pastas varridas por adapters LoRA (sem duplicatas, em ordem de prioridade).

    A pasta do repo entra SEMPRE (mesmo que `PTBR_ADAPTERS_DIR` aponte para outro
    lugar), porque e o destino do download e onde o usuario costuma largar o
    checkpoint a mao.
    """
    out: list[Path] = []
    for d in (REPO_ADAPTERS_DIR, ADAPTERS_DIR, TRAINING_RUNS_DIR, HF_ADAPTERS_DIR):
        if d not in out:
            out.append(d)
    return out


BREEZE_PY = os.environ.get("BREEZE_PY") or sys.executable

# ------------------------------------------------------------------ downloads
# Repos do Hugging Face usados no primeiro uso (auto-download).
# Troque PTBR_ADAPTER_REPO pelo seu repo quando publicar o adapter.
BASE_MODEL_REPO = os.environ.get("BREEZE_BASE_MODEL_REPO", "BreezeBlue/Breeze-TTS-2")
ADAPTER_REPO = os.environ.get("PTBR_ADAPTER_REPO", "EdnilsonMonts/Breeze-tts-2-brazillian-lora")
ADAPTER_REPO_PAGE = f"https://huggingface.co/{ADAPTER_REPO}" if ADAPTER_REPO else ""
ADAPTER_REPO_URL = f"{ADAPTER_REPO_PAGE}/tree/main" if ADAPTER_REPO else ""
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
