"""common_breeze.py — loaders do Breeze TTS 2 para inferencia/UI (versao slim).

Fonte unica de caminhos e de carga do modelo/tokenizers. O engine vive no
submódulo `breeze-tts/` (paths.BREEZE_REPO), que e adicionado ao sys.path.
"""
from __future__ import annotations

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

CORE_DIR = Path(__file__).resolve().parent
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

import paths  # noqa: E402

# O engine fica no submódulo do repo; garante o import de `breeze_infer`/`models`.
REPO = paths.BREEZE_REPO
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

ROOT = paths.REPO
CKPT = paths.CKPT
TRAINING = paths.TRAINING
ADAPTERS_DIR = paths.ADAPTERS_DIR
OUT_DIR = paths.OUT_DIR


def load_text_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(CKPT)


def load_audio_tokenizer(device: str = "cuda"):
    from qwen_tts import Qwen3TTSTokenizer

    return Qwen3TTSTokenizer.from_pretrained(str(CKPT / "audio_tokenizer"), device_map=device)


def load_breeze_model(device: str = "cuda", attn: str = "eager"):
    import torch
    from models.breeze import BreezeForConditionalGeneration

    model = BreezeForConditionalGeneration.from_pretrained(
        CKPT, dtype=torch.bfloat16, attn_implementation=attn
    )
    return model.to(device).eval()
