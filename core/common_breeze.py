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
HF_ADAPTERS_DIR = paths.HF_ADAPTERS_DIR
OUT_DIR = paths.OUT_DIR

_base_ensured = False


def _hf_download(repo_id: str, local_dir) -> str:
    from huggingface_hub import snapshot_download

    dest = Path(local_dir)
    dest.mkdir(parents=True, exist_ok=True)
    return snapshot_download(
        repo_id=repo_id, local_dir=str(dest), token=paths.HF_TOKEN, max_workers=8
    )


def ensure_base_model() -> None:
    """Baixa o checkpoint base do Hugging Face se ainda nao existir (1a execucao)."""
    global _base_ensured
    if _base_ensured:
        return
    _base_ensured = True
    if (CKPT / "config.json").is_file():
        return
    print(f"[setup] checkpoint base ausente em {CKPT}", flush=True)
    print(f"[setup] baixando {paths.BASE_MODEL_REPO} do Hugging Face "
          f"(pode levar alguns minutos)...", flush=True)
    _hf_download(paths.BASE_MODEL_REPO, CKPT)
    if not (CKPT / "config.json").is_file():
        raise RuntimeError(
            f"nao foi possivel baixar o modelo base para {CKPT}. Baixe "
            "'BreezeBlue/Breeze-TTS-2' manualmente ou ajuste PTBR_ARTIFACTS/BREEZE_CKPT."
        )


def ensure_adapter(repo_id: str | None = None, dest=None) -> Path | None:
    """Baixa (best-effort) o adapter LoRA do Hugging Face. Retorna a pasta ou None."""
    repo_id = repo_id if repo_id is not None else paths.ADAPTER_REPO
    if not repo_id:
        return None
    target = Path(dest) if dest else HF_ADAPTERS_DIR / repo_id.split("/")[-1]
    if (target / "adapter_config.json").is_file():
        return target
    try:
        print(f"[setup] baixando adapter {repo_id} -> {target}", flush=True)
        _hf_download(repo_id, target)
    except Exception as exc:  # noqa: BLE001
        msg = str(exc)
        if "404" in msg or "Repository Not Found" in msg:
            print(f"[setup] adapter '{repo_id}' ainda nao esta publicado no Hugging Face; "
                  "seguindo com os adapters locais. Para desativar o download, deixe "
                  "PTBR_ADAPTER_REPO vazio no .env.", flush=True)
        else:
            print(f"[setup] (aviso) nao baixei o adapter {repo_id}: {msg}", flush=True)
        return None
    return target if (target / "adapter_config.json").is_file() else None


def load_text_tokenizer():
    ensure_base_model()
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(CKPT)


def load_audio_tokenizer(device: str = "cuda"):
    ensure_base_model()
    from qwen_tts import Qwen3TTSTokenizer

    return Qwen3TTSTokenizer.from_pretrained(str(CKPT / "audio_tokenizer"), device_map=device)


def load_breeze_model(device: str = "cuda", attn: str = "eager"):
    ensure_base_model()
    import torch
    from models.breeze import BreezeForConditionalGeneration

    model = BreezeForConditionalGeneration.from_pretrained(
        CKPT, dtype=torch.bfloat16, attn_implementation=attn
    )
    return model.to(device).eval()
