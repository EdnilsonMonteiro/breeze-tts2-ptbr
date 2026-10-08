"""common_breeze.py — loaders do Breeze TTS 2 para inferencia/UI (versao slim).

Fonte unica de caminhos e de carga do modelo/tokenizers. O engine vive no
submódulo `breeze-tts/` (paths.BREEZE_REPO), que e adicionado ao sys.path.
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import logging
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


class _DropMessage(logging.Filter):
    """Descarta uma mensagem de log especifica (por substring)."""

    def __init__(self, needle: str) -> None:
        super().__init__()
        self._needle = needle

    def filter(self, record: logging.LogRecord) -> bool:
        return self._needle not in record.getMessage()


def _silence_known_warnings() -> None:
    # `transformers` avisa "incorrect regex pattern" e sugere fix_mistral_regex=True,
    # mas esse flag quebra na versao instalada (TypeError no tokenizers). O modelo foi
    # treinado com este tokenizer; mantemos o comportamento padrao e silenciamos apenas
    # esta mensagem especifica.
    logging.getLogger("transformers.tokenization_utils_base").addFilter(
        _DropMessage("incorrect regex pattern")
    )


_silence_known_warnings()


@contextlib.contextmanager
def _silence_stdout():
    """Silencia prints de bibliotecas durante um import (ex.: banner do qwen_tts)."""
    with contextlib.redirect_stdout(io.StringIO()):
        yield


def import_qwen_tts():
    """Importa Qwen3TTSTokenizer suprimindo o banner de flash-attn do qwen_tts.

    O `qwen_tts` imprime, no import, um aviso quando flash-attn nao esta instalado
    (o tokenizer de audio cai no caminho manual em PyTorch — funcional, so mais
    lento). Instalar flash-attn no Windows nao e pratico, entao suprimimos o banner.
    """
    with _silence_stdout():
        from qwen_tts import Qwen3TTSTokenizer

    return Qwen3TTSTokenizer


CORE_DIR = Path(__file__).resolve().parent
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

import adapter_store as ADS  # noqa: E402
import paths  # noqa: E402

# O engine fica no submódulo do repo; garante o import de `breeze_infer`/`models`.
REPO = paths.BREEZE_REPO
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

ROOT = paths.REPO
ARTIFACTS = paths.ARTIFACTS
CKPT = paths.CKPT
TRAINING = paths.TRAINING
# Adapters LoRA: pasta do repo (principal), runs de treino e o legado fora do repo.
REPO_ADAPTERS_DIR = paths.REPO_ADAPTERS_DIR
ADAPTERS_DIR = paths.ADAPTERS_DIR
TRAINING_RUNS_DIR = paths.TRAINING_RUNS_DIR
HF_ADAPTERS_DIR = paths.HF_ADAPTERS_DIR
OUT_DIR = paths.OUT_DIR

# Adapter LoRA padrao (pt-BR) publicado no Hugging Face.
ADAPTER_REPO = paths.ADAPTER_REPO
ADAPTER_REPO_PAGE = paths.ADAPTER_REPO_PAGE
ADAPTER_REPO_URL = paths.ADAPTER_REPO_URL

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
    """Baixa (best-effort) o adapter LoRA do Hugging Face para a pasta do repo.

    Retorna a pasta local do adapter ou None se nao deu para baixar. Quem quiser a
    falha explicita (para mostrar na UI) deve usar `download_adapter()`.
    """
    repo_id = repo_id if repo_id is not None else paths.ADAPTER_REPO
    if not repo_id:
        return None
    try:
        return download_adapter(repo_id, dest=dest)
    except Exception as exc:  # noqa: BLE001
        msg = str(exc)
        if "404" in msg or "Repository Not Found" in msg:
            print(f"[setup] adapter '{repo_id}' nao encontrado no Hugging Face (repo inexistente ou privado); "
                  "seguindo com os adapters locais. Para desativar o download, deixe "
                  "PTBR_ADAPTER_REPO vazio no .env.", flush=True)
        else:
            print(f"[setup] (aviso) nao baixei o adapter {repo_id}: {msg}", flush=True)
        return None


# --------------------------------------------------------------- adapters LoRA
def adapter_sources() -> list[tuple[str | None, Path]]:
    """Fontes varridas por adapters: (prefixo do rotulo, pasta), em ordem de busca.

    A pasta `adapters/` da raiz do repo vem sempre primeiro; o legado
    `HF_ADAPTERS_DIR` (downloads antigos, fora do repo) leva o prefixo `hf/`.
    """
    return [
        ("hf", d) if d == HF_ADAPTERS_DIR else (None, d)
        for d in adapter_search_dirs()
    ]


def adapter_search_dirs() -> list[Path]:
    """Pastas varridas por adapters LoRA (pasta do repo, runs de treino e legado)."""
    return paths.adapter_search_dirs()


def find_adapters() -> dict[str, Path]:
    """Rotulo -> pasta de cada adapter LoRA encontrado nas fontes conhecidas."""
    return ADS.find_adapters(adapter_sources())


def adapter_dir_for(repo_id: str | None = None) -> Path:
    """Pasta local (dentro do repo) onde mora o adapter de `repo_id`."""
    repo_id = repo_id if repo_id is not None else ADAPTER_REPO
    return ADS.adapter_dir_in(REPO_ADAPTERS_DIR, repo_id or "")


def is_adapter_installed(path) -> bool:
    """True se a pasta tem `adapter_config.json` + pesos."""
    return ADS.is_installed(path)


def adapter_installed(repo_id: str | None = None, dest=None) -> Path | None:
    """Pasta do adapter se ele ja estiver baixado/instalado; None se faltar."""
    target = Path(dest) if dest else adapter_dir_for(repo_id)
    return target if ADS.is_installed(target) else None


def prepare_adapter(path) -> str:
    """Deixa a pasta do adapter pronta para o PEFT (nome de pesos aceito) e devolve str.

    Idempotente e barato; pastas que nao existem (ex.: um id do Hugging Face) passam
    direto, sem alteracao.
    """
    p = Path(str(path)).expanduser()
    if p.is_dir():
        ADS.normalize_adapter_dir(p, log=lambda m: print(m, flush=True))
    return str(path)


def download_adapter(repo_id: str | None = None, dest=None, log=None, force: bool = False) -> Path:
    """Baixa o adapter LoRA do Hugging Face para `<raiz do repo>/adapters/<nome>`.

    Explica no log de ONDE o modelo esta sendo baixado (repo + destino), salva arquivo
    por arquivo (progresso no terminal) e normaliza o nome dos pesos para o PEFT.
    Levanta RuntimeError com mensagem clara se o download nao produzir um adapter valido.
    """
    import huggingface_hub as hf

    repo_id = (repo_id if repo_id is not None else ADAPTER_REPO or "").strip()
    if not repo_id:
        raise RuntimeError("nenhum repo de adapter configurado (PTBR_ADAPTER_REPO vazio no .env)")
    log = log or (lambda m: print(m, flush=True))
    target = Path(dest) if dest else adapter_dir_for(repo_id)
    if ADS.is_installed(target) and not force:
        ADS.normalize_adapter_dir(target, log=log)
        log(f"[adapter] ja instalado em {target} (nada a baixar)")
        return target

    target.mkdir(parents=True, exist_ok=True)
    log(f"[adapter] baixando o modelo LoRA de '{repo_id}' (Hugging Face)")
    log(f"[adapter] origem:  https://huggingface.co/{repo_id}/tree/main")
    log(f"[adapter] destino: {target}")

    files: list[str] = []
    try:
        files = [
            f for f in hf.HfApi().list_repo_files(repo_id, token=paths.HF_TOKEN)
            if not f.startswith(".") and not f.endswith("/")
        ]
    except Exception as exc:  # noqa: BLE001
        log(f"[adapter] (aviso) nao listei os arquivos do repo ({exc}); baixando o repo inteiro")
    if files:
        for i, name in enumerate(files, 1):
            log(f"[adapter] ({i}/{len(files)}) {name}")
            hf.hf_hub_download(repo_id=repo_id, filename=name,
                               local_dir=str(target), token=paths.HF_TOKEN)
    else:
        _hf_download(repo_id, target)

    ADS.normalize_adapter_dir(target, log=log)
    if not ADS.is_installed(target):
        raise RuntimeError(
            f"o download de '{repo_id}' terminou mas {target} nao tem "
            f"'{ADS.CONFIG_NAME}' + pesos. Baixe manualmente em "
            f"https://huggingface.co/{repo_id}/tree/main e coloque os arquivos nessa pasta."
        )
    log(f"[adapter] pronto para uso: {target}")
    return target


def load_text_tokenizer():
    ensure_base_model()
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(CKPT)


def load_audio_tokenizer(device: str = "cuda"):
    ensure_base_model()
    Qwen3TTSTokenizer = import_qwen_tts()

    return Qwen3TTSTokenizer.from_pretrained(str(CKPT / "audio_tokenizer"), device_map=device)


def load_breeze_model(device: str = "cuda", attn: str = "eager"):
    ensure_base_model()
    import torch
    from models.breeze import BreezeForConditionalGeneration

    model = BreezeForConditionalGeneration.from_pretrained(
        CKPT, dtype=torch.bfloat16, attn_implementation=attn
    )
    return model.to(device).eval()


# ---------------------------------------------------- cache de codes da referencia
# O Breeze nao tem embedding de falante puro: a voz vem dos codes do codec da
# referencia + transcript. Codificamos a referencia UMA vez e reutilizamos os
# codes (equivalente ao .breeze do breeze-cli): pula o encode e garante codes
# identicos a cada geracao. Cache em memoria + .npz persistente.
_code_cache: dict[str, "object"] = {}
_orig_encode_prompt_audio = None


def enable_code_cache() -> None:
    """Patch em breeze_infer.templates._encode_prompt_audio para usar o cache."""
    global _orig_encode_prompt_audio
    if _orig_encode_prompt_audio is not None:
        return
    import breeze_infer.templates as T

    def _patched(audio_tokenizer, audio_path):
        hit = _code_cache.get(str(Path(audio_path)))
        if hit is not None:
            return hit.clone()
        return _orig_encode_prompt_audio(audio_tokenizer, audio_path)

    _orig_encode_prompt_audio = T._encode_prompt_audio
    T._encode_prompt_audio = _patched


def _codes_npz(wav_path, cache_dir=None) -> Path:
    d = Path(cache_dir) if cache_dir else (paths.ARTIFACTS / "voice_cache")
    d.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1(str(Path(wav_path).resolve()).encode("utf-8")).hexdigest()[:16]
    return d / f"{key}.npz"


def reference_codes_path(wav_path, cache_dir=None) -> Path:
    return _codes_npz(wav_path, cache_dir)


def cache_reference(audio_tok, wav_path, cache_dir=None):
    """Codifica a referencia 1x (npz persistente) e registra no cache. Retorna o tensor."""
    import numpy as np
    import torch

    enable_code_cache()
    key = str(Path(wav_path))
    if key in _code_cache:
        return _code_cache[key]

    npz = _codes_npz(wav_path, cache_dir)
    if npz.is_file():
        arr = np.load(npz)["codes"]
    else:
        from breeze_infer.audio import encode_prompt_audio

        codes = encode_prompt_audio(audio_tok, str(wav_path))
        arr = codes.numpy().astype(np.int16)
        np.savez_compressed(npz, codes=arr)
    _code_cache[key] = torch.from_numpy(np.ascontiguousarray(arr))
    return _code_cache[key]


def cached_reference(wav_path):
    return _code_cache.get(str(Path(wav_path)))


# ------------------------------------------- padronizacao da referencia em 48 kHz
REF_SR = 48_000


def reference_48k(wav_path, cache_dir=None) -> Path:
    """Devolve um WAV mono **48 kHz** do arquivo de referencia (cacheado).

    Por que: a taxa do arquivo de referencia muda os *codes* do codec (medimos 24k
    vs 48k = ~70% de codes iguais), porque o codec reamostra internamente. Entao
    padronizamos a referencia em 48 kHz para nao descartar o conteudo agudo. Se a
    origem for < 48 kHz, reamostra para cima (padronizacao; nao recupera o que nao
    existe). O arquivo original nao e alterado.
    """
    import hashlib

    import librosa
    import numpy as np
    import soundfile as sf

    src = Path(wav_path)
    d = Path(cache_dir) if cache_dir else (paths.ARTIFACTS / "voice_cache")
    d.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1(
        f"{src.resolve()}|{src.stat().st_mtime_ns}".encode("utf-8")
    ).hexdigest()[:16]
    out = d / f"ref48_{key}.wav"
    if out.is_file():
        return out
    wav, sr = sf.read(str(src), dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    if sr != REF_SR:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=REF_SR)
    sf.write(str(out), np.clip(wav, -1.0, 1.0).astype("float32"), REF_SR, subtype="PCM_16")
    return out


# --------------------------------------------- similaridade de locutor (ECAPA)
_ecapa = None


def _load_ecapa(device: str = "cpu"):
    global _ecapa
    if _ecapa is None:
        try:
            from speechbrain.inference.speaker import EncoderClassifier
        except Exception:  # noqa: BLE001
            from speechbrain.pretrained import EncoderClassifier
        kw = {}
        try:  # Windows sem privilegio de admin/modo desenvolvedor nao cria symlink (WinError 1314)
            from speechbrain.utils.fetching import LocalStrategy

            kw["local_strategy"] = LocalStrategy.COPY
        except Exception:  # noqa: BLE001  (speechbrain antigo: sem o parametro)
            pass
        _ecapa = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=str(paths.ARTIFACTS / "voice_cache" / "spkrec-ecapa"),
            run_opts={"device": device},
            **kw,
        )
    return _ecapa


def speaker_embed(wav_path, device: str = "cpu"):
    """Embedding ECAPA (cos) de um WAV/array — usado para escolher a melhor seed."""
    import librosa
    import numpy as np
    import torch

    clf = _load_ecapa(device)
    wav, _sr = librosa.load(str(wav_path), sr=16000, mono=True)
    x = torch.from_numpy(np.asarray(wav, dtype=np.float32)).unsqueeze(0).to(device)
    with torch.no_grad():
        e = clf.encode_batch(x).squeeze()
    e = e / (e.norm() + 1e-9)
    return e.detach().cpu().numpy().astype(np.float32)


def cos(a, b) -> float:
    import numpy as np

    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
