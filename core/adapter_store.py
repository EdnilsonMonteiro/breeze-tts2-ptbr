"""adapter_store.py — localiza e prepara adapters LoRA no disco (sem GPU, sem torch).

O adapter (PEFT/LoRA) pode estar em tres layouts; todos identificados pela presenca
de `adapter_config.json`:

  1) <pasta>/<nome>/adapter_config.json                   plano: baixado do HF ou copiado
  2) <pasta>/<run>/checkpoints/<ckpt>/adapter_config.json layout de treino
  3) <pasta>/<nome>/<sub>/adapter_config.json             subpasta do repo HF (busca rasa)

Este modulo NAO conhece caminhos do repo: quem chama passa as pastas
(`core/common_breeze.py` usa `paths.adapter_search_dirs()`), o que o torna testavel
com pastas temporarias.

Detalhe que ja quebrou o carregamento: o PEFT so aceita os pesos chamados
`adapter_model.safetensors` (ou `adapter_model.bin`). Varios repos publicam com outro
nome — o `EdnilsonMonts/Breeze-tts-2-brazillian-lora`, por exemplo, publica
`breeze-tts-2-pt-br-lora.safetensors`. `normalize_adapter_dir()` cria o nome esperado
por **hardlink** (mesmo inode; nao duplica os ~570 MB em disco).
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Callable, Iterable, Sequence

CONFIG_NAME = "adapter_config.json"
SAFETENSORS_NAME = "adapter_model.safetensors"
BIN_NAME = "adapter_model.bin"

# Ordem de preferencia dos nomes de pesos ja prontos para o PEFT.
_READY_NAMES = (SAFETENSORS_NAME, BIN_NAME)
# Padroes aceitos como "pesos publicados com outro nome".
_RAW_GLOBS = ("*.safetensors", "*.bin")

Log = Callable[[str], None]


# ------------------------------------------------------------------ identificacao
def is_adapter_dir(path) -> bool:
    """True se a pasta tem `adapter_config.json` (e portanto e um adapter)."""
    return (Path(path) / CONFIG_NAME).is_file()


def adapter_weight_file(path) -> Path | None:
    """Pesos com o nome que o PEFT exige, ou None."""
    p = Path(path)
    for name in _READY_NAMES:
        f = p / name
        if f.is_file():
            return f
    return None


def published_weight_file(path) -> Path | None:
    """Pesos publicados com outro nome (ex.: `breeze-tts-2-pt-br-lora.safetensors`)."""
    cands = published_weight_files(path)
    return cands[0] if cands else None


def published_weight_files(path) -> list[Path]:
    """Todos os candidatos a pesos publicados, do mais provavel ao menos provavel."""
    p = Path(path)
    cands: list[Path] = []
    for pat in _RAW_GLOBS:
        for f in p.glob(pat):
            if f.is_file() and f.name not in _READY_NAMES:
                cands.append(f)
    # Mais de um = provavelmente checkpoint fatiado (shards) ou varios treinos juntos;
    # prefere o que parece LoRA e, em empate, o maior.
    cands.sort(key=lambda f: ("lora" not in f.name.lower(), -_size(f), f.name))
    return cands


def has_weights(path) -> bool:
    """A pasta tem pesos (com nome pronto para o PEFT ou publicado com outro nome)?"""
    return adapter_weight_file(path) is not None or published_weight_file(path) is not None


def is_installed(path) -> bool:
    """Pasta pronta para uso: `adapter_config.json` + algum arquivo de pesos."""
    return is_adapter_dir(path) and has_weights(path)


def _size(f: Path) -> int:
    try:
        return f.stat().st_size
    except OSError:
        return 0


# ------------------------------------------------------------------ normalizacao
def normalize_adapter_dir(path, log: Log | None = None) -> Path | None:
    """Garante que exista o nome de pesos que o PEFT carrega.

    Cria `adapter_model.safetensors` (ou `.bin`) a partir do arquivo publicado, por
    hardlink quando o sistema de arquivos permite (sem duplicar espaco); senao copia.
    Retorna o arquivo de pesos pronto para uso, ou None se nao houver pesos.
    """
    p = Path(path)
    ready = adapter_weight_file(p)
    if ready is not None:
        return ready
    cands = published_weight_files(p)
    if not cands:
        return None
    src = cands[0]
    if len(cands) > 1 and log:
        log(f"[adapter] (aviso) {p} tem {len(cands)} arquivos de pesos; usando '{src.name}'. "
            "Se o checkpoint for fatiado (shards), carregue-o pelo nome correto/index.")
    dst = p / (SAFETENSORS_NAME if src.suffix.lower() == ".safetensors" else BIN_NAME)
    if dst.exists():
        return dst
    how = "hardlink"
    try:
        os.link(src, dst)                      # mesmo inode: nao duplica os ~570 MB
    except OSError:
        how = "copia"
        shutil.copy2(src, dst)
    if log:
        log(f"[adapter] pesos publicados como '{src.name}'; criado '{dst.name}' "
            f"({how}) porque o PEFT so procura por esse nome")
    return dst


# ------------------------------------------------------------------ descoberta
def iter_adapter_dirs(root, max_depth: int = 3) -> list[Path]:
    """Pastas com `adapter_config.json` sob `root`, ate `max_depth` niveis.

    `max_depth=3` cobre os tres layouts (plano, `<run>/checkpoints/<ckpt>` e subpasta
    de repo) sem varrer a arvore inteira de uma pasta grande.
    """
    base = Path(root) if root else None
    if base is None or not base.is_dir():
        return []
    found: list[Path] = []

    def walk(d: Path, depth: int) -> None:
        try:
            entries = sorted(d.iterdir(), key=lambda e: e.name)
        except OSError:
            return
        for e in entries:
            try:
                if not e.is_dir() or e.name.startswith("."):   # .git, .cache do hub, ...
                    continue
            except OSError:
                continue
            if is_adapter_dir(e):
                found.append(e)
            if depth < max_depth:
                walk(e, depth + 1)

    walk(base, 1)
    return found


def find_adapters(sources: Sequence[tuple[str | None, Path]], max_depth: int = 3) -> dict:
    """Mapeia rotulo -> pasta do adapter.

    `sources` e uma sequencia de `(prefixo, pasta)`: o rotulo e o caminho relativo
    dentro da pasta, opcionalmente prefixado (ex.: `("hf", dir)` -> `hf/<nome>`).
    Pastas repetidas (mesmo adapter achado em duas raizes) entram uma vez so.
    """
    out: dict[str, Path] = {}
    seen: set[str] = set()
    for prefix, root in sources:
        if not root:
            continue
        base = Path(root)
        for ad in iter_adapter_dirs(base, max_depth):
            try:
                key = str(ad.resolve())
                rel = ad.relative_to(base).as_posix()
            except (OSError, ValueError):
                key, rel = str(ad), ad.name
            if key in seen:
                continue
            seen.add(key)
            label = f"{prefix}/{rel}" if prefix else rel
            if label in out:                        # homonimos em raizes diferentes
                n = 2
                while f"{label} ({n})" in out:
                    n += 1
                label = f"{label} ({n})"
            out[label] = ad
    return out


def repo_name(repo_id: str) -> str:
    """`org/nome` -> `nome` (nome da pasta local do adapter)."""
    return (repo_id or "").rstrip("/").split("/")[-1]


def adapter_dir_in(base, repo_id_or_name: str) -> Path:
    """Pasta local de um adapter dentro de `base` (`<base>/<nome>`)."""
    return Path(base) / repo_name(repo_id_or_name)


def labels_for_path(labels: Iterable[str], paths: dict, path) -> str | None:
    """Rotulo cujo caminho (em `paths`) aponta para `path` (comparacao por resolve)."""
    try:
        want = str(Path(str(path)).resolve())
    except OSError:
        want = str(path)
    for label in labels:
        p = paths.get(label)
        if not p:
            continue
        try:
            if str(Path(str(p)).resolve()) == want:
                return label
        except OSError:
            continue
    return None
