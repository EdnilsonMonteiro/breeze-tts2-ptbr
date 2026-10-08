"""Testes da descoberta/normalizacao de adapters LoRA (core/adapter_store.py).

O caso que motivou `normalize_adapter_dir`: o repo
`EdnilsonMonts/Breeze-tts-2-brazillian-lora` publica os pesos como
`breeze-tts-2-pt-br-lora.safetensors`, mas o PEFT so procura
`adapter_model.safetensors` (ou `adapter_model.bin`).
"""
from __future__ import annotations

import os

import adapter_store as ADS


def _adapter(d, weights=ADS.SAFETENSORS_NAME, size=1024):
    """Cria uma pasta de adapter com `adapter_config.json` + pesos."""
    d.mkdir(parents=True, exist_ok=True)
    (d / ADS.CONFIG_NAME).write_text("{}", encoding="utf-8")
    if weights:
        (d / weights).write_bytes(b"0" * size)
    return d


# ------------------------------------------------------------------ identificacao
def test_is_adapter_dir_and_installed(workdir):
    full = _adapter(workdir / "cheio")
    only_cfg = workdir / "so_config"
    only_cfg.mkdir()
    (only_cfg / ADS.CONFIG_NAME).write_text("{}", encoding="utf-8")
    (workdir / "vazia").mkdir()

    assert ADS.is_adapter_dir(full) and ADS.is_installed(full)
    assert ADS.is_adapter_dir(only_cfg) and not ADS.is_installed(only_cfg)   # sem pesos
    assert not ADS.is_adapter_dir(workdir / "vazia")
    assert not ADS.is_installed(workdir / "nao_existe")


def test_published_weight_prefers_lora_and_then_biggest(workdir):
    d = _adapter(workdir / "mix", weights=None)
    (d / "optimizer.bin").write_bytes(b"0" * 10)
    (d / "breeze-tts-2-pt-br-lora.safetensors").write_bytes(b"0" * 20)
    (d / "outro.safetensors").write_bytes(b"0" * 9999)
    assert ADS.published_weight_file(d).name == "breeze-tts-2-pt-br-lora.safetensors"
    assert [f.name for f in ADS.published_weight_files(d)][0] == "breeze-tts-2-pt-br-lora.safetensors"


def test_normalize_warns_when_there_are_several_weight_files(workdir):
    """Varios candidatos (ex.: checkpoint fatiado) nao passam em silencio."""
    d = _adapter(workdir / "varios", weights=None)
    (d / "a.safetensors").write_bytes(b"0" * 10)
    (d / "b.safetensors").write_bytes(b"0" * 20)
    logs = []
    out = ADS.normalize_adapter_dir(d, log=logs.append)
    assert out == d / ADS.SAFETENSORS_NAME and any("aviso" in m for m in logs)


# ------------------------------------------------------------------ normalizacao
def test_normalize_creates_peft_name_without_duplicating(workdir, monkeypatch):
    src_name = "breeze-tts-2-pt-br-lora.safetensors"
    d = _adapter(workdir / "hf", weights=src_name, size=2048)
    used = []
    real_link = os.link

    def spy(s, t):
        used.append((str(s), str(t)))
        return real_link(s, t)

    monkeypatch.setattr(ADS.os, "link", spy)        # hardlink e o caminho preferido
    logs = []
    out = ADS.normalize_adapter_dir(d, log=logs.append)

    assert out == d / ADS.SAFETENSORS_NAME
    assert out.is_file() and out.stat().st_size == 2048
    assert used == [(str(d / src_name), str(out))]  # mesmo inode: zero espaco extra
    assert out.stat().st_ino == (d / src_name).stat().st_ino
    assert ADS.is_installed(d) and logs and "adapter_model.safetensors" in logs[0]


def test_normalize_copies_when_hardlink_is_not_possible(workdir, monkeypatch):
    d = _adapter(workdir / "hf", weights="pesos-ptbr.safetensors", size=16)

    def no_link(*_a, **_k):
        raise OSError("hardlink indisponivel")

    monkeypatch.setattr(ADS.os, "link", no_link)
    out = ADS.normalize_adapter_dir(d)
    assert out.is_file() and out.read_bytes() == b"0" * 16


def test_normalize_is_idempotent_and_handles_missing_weights(workdir):
    ready = _adapter(workdir / "pronto")                 # ja tem adapter_model.safetensors
    before = (ready / ADS.SAFETENSORS_NAME).stat().st_mtime_ns
    assert ADS.normalize_adapter_dir(ready) == ready / ADS.SAFETENSORS_NAME
    assert (ready / ADS.SAFETENSORS_NAME).stat().st_mtime_ns == before

    sem = workdir / "sem_pesos"
    sem.mkdir()
    (sem / ADS.CONFIG_NAME).write_text("{}", encoding="utf-8")
    assert ADS.normalize_adapter_dir(sem) is None and not ADS.is_installed(sem)


def test_normalize_uses_bin_name_for_bin_weights(workdir):
    d = _adapter(workdir / "bin", weights="pesos.bin", size=8)
    assert ADS.normalize_adapter_dir(d) == d / ADS.BIN_NAME


# ------------------------------------------------------------------ descoberta
def test_iter_adapter_dirs_covers_flat_training_and_nested(workdir):
    root = workdir / "adapters"
    flat = _adapter(root / "Breeze-tts-2-brazillian-lora")
    treino = _adapter(root / "r76" / "checkpoints" / "step1500")
    aninhado = _adapter(root / "outro-repo" / "checkpoint-500")
    (root / "rascunho").mkdir(parents=True)                  # pasta sem adapter: ignorada
    (root / ".git").mkdir()
    _adapter(root / ".git" / "lixo")                         # pastas ocultas: ignoradas

    found = {p.relative_to(root).as_posix() for p in ADS.iter_adapter_dirs(root)}
    assert found == {
        "Breeze-tts-2-brazillian-lora",
        "r76/checkpoints/step1500",
        "outro-repo/checkpoint-500",
    }
    assert flat in ADS.iter_adapter_dirs(root) and treino in ADS.iter_adapter_dirs(root)
    # raso (2 niveis) acha o plano e a subpasta do repo, mas nao o layout de treino
    raso = ADS.iter_adapter_dirs(root, max_depth=2)
    assert aninhado in raso and flat in raso and treino not in raso


def test_iter_adapter_dirs_tolerates_missing_root(workdir):
    assert ADS.iter_adapter_dirs(workdir / "nao_existe") == []
    assert ADS.iter_adapter_dirs(None) == []


def test_find_adapters_labels_prefixes_and_dedup(workdir):
    repo = workdir / "adapters"
    runs = workdir / "runs"
    hf = workdir / "hf"
    _adapter(repo / "Breeze-tts-2-brazillian-lora")
    _adapter(runs / "r76" / "checkpoints" / "step1500")
    _adapter(hf / "adapter-antigo")
    _adapter(repo / "duplicado")
    _adapter(runs / "duplicado")                              # mesma pasta em raiz diferente

    found = ADS.find_adapters([(None, repo), (None, runs), ("hf", hf)])
    assert list(found) == [
        "Breeze-tts-2-brazillian-lora",       # pasta do repo (prioridade)
        "duplicado",
        "duplicado (2)",                      # homonimo: desambiguado
        "r76/checkpoints/step1500",           # layout de treino (rotulo igual ao de antes)
        "hf/adapter-antigo",                  # legado: prefixo hf/
    ]
    assert found["Breeze-tts-2-brazillian-lora"] == repo / "Breeze-tts-2-brazillian-lora"
    assert found["r76/checkpoints/step1500"] == runs / "r76" / "checkpoints" / "step1500"


def test_find_adapters_ignores_repeated_path_and_missing_sources(workdir):
    repo = workdir / "adapters"
    _adapter(repo / "a")
    assert list(ADS.find_adapters([(None, repo), (None, repo), (None, None)])) == ["a"]


def test_find_adapters_skips_non_adapter_dirs(workdir):
    root = workdir / "adapters"
    (root / "vazia").mkdir(parents=True)
    (root / "so_config" / ADS.CONFIG_NAME).parent.mkdir(parents=True)
    (root / "so_config" / ADS.CONFIG_NAME).write_text("{}", encoding="utf-8")
    # entra na lista (tem adapter_config.json); quem decide se carrega e o PEFT/UI
    assert list(ADS.find_adapters([(None, root)])) == ["so_config"]


# ------------------------------------------------------------------ utilidades
def test_repo_name_and_adapter_dir_in(workdir):
    assert ADS.repo_name("EdnilsonMonts/Breeze-tts-2-brazillian-lora") == "Breeze-tts-2-brazillian-lora"
    assert ADS.repo_name("Breeze-tts-2-brazillian-lora/") == "Breeze-tts-2-brazillian-lora"
    assert ADS.adapter_dir_in(workdir, "org/nome") == workdir / "nome"


def test_labels_for_path_matches_by_resolved_path(workdir):
    d = _adapter(workdir / "adapters" / "meu")
    paths = {"meu": str(d), "outro": str(workdir / "nada")}
    assert ADS.labels_for_path(["meu", "outro"], paths, d) == "meu"
    assert ADS.labels_for_path(["meu"], paths, workdir / "nao_existe") is None
