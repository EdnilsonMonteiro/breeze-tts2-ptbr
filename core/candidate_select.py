"""candidate_select.py — escolhe o MELHOR de N candidatos de TTS. Modulo puro (sem torch/gradio).

Criterio (nesta ordem):
  1. duracao plausivel para o texto (descarta fala arrastada/truncada), se algum candidato passa;
  2. sem erro de palavra: WER (Whisper) <= menor WER + `wer_tol` (o 1o criterio do usuario:
     "o portugues erra palavras");
  3. entre os que sobram, maior SECS (cos ECAPA contra a referencia, com RMS igualado).
Sem referencia, o SECS vira "centralidade" (cos contra o centroide dos candidatos = medoid).

`transcribe(wav, sr) -> str` e `embed(wav, sr) -> np.ndarray` sao injetados (a UI passa o
faster-whisper e o ECAPA; os testes passam fakes). Se `transcribe` for None ou falhar, o
criterio 2 e ignorado; se `embed` falhar, escolhe so por WER/duracao.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np


@dataclass
class Cand:
    idx: int
    seed: int
    dur: float
    dur_ok: bool = True
    wer: float | None = None
    secs: float | None = None
    hyp: str = ""
    chosen: bool = False
    error: str = field(default="", repr=False)


# ---------------------------------------------------------------- texto / WER
def words(s: str) -> list[str]:
    """minusculas, sem acento nem pontuacao; hifen vira espaco."""
    s = unicodedata.normalize("NFKD", (s or "").lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[-_/]", " ", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return s.split()


def wer(ref: str, hyp: str, norm: Callable[[str], str] | None = None) -> float:
    """WER por palavras (Levenshtein / n palavras da referencia). `norm` (ex.: numeros por
    extenso) e aplicado aos dois lados antes de comparar."""
    if norm is not None:
        ref, hyp = norm(ref), norm(hyp)
    r, h = words(ref), words(hyp)
    if not r:
        return 0.0 if not h else 1.0
    prev = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        cur = [i] + [0] * len(h)
        for j, hw in enumerate(h, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rw != hw))
        prev = cur
    return prev[-1] / len(r)


# ---------------------------------------------------------------- audio / voz
def rms(w: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(w)))) if len(w) else 0.0


def equalize_rms(wav: np.ndarray, target: float, max_gain: float = 8.0) -> np.ndarray:
    """Iguala o RMS ao do alvo (o ECAPA e sensivel ao nivel; compara-se com o mesmo volume)."""
    r = rms(wav)
    if r < 1e-8 or target < 1e-8:
        return wav
    g = min(max_gain, target / r)
    return np.clip(wav * g, -1.0, 1.0).astype(np.float32)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def centroid(embs: Sequence[np.ndarray]) -> np.ndarray:
    E = np.stack([e / (np.linalg.norm(e) + 1e-9) for e in embs])
    c = E.mean(axis=0)
    return c / (np.linalg.norm(c) + 1e-9)


# ---------------------------------------------------------------- escolha
def choose(cands: Sequence[Cand], wer_tol: float = 0.02) -> int:
    """Posicao (em `cands`) do melhor candidato. Empates -> menor indice (seed base)."""
    pool = [c for c in cands if c.dur_ok] or list(cands)
    with_wer = [c for c in pool if c.wer is not None]
    if with_wer:
        best_wer = min(c.wer for c in with_wer)          # type: ignore[type-var]
        pool = [c for c in with_wer if c.wer <= best_wer + wer_tol]   # type: ignore[operator]
    with_secs = [c for c in pool if c.secs is not None]
    if with_secs:
        top = max(with_secs, key=lambda c: (c.secs, -c.idx))
        return list(cands).index(top)
    return list(cands).index(min(pool, key=lambda c: (c.wer if c.wer is not None else 0.0, c.idx)))


def evaluate(
    wavs: Sequence[np.ndarray],
    sr: int,
    text: str,
    seeds: Sequence[int],
    *,
    dur_ok: Callable[[float], bool],
    transcribe: Callable[[np.ndarray, int], str] | None = None,
    embed: Callable[[np.ndarray, int], np.ndarray] | None = None,
    ref_emb: np.ndarray | None = None,
    ref_rms: float | None = None,
    norm: Callable[[str], str] | None = None,
    wer_tol: float = 0.02,
) -> tuple[int, list[Cand]]:
    """Pontua os candidatos e devolve (posicao do escolhido, lista de Cand com .chosen)."""
    cands = [Cand(idx=i, seed=int(seeds[i]), dur=len(w) / float(sr), dur_ok=bool(dur_ok(len(w) / float(sr))))
             for i, w in enumerate(wavs)]
    if transcribe is not None:
        for c, w in zip(cands, wavs):
            try:
                c.hyp = transcribe(w, sr)
                c.wer = wer(text, c.hyp, norm)
            except Exception as exc:  # noqa: BLE001
                c.error = f"asr: {exc}"
    if embed is not None:
        try:
            tgt = ref_rms if (ref_rms and ref_emb is not None) else float(np.mean([rms(w) for w in wavs]))
            embs = [embed(equalize_rms(w, tgt), sr) for w in wavs]
            anchor = ref_emb if ref_emb is not None else centroid(embs)
            for c, e in zip(cands, embs):
                c.secs = cosine(e, anchor)
        except Exception as exc:  # noqa: BLE001
            for c in cands:
                c.error += f" embed: {exc}"
    pick = choose(cands, wer_tol)
    cands[pick].chosen = True
    return pick, cands


# ---------------------------------------------------------------- relatorio
def format_block(bi: int, cands: Sequence[Cand], has_ref: bool, has_wer: bool) -> str:
    """Uma linha Markdown por bloco: '#k (seed) WER x% | voz y' com o escolhido em negrito."""
    label = "voz" if has_ref else "centralidade"
    parts = []
    for c in cands:
        bits = [f"#{c.idx} (seed {c.seed})"]
        if has_wer and c.wer is not None:
            bits.append(f"erro {c.wer * 100:.0f}%")
        if c.secs is not None:
            bits.append(f"{label} {c.secs:.3f}")
        if not c.dur_ok:
            bits.append("duracao fora")
        txt = " · ".join(bits)
        parts.append(f"**{txt} ✔**" if c.chosen else txt)
    return f"Bloco {bi + 1}: " + " | ".join(parts)
