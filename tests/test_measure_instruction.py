"""Testes da analise de `scripts/measure_instruction.py` (logica pura, sem GPU/audio).

Caso real que motivou a ferramenta: no adaptador pt-BR, trocar "neutro" por "empolgado"
mexe no audio menos do que trocar a seed (a instrucao e inerte); no modelo base, a
instrucao domina. O `analyse()` tem de separar os dois casos.
"""
from __future__ import annotations

import measure_instruction as MI


def _row(instruction, seed, *, cfg=4.0, f0=200.0, dur=5.0, rms=0.05, words=3.0, mfcc=0.0):
    """Amostra sintetica com as metricas que `analyse` compara."""
    return {
        "tag": f"{instruction[:6]}_{seed}", "instruction": instruction, "seed": seed, "cfg": cfg,
        "f0_med": f0, "dur_s": dur, "rms_mean": rms, "words_s": words,
        "mfcc": [mfcc] * 26,
    }


def test_instrucao_que_funciona_ganha_do_ruido():
    """Modelo base: instrucao muda F0/duracao muito mais que a seed."""
    rows = [
        _row("neutro", 42, f0=225.0, dur=5.04),
        _row("neutro", 43, f0=200.0, dur=5.30),      # ruido de seed
        _row("empolgado", 42, f0=353.0, dur=3.28),
        _row("empolgado", 43, f0=351.0, dur=3.68),   # ruido de seed
    ]
    res = MI.analyse(rows)
    # 2 instrucoes x 2 seeds: 1 par de instrucao por seed e 1 par de seed por instrucao
    assert res["n_instrucao"] == 2 and res["n_ruido"] == 2
    assert res["razao"]["f0_med"] > 2 and res["razao"]["dur_s"] > 2
    assert res["veredito"].startswith("INSTRUCAO FUNCIONA")


def test_instruncao_inerte_quando_nenhuma_metrica_supera_o_ruido():
    """Adaptador pt-BR: efeito da instrucao abaixo do ruido de seed em tudo."""
    rows = [
        _row("neutro", 42, f0=183.0, dur=5.28),
        _row("neutro", 43, f0=178.0, dur=8.88),      # ruido de seed grande
        _row("empolgado", 42, f0=207.0, dur=6.16),
        _row("empolgado", 43, f0=170.0, dur=5.76),
    ]
    res = MI.analyse(rows)
    assert res["razao"]["f0_med"] <= 1.0 and res["razao"]["dur_s"] <= 1.0
    assert res["veredito"].startswith("INSTRUCAO INERTE")


def test_efeito_fraco_quando_uma_metrica_so_supera_o_ruido():
    """F0 com 1,8x o ruido (mas sem folga forte) e duracao dentro do ruido -> fraco."""
    rows = [
        _row("neutro", 42, f0=200.0, dur=5.0),
        _row("neutro", 43, f0=190.0, dur=6.0),       # ruido de seed
        _row("empolgado", 42, f0=218.0, dur=4.4),    # efeito: +18 Hz (1,8x), duracao 0,6s
        _row("empolgado", 43, f0=208.0, dur=5.4),
    ]
    res = MI.analyse(rows)
    assert 1.5 <= res["razao"]["f0_med"] < MI.RAZAO_FORTE
    assert res["veredito"].startswith("EFEITO FRACO")


def test_sem_pares_suficientes_e_inconclusivo():
    uma_seed = [_row("a", 42), _row("b", 42)]        # sem par de seed -> sem piso de ruido
    assert MI.analyse(uma_seed)["veredito"].startswith("INCONCLUSIVO")
    sem_par = [_row("neutro", 42), _row("neutro", 43)]   # so uma instrucao
    assert MI.analyse(sem_par)["veredito"].startswith("INCONCLUSIVO")


def test_cfg_diferente_nao_conta_como_par_de_instrucao():
    """Trocar o cfg nao e trocar a instrucao (nao pode inflar o efeito medido)."""
    rows = [
        _row("neutro", 42, cfg=1.0, f0=180.0),
        _row("neutro", 42, cfg=4.0, f0=175.0),       # mesma instrucao/seed, cfg diferente
        _row("neutro", 43, cfg=4.0, f0=178.0),
        _row("empolgado", 42, cfg=4.0, f0=185.0),
        _row("empolgado", 43, cfg=4.0, f0=182.0),
    ]
    res = MI.analyse(rows)
    assert res["n_instrucao"] == 2                   # so os pares mesma seed+cfg
    assert res["n_ruido"] == 2


def test_distance_e_slug():
    a = _row("x", 1, f0=100.0, dur=2.0, rms=0.1, words=2.0, mfcc=1.0)
    b = _row("y", 2, f0=140.0, dur=3.0, rms=0.05, words=4.0, mfcc=3.0)
    d = MI.distance(a, b)
    assert d["f0_med"] == 40.0 and d["dur_s"] == 1.0 and d["words_s"] == 2.0
    assert abs(d["rms_mean"] - 0.5) < 1e-9          # relativa ao maior
    assert d["mfcc"] == 2.0
    assert MI.slug("Fale com muita empolgacao, energia!") == "fale-com-muita-empolgacao-en"
    assert MI.slug("", ) == "x"
