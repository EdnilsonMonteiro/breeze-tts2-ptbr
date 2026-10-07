import numpy as np

import candidate_select as CS


def _w(n=2400, v=0.1):
    return np.full(n, v, dtype=np.float32)


def test_wer_basic_and_normalization():
    assert CS.wer("Olá, mundo!", "ola mundo") == 0.0
    assert CS.wer("um dois tres quatro", "um dois quatro") == 0.25
    assert CS.wer("a b", "") == 1.0
    assert CS.wer("", "") == 0.0
    # norm injetada aplicada nos dois lados
    assert CS.wer("tenho dois gatos", "tenho 2 gatos", norm=lambda s: s.replace("2", "dois")) == 0.0
    assert CS.wer("bem-vindo ao show", "bem vindo ao show") == 0.0


def test_equalize_rms_clips_and_limits_gain():
    w = _w(v=0.05)
    out = CS.equalize_rms(w, 0.1)
    assert abs(CS.rms(out) - 0.1) < 1e-3
    assert abs(CS.rms(CS.equalize_rms(w, 10.0, max_gain=2.0)) - 0.1) < 1e-3   # ganho limitado
    z = np.zeros(10, np.float32)
    assert (CS.equalize_rms(z, 0.1) == z).all()


def _c(i, wer=None, secs=None, ok=True):
    return CS.Cand(idx=i, seed=i, dur=1.0, dur_ok=ok, wer=wer, secs=secs)


def test_choose_prefers_no_word_error_over_higher_secs():
    cs = [_c(0, 0.05, 0.70), _c(1, 0.0, 0.62), _c(2, 0.0, 0.66), _c(3, 0.10, 0.80)]
    assert CS.choose(cs) == 2                                    # sem erro e maior SECS entre elas


def test_choose_tolerance_and_ties():
    cs = [_c(0, 0.0, 0.60), _c(1, 0.01, 0.65)]                   # dentro da tolerancia -> SECS decide
    assert CS.choose(cs, wer_tol=0.02) == 1
    assert CS.choose([_c(0, 0.0, 0.6), _c(1, 0.0, 0.6)]) == 0     # empate -> menor indice


def test_choose_duration_gate_and_fallbacks():
    cs = [_c(0, 0.0, 0.9, ok=False), _c(1, 0.0, 0.5, ok=True)]
    assert CS.choose(cs) == 1                                    # gate de duracao
    assert CS.choose([_c(0, 0.0, 0.9, ok=False), _c(1, 0.0, 0.5, ok=False)]) == 0   # ninguem passa -> ignora
    assert CS.choose([_c(0, 0.2), _c(1, 0.0), _c(2, 0.0)]) == 1  # sem SECS: menor WER
    assert CS.choose([_c(0, None, 0.5), _c(1, None, 0.7)]) == 1  # sem WER: SECS


def test_evaluate_full_pipeline_with_fakes():
    wavs = [_w(24000, 0.1), _w(24000, 0.2), _w(24000, 0.05)]
    hyps = {0.1: "o gato subiu", 0.2: "o gato subiu no", 0.05: "o gato subiu"}

    def asr(w, sr):
        return hyps[round(float(w[0]), 2)]

    ref = np.array([1.0, 0.0], np.float32)
    emb_list = iter([np.array([0.6, 0.8]), np.array([1.0, 0.1]), np.array([0.9, 0.2])])
    seen = []

    def emb(w, sr):
        seen.append(round(CS.rms(w), 3))
        return next(emb_list).astype(np.float32)

    pick, cands = CS.evaluate(
        wavs, 24000, "O gato subiu", [10, 11, 12], dur_ok=lambda d: True,
        transcribe=asr, embed=emb, ref_emb=ref, ref_rms=0.1,
    )
    assert len(cands) == 3 and cands[pick].chosen and sum(c.chosen for c in cands) == 1
    assert cands[1].wer is not None and cands[1].wer > 0                  # "no" extra
    assert pick == 2                                                       # sem erro + maior SECS entre 0 e 2
    assert all(abs(r - 0.1) < 1e-3 for r in seen)                          # RMS igualado ao da referencia
    assert [c.seed for c in cands] == [10, 11, 12]


def test_evaluate_without_ref_uses_centroid_and_survives_failures():
    wavs = [_w(2400), _w(2400), _w(2400)]
    E = [np.array([1.0, 0.0]), np.array([0.9, 0.1]), np.array([0.0, 1.0])]
    it = iter(E)
    pick, cands = CS.evaluate(wavs, 24000, "x", [0, 1, 2], dur_ok=lambda d: True,
                              embed=lambda w, sr: next(it).astype(np.float32))
    assert pick in (0, 1) and cands[2].secs < cands[0].secs              # outlier perde (medoid)

    def boom(w, sr):
        raise RuntimeError("sem whisper")

    pick, cands = CS.evaluate(wavs, 24000, "x", [0, 1, 2], dur_ok=lambda d: True, transcribe=boom,
                              embed=lambda w, sr: (_ for _ in ()).throw(RuntimeError("sem ecapa")))
    assert pick == 0 and "asr" in cands[0].error                         # degrada sem quebrar


def test_format_block_marks_chosen():
    cs = [CS.Cand(0, 5, 2.0, wer=0.0, secs=0.61), CS.Cand(1, 6, 2.0, wer=0.1, secs=0.7, dur_ok=False)]
    cs[0].chosen = True
    s = CS.format_block(0, cs, has_ref=True, has_wer=True)
    assert s.startswith("Bloco 1:") and "**#0 (seed 5) · erro 0% · voz 0.610 ✔**" in s and "duracao fora" in s
