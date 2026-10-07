import os

import numpy as np
import pytest

pytest.importorskip("soundfile")
import soundfile as sf  # noqa: E402

import app  # noqa: E402  (ui/app.py com stubs)


def test_default_adapter_prefers_final_then_env(tmp_path, monkeypatch):
    a, b, c = tmp_path / "a", tmp_path / "b", tmp_path / "c"
    for d in (a, b, c):
        d.mkdir()
    os.utime(a, (100, 100))
    os.utime(b, (200, 200))
    os.utime(c, (300, 300))               # step4500 = mais novo
    labels = [app.BASE_LABEL, "r1/checkpoints/final", "r1/checkpoints/best", "r1/checkpoints/step4500"]
    app._STATE["adapter_paths"] = {labels[1]: str(a), labels[2]: str(b), labels[3]: str(c)}
    monkeypatch.setattr(app, "_load_ui_state", lambda: {})
    monkeypatch.delenv("PTBR_DEFAULT_ADAPTER", raising=False)
    assert app._default_adapter_label(labels) == "r1/checkpoints/best"      # final/best > step mais novo
    monkeypatch.setenv("PTBR_DEFAULT_ADAPTER", str(a))
    assert app._default_adapter_label(labels) == "r1/checkpoints/final"     # env por caminho
    monkeypatch.setattr(app, "_load_ui_state", lambda: {"adapter": labels[3]})
    assert app._default_adapter_label(labels) == labels[3]                  # ultimo usado ganha


def test_default_adapter_falls_back_to_base():
    app._STATE["adapter_paths"] = {}
    assert app._default_adapter_label([app.BASE_LABEL]) == app.BASE_LABEL


def test_generate_long_chunks_and_keeps_seed_for_single_block(monkeypatch):
    calls = []

    def fake_generate(text, seed, **kw):
        calls.append((text, seed, kw.get("ref_format")))
        return np.full(24000, 0.1, dtype=np.float32), 24000

    monkeypatch.setattr(app, "_generate", fake_generate)
    wav, sr, n = app._generate_long("Frase curta.", seed=42, ref_format="train")
    assert n == 1 and calls[0][1] == 42                                   # comportamento anterior
    calls.clear()
    long_text = " ".join(["Esta e uma frase razoavelmente longa para o teste de blocos."] * 6)
    wav, sr, n = app._generate_long(long_text, seed=7, max_block_s=4.0, gap_ms=100)
    assert n > 1 and len(calls) == n
    assert [c[1] for c in calls] == [7 + i * 1000 for i in range(n)]      # seeds por bloco
    assert len(wav) > n * 24000                                           # blocos + pausas


def test_generate_long_no_chunk(monkeypatch):
    monkeypatch.setattr(app, "_generate", lambda text, seed, **kw: (np.zeros(2400, np.float32), 24000))
    _, _, n = app._generate_long("a. " * 200, seed=1, auto_chunk=False)
    assert n == 1


def test_stash_reference_copies_and_is_stable(tmp_path):
    src = tmp_path / "in.wav"
    sf.write(str(src), np.zeros(2400, np.float32), 24000)
    p1 = app._stash_reference(str(src), "Minha Voz")
    p2 = app._stash_reference(str(src), "Minha Voz")
    assert p1 == p2 and os.path.isfile(p1) and "minha-voz" in p1
    assert p1 != str(src)


def test_save_voice_uses_upload_when_no_path(tmp_path, monkeypatch):
    src = tmp_path / "up.wav"
    sf.write(str(src), np.zeros(2400, np.float32), 24000)
    saved = {}
    monkeypatch.setattr(app, "_load_voices", lambda: dict(saved))
    monkeypatch.setattr(app, "_save_voices", lambda d: saved.update(d))
    app.save_voice("v1", str(src), "", "texto ref", app.BASE_LABEL, "Neutro / natural", "x", "S0",
                   0.7, 50, 1.0, 400, 1.0, False, 3.0, 3.0, 42, 0.8, "train")
    v = saved["v1"]
    assert v["ref_path"] and os.path.isfile(v["ref_path"])          # antes o upload era ignorado
    assert v["adapter_scale"] == 0.8 and v["ref_format"] == "train"
    with pytest.raises(Exception):
        app.save_voice("", None, "", "", app.BASE_LABEL, "", "", "S0", 0.7, 50, 1.0, 400, 1.0,
                       False, 3.0, 3.0, 42, 1.0, "train")


def test_parse_segments():
    segs = app.parse_segments("sad | oi\nsó texto\n# comentario\n", "Neutro")
    assert len(segs) == 2 and segs[1] == ("Neutro", "só texto")


def _fake_gen(values):
    """_generate falso: cada chamada devolve um wav de amplitude diferente (identifica o candidato)."""
    calls = []

    def fake(text, seed, **kw):
        calls.append(seed)
        return np.full(24000, values[len(calls) - 1], dtype=np.float32), 24000

    return fake, calls


def test_generate_long_best_of_picks_scored_candidate_and_reports(monkeypatch):
    fake, calls = _fake_gen([0.1, 0.2, 0.3, 0.4])
    monkeypatch.setattr(app, "_generate", fake)
    import candidate_select as CS

    def scorer(wavs, sr, btxt, seeds, gen_kwargs):
        assert seeds == [42, 43, 44, 45] and gen_kwargs["ref_format"] == "train"
        cs = [CS.Cand(i, s, 1.0, wer=0.0, secs=0.5 + 0.1 * i) for i, s in enumerate(seeds)]
        cs[2].chosen = True
        return 2, cs

    monkeypatch.setattr(app, "_score_block", scorer)
    rep = []
    wav, sr, n = app._generate_long("Frase curta.", seed=42, n_candidates=4, best_of=True,
                                    report=rep, ref_format="train")
    assert calls == [42, 43, 44, 45] and n == 1
    assert abs(float(wav[0]) - 0.3) < 1e-6                       # o escolhido (indice 2)
    assert len(rep) == 1 and rep[0].startswith("Bloco 1:")
    assert "Melhor de N" in app._report_md(rep)
    assert app._report_md([]) == ""


def test_generate_long_best_of_off_or_single_candidate_skips_scoring(monkeypatch):
    fake, calls = _fake_gen([0.1, 0.2])
    monkeypatch.setattr(app, "_generate", fake)
    monkeypatch.setattr(app, "_score_block", lambda *a, **k: (_ for _ in ()).throw(AssertionError("nao deve pontuar")))
    wav, _, _ = app._generate_long("Frase curta.", seed=5, n_candidates=1, best_of=True)
    assert calls == [5] and abs(float(wav[0]) - 0.1) < 1e-6
    calls.clear()
    app._generate_long("Frase curta.", seed=5, n_candidates=2, best_of=False)   # 2 candidatos sem best_of: pega o 0
    assert calls == [5, 6]


def test_generate_long_best_of_failure_falls_back(monkeypatch):
    fake, _ = _fake_gen([0.1, 0.2, 0.3])
    monkeypatch.setattr(app, "_generate", fake)
    monkeypatch.setattr(app, "_score_block", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr(app, "_medoid_pick", lambda wavs, sr, n: 1)
    wav, _, _ = app._generate_long("Frase curta.", seed=1, n_candidates=3, best_of=True)
    assert abs(float(wav[0]) - 0.2) < 1e-6                         # medoid como rede de seguranca


def test_score_block_wires_whisper_ecapa_and_reference(monkeypatch):
    texts = {0.1: "o gato subiu no telhado", 0.2: "o gato subiu no telhado", 0.3: "o pato subiu telhado"}
    monkeypatch.setattr(app, "_get_asr", lambda: object())
    monkeypatch.setattr(app, "_asr_text", lambda w, sr: texts[round(float(w[0]), 1)])
    embs = {0.1: np.array([0.6, 0.8]), 0.2: np.array([1.0, 0.1]), 0.3: np.array([1.0, 0.0])}
    # o embed recebe o wav ja com RMS igualado; identificamos pela ordem de chamada
    order = iter([0.1, 0.2, 0.3])
    monkeypatch.setattr(app, "_embed_wav", lambda w, sr: embs[next(order)].astype(np.float32))
    monkeypatch.setattr(app, "_ref_profile",
                        lambda ra, rt, rf: (np.array([1.0, 0.0], np.float32), 0.1))
    wavs = [np.full(24000 * 2, v, np.float32) for v in (0.1, 0.2, 0.3)]
    pick, cands = app._score_block(wavs, 24000, "O gato subiu no telhado.", [7, 8, 9],
                                   {"ref_audio": "r.wav", "ref_text": "x", "ref_format": "train"})
    assert pick == 1                                                # 0 e 1 sem erro; 1 mais perto da ref
    assert cands[2].wer > 0 and cands[1].wer == 0.0 and cands[1].chosen


def test_score_block_without_whisper_still_scores_by_voice(monkeypatch):
    def no_asr():
        raise ImportError("sem faster_whisper")

    monkeypatch.setitem(app._STATE, "asr_failed", False)
    monkeypatch.setattr(app, "_get_asr", no_asr)
    monkeypatch.setattr(app, "_ref_profile", lambda *a: (np.array([1.0, 0.0], np.float32), 0.1))
    order = iter([np.array([0.0, 1.0]), np.array([1.0, 0.0])])
    monkeypatch.setattr(app, "_embed_wav", lambda w, sr: next(order).astype(np.float32))
    wavs = [np.full(48000, 0.1, np.float32)] * 2
    pick, cands = app._score_block(wavs, 24000, "Duas palavras ok.", [1, 2], {"ref_audio": "r", "ref_text": "t"})
    assert pick == 1 and all(c.wer is None for c in cands)
    assert app._STATE["asr_failed"] is True
    app._STATE["asr_failed"] = False
