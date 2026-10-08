import os

import numpy as np
import pytest

pytest.importorskip("soundfile")
import soundfile as sf  # noqa: E402

import app  # noqa: E402  (ui/app.py com stubs)


def test_default_adapter_prefers_final_then_env(workdir, monkeypatch):
    a, b, c = workdir / "a", workdir / "b", workdir / "c"
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


# --------------------------------------------------- adapters na pasta do repo
def _fake_adapter(d, weights="adapter_model.safetensors"):
    d.mkdir(parents=True, exist_ok=True)
    (d / "adapter_config.json").write_text("{}", encoding="utf-8")
    (d / weights).write_bytes(b"0" * 32)
    return d


def test_refresh_adapters_finds_repo_folder_and_training_runs():
    """A lista vem da pasta `adapters/` do repo, dos runs de treino e do legado hf/<nome>."""
    repo_dir = app.CB.ADAPTERS_DIR
    runs_dir = app.CB.TRAINING_RUNS_DIR
    hf_dir = app.CB.HF_ADAPTERS_DIR
    for d in (repo_dir, runs_dir, hf_dir):
        if d.exists():
            __import__("shutil").rmtree(d)
    _fake_adapter(repo_dir / "Breeze-tts-2-brazillian-lora")
    _fake_adapter(runs_dir / "r76" / "checkpoints" / "step1500")
    _fake_adapter(hf_dir / "adapter-antigo")

    labels = app.refresh_adapters()
    assert labels == [
        app.BASE_LABEL,
        "Breeze-tts-2-brazillian-lora",        # pasta do repo (prioridade)
        "r76/checkpoints/step1500",            # layout de treino (rotulo preservado)
        "hf/adapter-antigo",                   # downloads antigos
    ]
    assert app._STATE["adapter_paths"]["Breeze-tts-2-brazillian-lora"] == str(
        repo_dir / "Breeze-tts-2-brazillian-lora"
    )
    assert app._resolve_adapter_path("r76/checkpoints/step1500") == (
        runs_dir / "r76" / "checkpoints" / "step1500"
    )
    # rotulo do repo -> caminho do repo (mesmo sem a varredura no estado)
    app._STATE["adapter_paths"] = {}
    assert app._resolve_adapter_path("Breeze-tts-2-brazillian-lora") == (
        repo_dir / "Breeze-tts-2-brazillian-lora"
    )
    assert app._resolve_adapter_path("nao/existe") is None


def test_adapter_download_state_offers_download_and_cites_source(monkeypatch):
    monkeypatch.setattr(app.CB, "adapter_installed", lambda repo_id=None, dest=None: None)
    missing, msg = app.adapter_download_state()
    assert missing is True
    assert "EdnilsonMonts/Breeze-tts-2-brazillian-lora" in msg      # diz de onde vem
    assert app.CB.ADAPTER_REPO_URL in msg
    assert str(app.CB.adapter_dir_for()) in msg                     # e onde fica


def test_adapter_download_state_hides_option_when_installed(workdir, monkeypatch):
    target = app.CB.adapter_dir_for()
    monkeypatch.setattr(app.CB, "adapter_installed", lambda repo_id=None, dest=None: target)
    missing, msg = app.adapter_download_state()
    assert missing is False and "instalado" in msg and str(target) in msg


def test_download_model_hides_button_and_selects_adapter(workdir, monkeypatch):
    """Depois do download o botao some e o adapter entra no dropdown ja selecionado."""
    target = _fake_adapter(app.CB.ADAPTERS_DIR / "Breeze-tts-2-brazillian-lora")
    monkeypatch.setattr(app.CB, "download_adapter", lambda *a, **k: target)
    monkeypatch.setattr(app.CB, "adapter_installed",
                        lambda repo_id=None, dest=None: target if (dest or target).exists() else None)
    monkeypatch.setattr(app, "_load_ui_state", lambda: {})
    app.gr.update.reset_mock()

    steps = list(app.download_model())
    assert len(steps) >= 2
    status, btn, dropdown = steps[-1]
    assert "instalado" in status
    assert {"visible": False} in [c.kwargs for c in app.gr.update.call_args_list]
    assert {"visible": True} not in [c.kwargs for c in app.gr.update.call_args_list]
    assert dropdown is not None                                     # dropdown atualizado


def test_download_model_reports_failure_and_keeps_button(monkeypatch):
    def boom(*_a, **_k):
        raise RuntimeError("sem internet")

    monkeypatch.setattr(app.CB, "download_adapter", boom)
    monkeypatch.setattr(app.CB, "adapter_installed", lambda repo_id=None, dest=None: None)
    app.gr.update.reset_mock()

    steps = list(app.download_model())
    status, btn, dropdown = steps[-1]
    assert "Falha no download" in status and "sem internet" in status
    assert app.CB.ADAPTER_REPO_URL in status                        # aponta o download manual
    assert {"visible": True} in [c.kwargs for c in app.gr.update.call_args_list]


def test_build_ui_smoke_with_and_without_download_option(monkeypatch):
    """build_ui monta sem Gradio/GPU reais â€” pega erro de fiacao no bloco do modelo."""
    monkeypatch.setattr(app.CB, "adapter_installed", lambda repo_id=None, dest=None: None)
    assert app.build_ui() is not None                     # sem adapter: mostra o download

    monkeypatch.setattr(app.CB, "adapter_installed",
                        lambda repo_id=None, dest=None: app.CB.adapter_dir_for())
    assert app.build_ui() is not None                     # com adapter: sem botao de download


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


def test_stash_reference_copies_and_is_stable(workdir):
    src = workdir / "in.wav"
    sf.write(str(src), np.zeros(2400, np.float32), 24000)
    p1 = app._stash_reference(str(src), "Minha Voz")
    p2 = app._stash_reference(str(src), "Minha Voz")
    assert p1 == p2 and os.path.isfile(p1) and "minha-voz" in p1
    assert p1 != str(src)


def test_save_voice_uses_upload_when_no_path(workdir, monkeypatch):
    src = workdir / "up.wav"
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
    segs = app.parse_segments("sad | oi\nsÃ³ texto\n# comentario\n", "Neutro")
    assert len(segs) == 2 and segs[1] == ("Neutro", "sÃ³ texto")


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
