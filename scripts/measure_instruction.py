"""measure_instruction.py — a INSTRUCAO (emocao/estilo) muda mesmo o audio?

Gera o MESMO texto/seed/referencia variando apenas a instrucao e o `cfg_scale`, mede
prosodia (F0, energia, duracao, ritmo, MFCC) e compara com o **piso de ruido**: a
variacao obtida trocando apenas a *seed*, com a mesma instrucao. Se o efeito da
instrucao nao passar do ruido, a instrucao nao esta dirigindo a voz.

Por que existe: o adaptador pt-BR (v2/v3) foi treinado com uma instrucao **fixa**
("Fale com clareza e naturalidade." — ver `docs/EMOTION.md`), entao medir e a unica
forma honesta de dizer se ele segue emocao. O modelo BASE segue, mas exige instrucao
**descritiva**, no **idioma do texto** e `--cfg 4`.

Uso:
  # modelo base (Voice Design), instrucoes EN, cfg 4, 2 seeds (piso de ruido)
  python scripts/measure_instruction.py --text "Good morning! Nice day, isn't it?" \
      --instructions "Speak clearly and naturally.|Speak with great excitement and high energy." \
      --cfg 4 --seeds 42,43

  # com o adapter pt-BR e a referencia (Voice Direction)
  python scripts/measure_instruction.py --adapter adapters/Breeze-tts-2-brazillian-lora \
      --ref-audio ref.wav --ref-text "transcricao exata da referencia" \
      --text "Bom dia! Que dia maravilhoso, nao e mesmo?" \
      --instructions "Fale com clareza e naturalidade.|Fale com muita empolgacao e energia." \
      --cfg 4 --seeds 42,43 --out out/instrucao

Saida: WAVs + `metricas.json` em `--out`, tabela comparativa no terminal e um veredito
(razao efeito-instrucao / ruido-de-seed por metrica). `--reuse` reaproveita WAVs ja
gerados (permite so re-medir, sem carregar o modelo).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np
import soundfile as sf

_ROOT = Path(__file__).resolve().parents[1]
_CORE = _ROOT / "core"
if str(_CORE) not in sys.path:
    sys.path.insert(0, str(_CORE))

import adapter_scale as AS  # noqa: E402
import common_breeze as CB  # noqa: E402
import reference_prep as RP  # noqa: E402
import text_norm as TN  # noqa: E402

# Metricas comparadas (F0/duracao/ritmo sao as que mais mudam com emocao; energia e
# timbre entram como apoio). `ecapa` entra com --ecapa (identidade do locutor).
METRICS = ("f0_med", "dur_s", "words_s", "rms_mean", "mfcc")
# "Funciona" = alguma metrica com folga forte (RAZAO_FORTE) + pelo menos 2 metricas acima
# de RAZAO_OK + pelo menos uma delas de prosodia (F0/duracao/ritmo). "Inerte" = nenhuma
# metrica chega a RAZAO_OK. Entre os dois = fraco/duvidoso.
RAZAO_OK = 1.5
RAZAO_FORTE = 2.0
PROSODIA = ("f0_med", "dur_s", "words_s")

DEFAULT_INSTRUCTIONS = (
    "Fale com clareza e naturalidade.|"
    "Fale com muita empolgacao, energia e ritmo acelerado.|"
    "Fale com tristeza e melancolia, em tom baixo e pausado."
)


def slug(text: str, limit: int = 28) -> str:
    s = unicodedata.normalize("NFKD", text or "")
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:limit] or "x"


# ------------------------------------------------------------------ medicao
def measure_wav(path: Path, text: str, *, ecapa: bool = False) -> dict:
    """Metricas acusticas de um WAV (F0 via pyin, energia, duracao, ritmo, MFCC)."""
    import librosa

    wav, sr = librosa.load(str(path), sr=24000, mono=True)
    dur = max(1e-6, len(wav) / sr)
    f0, _voiced, _ = librosa.pyin(wav, fmin=60, fmax=500, sr=sr, frame_length=1024)
    f0v = f0[~np.isnan(f0)]
    rms = librosa.feature.rms(y=wav, frame_length=1024, hop_length=256)[0]
    mfcc = librosa.feature.mfcc(y=wav, sr=sr, n_mfcc=13)
    out = {
        "dur_s": round(float(dur), 2),
        "words_s": round(len(text.split()) / dur, 2),
        "f0_med": round(float(np.median(f0v)), 1) if len(f0v) else 0.0,
        "rms_mean": round(float(np.mean(rms)), 4),
        "mfcc": np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)]).tolist(),
    }
    if ecapa:
        emb = CB.speaker_embed(str(path), "cpu")
        out["ecapa"] = (emb / (np.linalg.norm(emb) + 1e-9)).tolist()
    return out


def distance(a: dict, b: dict) -> dict:
    """Distancias entre duas medicoes (`rms_mean` relativa; `mfcc` = RMS da diferenca)."""
    d = {
        "f0_med": abs(a["f0_med"] - b["f0_med"]),
        "dur_s": abs(a["dur_s"] - b["dur_s"]),
        "words_s": abs(a["words_s"] - b["words_s"]),
        "rms_mean": abs(a["rms_mean"] - b["rms_mean"]) / max(a["rms_mean"], b["rms_mean"], 1e-9),
        "mfcc": float(np.linalg.norm(np.array(a["mfcc"]) - np.array(b["mfcc"]))
                      / np.sqrt(len(a["mfcc"]))),
    }
    if "ecapa" in a and "ecapa" in b:
        va, vb = np.array(a["ecapa"]), np.array(b["ecapa"])
        d["ecapa"] = 1.0 - float(np.dot(va, vb))
    return d


def _pairs(rows: list[dict], same: tuple[str, ...], diff: str) -> list[tuple[dict, dict]]:
    """Pares de amostras que compartilham todas as chaves de `same` e diferem em `diff`."""
    out: list[tuple[dict, dict]] = []
    for i, a in enumerate(rows):
        for b in rows[i + 1:]:
            if all(a[k] == b[k] for k in same) and a[diff] != b[diff]:
                out.append((a, b))
    return out


def analyse(rows: list[dict]) -> dict:
    """Efeito da INSTRUCAO vs ruido de SEED (funcao pura, sem GPU: testavel).

    - pares de INSTRUCAO: mesma seed/cfg, instrucao diferente.
    - pares de RUIDO: mesma instrucao/cfg, seed diferente.
    """
    metrics = [m for m in METRICS] + (["ecapa"] if rows and "ecapa" in rows[0] else [])
    inst = _pairs(rows, ("seed", "cfg"), "instruction")
    noise = _pairs(rows, ("instruction", "cfg"), "seed")

    def mean_dist(pairs, metric):
        vals = [distance(a, b)[metric] for a, b in pairs]
        return float(np.mean(vals)) if vals else 0.0

    res = {
        "n_instrucao": len(inst), "n_ruido": len(noise),
        "instrucao": {m: mean_dist(inst, m) for m in metrics},
        "ruido": {m: mean_dist(noise, m) for m in metrics},
    }
    # Sem pares de ruido nao ha como separar efeito de sorteio: razao fica indefinida.
    res["razao"] = {
        m: (res["instrucao"][m] / res["ruido"][m]) if res["ruido"][m] > 1e-9 else None
        for m in metrics
    }
    ratios = {m: r for m, r in res["razao"].items() if r is not None}
    if not inst or not noise or not ratios:
        res["veredito"] = "INCONCLUSIVO (precisa de >=2 instrucoes e >=2 seeds)"
        return res
    forte = max(ratios.values())
    bons = [m for m, r in ratios.items() if r >= RAZAO_OK]
    prosodia = [m for m in PROSODIA if ratios.get(m, 0.0) >= RAZAO_OK]
    if forte >= RAZAO_FORTE and len(bons) >= 2 and prosodia:
        res["veredito"] = "INSTRUCAO FUNCIONA (efeito acima do ruido de seed)"
    elif forte < RAZAO_OK:
        res["veredito"] = "INSTRUCAO INERTE (nenhuma metrica supera o ruido de seed)"
    else:
        res["veredito"] = "EFEITO FRACO/DUVIDOSO (perto do ruido de seed)"
    return res


def print_report(rows: list[dict], res: dict) -> None:
    print("\n-- amostras")
    for r in sorted(rows, key=lambda x: (x["instruction"], x["seed"])):
        print(f"   ins={r['instruction'][:44]:44s} seed={r['seed']:>4} cfg={r['cfg']:<4} "
              f"dur={r['dur_s']:5.2f}s f0={r['f0_med']:6.1f}Hz rms={r['rms_mean']:.4f} "
              f"ritmo={r['words_s']:.2f} w/s")
    print(f"\n-- efeito medio por metrica ({res['n_instrucao']} pares de instrucao, "
          f"{res['n_ruido']} pares de seed)")
    print(f"   {'metrica':10s} {'instrucao':>10s} {'ruido(seed)':>12s} {'razao':>7s}")
    for m, val in res["instrucao"].items():
        razao = res["razao"][m]
        print(f"   {m:10s} {val:10.3f} {res['ruido'][m]:12.3f} "
              f"{(f'{razao:.2f}' if razao is not None else '  --'):>7s}")
    print(f"\nVEREDITO: {res['veredito']}")


# ------------------------------------------------------------------ geracao
def build_model(adapter: str | None):
    from breeze_infer.runtime import update_generation_config_for_breeze

    print("[med] carregando base...", flush=True)
    model = CB.load_breeze_model("cuda", attn="eager")
    if adapter:
        from peft import PeftModel

        adapter_path = adapter
        if not Path(adapter).is_dir():
            dl = CB.ensure_adapter(repo_id=adapter)      # aceita id do Hugging Face
            if dl is None:
                sys.exit(f"[med] nao consegui resolver o adapter '{adapter}'")
            adapter_path = str(dl)
        adapter_path = CB.prepare_adapter(adapter_path)
        model = PeftModel.from_pretrained(model, adapter_path, adapter_name="a0")
        AS.apply_adapter_scale(model, 1.0)
        print(f"[med] adapter: {adapter_path}", flush=True)
    model.eval()
    update_generation_config_for_breeze(model)
    return model


def generate(model, tokenizer, audio_tok, text: str, instruction: str, *, cfg: float,
             seed: int, ref: dict | None):
    import torch
    from breeze_infer.runtime import set_all_seeds
    from breeze_infer.templates import get_template, prepare_inputs

    request = {"id": "med", "text": text, "instruction": instruction, "speaker": "S0"}
    template = "tts_instruction"
    if ref:
        request.update({"ref_audio_path": ref["path"], "ref_text": ref["text"]})
        template = "ref_edit_tata"
    set_all_seeds(int(seed))
    inputs = prepare_inputs(tokenizer, audio_tok, model, [request], get_template(template),
                            guidance_scale=float(cfg), guidance_scale_ref=None,
                            guidance_scale_ins=None)
    with torch.inference_mode():
        res = model.generate(**inputs, output_audio=True, audio_tokenizer=audio_tok,
                             max_new_tokens=400, do_sample=True, temperature=0.7,
                             top_k=50, top_p=1.0)
    wav_t = res[0] if isinstance(res, (list, tuple)) else getattr(res, "audio", [None])[0]
    if wav_t is None:
        raise RuntimeError("o modelo nao retornou audio")
    while wav_t.dim() > 1:
        wav_t = wav_t[0]
    return wav_t.detach().float().cpu().numpy(), audio_tok.get_output_sample_rate()


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Mede se a instrucao (emocao) muda o audio, contra o piso de ruido da seed.")
    ap.add_argument("--text", required=True)
    ap.add_argument("--instructions", default=DEFAULT_INSTRUCTIONS,
                    help="instrucoes separadas por '|'")
    ap.add_argument("--cfg", type=float, default=4.0, help="cfg_scale (4 = receita oficial)")
    ap.add_argument("--seeds", default="42,43", help="seeds (>=2 para o piso de ruido)")
    ap.add_argument("--adapter", default=None, help="pasta do LoRA ou id do Hugging Face")
    ap.add_argument("--ref-audio", default=None, help="referencia (ativa Voice Direction)")
    ap.add_argument("--ref-text", default=None, help="transcricao exata da referencia")
    ap.add_argument("--out", default=None, help="pasta de saida (default: <ui_out>/instrucao)")
    ap.add_argument("--ref-format", choices=list(RP.MODES), default="train")
    ap.add_argument("--ecapa", action="store_true", help="mede tambem similaridade de locutor")
    ap.add_argument("--reuse", action="store_true", help="reaproveita WAVs ja gerados")
    a = ap.parse_args()

    if bool(a.ref_audio) != bool(a.ref_text):
        sys.exit("[med] --ref-audio e --ref-text devem ser usados juntos")
    instructions = [s.strip() for s in a.instructions.split("|") if s.strip()]
    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    if len(instructions) < 2 or len(seeds) < 2:
        print("[med] (aviso) com <2 instrucoes ou <2 seeds nao da para separar efeito de ruido",
              flush=True)

    out_dir = Path(a.out) if a.out else (Path(CB.OUT_DIR) / "instrucao")
    out_dir.mkdir(parents=True, exist_ok=True)
    text = TN.normalize(a.text)
    state: dict = {"model": None, "tokenizer": None, "audio_tok": None, "ref": None}

    def loaded() -> dict:
        """Carrega modelo/tokenizers (e prepara a referencia) so na 1a geracao."""
        if state["model"] is None:
            state["model"] = build_model(a.adapter)
            state["tokenizer"] = CB.load_text_tokenizer()
            state["audio_tok"] = CB.load_audio_tokenizer("cuda")
            if a.ref_audio:
                ref_path = Path(a.ref_audio)
                if not ref_path.is_file():
                    sys.exit(f"[med] ref-audio nao encontrado: {ref_path}")
                prepared, dur = RP.prepare_reference(ref_path, CB.ARTIFACTS / "voice_cache",
                                                     a.ref_format)
                warn = RP.duration_warning(dur)
                if warn:
                    print(f"[med] (aviso) {warn}", flush=True)
                CB.cache_reference(state["audio_tok"], prepared)
                state["ref"] = {"path": str(prepared), "text": TN.normalize(a.ref_text.strip())}
        return state

    rows: list[dict] = []
    for instruction in instructions:
        for seed in seeds:
            tag = f"{slug(instruction)}_s{seed}_cfg{a.cfg:g}"
            wav_path = out_dir / f"{tag}.wav"
            if not (a.reuse and wav_path.is_file()):
                st = loaded()
                t0 = time.time()
                wav, sr = generate(st["model"], st["tokenizer"], st["audio_tok"], text,
                                   instruction, cfg=a.cfg, seed=seed, ref=st["ref"])
                sf.write(str(wav_path), np.clip(wav, -1.0, 1.0), sr, subtype="PCM_16")
                print(f"[med] {wav_path.name} ({time.time() - t0:.0f}s)", flush=True)
            m = measure_wav(wav_path, text, ecapa=a.ecapa)
            m.update({"tag": tag, "instruction": instruction, "seed": seed, "cfg": a.cfg,
                      "wav": str(wav_path)})
            rows.append(m)

    res = analyse(rows)
    print_report(rows, res)
    report = out_dir / "metricas.json"
    report.write_text(json.dumps(
        {"text": a.text, "adapter": a.adapter, "cfg": a.cfg, "ref_audio": a.ref_audio,
         "amostras": rows, "analise": res}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    print(f"\n[med] relatorio: {report}", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
