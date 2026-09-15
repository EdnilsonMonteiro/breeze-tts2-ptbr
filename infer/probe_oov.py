"""Sonda OOV: base vs adapter nas MESMAS frases (decode padrao validado: eager,
T=0.9, top_k=50 — o vencedor do A/B). Textos com palavras FORA do lexico de
treino (verificado por scan) + controles dentro do lexico.

Uso: python probe_oov.py --adapter <ckpt> --out <dir>
"""
import argparse
import re
import sys
from pathlib import Path

import soundfile as sf

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
_ROOT = Path(__file__).resolve().parents[1]
_CORE = _ROOT / "core"
if str(_CORE) not in sys.path:
    sys.path.insert(0, str(_CORE))
import common_breeze as CB

# OOV (0 ocorrencias no corpus, confirmado por scan) + controles (frequentes)
TEXTS = [
    ("oov-1", "O buzinaço do carro assustou o gatíneo do vizinho ontem à noite."),
    ("oov-2", "Ela papeou durante o almoçar até o entardecer na varanda."),
    ("oov-3", "O refrigerante triblopia é vendido no mercado da esquina."),
    ("ctrl-1", "A previsão do tempo indica pancadas de chuva à tarde."),
    ("ctrl-2", "O número da minha casa é quinze, no bairro Jardim Europa."),
    ("ctrl-3", "Ele trabalha no escritório desde o ano passado."),
]


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower())[:40]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=None, help="ckpt do LoRA; se ausente, gera com o BASE")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed0", type=int, default=7000)
    args = ap.parse_args()

    from breeze_infer.runtime import set_all_seeds, update_generation_config_for_breeze
    from breeze_infer.templates import get_template, prepare_inputs

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[probe] carregando base...", flush=True)
    raw = CB.load_breeze_model("cuda", attn="eager")
    adapter = args.adapter
    if adapter and not Path(adapter).is_dir():
        _dl = CB.ensure_adapter(repo_id=adapter)
        if _dl:
            adapter = str(_dl)
    if adapter:
        from peft import PeftModel

        raw = PeftModel.from_pretrained(raw, adapter)
        print(f"[probe] adapter: {adapter}", flush=True)
    raw.eval()
    update_generation_config_for_breeze(raw)
    tokenizer = CB.load_text_tokenizer()
    audio_tok = CB.load_audio_tokenizer("cuda")
    template = get_template("tts_instruction")

    for k, (name, text) in enumerate(TEXTS):
        request = {"id": f"oov-{k}", "text": text,
                   "instruction": "Fale com clareza e naturalidade.", "speaker": "S0"}
        set_all_seeds(args.seed0 + k)
        inputs = prepare_inputs(tokenizer, audio_tok, raw, [request], template,
                                guidance_scale=1.0, guidance_scale_ref=None,
                                guidance_scale_ins=None)
        inputs.pop("input_values", None)
        inputs.pop("cfg_scale", None)
        path = out_dir / f"{k:02d}_{slug(name)}.wav"
        res = raw.generate(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
            text_ids_mask=inputs["text_ids_mask"], text_ids_len=inputs["text_ids_len"],
            output_audio=True, audio_tokenizer=audio_tok,
            max_new_tokens=400, do_sample=True, temperature=0.9, top_k=50, top_p=1.0)
        wav_t = res[0] if isinstance(res, (list, tuple)) else getattr(res, "audio", [None])[0]
        if wav_t is None:
            print(f"[probe] {name}: sem audio")
            continue
        while wav_t.dim() > 1:
            wav_t = wav_t[0]
        sf.write(str(path), wav_t.detach().float().cpu().numpy().clip(-1, 1),
                 audio_tok.get_output_sample_rate(), subtype="PCM_16")
        print(f"[probe] {path.name} ok", flush=True)
    print(f"[probe] FIM -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
