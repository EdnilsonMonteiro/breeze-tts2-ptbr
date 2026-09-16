"""Experimento A/B de caminho de decode: regera os 6 textos fixos com o caminho
OFICIAL de inferencia (FastBreezeStreamingRuntime, repetition_penalty=1.1,
temperature=1.0, top_k=off, max_new_tokens=1500) sobre um adapter escolhido.

Uso: python ab_decode.py --adapter <pasta-ckpt> --out <dir> [--seed0 2000]
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

SAMPLE_TEXTS = [
    ("ola-pt", "Olá! Este é um teste de voz em português brasileiro."),
    ("numeros-pt", "O número da minha casa é quinze oh dois, no bairro Jardim Europa."),
    ("clima-pt", "A previsão do tempo indica pancadas de chuva à tarde, com temperaturas "
                 "entre dezesseis e vinte e três graus."),
    ("siglas-pt", "Atenção: CPF um dois três ponto quatro cinco seis ponto sete oito nove, traço zero um."),
    ("afetivo-pt", "Que saudade daquele café quentinho da vovó no fim da tarde!"),
    ("regressao-en", "The weather today is sunny with a gentle breeze from the east."),
]


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower())[:40]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed0", type=int, default=2000)
    args = ap.parse_args()

    from breeze_infer.runtime import set_all_seeds, update_generation_config_for_breeze
    from breeze_infer.templates import get_template, prepare_inputs
    from models.fast_streaming import FastBreezeStreamingRuntime, FastStreamingConfig
    from peft import PeftModel

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[ab] carregando modelo base...", flush=True)
    raw = CB.load_breeze_model("cuda", attn="eager")
    adapter = args.adapter
    if adapter and not Path(adapter).is_dir():
        _dl = CB.ensure_adapter(repo_id=adapter)
        if _dl:
            adapter = str(_dl)
    print(f"[ab] adapter: {adapter}", flush=True)
    model = PeftModel.from_pretrained(raw, adapter)
    model.eval()
    tokenizer = CB.load_text_tokenizer()
    audio_tok = CB.load_audio_tokenizer("cuda")
    update_generation_config_for_breeze(model)

    config = FastStreamingConfig(
        max_new_tokens=1500, max_seq_len=2048,
        fast_all=False, fast_text_encoder=False, fast_backbone_prefill=False,
        fast_backbone_decode=False, fast_depth_decoder=False, fast_codec=False,
        repetition_penalty=1.1,
    )
    runtime = FastBreezeStreamingRuntime(model, audio_tok, config, tokenizer=tokenizer)
    template = get_template("tts_instruction")

    for k, (name, text) in enumerate(SAMPLE_TEXTS):
        request = {"id": f"ab-{k}", "text": text,
                   "instruction": "Fale com clareza e naturalidade.", "speaker": "S0"}
        set_all_seeds(args.seed0 + k)
        inputs = prepare_inputs(tokenizer, audio_tok, model, [request], template,
                                guidance_scale=1.0, guidance_scale_ref=None,
                                guidance_scale_ins=None)
        path = out_dir / f"{k:02d}_{slug(name)}.wav"
        with sf.SoundFile(path, mode="w", samplerate=runtime.sample_rate,
                          channels=1, subtype="PCM_16") as f:
            for chunk in runtime.iter_audio_chunks(inputs, request_id=f"ab-{k}"):
                f.write(chunk.audio)
        print(f"[ab] {path.name} ok", flush=True)

    print(f"[ab] FIM -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
