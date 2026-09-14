"""clone_voice.py — clonagem de voz com um checkpoint LoRA do Breeze TTS 2.

Usa o template ref_edit (ref_audio + ref_text -> gera o texto-alvo no timbre da
referencia). Caminho EAGER validado (T=0.9, top_k=50, sem repetition_penalty).

Uso:
  python clone_voice.py --adapter training/runs/r64_01/checkpoints/step4000 ^
      --ref-audio "C:\\caminho\\minha_voz.wav" ^
      --ref-text "transcricao exata do audio de referencia" ^
      --text "Texto que o modelo deve falar com a voz clonada." ^
      --out training/clone_out/my_clone.wav

Sem --adapter usa o modelo base (controle). --ref-text deve ser EXATAMENTE o que
é falado no audio de referência (limpo, 3-12 s).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

_ROOT = Path(__file__).resolve().parents[1]
_CORE = _ROOT / "core"
if str(_CORE) not in sys.path:
    sys.path.insert(0, str(_CORE))
import common_breeze as CB  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Voice clone com LoRA (ref_edit)")
    ap.add_argument("--adapter", default=None, help="pasta do checkpoint LoRA")
    ap.add_argument("--ref-audio", required=True)
    ap.add_argument("--ref-text", default=None)
    ap.add_argument("--ref-text-file", default=None)
    ap.add_argument("--text", default=None)
    ap.add_argument("--text-file", default=None)
    ap.add_argument("--instruction", default="Fale com clareza e naturalidade.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--max-new-tokens", type=int, default=800)
    args = ap.parse_args()

    ref_audio = Path(args.ref_audio)
    if not ref_audio.is_file():
        sys.exit(f"[clone] ref-audio nao encontrado: {ref_audio}")

    def _resolve(text: str | None, path: str | None, label: str) -> str:
        if path:
            return Path(path).read_text(encoding="utf-8").strip()
        if text:
            return text.strip()
        sys.exit(f"[clone] falta --{label} ou --{label}-file")

    ref_text = _resolve(args.ref_text, args.ref_text_file, "ref-text")
    target_text = _resolve(args.text, args.text_file, "text")

    from breeze_infer.runtime import set_all_seeds, update_generation_config_for_breeze
    from breeze_infer.templates import get_template, prepare_inputs

    print("[clone] carregando base...", flush=True)
    raw = CB.load_breeze_model("cuda", attn="eager")
    if args.adapter:
        from peft import PeftModel

        raw = PeftModel.from_pretrained(raw, args.adapter)
        print(f"[clone] adapter: {args.adapter}", flush=True)
    raw.eval()
    update_generation_config_for_breeze(raw)
    tokenizer = CB.load_text_tokenizer()
    audio_tok = CB.load_audio_tokenizer("cuda")

    request = {
        "id": "clone",
        "text": target_text,
        "instruction": args.instruction,
        "speaker": "S0",
        "ref_audio_path": str(ref_audio),
        "ref_text": ref_text,
    }
    set_all_seeds(args.seed)
    inputs = prepare_inputs(
        tokenizer, audio_tok, raw, [request], get_template("ref_edit_tata"),
        guidance_scale=1.0, guidance_scale_ref=None, guidance_scale_ins=None,
    )
    # NAO descartar input_values: sao os CODES DO AUDIO DE REFERENCIA (essenciais
    # para a clonagem). Sem eles o modelo ignora a referencia.
    ref_codes = inputs.get("input_values")
    if ref_codes is None or ref_codes.numel() == 0:
        sys.exit("[clone] input_values (codes da referencia) vazio -- abortando")

    with torch.no_grad():
        res = raw.generate(
            input_ids=inputs["input_ids"],
            input_values=ref_codes,
            attention_mask=inputs["attention_mask"],
            text_ids_mask=inputs["text_ids_mask"],
            text_ids_len=inputs["text_ids_len"],
            output_audio=True,
            audio_tokenizer=audio_tok,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=1.0,
        )
    wav_t = res[0] if isinstance(res, (list, tuple)) else getattr(res, "audio", [None])[0]
    if wav_t is None:
        sys.exit("[clone] sem audio retornado")
    while wav_t.dim() > 1:
        wav_t = wav_t[0]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sr = audio_tok.get_output_sample_rate()
    wav = wav_t.detach().float().cpu().numpy()
    sf.write(str(out), np.clip(wav, -1.0, 1.0), sr, subtype="PCM_16")
    print(f"[clone] salvo: {out}  ({len(wav)/sr:.2f}s @ {sr}Hz)")

    # opcional: copia a referencia ao lado para comparacao A/B
    try:
        cmd = out.with_name(out.stem + "_REF" + out.suffix)
        ref_wav, ref_sr = sf.read(str(ref_audio), dtype="float32", always_2d=True)
        sf.write(str(cmd), np.clip(ref_wav[:, 0], -1, 1), ref_sr, subtype="PCM_16")
        print(f"[clone] referencia copiada: {cmd}")
    except Exception as exc:  # noqa: BLE001
        print(f"[clone] (aviso) nao copiei referencia: {exc}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
