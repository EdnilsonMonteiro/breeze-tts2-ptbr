"""app.py — interface web (Gradio) para clonagem/TTS PT-BR com o Breeze TTS 2 + LoRA.

Roda no navegador. Carrega o modelo base uma vez, permite escolher um adapter LoRA,
gerar COM voz de referencia (clonagem) ou SEM referencia, com controle de emocao/estilo
(o "system prompt" = instrucao, igual ao modelo original) e CFG.

Uso:
  & $py lora\\ui\\app.py                 # abre em http://127.0.0.1:7860
  & $py lora\\ui\\app.py --port 7861
  & $py lora\\ui\\app.py --selftest      # gera 1 amostra sem abrir a UI (validacao)

Emocao: no Breeze, o estilo/emocao vem da INSTRUCAO (texto entre <ins_bos>...</ins_eos>)
e pode ser reforcada pelo CFG (cfg_scale>1 amplifica a instrucao). No modo clonagem com
dual-CFG, cfg_ref controla a fidelidade a voz e cfg_ins a aderencia a instrucao.
"""
from __future__ import annotations

import argparse
import gc
import logging
import os
import re
import sys
import time
import unicodedata
import warnings
from pathlib import Path

import numpy as np

# --- silencia ruido de bibliotecas (deprecations internas do Gradio/Starlette,
#     warnings de flash-attn e do regex do tokenizer do Breeze) -----------------
warnings.filterwarnings("ignore", message=".*HTTP_422_UNPROCESSABLE.*")
warnings.filterwarnings("ignore", message=".*flash-attn.*")
warnings.filterwarnings("ignore", message=".*incorrect regex pattern.*")
warnings.filterwarnings("ignore", message=".*fix_mistral_regex.*")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
for _lg in ("transformers", "huggingface_hub", "gradio", "urllib3"):
    logging.getLogger(_lg).setLevel(logging.ERROR)

_ROOT = Path(__file__).resolve().parents[1]
_CORE = _ROOT / "core"
if str(_CORE) not in sys.path:
    sys.path.insert(0, str(_CORE))

import common_breeze as CB  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402

OUT_DIR = CB.OUT_DIR

BASE_LABEL = "(base - sem adapter)"

EMOTIONS = {
    "Neutro / natural": "Fale de forma clara e natural.",
    "Alegre / sorrindo": "Fale com alegria, entusiasmo e um sorriso na voz.",
    "Empolgado": "Fale com muita empolgacao, energia e ritmo acelerado.",
    "Triste / melancolico": "Fale com tristeza e melancolia, em tom baixo e pausado.",
    "Raiva / irritado": "Fale com raiva, irritacao e enfase forte.",
    "Medo / apreensivo": "Fale com medo e apreensao, em voz tremula.",
    "Sussurro": "Fale em um sussurro suave e intimo.",
    "Calmo / sereno": "Fale de forma calma, pausada e tranquila.",
    "Narrador de audiolivro": "Narre de forma envolvente, como um audiolivro.",
    "Locutor de noticia": "Leia como um locutor de telejornal, com diccao clara e seria.",
    "Romantico / sedutor": "Fale de forma romantica, suave e sedutora.",
    "Formal / profissional": "Fale de forma formal e profissional.",
    "Personagem infantil": "Fale com voz doce e animada, como em desenho infantil.",
}

# Apelidos (PT e EN, com/sem acento) aceitos no modo segmentado alem dos nomes
# completos do EMOTIONS. Ex.: "Sad", "triste", "Alegre", "whisper", "calmo".
EMOTION_ALIASES = {
    "neutro": "Neutro / natural", "natural": "Neutro / natural", "normal": "Neutro / natural",
    "neutral": "Neutro / natural",
    "alegre": "Alegre / sorrindo", "feliz": "Alegre / sorrindo", "sorrindo": "Alegre / sorrindo",
    "happy": "Alegre / sorrindo", "joy": "Alegre / sorrindo", "joyful": "Alegre / sorrindo",
    "empolgado": "Empolgado", "animado": "Empolgado", "excited": "Empolgado",
    "triste": "Triste / melancolico", "tristeza": "Triste / melancolico",
    "melancolico": "Triste / melancolico", "sad": "Triste / melancolico",
    "melancholic": "Triste / melancolico",
    "raiva": "Raiva / irritado", "irritado": "Raiva / irritado", "bravo": "Raiva / irritado",
    "angry": "Raiva / irritado", "anger": "Raiva / irritado",
    "medo": "Medo / apreensivo", "apreensivo": "Medo / apreensivo", "ansioso": "Medo / apreensivo",
    "scared": "Medo / apreensivo", "afraid": "Medo / apreensivo", "fear": "Medo / apreensivo",
    "sussurro": "Sussurro", "sussurrando": "Sussurro", "whisper": "Sussurro",
    "whispering": "Sussurro",
    "calmo": "Calmo / sereno", "sereno": "Calmo / sereno", "tranquilo": "Calmo / sereno",
    "calm": "Calmo / sereno",
    "narrador": "Narrador de audiolivro", "audiolivro": "Narrador de audiolivro",
    "narrator": "Narrador de audiolivro", "audiobook": "Narrador de audiolivro",
    "locutor": "Locutor de noticia", "noticia": "Locutor de noticia",
    "news": "Locutor de noticia", "anchor": "Locutor de noticia",
    "romantico": "Romantico / sedutor", "sedutor": "Romantico / sedutor",
    "romantic": "Romantico / sedutor", "seductive": "Romantico / sedutor",
    "formal": "Formal / profissional", "profissional": "Formal / profissional",
    "professional": "Formal / profissional",
    "infantil": "Personagem infantil", "crianca": "Personagem infantil",
    "child": "Personagem infantil", "kid": "Personagem infantil",
}


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c))
    return " ".join(s.lower().split())


_EMO_LOOKUP: dict[str, str] = {}
for _k, _v in EMOTIONS.items():
    _EMO_LOOKUP[_norm(_k)] = _v
for _alias, _k in EMOTION_ALIASES.items():
    _EMO_LOOKUP.setdefault(_norm(_alias), EMOTIONS[_k])


def resolve_emotion(name: str) -> str | None:
    """Mapeia 'Sad'/'triste'/'Alegre' etc. -> instrucao. None se nao for preset."""
    n = _norm(name)
    if not n:
        return None
    if n in _EMO_LOOKUP:
        return _EMO_LOOKUP[n]
    for key, val in _EMO_LOOKUP.items():  # match parcial: "sad mood", "bem triste"
        if key in n or n in key:
            return val
    return None

_STATE = {
    "raw": None,
    "peft": None,
    "adapters": {},  # nome interno -> caminho
    "adapter_paths": {},  # label da UI -> caminho absoluto
    "tokenizer": None,
    "audio_tok": None,
    "device": "cuda",
    "adapter_choices": [],
}


# ------------------------------------------------------------------ model
def _pick_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def refresh_adapters() -> list[str]:
    """Descobre adapters locais (runs/*/checkpoints/*) e baixados do HF (hf/*)."""
    found: dict[str, str] = {}
    runs = CB.ADAPTERS_DIR
    if runs.is_dir():
        for ck in sorted(runs.glob("*/checkpoints/*")):
            if (ck / "adapter_config.json").is_file():
                found[f"{ck.parent.parent.name}/checkpoints/{ck.name}"] = str(ck)
    hf = CB.HF_ADAPTERS_DIR
    if hf.is_dir():
        for ad in sorted(hf.iterdir()):
            if (ad / "adapter_config.json").is_file():
                found[f"hf/{ad.name}"] = str(ad)
    _STATE["adapter_paths"] = found
    return [BASE_LABEL, *found.keys()]


def list_adapters() -> list[str]:
    return refresh_adapters()


def load_model() -> None:
    """Carrega base + tokenizers (uma unica vez)."""
    if _STATE["raw"] is not None:
        return
    dev = _pick_device()
    print(f"[ui] carregando base ({dev})...", flush=True)
    t0 = time.time()
    if dev == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    raw = CB.load_breeze_model(dev, attn="eager")
    raw.eval()
    _STATE["raw"] = raw
    _STATE["device"] = dev
    _STATE["tokenizer"] = CB.load_text_tokenizer()
    _STATE["audio_tok"] = CB.load_audio_tokenizer(dev)
    print(f"[ui] base pronto em {time.time() - t0:.0f}s", flush=True)


def use_adapter(label: str):
    """Retorna o modelo a usar (base ou com adapter), trocando o adapter via PEFT."""
    load_model()
    raw = _STATE["raw"]
    peft = _STATE["peft"]

    if label in (None, "", BASE_LABEL, "base", "Base"):
        if peft is not None:
            peft.disable_adapter_layers()
            return peft
        return raw

    resolved = _STATE.get("adapter_paths", {}).get(label)
    path = (Path(resolved) if resolved else (CB.ADAPTERS_DIR / label)).resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"adapter nao encontrado: {path}")

    key = label
    if peft is None:
        from peft import PeftModel

        print(f"[ui] carregando adapter {label}...", flush=True)
        peft = PeftModel.from_pretrained(raw, str(path), adapter_name="a0")
        _STATE["peft"] = peft
        _STATE["adapters"]["a0"] = key
        return peft

    if key not in _STATE["adapters"].values():
        name = f"a{len(_STATE['adapters'])}"
        print(f"[ui] anexando adapter {label} como {name}...", flush=True)
        peft.load_adapter(str(path), adapter_name=name)
        _STATE["adapters"][name] = key
    name = next(k for k, v in _STATE["adapters"].items() if v == key)
    peft.set_adapter(name)
    peft.enable_adapter_layers()
    return peft


# ------------------------------------------------------------------ gen
def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (s or "audio").lower())[:32].strip("-") or "audio"


def _generate(
    text: str,
    instruction: str,
    ref_audio: str | None,
    ref_text: str,
    adapter_label: str,
    speaker: str,
    seed: int,
    temperature: float,
    top_k: int,
    top_p: float,
    max_new_tokens: int,
    cfg_scale: float,
    use_dual_cfg: bool,
    cfg_ref: float,
    cfg_ins: float,
) -> tuple[np.ndarray, int]:
    """Gera UMA locucao (uma instrucao global). Retorna (wav float32, sample_rate)."""
    has_ref = bool(ref_audio) and bool((ref_text or "").strip())
    model = use_adapter(adapter_label)
    tokenizer = _STATE["tokenizer"]
    audio_tok = _STATE["audio_tok"]

    from breeze_infer.runtime import set_all_seeds, update_generation_config_for_breeze
    from breeze_infer.templates import get_template, prepare_inputs

    update_generation_config_for_breeze(model)
    set_all_seeds(int(seed))

    request = {"id": "ui", "text": text, "instruction": instruction, "speaker": speaker or "S0"}
    if has_ref:
        request["ref_audio_path"] = str(ref_audio)
        request["ref_text"] = ref_text.strip()
        template = "ref_edit_tata"
    else:
        template = "tts_instruction"

    inputs = prepare_inputs(
        tokenizer, audio_tok, model, [request], get_template(template),
        guidance_scale=float(cfg_scale),
        guidance_scale_ref=float(cfg_ref) if use_dual_cfg else None,
        guidance_scale_ins=float(cfg_ins) if use_dual_cfg else None,
    )

    with torch.inference_mode():
        res = model.generate(
            **inputs,
            output_audio=True,
            audio_tokenizer=audio_tok,
            max_new_tokens=int(max_new_tokens),
            do_sample=True,
            temperature=float(temperature),
            top_k=int(top_k),
            top_p=float(top_p),
        )

    wav_t = res[0] if isinstance(res, (list, tuple)) else getattr(res, "audio", [None])[0]
    if wav_t is None:
        raise gr.Error("O modelo nao retornou audio.")
    while wav_t.dim() > 1:
        wav_t = wav_t[0]
    sr = audio_tok.get_output_sample_rate()
    return wav_t.detach().float().cpu().numpy().astype(np.float32), sr


def _save_wav(wav: np.ndarray, sr: int, slug: str, ref_audio: str | None) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = OUT_DIR / f"{stamp}_{_slug(slug)}.wav"
    sf.write(str(out), np.clip(wav, -1.0, 1.0), sr, subtype="PCM_16")
    if ref_audio and Path(str(ref_audio)).is_file():
        try:
            rw, rsr = sf.read(str(ref_audio), dtype="float32", always_2d=True)
            sf.write(str(out.with_name(out.stem + "_REF" + out.suffix)),
                     np.clip(rw[:, 0], -1, 1), rsr, subtype="PCM_16")
        except Exception:  # noqa: BLE001
            pass
    return out


def adjust_pauses(
    wav: np.ndarray,
    sr: int,
    target_ms: int,
    min_gap_ms: int = 90,
    rel_db: float = -38.0,
) -> tuple[np.ndarray, int]:
    """Ajusta a duracao das pausas NATURAIS do audio para `target_ms` (10..3000 ms).

    Deteccao por energia em janelas de 20 ms: trechos de silencio (RMS abaixo de
    `rel_db` relativo ao pico) com duracao >= min_gap_ms sao substituidos por silencio
    de exatamente target_ms. Nao gera nova inferencia -> o tom da fala e preservado.

    Retorna (novo_wav, n_pausas_ajustadas).
    """
    target_ms = int(max(10, min(3000, target_ms)))
    frame = max(1, int(sr * 0.02))
    n = len(wav) // frame
    if n < 3:
        return wav, 0
    rms = np.sqrt(np.mean(wav[: n * frame].reshape(n, frame) ** 2, axis=1))
    peak = float(rms.max())
    if peak <= 1e-6:
        return wav, 0
    thr = peak * (10.0 ** (rel_db / 20.0))
    silent = rms < thr

    gaps: list[tuple[int, int]] = []
    i = 0
    min_frames = max(1, int(sr * min_gap_ms / 1000.0 / frame))
    while i < n:
        if silent[i]:
            j = i
            while j < n and silent[j]:
                j += 1
            if (j - i) >= min_frames and i > 0 and j < n:
                gaps.append((i, j))
            i = j
        else:
            i += 1

    if not gaps:
        return wav, 0

    target = int(sr * target_ms / 1000.0)
    pieces: list[np.ndarray] = []
    prev = 0
    for i0, j0 in gaps:
        pieces.append(wav[prev : i0 * frame])
        pieces.append(np.zeros(target, dtype=wav.dtype))
        prev = j0 * frame
    pieces.append(wav[prev:])
    return np.concatenate(pieces), len(gaps)


def synthesize(
    text: str,
    emotion: str = "Neutro / natural",
    instruction_custom: str = "",
    ref_audio: str | None = None,
    ref_text: str = "",
    adapter_label: str = BASE_LABEL,
    speaker: str = "S0",
    seed: int = 42,
    temperature: float = 0.9,
    top_k: int = 50,
    top_p: float = 1.0,
    max_new_tokens: int = 800,
    cfg_scale: float = 1.0,
    use_dual_cfg: bool = False,
    cfg_ref: float = 1.0,
    cfg_ins: float = 1.0,
    pause_on: bool = False,
    pause_ms: int = 300,
):
    text = (text or "").strip()
    if not text:
        raise gr.Error("Informe o texto a ser falado.")

    instruction = (instruction_custom or "").strip() or EMOTIONS.get(emotion, EMOTIONS["Neutro / natural"])

    has_ref = bool(ref_audio) and bool((ref_text or "").strip())
    if bool(ref_audio) != bool((ref_text or "").strip()):
        raise gr.Error("Referencia e transcricao devem ser preenchidas juntas.")

    t0 = time.time()
    wav, sr = _generate(
        text=text, instruction=instruction, ref_audio=ref_audio, ref_text=ref_text,
        adapter_label=adapter_label, speaker=speaker, seed=seed, temperature=temperature,
        top_k=top_k, top_p=top_p, max_new_tokens=max_new_tokens, cfg_scale=cfg_scale,
        use_dual_cfg=use_dual_cfg, cfg_ref=cfg_ref, cfg_ins=cfg_ins,
    )
    n_pauses = 0
    if pause_on:
        wav, n_pauses = adjust_pauses(wav, sr, int(pause_ms))
    out = _save_wav(wav, sr, text, ref_audio if has_ref else None)

    mode = "clonagem (com referencia)" if has_ref else "voz padrao (sem referencia)"
    adapter_txt = adapter_label if adapter_label != BASE_LABEL else "base"
    persist_line = (f"  \n**Pausas:** {n_pauses} ajustada(s) para {int(pause_ms)} ms"
                    if pause_on else "")
    info = (
        f"**Modo:** {mode}  \n"
        f"**Adapter:** {adapter_txt}  \n"
        f"**Emocao/instrucao:** {instruction}  \n"
        f"**CFG:** {cfg_scale}"
        + (f" (dual ref={cfg_ref}, ins={cfg_ins})" if use_dual_cfg else "")
        + persist_line
        + f"  \n**Tempo:** {time.time() - t0:.1f}s  \n**Arquivo:** `{out}`"
    )
    return str(out), info


def parse_segments(raw: str, default_instruction: str) -> list[tuple[str, str]]:
    """Le o roteiro por linhas. Formato por linha:

        emocao | texto          (emocao = preset EMOTIONS, apelido PT/EN, ou instrucao livre)
        texto                   (usa a emocao global)

    Linhas vazias sao ignoradas. Retorna [(instruction, text), ...].
    """
    out: list[tuple[str, str]] = []
    for line in (raw or "").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "|" in line:
            left, right = line.split("|", 1)
            left, right = left.strip(), right.strip()
            if not right:
                continue
            resolved = resolve_emotion(left)
            instr = resolved if resolved else left  # preset (PT/EN) OU instrucao livre
        else:
            instr, right = default_instruction, line
        out.append((instr, right))
    return out


def synthesize_segments(
    script: str,
    gap_ms: int = 120,
    emotion: str = "Neutro / natural",
    instruction_custom: str = "",
    ref_audio: str | None = None,
    ref_text: str = "",
    adapter_label: str = BASE_LABEL,
    speaker: str = "S0",
    seed: int = 42,
    temperature: float = 0.9,
    top_k: int = 50,
    top_p: float = 1.0,
    max_new_tokens: int = 800,
    cfg_scale: float = 1.0,
    use_dual_cfg: bool = False,
    cfg_ref: float = 1.0,
    cfg_ins: float = 1.0,
):
    default_instr = (instruction_custom or "").strip() or EMOTIONS.get(emotion, EMOTIONS["Neutro / natural"])
    segs = parse_segments(script, default_instr)
    if not segs:
        raise gr.Error("Escreva ao menos uma linha no roteiro segmentado.")

    has_ref = bool(ref_audio) and bool((ref_text or "").strip())
    if bool(ref_audio) != bool((ref_text or "").strip()):
        raise gr.Error("Referencia e transcricao devem ser preenchidas juntas.")

    t0 = time.time()
    sr = None
    parts: list[np.ndarray] = []
    for i, (instr, seg_text) in enumerate(segs):
        wav, sr = _generate(
            text=seg_text, instruction=instr, ref_audio=ref_audio, ref_text=ref_text,
            adapter_label=adapter_label, speaker=speaker, seed=int(seed) + i,
            temperature=temperature, top_k=top_k, top_p=top_p,
            max_new_tokens=max_new_tokens, cfg_scale=cfg_scale,
            use_dual_cfg=use_dual_cfg, cfg_ref=cfg_ref, cfg_ins=cfg_ins,
        )
        parts.append(wav)
        if gap_ms and i < len(segs) - 1:
            parts.append(np.zeros(int(sr * gap_ms / 1000.0), dtype=np.float32))

    full = np.concatenate(parts) if parts else np.zeros(1, dtype=np.float32)
    script_slug = "segmentado_" + _slug(" ".join(s[1] for s in segs))
    out = _save_wav(full, sr, script_slug, ref_audio if has_ref else None)

    adapter_txt = adapter_label if adapter_label != BASE_LABEL else "base"
    lines = "\n".join(f"- *{instr}* -> {txt}" for instr, txt in segs)
    info = (
        f"**Modo:** segmentado ({len(segs)} trechos, {'com' if has_ref else 'sem'} referencia)  \n"
        f"**Adapter:** {adapter_txt}  \n"
        f"**CFG:** {cfg_scale}"
        + (f" (dual ref={cfg_ref}, ins={cfg_ins})" if use_dual_cfg else "")
        + f"  \n**Duracao:** {len(full) / sr:.2f}s  \n**Tempo:** {time.time() - t0:.1f}s  \n"
        f"**Arquivo:** `{out}`\n\n{lines}"
    )
    return str(out), info


def transcribe_ref(ref_audio: str | None, ref_path: str | None = None) -> str:
    """Transcreve o audio de referencia. Aceita o upload OU o caminho digitado."""
    src = (ref_path or "").strip() or ref_audio
    if not src:
        gr.Warning("Envie um audio OU informe o caminho do arquivo de referencia.")
        return ""
    if not Path(str(src)).is_file():
        gr.Warning(f"Arquivo de referencia nao encontrado: {src}")
        return ""
    model = _STATE.get("asr")
    if model is None:
        from faster_whisper import WhisperModel

        model = WhisperModel("large-v3", device="cpu", compute_type="int8")
        _STATE["asr"] = model
    segments, _ = model.transcribe(str(src), language="pt", vad_filter=True)
    return " ".join(s.text.strip() for s in segments).strip()


# ------------------------------------------------------------------ ui
import gradio as gr  # noqa: E402


def build_ui() -> gr.Blocks:
    adapters = list_adapters()
    _STATE["adapter_choices"] = adapters
    default_adapter = next(
        (a for a in adapters if "r64_02" in a),
        next((a for a in adapters if a.startswith("hf/")), adapters[0]),
    )

    with gr.Blocks(title="Breeze TTS 2 - LoRA PT-BR") as demo:
        gr.Markdown(
            "# Breeze TTS 2 - LoRA PT-BR\n"
            "Clonagem de voz e TTS com controle de emocao (system prompt) e CFG. "
            "O modelo e carregado na primeira geracao (leva ~1 min)."
        )

        with gr.Row():
            # ---------------- coluna de entrada
            with gr.Column(scale=3):
                with gr.Tab("Com voz de referencia (clonagem)"):
                    ref_audio = gr.Audio(
                        label="Audio de referencia (limpo, 3-30 s)",
                        type="filepath", sources=["upload", "microphone"],
                    )
                    with gr.Row():
                        ref_path = gr.Textbox(
                            label="...ou caminho do arquivo de referencia",
                            placeholder=r"C:\caminho\referencia.wav", scale=4,
                        )
                        btn_tr = gr.Button("Transcrever", scale=1)
                    ref_text = gr.Textbox(
                        label="Transcricao EXATA da referencia",
                        placeholder="O que exatamente e falado no audio de referencia.",
                        lines=3,
                    )

                with gr.Tab("Sem referencia (voz padrao)"):
                    gr.Markdown(
                        "Neste modo o texto e falado com a voz padrao do modelo "
                        "(nao clona). Use instrucao/CFG para o estilo."
                    )

                modo_seg = gr.Checkbox(
                    value=False,
                    label="Modo segmentado (emocao por trecho / por frase)",
                    info="Ligue para dar uma emocao diferente a cada linha do roteiro.",
                )

                with gr.Column(visible=True) as single_col:
                    text = gr.Textbox(
                        label="Texto a ser falado", lines=4,
                        placeholder="Digite o texto que o modelo deve falar.",
                    )
                    with gr.Row():
                        pause_on = gr.Checkbox(
                            value=False, scale=1,
                            label="Ajustar pausas (dramaticidade)",
                            info="Uma unica inferencia; as pausas naturais do audio sao "
                                 "redimensionadas depois (tom preservado).",
                        )
                        pause_ms = gr.Slider(
                            10, 3000, value=400, step=10, scale=2,
                            label="Duracao de cada pausa (ms)",
                        )
                    gr.Markdown(
                        "Marque *Ajustar pausas* e use pontuacao forte (`.` `…` ou quebras "
                        "de frase) para criar as pausas; cada pausa detectada e ajustada para "
                        "a duracao escolhida (10 ms a 3 s)."
                    )

                with gr.Column(visible=False) as seg_col:
                    script = gr.Textbox(
                        label="Roteiro por trechos (uma linha por trecho)",
                        lines=7,
                        placeholder=(
                            "Neutro / natural | Esta parte eu falo normal.\n"
                            "Alegre / sorrindo | MAS ESSA PARTE EU FALO SUPER ALEGRE!\n"
                            "sad | e esta aqui bem triste.\n"
                            "Sussurro | e esta aqui bem baixinho.\n"
                            "Fale como um narrador epico | No fim, tudo mudou."
                        ),
                    )
                    gap_ms = gr.Slider(0, 600, value=120, step=10,
                                       label="Pausa entre trechos (ms)")
                    gr.Markdown(
                        "Formato: `emocao | texto` (uma linha por trecho). A emocao pode ser um "
                        "**preset** do dropdown, um **apelido** PT/EN (ex.: `triste`, `sad`, "
                        "`alegre`, `happy`, `whisper`, `calmo`) ou uma **instrucao livre** "
                        "(ex.: `gritando de empolgacao`). Sem `|`, usa a emocao global.\n\n"
                        "**Importante:** a emocao é dirigida pela *instrucao*. Para ela ter "
                        "efeito forte, aumente o **CFG scale** (ex.: 3–4) em *Configurações "
                        "avançadas* — com CFG 1.0 a diferença fica sutil. No modo clonagem, "
                        "use o *dual-CFG* (`cfg_ins`)."
                    )

                gr.Markdown("### Emocao / estilo")
                emotion = gr.Dropdown(
                    choices=list(EMOTIONS.keys()),
                    value="Neutro / natural",
                    label="Preset de emocao (preenche a instrucao)",
                )
                instruction = gr.Textbox(
                    label="System prompt / instrucao (edite livremente)",
                    value=EMOTIONS["Neutro / natural"], lines=2,
                )
                emotion.change(lambda e: EMOTIONS.get(e, ""), inputs=emotion, outputs=instruction)

                with gr.Row():
                    adapter = gr.Dropdown(
                        choices=adapters, value=default_adapter,
                        label="Adapter LoRA (checkpoint)", scale=4,
                    )
                    refresh_btn = gr.Button("Atualizar", scale=1)

                with gr.Accordion("Configuracoes avancadas", open=False):
                    gr.Markdown(
                        "**CFG (classifier-free guidance):** `cfg_scale > 1` amplifica a "
                        "aderencia a instrucao (emocao). No modo clonagem, o *dual-CFG* separa "
                        "`cfg_ref` (fidelidade a voz) de `cfg_ins` (aderencia a emocao)."
                    )
                    cfg_scale = gr.Slider(1.0, 5.0, value=1.0, step=0.05, label="CFG scale (instrucao)")
                    use_dual = gr.Checkbox(value=False, label="Usar dual-CFG (so no modo clonagem)")
                    with gr.Row():
                        cfg_ref = gr.Slider(1.0, 6.0, value=3.0, step=0.05, label="cfg_ref (voz)")
                        cfg_ins = gr.Slider(1.0, 6.0, value=3.0, step=0.05, label="cfg_ins (emocao)")
                    speaker = gr.Textbox(value="S0", label="Speaker id")
                    with gr.Row():
                        seed = gr.Number(value=42, precision=0, label="Seed")
                        max_new = gr.Number(value=800, precision=0, label="max_new_tokens")
                    with gr.Row():
                        temperature = gr.Slider(0.1, 1.5, value=0.9, step=0.05, label="Temperature")
                        top_k = gr.Number(value=50, precision=0, label="top_k")
                        top_p = gr.Slider(0.1, 1.0, value=1.0, step=0.05, label="top_p")
                    gr.Markdown(
                        "Dica: o texto de referencia deve ser a transcricao **exata** do audio. "
                        "Referencias longas e limpas melhoram muito a clonagem."
                    )

                btn = gr.Button("Gerar audio", variant="primary")

            # ---------------- coluna de saida
            with gr.Column(scale=2):
                out_audio = gr.Audio(label="Audio gerado", type="filepath", autoplay=False)
                out_info = gr.Markdown("Pronto. Preencha o texto e clique em **Gerar audio**.")
                gr.Markdown(
                    "---\n**Como usar**\n"
                    "1. (Opcional) Envie a referencia + transcricao para clonar.\n"
                    "2. Digite o texto e escolha a emocao.\n"
                    "3. Ajuste o adapter e, se quiser, o CFG em *Configuracoes avancadas*.\n"
                    "4. Clique em **Gerar audio**.\n\n"
                    "**Emocao por trecho:** ligue *Modo segmentado* e escreva uma linha por "
                    "trecho no formato `emocao | texto`. Cada trecho e gerado com a sua "
                    "emocao e depois concatenado num unico audio."
                )

        btn_tr.click(transcribe_ref, inputs=[ref_audio, ref_path], outputs=[ref_text])

        def _toggle_seg(on):
            return (
                gr.update(visible=on),        # script
                gr.update(visible=on),        # gap
                gr.update(visible=not on),    # texto simples
                gr.update(visible=on),        # coluna segmentada
                gr.update(visible=not on),    # coluna simples
            )

        modo_seg.change(
            _toggle_seg, inputs=[modo_seg],
            outputs=[script, gap_ms, text, seg_col, single_col],
        )

        def _run(text_, script_, gap_, modo_seg_, emotion_, instruction_, ref_audio_, ref_path_,
                 ref_text_, adapter_, speaker_, seed_, temperature_, top_k_, top_p_, max_new_,
                 cfg_scale_, use_dual_, cfg_ref_, cfg_ins_, pause_on_, pause_ms_):
            ref = ref_path_.strip() if (ref_path_ and ref_path_.strip()) else ref_audio_
            common = dict(
                emotion=emotion_, instruction_custom=instruction_, ref_audio=ref,
                ref_text=ref_text_, adapter_label=adapter_, speaker=speaker_, seed=seed_,
                temperature=temperature_, top_k=top_k_, top_p=top_p_,
                max_new_tokens=max_new_, cfg_scale=cfg_scale_, use_dual_cfg=use_dual_,
                cfg_ref=cfg_ref_, cfg_ins=cfg_ins_,
            )
            if modo_seg_:
                return synthesize_segments(script=script_, gap_ms=int(gap_), **common)
            return synthesize(text=text_, pause_on=pause_on_, pause_ms=int(pause_ms_), **common)

        btn.click(
            _run,
            inputs=[text, script, gap_ms, modo_seg, emotion, instruction, ref_audio, ref_path,
                    ref_text, adapter, speaker, seed, temperature, top_k, top_p, max_new,
                    cfg_scale, use_dual, cfg_ref, cfg_ins, pause_on, pause_ms],
            outputs=[out_audio, out_info],
        )

        demo.load(lambda: list_adapters(), outputs=[adapter])
        refresh_btn.click(lambda: gr.update(choices=refresh_adapters()), outputs=[adapter])

    return demo


def main() -> None:
    ap = argparse.ArgumentParser(description="UI web (Gradio) do Breeze TTS 2 + LoRA")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--selftest", action="store_true", help="gera 1 amostra e sai")
    args = ap.parse_args()

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if args.selftest:
        load_model()
        audio, info = synthesize(text="Este e um teste da interface.", emotion="Alegre / sorrindo")
        print("SELFTEST OK ->", audio)
        print(info)
        return

    # Primeiro uso: baixa o adapter LoRA do Hugging Face (best-effort).
    CB.ensure_adapter()

    demo = build_ui()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Os audios sao salvos nos artefatos (fora do cwd); o Gradio exige liberar essas
    # pastas em allowed_paths, senao ele recusa mover o arquivo para o cache.
    allowed = sorted({str(OUT_DIR), str(CB.TRAINING), str(CB.ADAPTERS_DIR)})
    demo.queue().launch(
        server_name=args.host, server_port=args.port,
        share=args.share, inbrowser=not args.no_browser,
        allowed_paths=allowed,
        theme=gr.themes.Soft(
            font=["Segoe UI", "system-ui", "Tahoma", "Verdana", "Arial", "sans-serif"],
            font_mono=["Consolas", "Cascadia Mono", "Courier New", "monospace"],
        ),
    )


if __name__ == "__main__":
    main()
