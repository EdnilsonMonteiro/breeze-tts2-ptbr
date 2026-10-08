"""app.py — interface web (Gradio) para clonagem/TTS PT-BR com o Breeze TTS 2 + LoRA.

Roda no navegador. Carrega o modelo base uma vez, permite escolher um adapter LoRA,
gerar COM voz de referencia (clonagem) ou SEM referencia, com controle de emocao/estilo
(o "system prompt" = instrucao, igual ao modelo original) e CFG.

Os adapters LoRA ficam em `adapters/` NA RAIZ DESTE REPO (pasta ignorada pelo git) e
podem ser baixados pela propria UI, de
https://huggingface.co/EdnilsonMonts/Breeze-tts-2-brazillian-lora/tree/main.

Uso (Windows):
  .\\ui\\run.bat                        # abre em http://127.0.0.1:7860
  .\\venv\\Scripts\\python.exe ui\\app.py --port 7861

Uso (Linux/macOS):
  ./ui/run.sh                          # abre em http://127.0.0.1:7860
  ./venv/bin/python ui/app.py --selftest

Emocao: no Breeze, o estilo/emocao vem da INSTRUCAO (texto entre <ins_bos>...</ins_eos>)
e pode ser reforcada pelo CFG (cfg_scale>1 amplifica a instrucao). No modo clonagem com
dual-CFG, cfg_ref controla a fidelidade a voz e cfg_ins a aderencia a instrucao.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import shutil
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
import text_norm as TN  # noqa: E402
import adapter_scale as AS  # noqa: E402
import adapter_store as ADS  # noqa: E402
import reference_prep as RP  # noqa: E402
import text_blocks as TB  # noqa: E402
import candidate_select as CS  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402

OUT_DIR = CB.OUT_DIR

BASE_LABEL = "(base - sem adapter)"

EMOTIONS = {
    "Neutro / natural": "Fale com clareza e naturalidade.",
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
    "ref_warned": set(),
}


# ------------------------------------------------------------------ model
def _pick_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def refresh_adapters() -> list[str]:
    """Descobre os adapters LoRA em disco, na ordem de prioridade:

    1. `adapters/` na RAIZ deste repo (pasta ignorada pelo git) — onde a UI baixa o
       modelo do Hugging Face e onde voce pode copiar um checkpoint a mao;
    2. `<training>/runs/<run>/checkpoints/<ckpt>/` (adapters gerados pelo treino);
    3. `<artifacts>/adapters/<nome>/` (downloads antigos, fora do repo) — rotulo `hf/<nome>`.
    """
    found = ADS.find_adapters(CB.adapter_sources())
    _STATE["adapter_paths"] = {label: str(p) for label, p in found.items()}
    return [BASE_LABEL, *found]


def list_adapters() -> list[str]:
    return refresh_adapters()


# ---------------------------------------------------- download do adapter (HF)
def adapter_download_hint() -> str:
    """Instrucao mostrada quando o modelo ainda nao esta na pasta do repo."""
    target = CB.adapter_dir_for()
    page = CB.ADAPTER_REPO_PAGE or CB.ADAPTER_REPO_URL
    return (
        f"**O modelo será baixado de:** [{CB.ADAPTER_REPO}]({CB.ADAPTER_REPO_URL}) "
        "— repositório oficial no Hugging Face (~570 MB).\n\n"
        f"Ele fica em `{target}`, **dentro da pasta deste projeto**, numa pasta que o "
        "git ignora (nenhum peso vai para o repositório). Assim que o download terminar, "
        "esta opção desaparece e o adapter aparece na lista *Adapter LoRA (checkpoint)* "
        "acima.\n\n"
        "Sem internet? Baixe `adapter_config.json` e o `*.safetensors` na página "
        f"[{CB.ADAPTER_REPO}]({page}) e coloque os dois arquivos nessa pasta — a UI "
        "encontra sozinha (o botão também some, porque o modelo já está lá)."
    )


def adapter_download_state() -> tuple[bool, str]:
    """(falta_baixar, mensagem) para o bloco de download da UI."""
    try:
        path = CB.adapter_installed()
    except Exception as exc:  # noqa: BLE001
        return True, f":warning: nao consegui verificar o adapter local ({exc})."
    if path is not None:
        return False, (
            f"**Modelo LoRA instalado:** `{path}`  \n"
            f"Origem: [{CB.ADAPTER_REPO}]({CB.ADAPTER_REPO_URL}) — "
            "escolha-o no campo *Adapter LoRA (checkpoint)* acima."
        )
    return True, adapter_download_hint()


def _label_for_path(labels: list[str], path) -> str | None:
    """Rotulo do dropdown que aponta para `path` (None se nao estiver na lista)."""
    return ADS.labels_for_path(labels, _STATE.get("adapter_paths", {}), path)


def download_model():
    """Baixa o adapter LoRA do Hugging Face para `adapters/` (generator: status ao vivo)."""
    page = CB.ADAPTER_REPO_URL or CB.ADAPTER_REPO
    yield (
        f"⏳ Baixando o modelo LoRA de **{CB.ADAPTER_REPO}** (`{page}`)... "
        "são ~570 MB; o progresso por arquivo aparece no terminal.",
        gr.update(visible=False),
        gr.update(),
    )
    try:
        target = CB.download_adapter()
    except Exception as exc:  # noqa: BLE001
        yield (
            f"❌ **Falha no download:** `{exc}`\n\n"
            f"Você pode baixar manualmente em {page} e colocar `adapter_config.json` + "
            f"`*.safetensors` em `{CB.adapter_dir_for()}`.",
            gr.update(visible=True),
            gr.update(),
        )
        return
    opts = refresh_adapters()
    label = _label_for_path(opts, target) or _default_adapter_label(opts)
    missing, status = adapter_download_state()
    yield status, gr.update(visible=missing), gr.update(choices=opts, value=label)


# ------------------------------------------------------------------ vozes salvas
VOICES_PATH = CB.ARTIFACTS / "voices.json"


def _load_voices() -> dict:
    if VOICES_PATH.is_file():
        try:
            return json.loads(VOICES_PATH.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return {}
    return {}


def _save_voices(data: dict) -> None:
    VOICES_PATH.parent.mkdir(parents=True, exist_ok=True)
    VOICES_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


# ------------------------------------------------------------------ estado da UI
UI_STATE_PATH = CB.ARTIFACTS / "ui_state.json"


def _load_ui_state() -> dict:
    if UI_STATE_PATH.is_file():
        try:
            return json.loads(UI_STATE_PATH.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return {}
    return {}


def _save_ui_state(**kw) -> None:
    data = _load_ui_state()
    data.update(kw)
    UI_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    UI_STATE_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


def _resolve_adapter_path(label: str) -> Path | None:
    """Pasta do adapter: a que a ultima varredura achou ou o rotulo sobre as pastas conhecidas."""
    p = _STATE.get("adapter_paths", {}).get(label)
    if p:
        return Path(p)
    for prefix, base in CB.adapter_sources():
        rel = label[len(prefix) + 1:] if (prefix and label.startswith(f"{prefix}/")) else label
        cand = Path(base) / rel
        if cand.is_dir():
            return cand
    return None


def _adapter_mtime(label: str) -> float | None:
    p = _resolve_adapter_path(label)
    try:
        return p.stat().st_mtime if p else None
    except OSError:
        return None


def _newest_adapter(adapters: list[str], prefer_final: bool = True) -> str | None:
    """Checkpoint mais recente (por data). Com `prefer_final`, `final`/`best` vem antes de
    um `stepN` mais novo (checkpoints intermediarios nao sao a entrega da run)."""
    def newest(cands: list[str]) -> str | None:
        best, best_t = None, None
        for a in cands:
            t = _adapter_mtime(a)
            if t is not None and (best_t is None or t > best_t):
                best, best_t = a, t
        return best

    real = [a for a in adapters if a != BASE_LABEL]
    if prefer_final:
        pick = newest([a for a in real if a.endswith("/final") or a.endswith("/best")])
        if pick:
            return pick
    return newest(real)


def _env_adapter(adapters: list[str]) -> str | None:
    """$PTBR_DEFAULT_ADAPTER: rotulo do dropdown OU caminho da pasta do adapter."""
    env = os.environ.get("PTBR_DEFAULT_ADAPTER", "").strip()
    if not env:
        return None
    if env in adapters:
        return env
    try:
        want = Path(env).resolve()
    except OSError:
        return None
    for label, p in _STATE.get("adapter_paths", {}).items():
        try:
            if Path(p).resolve() == want and label in adapters:
                return label
        except OSError:
            continue
    return None


def _default_adapter_label(adapters: list[str]) -> str:
    """Prioriza: ultimo adapter usado -> $PTBR_DEFAULT_ADAPTER -> `final`/`best` mais
    recente -> adapter mais recente -> primeiro da lista (base)."""
    last = _load_ui_state().get("adapter")
    if last in adapters:
        return last
    return _env_adapter(adapters) or _newest_adapter(adapters) or adapters[0]


def _stash_reference(src: str, name: str) -> str:
    """Copia a referencia para <ARTIFACTS>/voice_refs/ (o arquivo do Gradio e temporario e o
    caminho digitado pode mudar de lugar). Retorna o caminho da copia."""
    sp = Path(str(src))
    d = CB.ARTIFACTS / "voice_refs"
    d.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha1(f"{sp.resolve()}|{sp.stat().st_mtime_ns}".encode()).hexdigest()[:8]
    dst = d / f"{_slug(name)}_{h}{sp.suffix or '.wav'}"
    if not dst.exists():
        shutil.copyfile(sp, dst)
    return str(dst)


def _persist_voice(name, ref_path, ref_text, adapter_label, emotion, instruction, speaker,
                   temperature, top_k, top_p, max_new, cfg_scale, use_dual, cfg_ref, cfg_ins,
                   seed=None, adapter_scale=1.0, ref_format="train") -> str:
    name = (name or "").strip() or (Path(ref_path).stem if ref_path else "voz") or "voz"
    data = _load_voices()
    data[name] = {
        "ref_path": (ref_path or ""), "ref_text": (ref_text or ""), "adapter": adapter_label,
        "emotion": emotion, "instruction": instruction, "speaker": speaker or "S0",
        "temperature": float(temperature), "top_k": int(top_k), "top_p": float(top_p),
        "max_new_tokens": int(max_new), "cfg_scale": float(cfg_scale),
        "use_dual_cfg": bool(use_dual), "cfg_ref": float(cfg_ref), "cfg_ins": float(cfg_ins),
        "seed": int(seed) if seed is not None else None,
        "adapter_scale": float(adapter_scale), "ref_format": ref_format or "train",
    }
    _save_voices(data)
    return name


def save_voice(name, ref_audio, ref_path, ref_text, adapter_label, emotion, instruction, speaker,
               temperature, top_k, top_p, max_new, cfg_scale, use_dual, cfg_ref, cfg_ins, seed,
               adapter_scale, ref_format):
    name = (name or "").strip()
    if not name:
        raise gr.Error("Informe um nome para a voz.")
    # a referencia pode vir do UPLOAD/microfone ou do caminho digitado (antes so o caminho era salvo)
    src = (ref_path or "").strip() or ref_audio
    stored = ""
    if src:
        if not Path(str(src)).is_file():
            raise gr.Error(f"Arquivo de referencia nao encontrado: {src}")
        stored = _stash_reference(str(src), name)
    _persist_voice(name, stored, ref_text, adapter_label, emotion, instruction, speaker,
                   temperature, top_k, top_p, max_new, cfg_scale, use_dual, cfg_ref, cfg_ins, seed,
                   adapter_scale, ref_format)
    return gr.update(choices=sorted(_load_voices()), value=name)


def load_voice(name):
    v = _load_voices().get(name, {})
    return (
        gr.update(value=v.get("ref_path", "")),
        gr.update(value=v.get("ref_text", "")),
        gr.update(value=v.get("adapter")),
        gr.update(value=v.get("emotion", "Neutro / natural")),
        gr.update(value=v.get("instruction", EMOTIONS["Neutro / natural"])),
        gr.update(value=v.get("speaker", "S0")),
        gr.update(value=v.get("temperature", 0.7)),
        gr.update(value=v.get("top_k", 50)),
        gr.update(value=v.get("top_p", 1.0)),
        gr.update(value=v.get("max_new_tokens", 400)),
        gr.update(value=v.get("cfg_scale", 1.0)),
        gr.update(value=v.get("use_dual_cfg", False)),
        gr.update(value=v.get("cfg_ref", 3.0)),
        gr.update(value=v.get("cfg_ins", 3.0)),
        gr.update(value=v.get("seed") if v.get("seed") is not None else 42),
        gr.update(value=v.get("adapter_scale", 1.0)),
        gr.update(value=v.get("ref_format", "train")),
        gr.update(value=v.get("ref_path") or None),           # mostra a referencia no player
    )


def delete_voice(name):
    data = _load_voices()
    data.pop(name, None)
    _save_voices(data)
    return gr.update(choices=sorted(data), value=None), gr.update(value="")


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
    # Robustez: alguns componentes (ou vozes salvas) podem entregar lista; pega o 1o.
    if isinstance(label, (list, tuple)):
        label = label[0] if label else BASE_LABEL
    label = "" if label is None else str(label)
    raw = _STATE["raw"]
    peft = _STATE["peft"]

    if label in (None, "", BASE_LABEL, "base", "Base"):
        if peft is not None:
            peft.disable_adapter_layers()
            return peft
        return raw

    path = _resolve_adapter_path(label)
    if path is None or not path.is_dir():
        raise FileNotFoundError(
            f"adapter nao encontrado: {label}. Ele deveria estar em {CB.ADAPTERS_DIR} "
            f"(pasta do repo, ignorada pelo git) ou em {CB.TRAINING_RUNS_DIR}. "
            "Use o botao 'Baixar modelo do Hugging Face' ou clique em Atualizar."
        )
    # O repositorio publica os pesos com outro nome (breeze-tts-2-pt-br-lora.safetensors);
    # o PEFT so carrega `adapter_model.*`, entao criamos esse nome (hardlink) se faltar.
    CB.prepare_adapter(path)

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


def _set_adapter_scale(model, factor: float) -> None:
    """Escala = escala TREINADA x `factor` (1.0 = como treinado; v2/v3: alpha/r = 4,0).

    Implementacao unica em core/adapter_scale.py: base registrado POR CHAVE (adapter carregado
    depois nao herda escala errada) e idempotente.
    """
    try:
        AS.apply_adapter_scale(model, float(factor))
    except Exception as exc:  # noqa: BLE001
        print(f"[ui] (aviso) nao foi possivel ajustar a escala do adapter: {exc}", flush=True)


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
    adapter_scale: float = 1.0,
    ref_format: str = "train",
) -> tuple[np.ndarray, int]:
    """Gera UMA locucao (uma instrucao global). Retorna (wav float32, sample_rate)."""
    text = TN.normalize(text)  # numeros por extenso (o modelo fala melhor)
    has_ref = bool(ref_audio) and bool((ref_text or "").strip())
    model = use_adapter(adapter_label)
    _set_adapter_scale(model, adapter_scale)
    tokenizer = _STATE["tokenizer"]
    audio_tok = _STATE["audio_tok"]

    from breeze_infer.runtime import set_all_seeds, update_generation_config_for_breeze
    from breeze_infer.templates import get_template, prepare_inputs

    update_generation_config_for_breeze(model)
    set_all_seeds(int(seed))

    request = {"id": "ui", "text": text, "instruction": instruction, "speaker": speaker or "S0"}
    if has_ref:
        # prepara a referencia como o TREINO a viu (mono 24 kHz, trim, peak-norm); cacheado,
        # nao altera o arquivo original. "48k" = formato antigo; "raw" = arquivo como esta.
        try:
            ref_audio, ref_dur = RP.prepare_reference(ref_audio, CB.ARTIFACTS / "voice_cache",
                                                      ref_format or "train")
            warn = RP.duration_warning(ref_dur)
            if warn and str(ref_audio) not in _STATE["ref_warned"]:
                _STATE["ref_warned"].add(str(ref_audio))
                gr.Warning(warn)
        except Exception as exc:  # noqa: BLE001
            print(f"[ui] (aviso) preparo da referencia falhou: {exc}", flush=True)
        request["ref_audio_path"] = str(ref_audio)
        request["ref_text"] = TN.normalize(ref_text.strip())   # mesma normalizacao do texto-alvo
        template = "ref_edit_tata"
        try:
            CB.cache_reference(audio_tok, ref_audio)  # codifica a referencia 1x
        except Exception as exc:  # noqa: BLE001
            print(f"[ui] (aviso) cache da referencia falhou: {exc}", flush=True)
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


def _rms(w: np.ndarray) -> float:
    return float(np.sqrt(np.mean(w ** 2))) if len(w) else 0.0


def _medoid_pick(wavs: list[np.ndarray], sr: int, n_words: int) -> int:
    """Indice do candidato mais CENTRAL (ECAPA) entre os de duracao plausivel para o texto.

    A escolha por medoid (nao pelo maior cos com a referencia) evita premiar ruido de
    amostragem; o gate de duracao descarta fala arrastada/truncada.
    """
    import tempfile

    ok = [i for i, w in enumerate(wavs) if TB.dur_ok(len(w) / sr, n_words)] or list(range(len(wavs)))
    if len(ok) <= 2:
        return ok[0]
    embs = []
    with tempfile.TemporaryDirectory() as td:
        for i in ok:
            pth = Path(td) / f"c{i}.wav"
            sf.write(str(pth), np.clip(wavs[i], -1.0, 1.0), sr, subtype="PCM_16")
            embs.append(CB.speaker_embed(pth, "cpu"))
    E = np.stack(embs)
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
    c = E.mean(axis=0)
    c = c / (np.linalg.norm(c) + 1e-9)
    return ok[int(np.argmax(E @ c))]


def _log_error(where: str) -> None:
    """Grava o traceback atual em <ui_out>/ui_error.log (a janela do run.bat perde o historico)."""
    import traceback

    try:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        with open(OUT_DIR / "ui_error.log", "a", encoding="utf-8") as f:
            f.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} [{where}]\n{traceback.format_exc()}")
    except Exception:  # noqa: BLE001
        pass


def _get_asr():
    """faster-whisper (carregado 1x, compartilhado com 'Transcrever').

    Padrao: CPU int8 (nao disputa VRAM com o Breeze e nao depende de cuBLAS/cuDNN).
    Opcoes: PTBR_WHISPER_DEVICE=cuda (float16; cai para CPU se falhar), PTBR_WHISPER_MODEL
    (padrao large-v3), PTBR_WHISPER_THREADS (CPU)."""
    model = _STATE.get("asr")
    if model is None:
        from faster_whisper import WhisperModel

        name = os.environ.get("PTBR_WHISPER_MODEL", "large-v3")
        dev = os.environ.get("PTBR_WHISPER_DEVICE", "cpu").lower()
        if dev == "cuda":
            try:
                model = WhisperModel(name, device="cuda", compute_type="float16")
            except Exception as exc:  # noqa: BLE001
                print(f"[ui] (aviso) whisper em cuda falhou ({exc}); usando CPU", flush=True)
        if model is None:
            threads = int(os.environ.get("PTBR_WHISPER_THREADS", "0") or 0)
            model = WhisperModel(name, device="cpu", compute_type="int8", cpu_threads=threads)
        _STATE["asr"] = model
    return model


def _asr_text(wav: np.ndarray, sr: int) -> str:
    """Transcreve um candidato SEM dar o texto-alvo como prompt (senao o Whisper "corrige" o erro)."""
    import librosa

    x = librosa.resample(np.asarray(wav, dtype=np.float32), orig_sr=sr, target_sr=16000) if sr != 16000 \
        else np.asarray(wav, dtype=np.float32)
    segments, _ = _get_asr().transcribe(x, language="pt", vad_filter=False,
                                        condition_on_previous_text=False)
    return " ".join(s.text.strip() for s in segments).strip()


def _embed_wav(wav: np.ndarray, sr: int) -> np.ndarray:
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        pth = Path(td) / "c.wav"
        sf.write(str(pth), np.clip(wav, -1.0, 1.0), sr, subtype="PCM_16")
        return CB.speaker_embed(pth, "cpu")


def _ref_profile(ref_audio: str | None, ref_text: str, ref_format: str):
    """(embedding ECAPA, RMS) da referencia PREPARADA (a mesma que o modelo ouviu). None sem ref."""
    if not (ref_audio and (ref_text or "").strip()):
        return None, None
    path, _dur = RP.prepare_reference(ref_audio, CB.ARTIFACTS / "voice_cache", ref_format or "train")
    p = Path(str(path))
    key = (str(p), p.stat().st_mtime_ns)
    cache = _STATE.setdefault("ref_profile", {})
    if key not in cache:
        w, sr = sf.read(str(p), dtype="float32", always_2d=True)
        cache[key] = (CB.speaker_embed(p, "cpu"), CS.rms(w[:, 0]))
    return cache[key]


def _score_block(wavs: list[np.ndarray], sr: int, btxt: str, seeds: list[int], gen_kwargs: dict):
    """Melhor de N por (palavras certas via Whisper) + (voz parecida com a referencia via ECAPA).

    Retorna (posicao escolhida, [Cand]). Whisper indisponivel -> so voz/duracao (avisa 1x)."""
    transcribe = None if _STATE.get("asr_failed") else _asr_text
    if transcribe is not None:
        try:
            _get_asr()
        except Exception as exc:  # noqa: BLE001
            _STATE["asr_failed"] = True
            transcribe = None
            print(f"[ui] (aviso) faster-whisper indisponivel: {exc}", flush=True)
            gr.Warning("Whisper indisponivel: escolhendo so pela voz (instale faster-whisper).")
    ref_emb, ref_rms = _ref_profile(gen_kwargs.get("ref_audio"), gen_kwargs.get("ref_text", ""),
                                    gen_kwargs.get("ref_format", "train"))
    n_words = len(btxt.split())
    return CS.evaluate(
        wavs, sr, TN.normalize(btxt), seeds,
        dur_ok=lambda d: TB.dur_ok(d, n_words),
        transcribe=transcribe, embed=_embed_wav, ref_emb=ref_emb, ref_rms=ref_rms,
        norm=TN.normalize,
    )


def _generate_long(
    text: str,
    *,
    seed: int,
    auto_chunk: bool = True,
    max_block_s: float = 10.0,
    n_candidates: int = 1,
    gap_ms: int = 200,
    best_of: bool = True,
    report: list | None = None,
    **gen_kwargs,
) -> tuple[np.ndarray, int, int]:
    """Texto longo -> blocos de <= max_block_s (o treino so viu clipes <= ~10 s; gerar 20-30 s
    de uma vez e extrapolar o comprimento e a voz "escorrega"). Cada bloco usa a MESMA
    referencia. Com n_candidates > 1: `best_of` escolhe por Whisper (palavras) + SECS (voz);
    sem `best_of` usa o medoid (so age com >= 3). Seeds: seed + bloco*1000 + k (k=0 = a seed
    exata de antes, entao o melhor-de-N nunca fica pior que o candidato unico).
    `report` (lista) recebe 1 linha Markdown por bloco. Retorna (wav, sr, n_blocos).

    Com 1 bloco e 1 candidato o comportamento e o de antes (seed exata)."""
    blocks = TB.blocks_for_seconds(text, float(max_block_s)) if auto_chunk else [text]
    blocks = blocks or [text]
    parts: list[np.ndarray] = []
    sr = 24_000
    for bi, btxt in enumerate(blocks):
        cands: list[np.ndarray] = []
        seeds = [int(seed) + bi * 1000 + k for k in range(max(1, int(n_candidates)))]
        for sd in seeds:
            wav, sr = _generate(text=btxt, seed=sd, **gen_kwargs)
            cands.append(wav)
        pick = 0
        if len(cands) > 1 and best_of:
            try:
                pick, scored = _score_block(cands, sr, btxt, seeds, gen_kwargs)
                if report is not None:
                    has_ref = bool(gen_kwargs.get("ref_audio")) and bool((gen_kwargs.get("ref_text") or "").strip())
                    report.append(CS.format_block(bi, scored, has_ref,
                                                  any(c.wer is not None for c in scored)))
            except Exception as exc:  # noqa: BLE001
                _log_error("melhor-de-N")
                print(f"[ui] (aviso) melhor-de-N falhou ({exc}); usando medoid", flush=True)
                try:
                    pick = _medoid_pick(cands, sr, len(btxt.split())) if len(cands) > 2 else 0
                except Exception:  # noqa: BLE001  (sem ECAPA: fica com a seed exata, nao derruba a geracao)
                    _log_error("medoid")
                    pick = 0
        elif len(cands) > 2:
            pick = _medoid_pick(cands, sr, len(btxt.split()))
        parts.append(cands[pick])
    if len(parts) > 1:                                  # iguala o volume entre blocos
        tgt = float(np.mean([_rms(w) for w in parts]))
        parts = [w * min(4.0, tgt / _rms(w)) if _rms(w) > 1e-6 else w for w in parts]
        gap = np.zeros(int(sr * max(0, int(gap_ms)) / 1000.0), dtype=np.float32)
        joined: list[np.ndarray] = []
        for i, w in enumerate(parts):
            joined.append(w.astype(np.float32))
            if i < len(parts) - 1:
                joined.append(gap)
        return np.concatenate(joined), sr, len(blocks)
    return parts[0], sr, 1


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


def _report_md(report: list[str], max_lines: int = 8) -> str:
    """Bloco Markdown 'Melhor de N' para o painel de info (vazio se nao houve selecao)."""
    if not report:
        return ""
    shown = report[:max_lines]
    more = f"\n- ... (+{len(report) - max_lines} blocos)" if len(report) > max_lines else ""
    return "  \n**Melhor de N** (✔ = escolhido):\n" + "\n".join(f"- {r}" for r in shown) + more + "\n"


def synthesize(
    text: str,
    emotion: str = "Neutro / natural",
    instruction_custom: str = "",
    ref_audio: str | None = None,
    ref_text: str = "",
    adapter_label: str = BASE_LABEL,
    speaker: str = "S0",
    seed: int = 42,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 1.0,
    max_new_tokens: int = 400,
    cfg_scale: float = 1.0,
    use_dual_cfg: bool = False,
    cfg_ref: float = 1.0,
    cfg_ins: float = 1.0,
    pause_on: bool = False,
    pause_ms: int = 300,
    adapter_scale: float = 1.0,
    ref_format: str = "train",
    auto_chunk: bool = True,
    max_block_s: float = 10.0,
    n_candidates: int = 1,
    best_of: bool = True,
):
    text = (text or "").strip()
    if not text:
        raise gr.Error("Informe o texto a ser falado.")

    instruction = (instruction_custom or "").strip() or EMOTIONS.get(emotion, EMOTIONS["Neutro / natural"])

    has_ref = bool(ref_audio) and bool((ref_text or "").strip())
    if bool(ref_audio) != bool((ref_text or "").strip()):
        raise gr.Error("Referencia e transcricao devem ser preenchidas juntas.")

    t0 = time.time()
    report: list[str] = []
    wav, sr, n_blocks = _generate_long(
        text, seed=int(seed), auto_chunk=auto_chunk, max_block_s=max_block_s,
        n_candidates=n_candidates, best_of=best_of, report=report,
        instruction=instruction, ref_audio=ref_audio, ref_text=ref_text,
        adapter_label=adapter_label, speaker=speaker, temperature=temperature,
        top_k=top_k, top_p=top_p, max_new_tokens=max_new_tokens, cfg_scale=cfg_scale,
        use_dual_cfg=use_dual_cfg, cfg_ref=cfg_ref, cfg_ins=cfg_ins,
        adapter_scale=adapter_scale, ref_format=ref_format,
    )
    n_pauses = 0
    if pause_on:
        wav, n_pauses = adjust_pauses(wav, sr, int(pause_ms))
    out = _save_wav(wav, sr, text, ref_audio if has_ref else None)

    mode = "clonagem (com referencia)" if has_ref else "voz padrao (sem referencia)"
    adapter_txt = adapter_label if adapter_label != BASE_LABEL else "base"
    persist_line = (f"  \n**Pausas:** {n_pauses} ajustada(s) para {int(pause_ms)} ms"
                    if pause_on else "")
    if n_blocks > 1 or n_candidates > 1:
        persist_line += (f"  \n**Blocos:** {n_blocks} (<= {float(max_block_s):.0f} s cada) x "
                         f"{int(n_candidates)} candidato(s)/bloco")
    persist_line += _report_md(report)
    if adapter_label == BASE_LABEL:
        persist_line += "  \n**Aviso:** modelo base (sem adapter) nao fala pt-BR bem."
    if not has_ref and adapter_label != BASE_LABEL:
        persist_line += ("  \n**Aviso:** sem referencia — adapters v2 nao foram treinados nesse modo "
                         "(so os v3 cobrem ~15 %); prefira clonar com uma referencia.")
    # Instrucao/emocao: numeros medidos em docs/EMOTION.md (mesmo texto/seed/referencia,
    # so a instrucao e o cfg mudam; comparado com o ruido de trocar a seed).
    if float(cfg_scale) < 2.0:
        persist_line += ("  \n**Dica:** com CFG ~1 a instrucao dirige pouco a voz (energia/ritmo "
                         "quase nao mudam) — para emocao/estilo use CFG 3-4.")
    if adapter_label != BASE_LABEL:
        persist_line += ("  \n**Aviso:** adapters pt-BR v2/v3 foram treinados com instrucao "
                         "FIXA e nao seguem emocao (a voz sai neutra; medido em docs/EMOTION.md). "
                         "Para emocao: modelo base com instrucao descritiva, ou uma referencia "
                         "que ja tenha a emocao desejada.")
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
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 1.0,
    max_new_tokens: int = 400,
    cfg_scale: float = 1.0,
    use_dual_cfg: bool = False,
    cfg_ref: float = 1.0,
    cfg_ins: float = 1.0,
    adapter_scale: float = 1.0,
    ref_format: str = "train",
    auto_chunk: bool = True,
    max_block_s: float = 10.0,
    n_candidates: int = 1,
    best_of: bool = True,
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
    report: list[str] = []
    for i, (instr, seg_text) in enumerate(segs):
        wav, sr, _nb = _generate_long(
            seg_text, seed=int(seed) + i, auto_chunk=auto_chunk, max_block_s=max_block_s,
            n_candidates=n_candidates, best_of=best_of, report=report,
            instruction=instr, ref_audio=ref_audio, ref_text=ref_text,
            adapter_label=adapter_label, speaker=speaker,
            temperature=temperature, top_k=top_k, top_p=top_p,
            max_new_tokens=max_new_tokens, cfg_scale=cfg_scale,
            use_dual_cfg=use_dual_cfg, cfg_ref=cfg_ref, cfg_ins=cfg_ins,
            adapter_scale=adapter_scale, ref_format=ref_format,
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
        f"**Arquivo:** `{out}`{_report_md(report)}\n\n{lines}"
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
    model = _get_asr()
    segments, _ = model.transcribe(str(src), language="pt", vad_filter=True)
    return " ".join(s.text.strip() for s in segments).strip()


# ------------------------------------------------------------------ ui
import gradio as gr  # noqa: E402


def build_ui() -> gr.Blocks:
    adapters = list_adapters()
    _STATE["adapter_choices"] = adapters
    default_adapter = _default_adapter_label(adapters)
    # Se o adapter ainda nao estiver em adapters/, a UI oferece o download (e diz de
    # onde ele vem); depois de baixado a opcao desaparece sozinha.
    need_download, dl_message = adapter_download_state()

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
                        label="Audio de referencia (limpo, ideal 3-10 s; o treino viu ate ~10 s)",
                        type="filepath", sources=["upload", "microphone"],
                    )
                    with gr.Row():
                        ref_path = gr.Textbox(
                            label="...ou caminho do arquivo de referencia",
                            placeholder=str(Path.home() / "referencia.wav"), scale=4,
                        )
                        btn_tr = gr.Button("Transcrever", scale=1)
                    ref_text = gr.Textbox(
                        label="Transcricao EXATA da referencia",
                        placeholder="O que exatamente e falado no audio de referencia.",
                        lines=3,
                    )
                    ref_info = gr.Markdown("")

                with gr.Tab("Sem referencia (voz padrao)"):
                    gr.Markdown(
                        "Neste modo o texto e falado com a voz padrao do modelo "
                        "(nao clona). Use instrucao/CFG para o estilo.\n\n"
                        "**O modo e decidido por haver (ou nao) audio + transcricao preenchidos "
                        "na aba de clonagem, nao pela aba aberta.**\n\n"
                        "**Aviso:** adapters treinados na receita v2 foram treinados 100 % com "
                        "referencia; sem referencia a qualidade pode ser pior que a do modelo "
                        "base. Os adapters v3 cobrem esse modo (~15 % do treino)."
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
                    info="Quem segue esta instrucao e o MODELO, nao a UI: o modelo base segue bem "
                         "(instrucao descritiva, no idioma do texto, com CFG 3-4), mas os adapters "
                         "pt-BR v2/v3 foram treinados com instrucao FIXA e praticamente nao mudam a "
                         "voz por emocao. Medido em docs/EMOTION.md.",
                )
                # .input (so acao do usuario): com .change, carregar uma voz salva disparava este evento e
                # sobrescrevia a instrucao personalizada salva junto com a voz.
                emotion.input(lambda e: EMOTIONS.get(e, ""), inputs=emotion, outputs=instruction)
                with gr.Accordion("Emocao: o que funciona (medido)", open=False):
                    gr.Markdown(
                        "- **Modelo base** (`(base - sem adapter)`): segue instrucao de verdade. "
                        "No teste controlado (mesmo texto/seed), *\"Speak with great excitement…\"* "
                        "com **CFG 4** levou a F0 de 225 → 353 Hz, −35 % de duracao e +133 % de "
                        "energia; *sussurro* saiu sem vozeamento e 10× mais baixo. Com **CFG 1** o "
                        "efeito cai muito (F0 +37 Hz em vez de +128 Hz).\n"
                        "- **Adapters pt-BR v2/v3** (inclui o publicado): **nao seguem emocao**. "
                        "Foram treinados com uma unica instrucao neutra fixa (*\"Fale com clareza e "
                        "naturalidade.\"*), entao trocar por *\"empolgado\"*/*\"triste\"* mexe no "
                        "audio menos do que trocar a seed. Use-os pelo portugues/clonagem; a emocao "
                        "tem de vir da **referencia** (referencia alegre clona alegre) ou de "
                        "pos-processamento.\n"
                        "- Detalhes, numeros e como remedir: `docs/EMOTION.md` e "
                        "`python scripts/measure_instruction.py --help`."
                    )

                with gr.Row():
                    adapter = gr.Dropdown(
                        choices=adapters, value=default_adapter,
                        label="Adapter LoRA (checkpoint)", scale=4,
                    )
                    refresh_btn = gr.Button("Atualizar", scale=1)

                # A pasta e a `adapters/` da raiz do repo (ignorada pelo git); o bloco so
                # aparece aberto enquanto o modelo nao estiver baixado.
                with gr.Accordion("Modelo LoRA (pasta adapters/ do projeto)", open=need_download):
                    dl_status = gr.Markdown(dl_message)
                    dl_btn = gr.Button(
                        f"Baixar modelo de {CB.ADAPTER_REPO} (Hugging Face)",
                        variant="primary", visible=need_download,
                    )

                with gr.Accordion("Vozes salvas", open=False):
                    with gr.Row():
                        voice_name = gr.Textbox(label="Nome da voz", placeholder="minha_voz", scale=3)
                        save_voice_btn = gr.Button("Salvar voz", scale=1)
                    with gr.Row():
                        voice_pick = gr.Dropdown(label="Vozes salvas",
                                                 choices=sorted(_load_voices()), scale=3)
                        load_voice_btn = gr.Button("Carregar", scale=1)
                        del_voice_btn = gr.Button("Excluir", scale=1)
                    gr.Markdown(
                        "**Salvar voz** grava a referência + a config atual (temperature/CFG/seed). "
                        "**Carregar** preenche a referência/transcrição e os parâmetros salvos."
                    )

                with gr.Accordion("Configuracoes avancadas", open=False):
                    with gr.Accordion("(i) O que faz cada parametro", open=False):
                        gr.Markdown(
                            "- **Temperature** (0,1–1,5): aleatoriedade do sorteio de tokens. "
                            "Baixa (~0,5–0,7) = mais estável e fiel, menos variada; alta (>1,0) = "
                            "mais expressiva, porém imprevisível e pode 'perder' a voz. Para "
                            "clonagem, **0,7** costuma sair melhor.\n"
                            "- **top_k**: sorteia só entre os K tokens mais prováveis. Menor "
                            "(20–50) = foco/estabilidade; maior = mais liberdade.\n"
                            "- **top_p** (nucleus): sorteia do menor conjunto cuja probabilidade "
                            "soma `p`. `1,0` = desligado; `0,9` = mais foco. Ajuste **top_k ou "
                            "top_p**, não os dois ao extremo.\n"
                            "- **CFG scale** (guidance): quanto reforçar a **instrução** (emoção/"
                            "estilo). `1,0` = sem reforço; `3–4` = forte (recomendado p/ direção). "
                            "Com CFG 1 a instrução quase não muda a voz — medido em "
                            "`docs/EMOTION.md`.\n"
                            "- **cfg_ref** (dual-CFG): fidelidade à **voz de referência**. Maior = "
                            "mais parecido com a referência.\n"
                            "- **cfg_ins** (dual-CFG): aderência à **instrução** (emoção). Maior = "
                            "segue mais a emoção.\n"
                            "- **Seed**: semente do sorteio. Mesma seed + mesmos parâmetros = "
                            "mesmo áudio. Seeds diferentes mudam a realização — por isso algumas "
                            "'acertam' mais que outras.\n"
                            "- **max_new_tokens**: limite de tokens de áudio do BLOCO (o codec roda a "
                            "12,5 tokens ≈ 1 s: 400 ≈ 32 s). Com *Dividir texto longo* ligado cada "
                            "bloco tem ≤ 10 s, então 400 sobra.\n"
                            "- **Dividir texto longo** (recomendado): o treino só viu clipes de até "
                            "~10 s; gerar 20–30 s de uma vez faz a voz derivar. O texto é fatiado em "
                            "blocos de ≤ N s (pontuação forte) e cada bloco usa a mesma referência.\n"
                            "- **Escolher o melhor de N candidatos** (ligado por padrão, N = 4): gera N "
                            "variações do bloco (seeds `seed`, `seed+1`, …) e escolhe: (1) duração "
                            "plausível; (2) sem erro de palavra — o Whisper transcreve cada candidato e "
                            "compara com o texto; (3) entre os sem erro, a **mais parecida com a "
                            "referência** (SECS/ECAPA, volume igualado). Sem referência escolhe a mais "
                            "'central'. O candidato 0 é o áudio que você teria com 1 só geração. N× mais "
                            "lento; Whisper roda na CPU (`PTBR_WHISPER_DEVICE=cuda` para usar a GPU).\n"
                            "- **Formato da referência**: *Como no treino* (24 kHz, corta silêncio das "
                            "pontas, normaliza pico) é o que o adapter viu; *48 kHz* é o formato "
                            "anterior desta UI; *Original* não mexe no arquivo.\n"
                            "- **Speaker id**: tag de locutor; mantenha `S0`.\n"
                            "- **Escala do adapter** (0,3–1,0): atenua o LoRA em inferência. "
                            "`1,0` = escala treinada (adapters v2/v3: alpha/r = 4,0) e é o recomendado: "
                            "nos testes desta receita valores < 1,0 pioraram o português."
                        )
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
                        max_new = gr.Number(value=400, precision=0, label="max_new_tokens (12,5 tokens = 1 s)")
                    with gr.Row():
                        temperature = gr.Slider(0.1, 1.5, value=0.7, step=0.05, label="Temperature")
                        top_k = gr.Number(value=50, precision=0, label="top_k")
                        top_p = gr.Slider(0.1, 1.0, value=1.0, step=0.05, label="top_p")
                    adapter_scale = gr.Slider(
                        0.3, 1.0, value=1.0, step=0.05,
                        label="Escala do adapter (1.0 = treinada; <1.0 atenua)",
                    )
                    with gr.Row():
                        auto_chunk = gr.Checkbox(value=True, label="Dividir texto longo em blocos")
                        max_block_s = gr.Slider(5.0, 10.0, value=10.0, step=0.5,
                                                label="Duracao maxima do bloco (s)")
                    ref_format = gr.Dropdown(
                        choices=[("Como no treino (24 kHz, corta silencio, normaliza)", "train"),
                                 ("48 kHz (formato anterior da UI)", "48k"),
                                 ("Original (sem preparo)", "raw")],
                        value="train", label="Formato da referencia",
                    )
                    gr.Markdown(
                        "Dica: o texto de referencia deve ser a transcricao **exata** do audio. "
                        "Referencias **limpas de 3-10 s** funcionam melhor que as longas (o treino "
                        "nao viu mais de ~10 s)."
                    )

                with gr.Row():
                    best_of = gr.Checkbox(
                        value=True, scale=2,
                        label="Escolher o melhor de N candidatos",
                        info="Gera N audios (seed, seed+1, ...) e fica com o melhor: sem erro de "
                             "palavra (Whisper) e mais parecido com a referencia (ECAPA). "
                             "Desligado = 1 audio so. Leva ~N x mais tempo.",
                    )
                    n_cand = gr.Slider(2, 8, value=4, step=1, scale=2,
                                       label="N (candidatos por bloco)")

                btn = gr.Button("Gerar audio", variant="primary")

            # ---------------- coluna de saida
            with gr.Column(scale=2):
                out_audio = gr.Audio(label="Audio gerado", type="filepath", autoplay=False)
                out_info = gr.Markdown("Pronto. Preencha o texto e clique em **Gerar audio**.")
                gr.Markdown(
                    "---\n**Como usar**\n"
                    "1. (Opcional) Envie a referencia + transcricao para clonar.\n"
                    "2. Digite o texto e escolha a emocao.\n"
                    "3. Ajuste o adapter (a lista vem da pasta `adapters/` deste projeto e "
                    "dos checkpoints de treino) e, se quiser, o CFG em *Configuracoes "
                    "avancadas*.\n"
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
                 cfg_scale_, use_dual_, cfg_ref_, cfg_ins_, pause_on_, pause_ms_, adapter_scale_,
                 ref_format_, auto_chunk_, max_block_s_, n_cand_, best_of_):
            ref = ref_path_.strip() if (ref_path_ and ref_path_.strip()) else ref_audio_
            if isinstance(adapter_, (list, tuple)):
                adapter_ = adapter_[0] if adapter_ else BASE_LABEL
            try:
                _save_ui_state(adapter=adapter_)
            except Exception:  # noqa: BLE001
                pass
            common = dict(
                emotion=emotion_, instruction_custom=instruction_, ref_audio=ref,
                ref_text=ref_text_, adapter_label=adapter_, speaker=speaker_, seed=seed_,
                temperature=temperature_, top_k=top_k_, top_p=top_p_,
                max_new_tokens=max_new_, cfg_scale=cfg_scale_, use_dual_cfg=use_dual_,
                cfg_ref=cfg_ref_, cfg_ins=cfg_ins_, adapter_scale=adapter_scale_,
                ref_format=ref_format_, auto_chunk=bool(auto_chunk_),
                max_block_s=float(max_block_s_),
                n_candidates=int(n_cand_) if best_of_ else 1, best_of=bool(best_of_),
            )
            if modo_seg_:
                return synthesize_segments(script=script_, gap_ms=int(gap_), **common)
            return synthesize(text=text_, pause_on=pause_on_, pause_ms=int(pause_ms_), **common)

        btn.click(
            _run,
            inputs=[text, script, gap_ms, modo_seg, emotion, instruction, ref_audio, ref_path,
                    ref_text, adapter, speaker, seed, temperature, top_k, top_p, max_new,
                    cfg_scale, use_dual, cfg_ref, cfg_ins, pause_on, pause_ms, adapter_scale,
                    ref_format, auto_chunk, max_block_s, n_cand, best_of],
            outputs=[out_audio, out_info],
        )

        def _check_ref(audio_, path_):
            src = (path_ or "").strip() or audio_
            if not src or not Path(str(src)).is_file():
                return ""
            try:
                info = sf.info(str(src))
                dur = info.frames / float(info.samplerate)
            except Exception:  # noqa: BLE001
                return ""
            warn = RP.duration_warning(dur)
            base = f"Referencia: **{dur:.1f} s** ({info.samplerate} Hz, {info.channels} canal(is))."
            return base + (f"  \n:warning: {warn}" if warn else "  \nDuracao adequada.")

        ref_audio.change(_check_ref, inputs=[ref_audio, ref_path], outputs=[ref_info])
        ref_path.change(_check_ref, inputs=[ref_audio, ref_path], outputs=[ref_info])
        # enviar/gravar um audio novo limpa o caminho digitado (senao o caminho antigo ganhava)
        ref_audio.upload(lambda _a: "", inputs=[ref_audio], outputs=[ref_path])
        ref_audio.stop_recording(lambda _a: "", inputs=[ref_audio], outputs=[ref_path])

        def _refresh_init():
            """No load: define o valor padrao e refaz o estado do bloco de download."""
            opts = refresh_adapters()
            missing, msg = adapter_download_state()
            return (
                gr.update(choices=opts, value=_default_adapter_label(opts)),
                gr.update(value=msg),
                gr.update(visible=missing),
            )

        def _refresh_keep(current=None):
            """No botao Atualizar: mantem a escolha atual se ela ainda existir."""
            opts = refresh_adapters()
            val = current if current in opts else _default_adapter_label(opts)
            return gr.update(choices=opts, value=val)

        demo.load(_refresh_init, outputs=[adapter, dl_status, dl_btn])
        demo.load(lambda: gr.update(choices=sorted(_load_voices())), outputs=[voice_pick])
        refresh_btn.click(_refresh_keep, inputs=[adapter], outputs=[adapter])
        dl_btn.click(download_model, outputs=[dl_status, dl_btn, adapter])

        save_voice_btn.click(
            save_voice,
            inputs=[voice_name, ref_audio, ref_path, ref_text, adapter, emotion, instruction,
                    speaker, temperature, top_k, top_p, max_new, cfg_scale, use_dual, cfg_ref,
                    cfg_ins, seed, adapter_scale, ref_format],
            outputs=[voice_pick],
        )
        load_voice_btn.click(
            load_voice, inputs=[voice_pick],
            outputs=[ref_path, ref_text, adapter, emotion, instruction, speaker,
                     temperature, top_k, top_p, max_new, cfg_scale, use_dual, cfg_ref, cfg_ins,
                     seed, adapter_scale, ref_format, ref_audio],
        )
        del_voice_btn.click(delete_voice, inputs=[voice_pick], outputs=[voice_pick, voice_name])

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

    # Nada e baixado automaticamente: se o adapter LoRA ainda nao estiver em adapters/,
    # a propria UI mostra o botao (com a origem do download) ou voce copia o checkpoint
    # para a pasta. Assim o usuario decide se/ quando baixar ~570 MB.
    if CB.adapter_installed() is None:
        print(
            f"[ui] adapter LoRA '{CB.ADAPTER_REPO}' ainda nao esta em {CB.ADAPTERS_DIR}.\n"
            f"[ui] use o botao 'Baixar modelo de {CB.ADAPTER_REPO}' na UI, ou baixe de "
            f"{CB.ADAPTER_REPO_URL} e coloque os arquivos nessa pasta.",
            flush=True,
        )

    demo = build_ui()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Os audios sao salvos nos artefatos (fora do cwd); o Gradio exige liberar essas
    # pastas em allowed_paths, senao ele recusa mover o arquivo para o cache.
    allowed = sorted(
        {str(OUT_DIR), str(CB.TRAINING), *(str(d) for d in CB.adapter_search_dirs())}
    )
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
