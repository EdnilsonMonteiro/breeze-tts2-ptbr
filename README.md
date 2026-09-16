# Breeze TTS 2 — PT-BR LoRA (inferência e UI)

Repositório de **inferência e interface web** para usar um adaptador **LoRA pt-BR**
treinado sobre o [Breeze TTS 2](https://github.com/breezeblue-ai/breeze-tts).
O engine entra como **submódulo pinado** (`breeze-tts/`); o adaptador é aplicado
via PEFT sobre o modelo base.

Na **primeira execução**, o programa baixa automaticamente do Hugging Face:
1. o **modelo base** `BreezeBlue/Breeze-TTS-2`;
2. o **adapter LoRA** `EdnilsonMonteiro/Breeze-TTS-2-lora-ptbr` (quando publicado).

> **Aviso — o adaptador LoRA ainda NÃO foi publicado.** Não há modelo treinado
> disponível neste repositório nem no Hugging Face. Este repo traz apenas o
> **código** de inferência/UI; o download automático do adapter já está pronto
> para quando ele for liberado (após treinos adicionais).

> **Derived from Breeze TTS 2 by BreezeBlue and licensed for research and
> non-commercial use only.** Veja `NOTICE` e `breeze-tts/MODEL_LICENSE`.
> Uso comercial exige licença separada da BreezeBlue.

---

## Requisitos

| Item | Mínimo |
|---|---|
| SO | Windows 10/11 ou Linux |
| Python | 3.10 – 3.12 |
| GPU | NVIDIA com CUDA (inferência eager ~8 GB VRAM; 12 GB recomendado) |
| Disco | ~8 GB (modelo base + adapters + ambiente) |
| Internet | para o primeiro download (HF) |

CPU funciona, mas é muito lento (não recomendado).

---

## Instalação

### Windows (PowerShell)

```powershell
git clone --recursive https://github.com/EdnilsonMonteiro/breeze-tts2-ptbr.git
cd breeze-tts2-ptbr
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

### Linux

```bash
git clone --recursive https://github.com/EdnilsonMonteiro/breeze-tts2-ptbr.git
cd breeze-tts2-ptbr
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

> Clonou sem `--recursive`? Rode:
> `git submodule update --init --recursive`

`requirements.txt` já puxa tudo do engine (torch, qwen-tts, transformers) e
adiciona `peft`, `gradio` e `faster-whisper`.

---

## Configuração (opcional)

Tudo funciona sem configurar nada: os artefatos vão para `./artifacts/` e o
download é automático. Para escolher pastas/repos, copie `.env.example` → `.env`
e edite:

```ini
# Onde ficam modelo base + adapters (fora do git)
PTBR_ARTIFACTS=C:\IA\Breeze-tts

# Repos do Hugging Face (troque quando publicar o seu adapter)
BREEZE_BASE_MODEL_REPO=BreezeBlue/Breeze-TTS-2
PTBR_ADAPTER_REPO=EdnilsonMonteiro/Breeze-TTS-2-lora-ptbr

# Necessário apenas se o repo for gated
# HF_TOKEN=hf_xxx
```

| Variável | Default | Para que serve |
|---|---|---|
| `PTBR_ARTIFACTS` | `./artifacts` | raiz de `models/`, `training/`, `adapters/` |
| `BREEZE_BASE_MODEL_REPO` | `BreezeBlue/Breeze-TTS-2` | modelo base no HF |
| `PTBR_ADAPTER_REPO` | `EdnilsonMonteiro/Breeze-TTS-2-lora-ptbr` | adapter LoRA no HF (vazio = não baixa) |
| `PTBR_ADAPTERS_DIR` | `<training>/runs` | adapters locais de treino |
| `PTBR_HF_ADAPTERS_DIR` | `<artifacts>/adapters` | adapters baixados do HF |
| `PTBR_OUT_DIR` | `<training>/ui_out` | áudio gerado pela UI |
| `HF_TOKEN` | — | token HF (repos gated/licença) |

> O modelo base tem licença **research/non-commercial**. Se o HF pedir, aceite os
> termos em https://huggingface.co/BreezeBlue/Breeze-TTS-2 e informe `HF_TOKEN`.

---

### Usar um adapter local (sem Hugging Face)

Se você já tem o adapter (ex.: seu checkpoint de treino), coloque em
`<PTBR_ARTIFACTS>/adapters/<nome>/`:

```
<PTBR_ARTIFACTS>/adapters/Breeze-TTS-2-lora-ptbr/
├─ adapter_config.json
└─ adapter_model.safetensors
```

A UI mostra esse adapter como `hf/Breeze-TTS-2-lora-ptbr` e o download é pulado
(a pasta já existe). Alternativas: apontar `PTBR_ADAPTERS_DIR` para uma pasta com
`<run>/checkpoints/<ckpt>/adapter_config.json`, ou deixar `PTBR_ADAPTER_REPO=`
vazio para não tentar baixar nada.

## Uso

### Interface web

```bash
python ui/app.py                 # abre em http://127.0.0.1:7860
python ui/app.py --port 7861
python ui/app.py --share         # link público temporário do Gradio
python ui/app.py --selftest      # gera 1 áudio e sai (checa a instalação)
```

Windows também: duplo-clique em `ui/run.bat`.

Recursos da UI:
- **Clonagem** com áudio de referência + transcrição exata;
- Geração **sem referência** (voz padrão + instrução);
- **Emoção/estilo** por *system prompt* e **CFG** (simples ou dual);
- **Modo segmentado** (emoção por frase) e **ajuste de pausas**;
- Escolha do **adapter LoRA** (lista os locais e os baixados; botão *Atualizar*).

Na primeira geração o modelo é carregado (baixa na 1ª vez e leva ~1 min depois).

### Linha de comando

```bash
# Clonar uma voz (template ref_edit)
python infer/clone_voice.py \
  --adapter EdnilsonMonteiro/Breeze-TTS-2-lora-ptbr \
  --ref-audio ref.wav --ref-text "transcrição exata da referência" \
  --text "Texto que o modelo deve falar." --out out/clone.wav

# Sonda base vs adapter em frases OOV
python infer/probe_oov.py --adapter EdnilsonMonteiro/Breeze-TTS-2-lora-ptbr --out out/probe

# A/B do caminho de decode (eager vs streaming oficial)
python infer/ab_decode.py --adapter EdnilsonMonteiro/Breeze-TTS-2-lora-ptbr --out out/ab
```

`--adapter` aceita **pasta local** ou **id do Hugging Face** (baixa na hora).
Sem `--adapter`, gera com o modelo base (controle).

---

## Estrutura

```
breeze-tts/     engine oficial (submódulo pinado em ca632ce)
core/           paths.py + loaders (e auto-download do base/adapter)
infer/          clone_voice.py, probe_oov.py, ab_decode.py
ui/             app.py (Gradio), run.bat
docs/           documentacao publica (ver docs/README.md)
```

---

## Solução de problemas

- **`InvalidPathError` do Gradio**: a UI já libera as pastas de artefatos
  (`allowed_paths`). Se gerar áudio em outro lugar, adicione a pasta lá.
- **Download do base falha (401/403)**: aceite os termos no HF e defina
  `HF_TOKEN`.
- **CUDA/VRAM**: feche outros processos; o caminho *eager* usa ~8 GB. Sem GPU,
  use `--selftest` para validar o ambiente (vai rodar em CPU, lento).
- **Adapter não aparece na lista**: clique em *Atualizar*; confira
  `PTBR_ADAPTER_REPO` e `PTBR_HF_ADAPTERS_DIR`; ou aponte `PTBR_ADAPTERS_DIR` para
  uma pasta com `<run>/checkpoints/<ckpt>/adapter_config.json`.

---

## Licença

- Código deste repo: Apache-2.0 (`LICENSE`).
- Pesos e derivados do Breeze TTS 2: **BreezeBlue Research and Non-Commercial**
  (`breeze-tts/MODEL_LICENSE`). **Sem uso comercial.**
- Não clone voz de pessoas sem consentimento explícito; não use para enganar,
  impersonar ou qualquer fim proibido. Sem vínculo oficial com a BreezeBlue.
