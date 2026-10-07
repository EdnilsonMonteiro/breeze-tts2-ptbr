# Breeze TTS 2 — PT-BR LoRA (inferência e UI)

> **Official project (upstream):** [**breezeblue-ai/breeze-tts**](https://github.com/breezeblue-ai/breeze-tts)
> · **Model:** [BreezeBlue/Breeze-TTS-2](https://huggingface.co/BreezeBlue/Breeze-TTS-2)
> · **Training repo:** [EdnilsonMonteiro/breeze-tts2-ptbr-lora-training](https://github.com/EdnilsonMonteiro/breeze-tts2-ptbr-lora-training)

Repositório de **inferência e interface web** para usar um adaptador **LoRA pt-BR**
treinado sobre o [Breeze TTS 2](https://github.com/breezeblue-ai/breeze-tts).
O engine entra como **submódulo pinado** (`breeze-tts/`); o adaptador é aplicado
via PEFT sobre o modelo base.

Na **primeira execução**, o programa baixa automaticamente do Hugging Face:
1. o **modelo base** `BreezeBlue/Breeze-TTS-2`;
2. o **adapter LoRA** `EdnilsonMonts/Breeze-tts-2-brazillian-lora`.

> **Adaptador publicado:** [EdnilsonMonts/Breeze-tts-2-brazillian-lora](https://huggingface.co/EdnilsonMonts/Breeze-tts-2-brazillian-lora) (checkpoint *r76, passo 1500*; veja o model card para dados de treino, avaliação e limites).

> **Derived from Breeze TTS 2 by BreezeBlue and licensed for research and
> non-commercial use only.** Veja `NOTICE`; a licença do modelo está em
> https://huggingface.co/BreezeBlue/Breeze-TTS-2/blob/main/LICENSE.
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
PTBR_ARTIFACTS=/path/to/artifacts

# Repos do Hugging Face (troque quando publicar o seu adapter)
BREEZE_BASE_MODEL_REPO=BreezeBlue/Breeze-TTS-2
PTBR_ADAPTER_REPO=EdnilsonMonts/Breeze-tts-2-brazillian-lora

# Necessário apenas se o repo for gated
# HF_TOKEN=hf_xxx
```

| Variável | Default | Para que serve |
|---|---|---|
| `PTBR_ARTIFACTS` | `./artifacts` | raiz de `models/`, `training/`, `adapters/` |
| `BREEZE_BASE_MODEL_REPO` | `BreezeBlue/Breeze-TTS-2` | modelo base no HF |
| `PTBR_ADAPTER_REPO` | `EdnilsonMonts/Breeze-tts-2-brazillian-lora` | adapter LoRA no HF (vazio = não baixa) |
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
<PTBR_ARTIFACTS>/adapters/Breeze-tts-2-brazillian-lora/
├─ adapter_config.json
└─ adapter_model.safetensors
```

A UI mostra esse adapter como `hf/Breeze-tts-2-brazillian-lora` e o download é pulado
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
- **Clonagem** com áudio de referência (**limpo, 3–10 s**) + transcrição exata; a referência é
  preparada **como no treino** (24 kHz, corte de silêncio, pico normalizado) e a UI avisa se
  passar de ~10 s. Texto **e** transcrição passam pela mesma normalização (números, moeda, hora,
  siglas → por extenso);
- Geração **sem referência** (voz padrão + instrução) — os adapters v2 não foram treinados nesse
  modo; o modo é decidido por haver (ou não) áudio + transcrição, não pela aba aberta;
- **Texto longo**: dividido automaticamente em blocos de ≤ 10 s (o treino só viu clipes até ~10 s),
  com **N candidatos por bloco + medoid** opcional;
- **Emoção/estilo** por *system prompt* e **CFG** (simples ou dual);
- **Modo segmentado** (emoção por frase) e **ajuste de pausas**;
- Escolha do **adapter LoRA** (lista os locais e os baixados; botão *Atualizar*);
- **Escala do adapter (multiplicador)**: `1,0` = como treinado (recomendado; aplicada por módulo
  e idempotente). Valores `< 1,0` **pioram** o resultado — o adapter da receita v2 (escala de treino 4,0) **não**
  está "over-steering". Detalhes e números em
  [ESTRATEGIA-PTBR.md](https://github.com/EdnilsonMonteiro/breeze-tts2-ptbr-lora-training/blob/main/docs/ESTRATEGIA-PTBR.md).

Na primeira geração o modelo é carregado (baixa na 1ª vez e leva ~1 min depois).

### Linha de comando

```bash
# Clonar uma voz (template ref_edit)
python infer/clone_voice.py \
  --adapter EdnilsonMonts/Breeze-tts-2-brazillian-lora \
  --ref-audio ref.wav --ref-text "transcrição exata da referência" \
  --text "Texto que o modelo deve falar." --out out/clone.wav

# Sonda base vs adapter em frases OOV
python infer/probe_oov.py --adapter EdnilsonMonts/Breeze-tts-2-brazillian-lora --out out/probe

# A/B do caminho de decode (eager vs streaming oficial)
python infer/ab_decode.py --adapter EdnilsonMonts/Breeze-tts-2-brazillian-lora --out out/ab
```

`--adapter` aceita **pasta local** ou **id do Hugging Face** (baixa na hora).
Sem `--adapter`, gera com o modelo base (controle).

---

## Estrutura

```
breeze-tts/     engine oficial (submódulo pinado em ca632ce)
core/           paths.py + loaders (e auto-download do base/adapter);
                text_norm/text_blocks/adapter_scale/reference_prep = cópias dos módulos
                compartilhados com o repo de treino (`scripts/sync_shared.py`)
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
  (https://huggingface.co/BreezeBlue/Breeze-TTS-2/blob/main/LICENSE). **Sem uso comercial.**
- Não clone voz de pessoas sem consentimento explícito; não use para enganar,
  impersonar ou qualquer fim proibido. Sem vínculo oficial com a BreezeBlue.
