# Configuração

O código é versionado no repo; os **artefatos** (modelo base, adapters, áudios
gerados) ficam **fora do git**. Os caminhos vêm de `core/paths.py`, dirigidos por
variáveis de ambiente ou por um `.env` na raiz.

## Configuração mínima

Copie `.env.example` para `.env` e edite:

```ini
PTBR_ARTIFACTS=C:\IA\Breeze-tts
```

Sem `.env`, usa `./artifacts` e baixa tudo automaticamente.

## Variáveis

| Variável | Default | Descrição |
|---|---|---|
| `PTBR_ARTIFACTS` | `./artifacts` | raiz de `models/`, `training/`, `adapters/` |
| `BREEZE_BASE_MODEL_REPO` | `BreezeBlue/Breeze-TTS-2` | modelo base no HF |
| `PTBR_ADAPTER_REPO` | `EdnilsonMonteiro/Breeze-TTS-2-lora-ptbr` | adapter LoRA no HF (vazio = não baixa) |
| `BREEZE_TTS_REPO` | `./breeze-tts` | engine (o submódulo) |
| `BREEZE_CKPT` | `<ARTIFACTS>/models/Breeze-TTS-2` | checkpoint base |
| `BREEZE_TRAINING_DIR` | `<ARTIFACTS>/training` | runs/adapters locais de treino |
| `PTBR_ADAPTERS_DIR` | `<training>/runs` | onde procurar adapters locais |
| `PTBR_HF_ADAPTERS_DIR` | `<ARTIFACTS>/adapters` | adapters baixados do HF |
| `PTBR_OUT_DIR` | `<training>/ui_out` | áudio gerado pela UI |
| `HF_TOKEN` | — | token HF (repos gated/licença) |

## Download automático (primeira execução)

Na 1ª vez, o programa baixa do Hugging Face:

1. **Modelo base** → `<PTBR_ARTIFACTS>/models/Breeze-TTS-2` (obrigatório).
2. **Adapter LoRA** → `<PTBR_ARTIFACTS>/adapters/<nome-do-repo>` (best-effort;
   se ainda não existir, segue com o base).

> **O adapter LoRA ainda não foi publicado** no Hugging Face. Enquanto isso, o
> download falha de forma silenciosa e você deve usar um adapter local
> (ver abaixo) ou o modelo base.

Se o repo for *gated*, aceite os termos e informe `HF_TOKEN`:
https://huggingface.co/settings/tokens

## Usar um adapter local (sem HF)

Coloque os arquivos em `<PTBR_ARTIFACTS>/adapters/<nome>/`:

```
<PTBR_ARTIFACTS>/adapters/Breeze-TTS-2-lora-ptbr/
├─ adapter_config.json
└─ adapter_model.safetensors
```

A UI mostra como `hf/Breeze-TTS-2-lora-ptbr` e pula o download (a pasta já
existe). Alternativas:

- `PTBR_ADAPTERS_DIR` apontando para uma pasta com
  `<run>/checkpoints/<ckpt>/adapter_config.json` (layout de treino); ou
- `PTBR_ADAPTER_REPO=` vazio para não tentar baixar nada.

## Árvore de artefatos

```
<PTBR_ARTIFACTS>/
├─ models/Breeze-TTS-2/              # base (download oficial)
├─ adapters/<nome>/                  # adapters baixados (ou colocados à mão)
├─ training/runs/<run>/checkpoints/  # adapters locais de treino
└─ training/ui_out/                  # áudios gerados pela UI
```
