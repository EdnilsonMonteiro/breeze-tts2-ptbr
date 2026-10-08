# Configuração

O código é versionado no repo; os **artefatos** (modelo base, adapters, áudios
gerados) ficam **fora do git**. Os caminhos vêm de `core/paths.py`, dirigidos por
variáveis de ambiente ou por um `.env` na raiz.

> Exceção: os **adapters LoRA** ficam em `adapters/`, uma pasta **na raiz do repo**
> cujo conteúdo é ignorado pelo git (`adapters/.gitignore`) — assim a UI encontra o
> checkpoint sozinha, sem que os pesos entrem no repositório.

## Configuração mínima

Copie `.env.example` para `.env` e edite:

```ini
PTBR_ARTIFACTS=/path/to/artifacts
```

Sem `.env`, usa `./artifacts` e baixa o modelo base automaticamente.

## Variáveis

| Variável | Default | Descrição |
|---|---|---|
| `PTBR_ARTIFACTS` | `./artifacts` | raiz de `models/`, `training/`, `ui_out/` |
| `BREEZE_BASE_MODEL_REPO` | `BreezeBlue/Breeze-TTS-2` | modelo base no HF |
| `PTBR_ADAPTER_REPO` | `EdnilsonMonts/Breeze-tts-2-brazillian-lora` | adapter LoRA no HF (vazio = não oferece download) |
| `BREEZE_TTS_REPO` | `./breeze-tts` | engine (o submódulo) |
| `BREEZE_CKPT` | `<ARTIFACTS>/models/Breeze-TTS-2` | checkpoint base |
| `BREEZE_TRAINING_DIR` | `<ARTIFACTS>/training` | runs/áudios do treino |
| `PTBR_REPO_ADAPTERS_DIR` | `<repo>/adapters` | pasta (git-ignorada) onde a UI procura/baixa adapters |
| `PTBR_ADAPTERS_DIR` | `<PTBR_REPO_ADAPTERS_DIR>` | pasta principal de adapters |
| `PTBR_TRAINING_RUNS_DIR` | `<training>/runs` | checkpoints de treino (`<run>/checkpoints/<ckpt>`) |
| `PTBR_HF_ADAPTERS_DIR` | `<ARTIFACTS>/adapters` | adapters baixados por versões antigas (legado) |
| `PTBR_OUT_DIR` | `<training>/ui_out` | áudio gerado pela UI |
| `HF_TOKEN` | — | token HF (repos gated/licença) |

## Download (primeira execução)

Na 1ª vez, o programa baixa do Hugging Face o **modelo base** →
`<PTBR_ARTIFACTS>/models/Breeze-TTS-2` (obrigatório).

O **adapter LoRA** não é mais baixado sozinho: a UI mostra um bloco
*Modelo LoRA (pasta adapters/ do projeto)* que diz **de onde** o download vem e oferece
o botão *Baixar modelo de EdnilsonMonts/Breeze-tts-2-brazillian-lora* — ele baixa para
`adapters/Breeze-tts-2-brazillian-lora/` (dentro do repo, fora do git). Depois de
baixado, **a opção desaparece** e o adapter aparece no dropdown.

> O adapter LoRA padrão (r76, passo 1500) está em
> [`EdnilsonMonts/Breeze-tts-2-brazillian-lora`](https://huggingface.co/EdnilsonMonts/Breeze-tts-2-brazillian-lora/tree/main).

Se o repo for *gated*, aceite os termos e informe `HF_TOKEN`:
https://huggingface.co/settings/tokens

## Usar um adapter local (sem HF)

Coloque os arquivos em `adapters/<nome>/` (na raiz do repo):

```
adapters/Breeze-tts-2-brazillian-lora/
├─ adapter_config.json
└─ adapter_model.safetensors        # se o repo publicar com outro nome, a UI cria
                                    # este por hardlink (o PEFT exige esse nome)
```

A UI mostra como `Breeze-tts-2-brazillian-lora` e o botão de download some (a pasta já
existe). Alternativas:

- `PTBR_ADAPTERS_DIR`/`PTBR_REPO_ADAPTERS_DIR` apontando para outra pasta (o primeiro
  layout aceito é `<nome>/adapter_config.json`);
- `PTBR_TRAINING_RUNS_DIR` para uma pasta com
  `<run>/checkpoints/<ckpt>/adapter_config.json` (layout de treino);
- `PTBR_ADAPTER_REPO=` vazio para não oferecer download nenhum.

## Árvore de artefatos

```
adapters/                            # NA RAIZ DO REPO (conteúdo ignorado pelo git)
├─ Breeze-tts-2-brazillian-lora/     # baixado pela UI (layout plano)
└─ r76/checkpoints/step1500/         # checkpoints de treino copiados para cá

<PTBR_ARTIFACTS>/
├─ models/Breeze-TTS-2/              # base (download oficial)
├─ adapters/<nome>/                  # adapters baixados por versões antigas (legado)
├─ training/runs/<run>/checkpoints/  # runs do treino
└─ training/ui_out/                  # áudios gerados pela UI
```
