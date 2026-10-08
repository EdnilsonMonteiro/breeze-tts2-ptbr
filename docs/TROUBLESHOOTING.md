# Solução de problemas

## Gradio: `InvalidPathError`

O Gradio recusa servir arquivos fora do diretório de trabalho. A UI já libera as
pastas de artefatos via `allowed_paths` (`ui/app.py`). Se você mudar
`PTBR_OUT_DIR` para outro lugar, a pasta já é liberada automaticamente
(`allowed_paths` inclui `OUT_DIR`, `TRAINING` e todas as pastas de adapters). Se ainda
assim ocorrer, verifique se `PTBR_OUT_DIR` está sob `PTBR_ARTIFACTS`.

## Download do modelo base falha (401/403/404)

- Repo *gated*: aceite os termos em https://huggingface.co/BreezeBlue/Breeze-TTS-2
  e defina `HF_TOKEN` no `.env`.
- Verifique a internet e o `repo_id` (`BREEZE_BASE_MODEL_REPO`).

## O download do adapter (LoRA) falhou

O botão *Baixar modelo de EdnilsonMonts/Breeze-tts-2-brazillian-lora* mostra o erro na
própria UI. Se não houver internet/proxy:

- baixe `adapter_config.json` e o `*.safetensors` em
  https://huggingface.co/EdnilsonMonts/Breeze-tts-2-brazillian-lora/tree/main e coloque
  os dois arquivos em `adapters/Breeze-tts-2-brazillian-lora/` (na raiz do repo) — o
  botão some sozinho e o adapter passa a aparecer no dropdown;
- confira se `PTBR_ADAPTER_REPO` aponta para o repo certo (`EdnilsonMonts/Breeze-tts-2-brazillian-lora`);
- `PTBR_ADAPTER_REPO=` (vazio) desliga a oferta de download.

## "Can't find weights for ... adapter_model" (ou `ValueError` ao carregar o adapter)

O PEFT só procura os pesos como `adapter_model.safetensors` (ou `adapter_model.bin`).
Este repo publica `breeze-tts-2-pt-br-lora.safetensors`; a UI cria o nome esperado por
hardlink automaticamente. Se você colocou o adapter à mão e o erro apareceu, confirme
que a pasta tem `adapter_config.json` **e** algum `*.safetensors`, e clique em
*Atualizar* (ou renomeie o arquivo para `adapter_model.safetensors`).

## O adapter não aparece na lista

- Clique em **Atualizar** no dropdown.
- Confira a pasta `adapters/` na **raiz do repo**: deve haver
  `<nome>/adapter_config.json` e os pesos.
- Confira `PTBR_TRAINING_RUNS_DIR` (layout `<run>/checkpoints/<ckpt>/adapter_config.json`)
  e `PTBR_HF_ADAPTERS_DIR` (`<nome>/adapter_config.json`, adapters de versões antigas).
- O adapter está lá, mas o `git status` mostra a pasta? O conteúdo de `adapters/` é
  ignorado pelo git por padrão (`adapters/.gitignore`).

## A emoção/instrução não muda a voz

Medido e documentado em [EMOTION.md](EMOTION.md). Resumo do diagnóstico:

- **`cfg_scale` baixo** (o padrão é 1,0): sem CFG a instrução quase não dirige a voz — use
  **3–4** (no modo clonagem, o *dual-CFG* com `cfg_ins`).
- **Instrução fora do regime do modelo base**: ele é bilíngue **EN/ZH** e espera instruções
  **descritivas**, no **idioma do texto** (ex.: *"A warm, thoughtful young woman with a clear
  voice and a calm, reflective delivery."*), não comandos curtos em português.
- **Adapter pt-BR v2/v3** (inclui o publicado `EdnilsonMonts/Breeze-tts-2-brazillian-lora`):
  foi treinado com **uma instrução neutra fixa**, então **não segue emoção** — trocar
  "neutro" por "empolgado"/"triste" muda o áudio menos do que trocar a seed. Alternativas:
  pôr a emoção na **referência** clonada, usar o modelo base (EN/ZH), pós-processar prosódia,
  ou treinar um v4 com pool de instruções emocionais.
- Para **medir** em vez de confiar na impressão:
  `python scripts/measure_instruction.py --help`.

## CUDA / VRAM

- O caminho *eager* usa ~8 GB. Feche outros processos de GPU.
- Sem GPU, `python ui/app.py --selftest` roda em CPU (muito lento, só para
  validar o ambiente).

## A emoção (modo segmentado) tem pouco efeito

- Confirme que o **Modo segmentado** está ligado.
- Aumente o **CFG scale** para **3–4** (padrão 1.0 quase não dirige).
- Use instruções/presets reconhecidos (ex.: `triste`, `sad`, `alegre`). Palavras
  fora da lista são tratadas como instrução livre.
- Compare com o **modelo base** (sem adapter): um adapter pt-BR treinado com
  instruções neutras pode ter direção emocional mais fraca.

## Áudio sai metálico / com duração errada

- Confirme que o áudio de referência está em 24 kHz (a UI reamostra, mas
  referências muito ruidosas atrapalham).
- Reduza `temperature` ou ajuste `top_k/top_p` em *Configurações avançadas*.
