# Solução de problemas

## Gradio: `InvalidPathError`

O Gradio recusa servir arquivos fora do diretório de trabalho. A UI já libera as
pastas de artefatos via `allowed_paths` (`ui/app.py`). Se você mudar
`PTBR_OUT_DIR` para outro lugar, a pasta já é liberada automaticamente
(`allowed_paths` inclui `OUT_DIR`, `TRAINING` e `ADAPTERS_DIR`). Se ainda assim
ocorrer, verifique se `PTBR_OUT_DIR` está sob `PTBR_ARTIFACTS`.

## Download do modelo base falha (401/403/404)

- Repo *gated*: aceite os termos em https://huggingface.co/BreezeBlue/Breeze-TTS-2
  e defina `HF_TOKEN` no `.env`.
- Verifique a internet e o `repo_id` (`BREEZE_BASE_MODEL_REPO`).

## Aviso: adapter "ainda nao esta publicado no Hugging Face"

Esperado enquanto `PTBR_ADAPTER_REPO` aponta para um repo que não existe. Opções:

- use um adapter local (ver [CONFIGURATION.md](CONFIGURATION.md));
- deixe `PTBR_ADAPTER_REPO=` vazio para não tentar baixar.

## O adapter não aparece na lista

- Clique em **Atualizar** no dropdown.
- Confira `PTBR_ADAPTERS_DIR` (layout `<run>/checkpoints/<ckpt>/adapter_config.json`)
  e `PTBR_HF_ADAPTERS_DIR` (`<nome>/adapter_config.json`).

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
