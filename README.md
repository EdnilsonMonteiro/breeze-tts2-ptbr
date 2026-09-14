# Breeze TTS 2 — PT-BR LoRA (inferência e UI)

Repositório de **inferência e interface web** para usar o adaptador **LoRA
pt-BR** treinado sobre o [Breeze TTS 2](https://github.com/breezeblue-ai/breeze-tts).
O engine entra como **submódulo pinado** (`breeze-tts/`); o adaptador é carregado
via PEFT por cima do modelo base.

> Derived from Breeze TTS 2 by BreezeBlue and licensed for research and
> non-commercial use only. See `NOTICE` and `breeze-tts/MODEL_LICENSE`.

O treino do adaptador vive no repo separado (`breeze-tts2-ptbr-lora-training`).

## Instalação

```bash
git clone --recursive <este-repo>
cd breeze-tts2-ptbr
python -m venv venv
./venv/Scripts/activate            # Windows
pip install -r requirements.txt
```

Se já clonou sem `--recursive`:

```bash
git submodule update --init --recursive
```

## Artefatos (fora do git)

Copie `.env.example` para `.env` e aponte os caminhos:

```
PTBR_ARTIFACTS=C:\IA\Breeze-tts
```

- **Modelo base**: baixe `BreezeBlue/Breeze-TTS-2` (Hugging Face) para
  `<PTBR_ARTIFACTS>/models/Breeze-TTS-2`.
- **Adapter LoRA**: coloque os checkpoints em
  `<PTBR_ARTIFACTS>/training/runs/<run>/checkpoints/<ckpt>` (a UI lista sozinha),
  ou defina `PTBR_ADAPTERS_DIR`. Também é possível baixar o adapter publicado no
  Hugging Face e apontar `PTBR_ADAPTERS_DIR` para a pasta baixada.

## Interface web

```bash
python ui/app.py                 # http://127.0.0.1:7860
python ui/app.py --port 7861
python ui/app.py --selftest      # gera 1 amostra e sai
```

Ou duplo-clique em `ui/run.bat` (usa `BREEZE_PY` ou `.venv`).

Recursos: clonagem com referência, modo sem referência, controle de emoção por
instrução, CFG (simples/dual), modo segmentado (emoção por frase), ajuste de
pausas e escolha do adapter.

## CLI

```bash
# clonar uma voz (template ref_edit)
python infer/clone_voice.py --adapter <ckpt> \
  --ref-audio ref.wav --ref-text "transcricao exata" \
  --text "Texto a falar." --out out/clone.wav

# sonda base vs adapter em frases OOV
python infer/probe_oov.py --adapter <ckpt> --out out/probe

# A/B do caminho de decode (eager vs streaming oficial)
python infer/ab_decode.py --adapter <ckpt> --out out/ab
```

Documentação operacional (copiada do repo de treino): `docs/tutorial/`.

## Licença

- Código deste repo: Apache-2.0 (`LICENSE`).
- Pesos e derivados do Breeze TTS 2: BreezeBlue Research and Non-Commercial
  (`breeze-tts/MODEL_LICENSE`). Uso comercial exige licença separada.
