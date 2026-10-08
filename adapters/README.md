# adapters/ — adapters LoRA na raiz do repo

Esta pasta é a **raiz do repositório** e é onde a UI (`ui/app.py`) procura os
adapters LoRA. Os arquivos aqui dentro são **ignorados pelo git** (ver
`.gitignore` desta pasta): nenhum peso/checkpoint é comitado.

## Layout esperado

Dois layouts são aceitos — basta existir um `adapter_config.json` dentro:

```
adapters/
├─ Breeze-tts-2-brazillian-lora/        # 1) plano (baixado do Hugging Face)
│  ├─ adapter_config.json
│  ├─ adapter_model.safetensors          # ou o nome publicado + adapter_model.*
│  └─ breeze-tts-2-pt-br-lora.safetensors
└─ r76/
   └─ checkpoints/
      └─ step1500/                       # 2) layout de treino
         ├─ adapter_config.json
         └─ adapter_model.safetensors
```

O dropdown *Adapter LoRA (checkpoint)* mostra o primeiro como
`Breeze-tts-2-brazillian-lora` e o segundo como `r76/checkpoints/step1500`.

## Como colocar um adapter aqui

1. **Pela UI** (recomendado): clique em *Baixar modelo deste repositório do Hugging Face*.
   O download vem de
   [`EdnilsonMonts/Breeze-tts-2-brazillian-lora`](https://huggingface.co/EdnilsonMonts/Breeze-tts-2-brazillian-lora/tree/main)
   e é salvo em `adapters/Breeze-tts-2-brazillian-lora/`.
   Depois de baixado, a opção some da UI (a pasta já existe).
2. **Manual**: baixe `adapter_config.json` + o `*.safetensors` na página do repositório
   no Hugging Face e coloque-os em `adapters/<nome>/`. Também vale copiar um checkpoint
   de treino seu para cá.
3. **Pelo terminal**:

   ```bash
   # Linux/macOS (usa o `hf` CLI ou o huggingface_hub do venv)
   ./venv/bin/python -c "import sys; sys.path.insert(0,'core'); import common_breeze as CB; print(CB.download_adapter())"
   ```

## Nome dos pesos

O PEFT só carrega `adapter_model.safetensors` (ou `adapter_model.bin`). Se o repo
publica com outro nome (é o caso deste adapter: `breeze-tts-2-pt-br-lora.safetensors`),
a UI cria `adapter_model.safetensors` automaticamente por **hardlink** — sem duplicar
o arquivo em disco.

> Nada aqui é versionado, exceto este README e o `.gitignore`. Não force
> `git add -f` em pesos.
