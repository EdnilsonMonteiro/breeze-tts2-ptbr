# Documentação — PT-BR LoRA (inferência e UI)

Como instalar, configurar e usar o adaptador LoRA pt-BR sobre o Breeze TTS 2.

| Documento | Conteúdo |
|---|---|
| [CONFIGURATION.md](CONFIGURATION.md) | artefatos, pasta `adapters/`, download e variáveis de ambiente |
| [USAGE.md](USAGE.md) | interface web, modo segmentado, emoção/CFG e linha de comando |
| [EMOTION.md](EMOTION.md) | **emoção/instrução: o que funciona, o que não e por quê** (com medições) |
| [TROUBLESHOOTING.md](TROUBLESHOOTING.md) | erros comuns e como resolver |

Instalação e visão geral: [`../README.md`](../README.md).

## Fluxo típico

```
[git clone --recursive] → [venv + pip install -r requirements.txt]
→ [python ui/app.py]  (baixa o base na 1ª vez; o adapter vem do botão em adapters/)
→ [gerar áudio]
```

> **Licença:** pesos e derivados do Breeze TTS 2 são *research/non-commercial*
> (https://huggingface.co/BreezeBlue/Breeze-TTS-2/blob/main/LICENSE).
> Veja `../NOTICE`.
