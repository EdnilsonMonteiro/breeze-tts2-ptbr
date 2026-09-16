# Documentação — PT-BR LoRA (inferência e UI)

Como instalar, configurar e usar o adaptador LoRA pt-BR sobre o Breeze TTS 2.

| Documento | Conteúdo |
|---|---|
| [CONFIGURATION.md](CONFIGURATION.md) | artefatos, download automático e variáveis de ambiente |
| [USAGE.md](USAGE.md) | interface web, modo segmentado, emoção/CFG e linha de comando |
| [TROUBLESHOOTING.md](TROUBLESHOOTING.md) | erros comuns e como resolver |

Instalação e visão geral: [`../README.md`](../README.md).

## Fluxo típico

```
[git clone --recursive] → [venv + pip install -r requirements.txt]
→ [python ui/app.py]  (baixa base+adapter na 1ª vez) → [gerar áudio]
```

> **Licença:** pesos e derivados do Breeze TTS 2 são *research/non-commercial*.
> Veja `../NOTICE` e `../breeze-tts/MODEL_LICENSE`.
