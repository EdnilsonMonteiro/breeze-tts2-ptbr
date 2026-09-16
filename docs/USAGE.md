# Uso

## Interface web

```bash
python ui/app.py                 # http://127.0.0.1:7860
python ui/app.py --port 7861
python ui/app.py --share         # link público temporário (Gradio)
python ui/app.py --selftest      # gera 1 áudio e sai (checa a instalação)
```

Windows também: duplo-clique em `ui/run.bat`.

### Recursos

- **Clonagem** (com referência): envie/indique um áudio limpo (3–30 s) e a
  **transcrição exata**. O modelo gera o texto-alvo no timbre da referência.
- **Sem referência**: voz padrão do modelo + instrução de estilo.
- **Emoção/estilo**: presets ou instrução livre (*system prompt*), reforçados
  pelo **CFG**.
- **Modo segmentado**: uma emoção por frase (ver abaixo).
- **Ajuste de pausas**: redimensiona as pausas naturais (10 ms–3 s).
- **Adapter LoRA**: escolha entre base, adapters locais de treino e baixados do
  HF; botão *Atualizar* re-escaneia.

### Emoção por trecho (modo segmentado)

Ligue **Modo segmentado** e escreva uma linha por trecho no formato
`emocao | texto`:

```
Neutro / natural | Esta parte eu falo normal.
alegre | MAS ESSA PARTE EU FALO SUPER ALEGRE!
sad | e esta aqui bem triste.
whisper | e esta aqui bem baixinho.
Fale como um narrador epico | No fim, tudo mudou.
```

- A emoção pode ser um **preset** do dropdown, um **apelido** PT/EN
  (`triste`, `sad`, `alegre`, `happy`, `whisper`, `calmo`, ...) ou uma
  **instrução livre**. Sem `|`, usa a emoção global.
- **A emoção é dirigida pela instrução.** Para efeito forte, aumente o
  **CFG scale** (ex.: **3–4**) em *Configurações avançadas*. Com CFG 1.0 a
  diferença fica sutil.
- No modo clonagem, use o **dual-CFG**: `cfg_ref` (fidelidade à voz) e
  `cfg_ins` (aderência à emoção).
- As emoções tendem a funcionar melhor com o **modelo base** do que com um
  adaptador pt-BR treinado com instruções majoritariamente neutras.

## Linha de comando

```bash
# Clonar uma voz (template ref_edit)
python infer/clone_voice.py \
  --adapter EdnilsonMonteiro/Breeze-TTS-2-lora-ptbr \
  --ref-audio ref.wav --ref-text "transcrição exata da referência" \
  --text "Texto que o modelo deve falar." --out out/clone.wav

# Sonda base vs adapter em frases OOV
python infer/probe_oov.py --adapter <adapter|pasta> --out out/probe

# A/B do caminho de decode (eager vs streaming oficial)
python infer/ab_decode.py --adapter <adapter|pasta> --out out/ab
```

`--adapter` aceita **pasta local** ou **id do Hugging Face** (baixa na hora).
Sem `--adapter`, gera com o modelo base (controle).

## Dica de referência (clonagem)

Referência longa e limpa melhora muito a clonagem. A transcrição precisa ser
**exata** (o botão *Transcrever* usa `faster-whisper` para ajudar).
