# Uso

## Interface web

```bash
python ui/app.py                 # http://127.0.0.1:7860
python ui/app.py --port 7861
python ui/app.py --share         # link público temporário (Gradio)
python ui/app.py --selftest      # gera 1 áudio e sai (checa a instalação)
```

Atalhos com a venv do repo: `ui\run.bat` (Windows, duplo-clique) e `./ui/run.sh`
(Linux/macOS — `chmod +x ui/run.sh` uma vez, ou `bash ui/run.sh`).

### Recursos

- **Clonagem** (com referência): envie/indique um áudio limpo (**3–10 s** — o treino não viu clipes maiores) e a
  **transcrição exata**. O modelo gera o texto-alvo no timbre da referência.
- **Sem referência**: voz padrão do modelo + instrução de estilo.
- **Emoção/estilo**: presets ou instrução livre (*system prompt*), reforçados
  pelo **CFG** — veja o aviso abaixo (**os adapters pt-BR v2/v3 não seguem emoção**).
- **Modo segmentado**: uma emoção por frase (ver abaixo).
- **Ajuste de pausas**: redimensiona as pausas naturais (10 ms–3 s).
- **Adapter LoRA**: a lista vem da pasta `adapters/` na raiz do repo (conteúdo
  ignorado pelo git), dos checkpoints de treino (`<run>/checkpoints/<ckpt>`) e de
  downloads antigos (`hf/<nome>`); botão *Atualizar* re-escaneia.
- **Download do modelo LoRA**: se `adapters/` ainda não tiver o adapter, aparece um
  bloco dizendo **de onde** o download vem
  ([EdnilsonMonts/Breeze-tts-2-brazillian-lora](https://huggingface.co/EdnilsonMonts/Breeze-tts-2-brazillian-lora/tree/main))
  com o botão *Baixar modelo…*. Depois de baixado, **a opção desaparece** e o adapter
  entra no dropdown já selecionado.

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
- **Limite medido (importante):** o **modelo base** segue instrução de verdade
  (instrução **descritiva**, no **idioma do texto**, com **CFG ~4**), mas os
  **adapters pt-BR v2/v3** foram treinados com uma instrução **fixa** e
  **não seguem emoção** — trocar "neutro" por "empolgado"/"triste" muda o áudio
  menos do que trocar a seed. Com eles, a emoção tem de vir da **referência**
  (referência alegre → clone alegre). Números, gráficos de medição e como
  remedir: [EMOTION.md](EMOTION.md).

### Medir se a emoção funciona (em vez de confiar na impressão)

```bash
python scripts/measure_instruction.py \
  --text "Good morning! What a wonderful day, isn't it?" \
  --instructions "Speak clearly and naturally.|Speak with great excitement and high energy." \
  --cfg 4 --seeds 42,43
```

Ele gera sempre o mesmo texto/seed mudando **só a instrução**, mede F0/duração/
ritmo/energia e compara com o **piso de ruído** (mesma instrução, seed diferente),
imprimindo um veredito (`INSTRUCAO FUNCIONA` / `EFEITO FRACO/DUVIDOSO` /
`INSTRUCAO INERTE`). Use `--adapter` e `--ref-audio/--ref-text` para testar um
LoRA no modo clonagem.

## Linha de comando

```bash
# Clonar uma voz (template ref_edit)
python infer/clone_voice.py \
  --adapter EdnilsonMonts/Breeze-tts-2-brazillian-lora \
  --ref-audio ref.wav --ref-text "transcrição exata da referência" \
  --text "Texto que o modelo deve falar." --out out/clone.wav

# Sonda base vs adapter em frases OOV
python infer/probe_oov.py --adapter <adapter|pasta> --out out/probe

# A/B do caminho de decode (eager vs streaming oficial)
python infer/ab_decode.py --adapter <adapter|pasta> --out out/ab
```

`--adapter` aceita **pasta local** ou **id do Hugging Face** (baixa na hora).
Sem `--adapter`, gera com o modelo base (controle).

## Texto longo, blocos e candidatos

A UI divide o texto em blocos de ≤ 10 s (*Dividir texto longo*, ligado por padrão) e gera cada
bloco com a mesma referência; *Candidatos por bloco* ≥ 3 escolhe a variação mais "central"
(medoid ECAPA) entre as de duração plausível. `max_new_tokens` vale por bloco (12,5 tokens = 1 s).

## Formato da referência

*Como no treino* (padrão): mono, 24 kHz, corte de silêncio nas pontas (top_db 40) e pico normalizado
para 0,95 — o formato que o adapter viu. *48 kHz* reproduz o comportamento antigo da UI; *Original*
usa o arquivo como está. O arquivo original nunca é alterado (cache em `voice_cache/`).

## Vozes salvas

*Salvar voz* copia a referência (upload/microfone **ou** caminho digitado) para
`<artifacts>/voice_refs/` e grava também escala do adapter e formato da referência.

## Dica de referência (clonagem)

Referência longa e limpa melhora muito a clonagem. A transcrição precisa ser
**exata** (o botão *Transcrever* usa `faster-whisper` para ajudar).
