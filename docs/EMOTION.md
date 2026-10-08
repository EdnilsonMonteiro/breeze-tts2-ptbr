# Emoção / instrução: o que funciona, o que não funciona e por quê

> **Resposta curta:** a UI está correta — a instrução é enviada ao modelo no lugar certo
> (dentro de `<ins_bos>…</ins_eos>`, com `[S0]`, CFG negativo sem instrução). O limite é do
> **modelo/adaptr**: o adaptador pt-BR (v2/v3, incluindo o publicado) foi treinado com uma
> instrução **fixa e neutra**, então **não segue emoção**; o modelo **base** segue bem, mas
> só no regime em que foi treinado (instrução **descritiva**, no **idioma do texto**, e
> `cfg_scale ≈ 4`).

Este documento registra a investigação, os números medidos e como remedir depois de
qualquer retreino.

---

## 1. O caminho da instrução (por que não é bug de implementação)

| Etapa | Onde | O que acontece |
|---|---|---|
| 1 | `ui/app.py` — `synthesize()` | `instruction = instruction_custom.strip() or EMOTIONS[preset]`: o texto **editado à mão tem precedência** sobre o preset |
| 2 | `ui/app.py` — `_generate()` | `request = {"text", "instruction", "speaker", "ref_audio_path", "ref_text"}` |
| 3 | `breeze_infer/templates.py` | monta `[S0]<ins_bos>{instrução}<ins_eos>{texto}` (`ref_edit_tata` com referência; `tts_instruction` sem) |
| 4 | tokenizer do modelo | `<ins_bos>` = id **262156**, `<ins_eos>` = **262157**, `[S0]` = **262146** — são *special tokens* de verdade, não texto qualquer |
| 5 | `breeze_infer/templates.py` | com `cfg_scale > 1` o **negativo** é o mesmo texto **sem** a instrução → o CFG amplifica exatamente a diferença que a instrução causa |
| 6 | `models/generation_breeze.py` | aplica o CFG (e o dual-CFG: `uncond` / `ref` / `ins`) |

Ou seja: template, tokens, CFG e passagem de parâmetros são o **caminho oficial do engine**
(o mesmo de `breeze-tts/infer.py` e da API streaming). Nada na UI descarta ou reescreve a
instrução.

Os dois únicos "atritos" reais na UI eram de **default/expectativa**, não de código:

1. **`cfg_scale` começa em 1,0** — com CFG 1 não existe ramo negativo e a instrução quase não
   dirige nada. A receita oficial do modelo base usa `--cfg-scale 4`.
2. Os **presets são frases curtas em português** ("Fale com alegria…"), enquanto o modelo base
   é **bilíngue EN/ZH** e espera descrições longas **no idioma do texto** ("A warm, thoughtful
   young woman…").

---

## 2. Medições (mesmo texto, mesma seed, mesma referência: só a instrução muda)

Experimento controlado, eager, `temperature=0.7`, `top_k=50`, 24 kHz. O **piso de ruído** é a
variação obtida trocando só a *seed* (mesma instrução) — se o efeito da instrução não passar
disso, ela não está dirigindo a voz.

### 2.1 Modelo BASE, sem referência (Voice Design), texto EN, `cfg 4`, seed 42

| instrução | duração | F0 mediana | RMS | ritmo |
|---|---:|---:|---:|---:|
| `Speak clearly and naturally.` | 5,04 s | 225 Hz | 0,046 | 2,78 pal/s |
| `Speak with great excitement, high energy…` | **3,28 s** | **353 Hz** | **0,108** | **4,27 pal/s** |
| `Speak in a soft, quiet whisper…` | 8,80 s | 179 Hz | 0,022 | 1,59 pal/s |
| empolgado, **mas com `cfg 1`** | 5,52 s | 262 Hz | 0,069 | 2,54 pal/s |

Efeito médio da **instrução** vs **ruído de seed** (todos os pares desta bateria):

| métrica | instrução | ruído (seed) | razão |
|---|---:|---:|---:|
| ΔF0 | 95,2 Hz | 35,8 Hz | **2,7×** |
| Δduração | 2,08 s | 0,48 s | **4,3×** |
| Δritmo | 1,04 pal/s | 0,41 pal/s | **2,5×** |
| ΔMFCC (timbre) | 26,1 | 16,6 | **1,6×** |

O par mais limpo: *empolgado* trocado entre seeds move a F0 em **2 Hz** (a instrução "trava" o
estilo), enquanto trocar a instrução move **128 Hz**.

### 2.2 Modelo BASE **com referência** (Voice Direction — o caminho da UI na clonagem), `cfg 4`

| instrução | duração | F0 mediana | RMS |
|---|---:|---:|---:|
| neutro | 4,00 s | 168 Hz | 0,021 |
| empolgado | 3,44 s | 223 Hz | 0,049 |
| **sussurro** | 6,80 s | **sem F0 detectável (não vozeado)** | **0,002** |
| empolgado com `cfg 1` | 4,88 s | 222 Hz | 0,026 |

Ruído de seed (mesma instrução): ΔF0 **6 Hz**. Efeito da instrução: ΔF0 **55 Hz** (8,7×),
+136 % de energia, −14 % de duração; o sussurro sai 10× mais baixo e sem vozeamento. **A
implementação entrega emoção quando o modelo sabe seguir instrução.**

### 2.3 Adaptador pt-BR (`EdnilsonMonts/Breeze-tts-2-brazillian-lora`) + referência, seed 42

| instrução | cfg | duração | F0 | RMS | ritmo |
|---|---:|---:|---:|---:|---:|
| `Fale com clareza e naturalidade.` | 1 | 5,28 s | 183 Hz | 0,051 | 2,46 |
| `Fale com clareza e naturalidade.` | 4 | 6,56 s | 174 Hz | 0,048 | 1,98 |
| `Fale com muita empolgacao, energia…` | 1 | 6,16 s | 207 Hz | 0,079 | 2,11 |
| `Fale com muita empolgacao, energia…` | 4 | 5,76 s | 189 Hz | 0,092 | 2,26 |
| `Fale com tristeza e melancolia…` | 4 | 8,72 s | 150 Hz | 0,029 | 1,49 |
| neutro (seed 43) | 1 | 8,88 s | 178 Hz | 0,019 | 1,46 |
| empolgado (seed 43) | 1 | 5,76 s | 170 Hz | 0,037 | 2,26 |

| métrica | instrução | ruído (seed) | razão |
|---|---:|---:|---:|
| ΔF0 | 19,5 Hz | 21,0 Hz | **0,93×** |
| Δduração | 1,48 s | 2,00 s | **0,74×** |
| Δritmo | 0,38 pal/s | 0,57 pal/s | **0,67×** |
| ΔMFCC | 18,5 | 15,9 | 1,16× |
| ΔECAPA (locutor) | 0,32 | 0,35 | 0,91× |

**Trocar "neutro" por "empolgado" ou "triste" mexe no áudio menos do que trocar só a seed.**
A instrução é inerte nesse adaptador (a variabilidade de amostragem domina).

---

## 3. Por que o adaptador não segue instrução (está documentado no próprio treino)

- `docs/TRAINING.md` (repo de treino): *"Instrução **fixa** (`Fale com clareza e naturalidade.`,
  igual à produção)"* e `--instruction-mode` default **`fixed`** (`pool` = "9 frases variadas
  (não recomendado)").
- `ptbr_lora/core/common_breeze.py`: `INSTRUCTION_POOL` tem **9 frases, todas neutras**
  ("Leia com dicção clara…", "Narre com voz neutra…") e o comentário explica que o pool v2
  "virava ruído" na inferência.
- `docs/AUDITORIA-2026-09.md`, achado 4: *"Instrução: pool de 9 frases, mas a inferência usa
  1"* → correção adotada na v3: instrução **fixa**.
- Model card do adaptador publicado: *"expressive style, **emotion** and very long texts are
  **not evaluated**"* — a avaliação mede SECS/WER, nunca prosódia.

Consequência: no treino o bloco `<ins_bos>…</ins_eos>` foi sempre o **mesmo token pattern**
(quase um token especial de "comece a falar"). Texto novo ali é fora de distribuição e o
adaptador não tem como mapeá-lo para prosódia — nenhum valor de CFG conserta isso, porque o CFG
só amplifica o que o modelo já distingue.

E o modelo base? `breeze-tts/README.md`: *"Bilingual Support — English and Chinese"*, *"Match the
instruction language to the target text. Use `--cfg-scale 4` to strengthen instruction-following."*
Ou seja, a UI mandando **preset em português, com CFG 1**, para um modelo EN/ZH, é duplamente
fora do regime — daí a impressão de que "não funciona nem no modelo em inglês".

---

## 4. O que fazer (do mais barato ao mais definitivo)

| Opção | Como | Custo | Limite |
|---|---|---|---|
| **A. Emoção na referência** | clone com uma referência que **já** tenha a emoção desejada (o adaptador aprendeu a copiar estilo/timbre da referência) | nenhum | precisa de áudio emocional por emoção |
| **B. Modelo base para emoção** | `(base - sem adapter)` + instrução **descritiva no idioma do texto** + **CFG 4** | nenhum | texto em EN/ZH (pt-BR degrada) |
| **C. Pós-processar prosódia** | F0/ritmo/energia no áudio gerado (pitch-shift + time-stretch) | baixo | "emoção" mais artificial; exagerar muda o timbre |
| **D. Retreinar (v4) com instrução variada** | pool com 10–20 instruções PT **emocionais** + a de produção, `--instruction-mode pool` | 1 run de treino | o corpus é leitura/podcast, com pouca emoção marcada — pode não haver alvo para aprender |
| **E. Eventos vocais no texto** | cues no próprio texto (ex.: `[risada]`, `(suspiro)`, CAIXA ALTA) | nenhum | efeito informal; **medir antes de confiar** |

Se for para o **D**, o critério de aceite objetivo já existe:
`scripts/measure_instruction.py` deve dizer **`INSTRUCAO FUNCIONA`** (alguma métrica com folga
≥ 2× o ruído, pelo menos **duas** métricas ≥ 1,5× e ao menos uma de prosódia — F0/duração/ritmo)
**sem** piorar WER/SECS no `eval_zero_shot.py` — e isso vale por estilo, não só na média.

---

## 5. Como remedir (ferramenta do repo)

```bash
# Modelo base (Voice Design): instrução EN, cfg 4, 2 seeds = piso de ruído
python scripts/measure_instruction.py \
  --text "Good morning! What a wonderful day, isn't it?" \
  --instructions "Speak clearly and naturally.|Speak with great excitement and high energy." \
  --cfg 4 --seeds 42,43

# Adaptador pt-BR com referência (Voice Direction)
python scripts/measure_instruction.py --adapter adapters/Breeze-tts-2-brazillian-lora \
  --ref-audio ref.wav --ref-text "transcrição exata da referência" \
  --text "Bom dia! Que dia maravilhoso, não é mesmo?" \
  --instructions "Fale com clareza e naturalidade.|Fale com muita empolgação, energia e ritmo acelerado." \
  --cfg 4 --seeds 42,43 --out out/instrucao
```

Saída: WAVs + `metricas.json` + tabela `instrução vs ruído` com razão por métrica e veredito
(`INSTRUCAO FUNCIONA` / `EFEITO FRACO/DUVIDOSO` / `INSTRUCAO INERTE`). `--reuse` reaproveita
WAVs e só re-mede.

Rodando nos áudios desta investigação, o critério separa os dois casos sem ambiguidade:

| caso | razão máx. instrução/ruído | veredito |
|---|---:|---|
| modelo base (Voice Design, EN, cfg 4) | **4,92×** (F0) | `INSTRUCAO FUNCIONA` |
| adaptador pt-BR (clonagem, PT, cfg 4) | **1,37×** (MFCC) | `INSTRUCAO INERTE` |

---

## 6. Limitações desta medição

- O caso do adaptador tem **poucas amostras** (1 seed por instrução/cfg; 2 pares de ruído), uma
  única voz de referência (`VozEdnilson`) e um único texto-alvo.
- F0/RMS/duração/MFCC são **proxies acústicos**, não teste perceptivo. O "triste" saiu na direção
  esperada (mais grave, mais lento, mais baixo), mas **dentro do ruído** — não dá para afirmar
  efeito.
- O modelo base foi medido com texto EN (é o regime dele); texto em pt-BR no base é fora de
  distribuição e os números de lá não devem ser generalizados.
