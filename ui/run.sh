#!/usr/bin/env bash
# run.sh - abre a UI web do Breeze TTS 2 + LoRA PT-BR (Linux/macOS).
#
# Uso:
#   ./ui/run.sh                 # http://127.0.0.1:7860
#   ./ui/run.sh --port 7861 --no-browser
#   ./ui/run.sh --selftest      # gera 1 amostra sem abrir a UI (validacao)
#   BREEZE_PY=/caminho/python ./ui/run.sh
#
# Precisa de bit de execucao uma vez: chmod +x ui/run.sh  (ou rode `bash ui/run.sh`).
# Python: usa a venv do repo (./venv) ou o interpretador de $BREEZE_PY.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"

PY="${BREEZE_PY:-$ROOT/venv/bin/python}"
if [ ! -x "$PY" ]; then
  # Sem venv no repo: cai para o python3 do PATH (com aviso) para nao travar o usuario.
  if command -v python3 >/dev/null 2>&1; then
    echo "[aviso] venv do repo nao encontrada em $ROOT/venv; usando $(command -v python3)"
    echo "[aviso] crie a venv com: python3 -m venv venv && ./venv/bin/pip install -r requirements.txt"
    PY="$(command -v python3)"
  else
    echo "[erro] python/venv nao encontrado: ${BREEZE_PY:-$ROOT/venv/bin/python}"
    echo "Crie a venv em ./venv (python3 -m venv venv) ou defina BREEZE_PY."
    exit 1
  fi
fi

echo "Abrindo a UI... (Ctrl+C encerra o servidor)"
echo "Python: $PY"
exec "$PY" "$ROOT/ui/app.py" "$@"
