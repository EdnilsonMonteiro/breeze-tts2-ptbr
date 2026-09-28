@echo off
REM run.bat - abre a UI web do Breeze TTS 2 + LoRA PT-BR.
REM Uso: duplo-clique, ou  run.bat --port 7861
REM Python: use a venv do repo (.\venv) ou defina BREEZE_PY.
setlocal
set "ROOT=%~dp0.."
if defined BREEZE_PY (set "PY=%BREEZE_PY%") else (set "PY=%ROOT%\venv\Scripts\python.exe")
if not exist "%PY%" (
  echo [erro] python/venv nao encontrado: %PY%
  echo Defina a variavel BREEZE_PY ou crie a venv em .\venv
  pause
  exit /b 1
)
echo Abrindo a UI... (a janela fica aberta enquanto o servidor roda; feche com Ctrl+C)
echo Python: %PY%
"%PY%" "%ROOT%\ui\app.py" %*
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" (
  echo.
  echo [erro] a UI terminou com codigo %RC%. Leia a mensagem acima.
)
pause
endlocal
