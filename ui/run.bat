@echo off
REM run.bat - abre a UI web do Breeze TTS 2 + LoRA PT-BR.
REM Uso: duplo-clique, ou  run.bat --port 7861
REM Python: use a venv do repo (.\venv) ou defina BREEZE_PY.
setlocal
set "ROOT=%~dp0.."
if defined BREEZE_PY (set "PY=%BREEZE_PY%") else (set "PY=%ROOT%\venv\Scripts\python.exe")
if not exist "%PY%" (echo [erro] python/venv nao encontrado: %PY% & echo Defina BREEZE_PY ou crie .\venv & pause & exit /b 1)
"%PY%" "%ROOT%\ui\app.py" %*
endlocal
