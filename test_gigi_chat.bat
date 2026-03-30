@echo off
setlocal
cd /d "%~dp0"
set "PYTHONPATH=%cd%\src;%cd%"
set "weight_root=models/rvc"
set "hubert_path=models/rvc/hubert_base.pt"
set "rmvpe_root=models/rvc"
set "index_root=models/rvc"

echo ===========================================
echo    Teste de Conversa Real (IA + Edge + RVC)
echo ===========================================
echo.
echo Conectando ao cérebro e voz da Gigi...
echo.

"e:\Apps\Project J.A.I.son\venv_jaison\python.exe" test_gigi_chat.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERRO] O teste de conversa falhou.
    pause
)
