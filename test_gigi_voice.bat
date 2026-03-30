@echo off
setlocal
cd /d "%~dp0"
set "PYTHONPATH=%cd%\src"
set "weight_root=models/rvc"
set "hubert_path=models/rvc/hubert_base.pt"
set "rmvpe_root=models/rvc"
set "index_root=models/rvc"

echo ===========================================
echo    Teste de Voz Completo (Edge + RVC)
echo ===========================================
echo.

"e:\Apps\Project J.A.I.son\venv_jaison\python.exe" test_gigi_voice.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERRO] O teste falhou. Veja o erro acima.
    pause
) else (
    echo.
    echo [SUCESSO] Teste concluido!
    pause
)
