@echo off
setlocal
cd /d "%~dp0"
set "PYTHONPATH=%cd%\src;%cd%"
set "weight_root=models/rvc"
set "hubert_path=models/rvc/hubert_base.pt"
set "rmvpe_root=models/rvc"
set "index_root=models/rvc"

echo ===========================================
echo    Gigi Lab - Interface Web do RVC
echo ===========================================
echo.
echo Iniciando servidor Gradio...
echo.

"e:\Apps\Project J.A.I.son\venv_jaison\python.exe" rvc_gui.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERRO] Nao foi possivel iniciar a interface.
    pause
)
