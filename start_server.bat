@echo off
setlocal
cd /d "%~dp0"

set "PYTHONPATH=%cd%\src"
set "weight_root=models/rvc"
set "hubert_path=models/rvc/hubert_base.pt"
set "rmvpe_root=models/rvc"
set "index_root=models/rvc"

echo ================================================
echo   Servidor Gigi a Sapeca - Iniciando...
echo ================================================
echo.
echo   Config:  config.yaml
echo   Porta:   5000
echo   RVC:     models/rvc/Gigi.pth
echo.
echo ================================================
echo.

"e:\Apps\Project J.A.I.son\venv_jaison\python.exe" src/main.py --config config

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERRO] O servidor encerrou com erro. Veja os logs acima.
    pause
)
