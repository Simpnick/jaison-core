@echo off
setlocal
cd /d "%~dp0"
set "PYTHONPATH=%cd%\src;%cd%"
set "weight_root=models/rvc"
set "hubert_path=models/rvc/hubert_base.pt"
set "rmvpe_root=models/rvc"
set "index_root=models/rvc"

echo ================================================
echo   Cliente de Voz - Gigi a Sapeca
echo ================================================
echo.
echo IMPORTANTE: Certifique-se que o start_server.bat
echo ja esta rodando em outra janela antes de continuar!
echo.
pause

"e:\Apps\Project J.A.I.son\venv_jaison\python.exe" client_voz.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERRO] O cliente falhou.
    pause
)
