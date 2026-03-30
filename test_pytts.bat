@echo off
setlocal
cd /d "e:\Apps\Gigi a Sapeca\jaison-core"

echo ===========================================
echo       Teste de Voz Gigi a Sapeca (SAPI5)
echo ===========================================
echo.

:: Caminho para o Python do seu venv conforme start_server.bat
set PYTHON_EXE="e:\Apps\Project J.A.I.son\venv_jaison\python.exe"

if exist %PYTHON_EXE% (
    %PYTHON_EXE% test_pytts.py
) else (
    echo [ERRO] Nao foi possivel encontrar o ambiente virtual em:
    echo %PYTHON_EXE%
    echo Verifique o caminho no arquivo start_server.bat e corrija-o se necessario.
)

echo.
pause
