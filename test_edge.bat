@echo off
setlocal
cd /d "e:\Apps\Gigi a Sapeca\jaison-core"

echo ===========================================
echo       Teste de Voz Gigi a Sapeca (EDGE)
echo ===========================================
echo.

set PYTHONPATH=src
set PYTHON_EXE="e:\Apps\Project J.A.I.son\venv_jaison\python.exe"

if exist %PYTHON_EXE% (
    %PYTHON_EXE% test_edge.py
) else (
    echo [ERRO] Nao foi possivel encontrar o ambiente virtual.
)

echo.
pause
