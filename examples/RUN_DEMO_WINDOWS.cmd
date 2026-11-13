@echo off
setlocal
cd /d "%~dp0.."
if exist ".venv\Scripts\python.exe" (set "PY=.venv\Scripts\python.exe") else (set "PY=py -3")
%PY% examples\demo_v7f.py
if errorlevel 1 (echo Demo failed. & pause & exit /b 1)
%PY% stage4\simulate_stage4.py --subjects 1 --out outputs\stage4_demo
if errorlevel 1 (echo Stage4 demo failed. & pause & exit /b 1)
echo Finished. Outputs in outputs\stage4_demo
pause
