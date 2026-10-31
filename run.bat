@echo off
cd /d "%~dp0"
set PY=
for %%V in (3.12 3.13 3.11 3.10) do if not defined PY py -%%V -c "import sys" >nul 2>&1 && set PY=py -%%V
if not defined PY py -3 -c "import sys" >nul 2>&1 && set PY=py -3
if not defined PY python -c "import sys" >nul 2>&1 && set PY=python
if not defined PY (
  echo Python not found. Install Python 3.12 from python.org and tick "Add python.exe to PATH".
  pause
  exit /b 1
)
%PY% -c "import numpy, sounddevice, pygame" >nul 2>&1 || %PY% -m pip install --disable-pip-version-check -r requirements.txt
%PY% -m pitch %*
if errorlevel 1 pause
