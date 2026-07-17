@echo off
rem ============================================================
rem  Colortina - Windows one-click launcher / bootstrapper
rem
rem  First run : downloads a private embedded Python (~11 MB),
rem              detects NVIDIA GPU vs CPU, installs the matching
rem              PyTorch build + all dependencies into .\runtime\
rem              (nothing is installed system-wide).
rem  Later runs: environment check passes instantly -> app starts.
rem
rem  To force a clean re-install: delete the "runtime" folder.
rem ============================================================
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul
cd /d "%~dp0"

set "RT=%~dp0runtime"
set "PY=%RT%\python\python.exe"
set "MARKER=%RT%\.installed"
set "PYVER=3.11.9"
set "PYZIP_URL=https://www.python.org/ftp/python/%PYVER%/python-%PYVER%-embed-amd64.zip"
set "GETPIP_URL=https://bootstrap.pypa.io/get-pip.py"

rem ---------- Fast path: already installed ----------
if exist "%MARKER%" if exist "%PY%" goto :prep_run

echo.
echo  ==============================================
echo   Colortina first-time setup  首次运行环境安装
echo   (only happens once / 只需执行一次)
echo  ==============================================
echo.

rem ---------- 1. Private embedded Python ----------
if not exist "%PY%" (
    echo  [1/4] Downloading embedded Python %PYVER% ...
    mkdir "%RT%\python" 2>nul
    call :download "%PYZIP_URL%" "%RT%\python.zip" || goto :fail
    powershell -NoProfile -Command "Expand-Archive -Force '%RT%\python.zip' '%RT%\python'" || goto :fail
    del "%RT%\python.zip" 2>nul
    rem Enable site-packages + pip in the embedded build
    for %%F in ("%RT%\python\python3*._pth") do (
        powershell -NoProfile -Command "(Get-Content '%%~F') -replace '#import site','import site' | Set-Content '%%~F'"
    )
    call :download "%GETPIP_URL%" "%RT%\get-pip.py" || goto :fail
    "%PY%" "%RT%\get-pip.py" --no-warn-script-location || goto :fail
    del "%RT%\get-pip.py" 2>nul
) else (
    echo  [1/4] Embedded Python already present.
)

rem ---------- 2. GPU detection ----------
echo  [2/4] Detecting hardware ...
set "TORCH_INDEX=https://download.pytorch.org/whl/cpu"
set "HW=CPU"
where nvidia-smi >nul 2>&1
if not errorlevel 1 (
    nvidia-smi -L >nul 2>&1
    if not errorlevel 1 (
        set "TORCH_INDEX=https://download.pytorch.org/whl/cu121"
        set "HW=NVIDIA GPU (CUDA 12.1)"
    )
)
echo         Detected: !HW!

rem ---------- 3. PyTorch (matched to hardware) ----------
echo  [3/4] Installing PyTorch for !HW! ... (this is the big one, please wait)
"%PY%" -m pip install --no-warn-script-location torch torchvision --index-url !TORCH_INDEX! || goto :fail

rem ---------- 4. Application dependencies ----------
echo  [4/4] Installing application dependencies ...
"%PY%" -m pip install --no-warn-script-location numpy opencv-contrib-python Pillow transformers gdown pymupdf pydantic PySide6 || goto :fail

echo done> "%MARKER%"
echo.
echo  Setup complete! 安装完成，正在启动...
echo.

:prep_run
rem The embedded distribution's ._pth file fully controls sys.path and
rem does NOT include the script's directory - without this line Python
rem cannot find the app's own "ui", "core", ... packages.  Add the app
rem root (..\.. relative to the python folder) once, idempotently.
for %%F in ("%RT%\python\python3*._pth") do (
    findstr /x /c:"..\.." "%%~F" >nul 2>&1 || echo ..\..>> "%%~F"
)

:run
"%PY%" "%~dp0main.py"
if errorlevel 1 (
    echo.
    echo  Colortina exited with an error. 程序异常退出。
    echo  If the environment is broken, delete the "runtime" folder and
    echo  double-click this file again to reinstall.
    echo  如环境损坏，删除 runtime 文件夹后重新双击本文件即可自动重装。
    pause
)
exit /b 0

rem ---------- helpers ----------
:download
rem %1 = url, %2 = output path. Tries curl (built into Win10+), falls back to PowerShell.
where curl >nul 2>&1
if not errorlevel 1 (
    curl -L -sS -o "%~2" "%~1" && exit /b 0
)
powershell -NoProfile -Command "[Net.ServicePointManager]::SecurityProtocol=[Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri '%~1' -OutFile '%~2'" && exit /b 0
exit /b 1

:fail
echo.
echo  Setup failed. 安装失败。
echo  Please check your internet connection and try again.
echo  请检查网络连接后重新双击本文件（已完成的步骤会自动跳过）。
pause
exit /b 1
