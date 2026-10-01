@echo off
rem ---------------------------------------------------------------------------
rem Road Texture Agent: update to a new version, keeping everything learned.
rem
rem Use: drag the new road-texture-agent.zip onto this file, or just double-click
rem it and it takes the newest road-texture-agent*.zip in your Downloads folder.
rem
rem Your 'workspace' folder (training, libraries, tiles, corrections, bridges)
rem is never touched. A backup copy of it is made first, just in case.
rem Stop the server (Ctrl+C in its window) before updating.
rem ---------------------------------------------------------------------------
setlocal
cd /d "%~dp0"

set "ZIP=%~1"
if "%ZIP%"=="" (
  for /f "delims=" %%F in ('dir /b /o-d "%USERPROFILE%\Downloads\road-texture-agent*.zip" 2^>nul') do (
    set "ZIP=%USERPROFILE%\Downloads\%%F"
    goto found
  )
)
:found
if not exist "%ZIP%" (
  echo Could not find the update zip.
  echo Drag the new road-texture-agent.zip onto update.bat, or put it in your Downloads folder.
  pause
  exit /b 1
)
echo Updating from: %ZIP%
echo Into this folder: %CD%
echo.

if exist "workspace" (
  echo Backing up your workspace first...
  powershell -NoProfile -Command "Compress-Archive -Path 'workspace' -DestinationPath ('workspace-backup-' + (Get-Date -Format 'yyyyMMdd-HHmm') + '.zip') -Force"
)

set "TMPDIR=%TEMP%\road_texture_agent_update"
if exist "%TMPDIR%" rmdir /s /q "%TMPDIR%"
powershell -NoProfile -Command "Expand-Archive -Path '%ZIP%' -DestinationPath '%TMPDIR%' -Force"
set "SRC=%TMPDIR%\road-texture-agent"
if not exist "%SRC%\server.py" (
  echo The zip does not look like a Road Texture Agent build.
  pause
  exit /b 1
)

echo Replacing the program files...
robocopy "%SRC%\app" "app" /MIR /NFL /NDL /NJH /NJS /NP >nul
robocopy "%SRC%\ui" "ui" /MIR /NFL /NDL /NJH /NJS /NP >nul
copy /y "%SRC%\server.py" . >nul
copy /y "%SRC%\requirements.txt" . >nul
copy /y "%SRC%\README.md" . >nul

echo Installing any new packages...
python -m pip install -r requirements.txt --quiet

rmdir /s /q "%TMPDIR%"
echo.
powershell -NoProfile -Command "$v = (Select-String -Path 'server.py' -Pattern '^VERSION').Line; 'Installed version: ' + $v.Split([char]34)[1]"
echo Done. Your training in 'workspace' was kept.
echo Start the server with:  python server.py
pause
