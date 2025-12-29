@echo off
setlocal

REM Project root (this bat location)
set ROOT=%~dp0

REM exe folder
set BIN=%ROOT%out\build\windows-debug

REM run with working directory = BIN
pushd "%BIN%"
PowerEngine.exe
popd

endlocal
