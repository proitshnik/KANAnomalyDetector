@echo off
setlocal enabledelayedexpansion

set "APP_ENTRY=main.py"
set "APP_NAME=Anomaly_Detector_KM"

set "PYTHON_BIN=python3"

cd /d "%~dp0"

echo 1 Using: 
%PYTHON_BIN% -V

echo 2 Installing build dependencies (PyInstaller)
%PYTHON_BIN% -m pip install --upgrade pip >nul
%PYTHON_BIN% -m pip install --upgrade pyinstaller >nul

echo 3 Cleaning old builds
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
if exist *.spec del /q *.spec

rem скрипт работает с --onedir
rem --onefile дает один бинарник, но сборка PKG может падать с struct.error из-за ограничений формата архива

echo 4 Building app with PyInstaller
%PYTHON_BIN% -m PyInstaller ^
  --noconfirm ^
  --clean ^
  --name "%APP_NAME%" ^
  --onedir ^
  --windowed ^
  --collect-all PyQt6 ^
  "%APP_ENTRY%"

echo 5 Done
echo Output: dist\%APP_NAME%\ (запуск: dist\%APP_NAME%\%APP_NAME%.exe)

endlocal
