#!/usr/bin/env bash
set -euo pipefail

APP_ENTRY="main.py"
APP_NAME="Anomaly_Detector_KM"

PYTHON_BIN="${PYTHON_BIN:-python3}"

cd "$(dirname "$0")"

echo "1 Using: $($PYTHON_BIN -V)"

echo "2 Installing build dependencies (PyInstaller)"
$PYTHON_BIN -m pip install --upgrade pip >/dev/null
$PYTHON_BIN -m pip install --upgrade pyinstaller >/dev/null

echo "3 Cleaning old builds"
rm -rf build dist *.spec

# скрипт работает с --onedir
# --onefile дает один бинарник, но сборка PKG может падать с struct.error из-за ограничений формата архива

echo "4 Building app with PyInstaller"
$PYTHON_BIN -m PyInstaller \
  --noconfirm \
  --clean \
  --name "$APP_NAME" \
  --onedir \
  --windowed \
  --collect-all PyQt6 \
  "$APP_ENTRY"

echo "5 Done"
echo "Output: dist/$APP_NAME/  (запуск: dist/$APP_NAME/$APP_NAME)"
