#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PY="${ROOT}/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  python3 -m venv .venv
  PY="${ROOT}/.venv/bin/python"
  "$PY" -m pip install -U pip
  "$PY" -m pip install -r requirements.txt "pyinstaller==6.22.3"
fi
"$PY" -c "import PyInstaller" 2>/dev/null || "$PY" -m pip install "pyinstaller==6.22.3"

rm -rf build dist
"$PY" -m PyInstaller --noconfirm --distpath dist --workpath build packaging/BrainTumorViewer.spec

APP="dist/BrainTumorViewer.app"
DMG="dist/BrainTumorViewer-macOS.dmg"
rm -f "$DMG"
create-dmg \
  --volname "Brain Tumor Viewer" \
  --window-pos 200 120 \
  --window-size 600 400 \
  --icon-size 100 \
  --icon "BrainTumorViewer.app" 150 180 \
  --app-drop-link 450 180 \
  --skip-jenkins \
  "$DMG" \
  "$APP"

echo "Built $DMG"
ls -lh "$DMG"
