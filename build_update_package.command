#!/bin/zsh
set -u
setopt NULL_GLOB

cd "$(dirname "$0")"

echo "Colortina package builder"
echo "Project: $(pwd)"
echo

if ! command -v rsync >/dev/null 2>&1; then
  echo "rsync was not found."
  echo "Press any key to close..."
  read -k 1
  exit 1
fi

if ! command -v zip >/dev/null 2>&1; then
  echo "zip was not found."
  echo "Press any key to close..."
  read -k 1
  exit 1
fi

PACKAGE_DIR="$HOME/Desktop/更新包"
STAMP=$(date +%Y%m%d-%H%M%S)
PACKAGE_NAME="Colortina-V5-${STAMP}"
TMP_PARENT=$(mktemp -d /private/tmp/colortina-package.XXXXXX)
STAGE_DIR="$TMP_PARENT/$PACKAGE_NAME"
ZIP_PATH="$PACKAGE_DIR/$PACKAGE_NAME.zip"

cleanup() {
  rm -rf "$TMP_PARENT"
}
trap cleanup EXIT

mkdir -p "$PACKAGE_DIR"
mkdir -p "$STAGE_DIR"

echo "Copying project files..."
rsync -a \
  --exclude '.git/' \
  --exclude '.DS_Store' \
  --exclude 'Thumbs.db' \
  --exclude 'venv/' \
  --exclude '.venv/' \
  --exclude 'runtime/' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude '.pytest_cache/' \
  --exclude '.mypy_cache/' \
  --exclude '.ruff_cache/' \
  --exclude '.coverage' \
  --exclude 'htmlcov/' \
  --exclude 'models/weights/' \
  --exclude 'models/*.pth' \
  --exclude 'models/*.zip' \
  --exclude '*.onnx' \
  --exclude 'outputs/' \
  --exclude 'characters/' \
  --exclude '*.ccproject' \
  --exclude '*_assets/' \
  ./ "$STAGE_DIR/"

chmod +x "$STAGE_DIR/run_colortina.command" "$STAGE_DIR/update_colortina.command" 2>/dev/null || true
chmod +x "$STAGE_DIR/build_update_package.command" 2>/dev/null || true

echo "Checking Python files..."
if [ -x "venv/bin/python" ]; then
  venv/bin/python -m compileall -q main.py pipeline.py core ui models tests tools test_pipeline.py
else
  python3 -m compileall -q main.py pipeline.py core ui models tests tools test_pipeline.py
fi

echo "Creating zip package..."
(
  cd "$TMP_PARENT"
  zip -qr "$ZIP_PATH" "$PACKAGE_NAME"
)

echo
echo "Package created:"
echo "$ZIP_PATH"
echo
echo "Put this zip in Desktop/更新包 on the target computer, then double-click update_colortina.command there."
echo
if [ -t 0 ]; then
  echo "Press any key to close..."
  read -k 1
fi
