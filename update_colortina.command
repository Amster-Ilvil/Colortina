#!/bin/zsh
set -u
setopt NULL_GLOB

cd "$(dirname "$0")"

echo "Colortina updater"
echo "Project: $(pwd)"
echo

PACKAGE_DIR="$HOME/Desktop/更新包"
ZIP_PATH="${1:-}"

if [ -z "$ZIP_PATH" ]; then
  zip_candidates=("$PACKAGE_DIR"/Colortina*.zip(N.om))
  latest_zip="${zip_candidates[1]:-}"
  if [ -n "$latest_zip" ]; then
    echo "Found latest package:"
    echo "$latest_zip"
    echo
    echo "Press Enter to use it, or paste another .zip path:"
    read -r typed_zip
    if [ -n "$typed_zip" ]; then
      ZIP_PATH="$typed_zip"
    else
      ZIP_PATH="$latest_zip"
    fi
  else
    echo "No Colortina package was found in:"
    echo "$PACKAGE_DIR"
    echo
    echo "Paste the full path of the Colortina .zip package:"
    read -r ZIP_PATH
  fi
fi

ZIP_PATH="${ZIP_PATH/#\~/$HOME}"
ZIP_PATH="${ZIP_PATH%\"}"
ZIP_PATH="${ZIP_PATH#\"}"
ZIP_PATH="${ZIP_PATH%\'}"
ZIP_PATH="${ZIP_PATH#\'}"

if [ ! -f "$ZIP_PATH" ]; then
  echo
  echo "Package not found:"
  echo "$ZIP_PATH"
  echo
  if [ -t 0 ]; then
    echo "Press any key to close..."
    read -k 1
  fi
  exit 1
fi

tmp_dir=$(mktemp -d /private/tmp/colortina-update.XXXXXX)
backup_dir=$(mktemp -d /private/tmp/colortina-keep.XXXXXX)
cleanup() {
  rm -rf "$tmp_dir"
  rm -rf "$backup_dir"
}
trap cleanup EXIT

echo
echo "Unzipping to temporary folder..."
unzip -q -o "$ZIP_PATH" -d "$tmp_dir"

root_dir=""
if [ -f "$tmp_dir/main.py" ]; then
  root_dir="$tmp_dir"
else
  for candidate in "$tmp_dir"/*(N/); do
    if [ -f "$candidate/main.py" ]; then
      root_dir="$candidate"
      break
    fi
  done
fi

if [ -z "$root_dir" ]; then
  echo
  echo "Could not find a Colortina project folder inside the zip."
  if [ -t 0 ]; then
    echo "Press any key to close..."
    read -k 1
  fi
  exit 1
fi

for keep_path in \
  "Colortina.app" \
  "run_colortina.command" \
  "update_colortina.command" \
  "build_update_package.command"
do
  if [ -e "$keep_path" ]; then
    ditto "$keep_path" "$backup_dir/$keep_path" 2>/dev/null || cp -R "$keep_path" "$backup_dir/$keep_path"
  fi
done

echo "Syncing update..."
rsync -a \
  --delete \
  --filter 'P run_colortina.command' \
  --filter 'P update_colortina.command' \
  --filter 'P build_update_package.command' \
  --filter 'P Colortina.app/' \
  --exclude 'run_colortina.command' \
  --exclude 'update_colortina.command' \
  --exclude 'build_update_package.command' \
  --exclude 'Colortina.app/' \
  --exclude 'models/weights/' \
  --exclude 'venv/' \
  --exclude '.venv/' \
  --exclude 'runtime/' \
  --exclude '.git/' \
  --exclude '.DS_Store' \
  --exclude '.pytest_cache/' \
  --exclude '.mypy_cache/' \
  --exclude '.ruff_cache/' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  "$root_dir/" \
  ./

for keep_path in \
  "Colortina.app" \
  "run_colortina.command" \
  "update_colortina.command" \
  "build_update_package.command"
do
  if [ -e "$backup_dir/$keep_path" ]; then
    rm -rf "$keep_path"
    ditto "$backup_dir/$keep_path" "$keep_path" 2>/dev/null || cp -R "$backup_dir/$keep_path" "$keep_path"
  fi
done

chmod +x run_colortina.command update_colortina.command 2>/dev/null || true
chmod +x build_update_package.command 2>/dev/null || true
chmod +x Colortina.app/Contents/MacOS/* 2>/dev/null || true

if [ ! -d "Colortina.app" ]; then
  app_zip_candidates=(
    "$HOME/Documents"/Colortina*.zip(N.om)
    "$HOME/Desktop/更新包"/Colortina*.zip(N.om)
  )
  APP_SOURCE_ZIP=""
  for candidate_zip in "${app_zip_candidates[@]}"; do
    if zipinfo -1 "$candidate_zip" '*/Colortina.app/*' >/dev/null 2>&1; then
      APP_SOURCE_ZIP="$candidate_zip"
      break
    fi
  done
  if [ -n "$APP_SOURCE_ZIP" ]; then
    echo "Restoring Colortina.app from local Colortina app package..."
    app_tmp=$(mktemp -d /private/tmp/colortina-app.XXXXXX)
    unzip -q -o "$APP_SOURCE_ZIP" '*/Colortina.app/*' -d "$app_tmp"
    app_found=("$app_tmp"/*/Colortina.app(N))
    if [ -d "${app_found[1]:-}" ]; then
      ditto "${app_found[1]}" "Colortina.app" 2>/dev/null || cp -R "${app_found[1]}" "Colortina.app"
      chmod +x Colortina.app/Contents/MacOS/* 2>/dev/null || true
    fi
    rm -rf "$app_tmp"
  fi
fi

if [ -x "venv/bin/python" ]; then
  echo "Checking Python files..."
  venv/bin/python -m compileall -q main.py pipeline.py core ui models tests tools test_pipeline.py
else
  echo "Skipping Python check because venv/bin/python was not found."
fi

echo
echo "Update complete."
echo "Kept local models/weights and venv."
echo
if [ -t 0 ]; then
  echo "Press any key to close..."
  read -k 1
fi
