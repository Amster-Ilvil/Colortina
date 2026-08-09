#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PRODUCT_NAME="${PRODUCT_NAME:-Colortina}"
VERSION="${VERSION:-$(python - <<'PY'
import pathlib, re
text = pathlib.Path('config.py').read_text(encoding='utf-8')
m = re.search(r'^\s*APP_VERSION\s*=\s*["\x27]([^"\x27]+)["\x27]', text, re.M)
if not m:
    raise SystemExit('APP_VERSION not found')
print(m.group(1))
PY
)}"
ARCH="${ARCH:-$(uname -m)}"
PACKAGE_DIR="${PACKAGE_DIR:-package}"
DMG_BASENAME="${DMG_BASENAME:-Colortina_${VERSION}_macOS_${ARCH}}"
APP="dist/${PRODUCT_NAME}.app"
APP_ZIP="${PACKAGE_DIR}/${DMG_BASENAME}.app.zip"
DMG="${PACKAGE_DIR}/${DMG_BASENAME}.dmg"

if [[ "${1:-}" != "--skip-build" ]]; then
  python scripts/build_desktop.py --clean
fi

test -d "$APP"
rm -rf "$PACKAGE_DIR" dmg-stage
mkdir -p "$PACKAGE_DIR" dmg-stage

xattr -cr "$APP" || true

# Ad-hoc sign nested Mach-O files first, then the top-level app. This keeps
# the release self-contained without requiring Developer ID secrets.
while IFS= read -r -d '' f; do
  if /usr/bin/file "$f" | grep -q 'Mach-O'; then
    /usr/bin/codesign --force --sign - --timestamp=none "$f" || true
  fi
done < <(find "$APP/Contents" -type f -print0)

while IFS= read -r bundle; do
  /usr/bin/codesign --force --sign - --timestamp=none "$bundle" || true
done < <(find "$APP/Contents" \( -name '*.framework' -o -name '*.dylib' -o -name '*.so' \) -print | awk '{ print length, $0 }' | sort -rn | cut -d' ' -f2-)

/usr/bin/codesign --force --sign - --timestamp=none "$APP"
/usr/bin/codesign --verify --deep --strict --verbose=2 "$APP"

ditto -c -k --sequesterRsrc --keepParent "$APP" "$APP_ZIP"

ditto "$APP" "dmg-stage/${PRODUCT_NAME}.app"
ln -s /Applications "dmg-stage/Applications"
hdiutil create \
  -volname "$PRODUCT_NAME" \
  -srcfolder dmg-stage \
  -ov \
  -format UDZO \
  "$DMG"
hdiutil verify "$DMG"

(
  cd "$PACKAGE_DIR"
  shasum -a 256 "$(basename "$APP_ZIP")" "$(basename "$DMG")" > SHA256SUMS.txt
  {
    echo "Colortina macOS build"
    echo "Version: $VERSION"
    echo "macOS: $(sw_vers -productVersion)"
    echo "Architecture: $(uname -m)"
    echo "Python: $(python --version 2>&1)"
    echo "Built: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    echo "Signing: ad-hoc"
    echo "Model weights bundled: no (downloaded on first use)"
    echo
    cat SHA256SUMS.txt
  } > BUILD-INFO.txt
)

rm -rf dmg-stage
printf 'Created: %s\n' "$DMG"
