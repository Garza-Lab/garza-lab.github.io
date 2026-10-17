#!/bin/bash
# Turn rendered frames into the three files the homepage uses.
#   tools/md-hero/encode.sh FRAMES_DIR OUT_DIR [FPS=30] [CRF=32] [PRORES_SCALE=100]
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
frames="$1"; out="$2"; fps="${3:-30}"; crf="${4:-32}"; scale="${5:-100}"
blender="${BLENDER:-/Applications/Blender.app/Contents/MacOS/Blender}"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
mkdir -p "$out"
"$blender" -b --factory-startup --python-exit-code 1 \
  --python "$here/encode_blender.py" -- \
  --frames "$frames" --fps "$fps" --crf "$crf" --prores-scale "$scale" \
  --webm "$out/e6-hero.webm" --prores "$tmp/prores.mov" \
  --poster "$out/e6-hero-poster.webp"
# Safari and iOS only show video transparency in HEVC with alpha, which only
# Apple's encoder writes.
avconvert --source "$tmp/prores.mov" --output "$out/e6-hero.mov" \
  --preset PresetHEVC1920x1080WithAlpha --replace
ls -lh "$out"
