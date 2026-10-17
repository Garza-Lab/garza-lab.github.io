#!/bin/bash
# Turn rendered frames into the three files the homepage uses.
#   tools/md-hero/encode.sh FRAMES_DIR OUT_DIR [FPS=30] [CRF=32] [HEVC_KBPS=1000]
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
frames="$1"; out="$2"; fps="${3:-30}"; crf="${4:-32}"; kbps="${5:-1000}"
blender="${BLENDER:-/Applications/Blender.app/Contents/MacOS/Blender}"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
mkdir -p "$out"
"$blender" -b --factory-startup --python-exit-code 1 \
  --python "$here/encode_blender.py" -- \
  --frames "$frames" --fps "$fps" --crf "$crf" \
  --webm "$out/e6-hero.webm" --poster "$out/e6-hero-poster.webp"
# Safari and iOS only show video transparency in HEVC with alpha, which only
# Apple's encoder writes; encode_hevc drives it with an explicit bitrate.
clang -fobjc-arc -framework Foundation -framework AVFoundation \
  -framework CoreMedia -framework CoreVideo -framework VideoToolbox \
  -framework ImageIO -framework CoreGraphics -framework Accelerate \
  "$here/encode_hevc.m" -o "$tmp/encode_hevc"
"$tmp/encode_hevc" "$frames" "$out/e6-hero.mov" "$fps" "$kbps"
ls -lh "$out"
