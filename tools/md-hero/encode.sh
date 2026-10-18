#!/bin/bash
# Turn rendered frames into the three files the homepage uses.
#   NAME=e6-hero tools/md-hero/encode.sh FRAMES_DIR OUT_DIR [FPS=30] [CRF=36] [HEVC_KBPS=900] [ALPHA_Q=0.7]
# Writes NAME.webm, NAME.mov and NAME-poster.webp. Defaults keep the 1080 px,
# 12 s hero loop under ~3 MB per video.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
frames="$1"; out="$2"; fps="${3:-30}"; crf="${4:-36}"; kbps="${5:-900}"; alpha="${6:-0.7}"
name="${NAME:-e6-hero}"
blender="${BLENDER:-/Applications/Blender.app/Contents/MacOS/Blender}"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
mkdir -p "$out"
"$blender" -b --factory-startup --python-exit-code 1 \
  --python "$here/encode_blender.py" -- \
  --frames "$frames" --fps "$fps" --crf "$crf" \
  --webm "$out/$name.webm" --poster "$out/$name-poster.webp"
# Safari and iOS only show video transparency in HEVC with alpha, which only
# Apple's encoder writes; encode_hevc drives it with an explicit bitrate.
clang -fobjc-arc -framework Foundation -framework AVFoundation \
  -framework CoreMedia -framework CoreVideo -framework VideoToolbox \
  -framework ImageIO -framework CoreGraphics -framework Accelerate \
  "$here/encode_hevc.m" -o "$tmp/encode_hevc"
"$tmp/encode_hevc" "$frames" "$out/$name.mov" "$fps" "$kbps" "$alpha"
ls -lh "$out"
