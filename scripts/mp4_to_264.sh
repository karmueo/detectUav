#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<EOF
Usage: $0 INPUT.mp4 [OUTPUT.264] [FPS]

Convert an MP4 (or other container) video to a raw H.264 bitstream (.264 / .h264).

Arguments:
  INPUT.mp4        Path to input video file.
  OUTPUT.264      Optional output path (default: same name as input with .264).
  FPS             Optional output frames per second (integer or float). If omitted, original FPS is kept.

Examples:
  $0 video.mp4                # -> video.264 using source FPS
  $0 video.mp4 out.264 25     # -> out.264 with 25 FPS
  $0 video.mp4 out.h264 30    # -> out.h264 with 30 FPS
EOF
}

if [[ ${1-} == "-h" || ${1-} == "--help" ]]; then
  usage
  exit 0
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "Error: ffmpeg is not installed or not in PATH." >&2
  exit 2
fi

if [ $# -lt 1 ]; then
  echo "Error: missing input file." >&2
  usage
  exit 2
fi

INPUT="$1"
if [ ! -f "$INPUT" ]; then
  echo "Error: input file '$INPUT' does not exist." >&2
  exit 2
fi

DEFAULT_OUT="${INPUT%.*}.264"
OUTPUT="${2:-$DEFAULT_OUT}"

FPS_ARG=""
if [ ${3+set} ]; then
  FPS_RAW="$3"
  if [ -n "$FPS_RAW" ]; then
    # Basic validation: allow integers or decimals
    if [[ ! $FPS_RAW =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
      echo "Error: FPS must be a positive number (e.g. 25 or 29.97)." >&2
      exit 2
    fi
    FPS_ARG=( -r "$FPS_RAW" )
  fi
fi

echo "Converting '$INPUT' -> '$OUTPUT'" 
if [ -n "${FPS_ARG[*]:-}" ]; then
  echo "  with FPS: ${FPS_RAW}"
else
  echo "  keeping source FPS"
fi

# Use libx264 encoder and force raw h264 output (-f h264). Remove audio (-an).
# -y to overwrite output if exists.
ffmpeg -y -i "$INPUT" "${FPS_ARG[@]:-}" -c:v libx264 -preset veryfast -crf 18 -an -f h264 "$OUTPUT"

RET=$?
if [ $RET -eq 0 ]; then
  echo "Done: $OUTPUT"
else
  echo "ffmpeg failed with exit code $RET" >&2
fi

exit $RET
