#!/usr/bin/env bash
# Batch-download Zhang Xiaojun episodes as MP3 into docs/videos/audio/.
# Usage: ./download_batch.sh           # download all VIDS listed below
#        ./download_batch.sh VID1 VID2 # download specific IDs

set -uo pipefail

cd "$(dirname "$0")"
mkdir -p audio

VIDS=(
  # Sequential backfill 131..121
  dm0Zsm9BiD8 ruVJ_5dObxs 9zSMTUUEfmU MW-ezf2RhVg SG90aehV3vU
  uOOB1azmbXk k82iFzvKFCQ EosO2Qd35Cw qZbzFZ2R_Nw wK0-m3rKgZ0
  2o281Zy5aZE
  # Technical paper-walkthroughs
  zrvnoYYPaWQ gQgKkUsx5q0 8dKBH4x0D9o vWrYHvSRz0s 3jI6F3M2ocU
  42CveqxzU5M j5CSpSqNCJw 0AImqp6KznY
)

if [[ $# -gt 0 ]]; then
  VIDS=("$@")
fi

dl_one() {
  local vid="$1"
  local out="audio/${vid}.mp3"
  if [[ -s "$out" ]]; then
    echo "SKIP $vid (exists, $(du -h "$out" | cut -f1))"
    return 0
  fi
  echo "GET  $vid  https://www.youtube.com/watch?v=${vid}"
  yt-dlp --no-warnings -x --audio-format mp3 --audio-quality 0 \
    -o "audio/%(id)s.%(ext)s" \
    "https://www.youtube.com/watch?v=${vid}" 2>&1 | tail -5
  if [[ -s "$out" ]]; then
    echo "OK   $vid  $(du -h "$out" | cut -f1)"
  else
    echo "FAIL $vid"
    return 1
  fi
}

export -f dl_one

echo "Starting ${#VIDS[@]} downloads at $(date)"
START=$(date +%s)

# 3-way parallel keeps the network busy without slamming YouTube.
printf "%s\n" "${VIDS[@]}" | xargs -I {} -P 3 bash -c 'dl_one "$@"' _ {}

END=$(date +%s)
echo "Done in $((END - START))s at $(date)"
echo "Audio dir size: $(du -sh audio | cut -f1)"
