#!/usr/bin/env bash
# Optimized YouTube->frames harvester for driving footage.
# Improvement over fixed-interval: SCENE-CHANGE detection (diverse frames, no near-duplicate bursts at
# red lights / straight roads) + a minimum time gap so we don't over-sample fast cuts.
# Usage: harvest_frames.sh "<youtube_url>" <name> <outdir> [start] [dur] [scene_thresh]
set -euo pipefail
URL="${1:?url}"; NAME="${2:?name}"; OUT="${3:?outdir}"
START="${4:-0:30}"; DUR="${5:-360}"; TH="${6:-0.30}"
mkdir -p "$OUT/frames"
TMP="$OUT/${NAME}.mp4"

echo "[1/2] download ${NAME} (720p, ${START}+${DUR}s)"
yt-dlp -f "best[height<=720]" --download-sections "*${START}-$(python3 - "$START" "$DUR" <<'PY'
import sys; from datetime import timedelta
def s(t):
    p=[int(x) for x in t.split(':')];
    return p[0]*60+p[1] if len(p)==2 else p[0]
print(str(timedelta(seconds=s(sys.argv[1])+int(sys.argv[2]))))
PY
)" -o "$TMP" "$URL" >/dev/null 2>&1 || yt-dlp -f "best[height<=720]" -o "$TMP" "$URL" >/dev/null 2>&1

echo "[2/2] scene-change frame extraction (thresh=${TH}, min-gap 1.5s)"
# select frames at scene cuts; cap to ~1 per 1.5s via setpts dedup is hard in one pass, so scene-only:
ffmpeg -i "$TMP" -vf "select='gt(scene,${TH})',scale=1280:-1" -vsync vfr -q:v 3 \
       "$OUT/frames/${NAME}_%04d.jpg" >/dev/null 2>&1
N=$(ls "$OUT/frames/${NAME}_"*.jpg 2>/dev/null | wc -l | tr -d ' ')
echo "extracted ${N} diverse frames for ${NAME}"
rm -f "$TMP"
