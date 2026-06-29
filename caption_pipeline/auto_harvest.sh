#!/usr/bin/env bash
# YouTube + VLM harvest pipeline — the reliable FALLBACK for supplementing driving data when external
# datasets are blocked (ZOD email-wall, S2TLD Baidu-only, etc.). One command:
#   search/download driving videos -> scene-change frame extraction -> push to GPU server -> multi_extract
#   (one VLM call/frame -> scene/light/sign/POI/text/VRU) -> per-use-case sharegpt.
#
# Usage:
#   auto_harvest.sh "<search query>"  <tag> [num_videos]      # auto-find via YouTube search
#   auto_harvest.sh "url1,url2,..."   <tag>                   # explicit video URLs
# Env: SRV=user@host  SRVDIR=/path/on/server  KEYFILE=server .vultr_keys.env path  PW=ssh_password
set -uo pipefail
IN="${1:?search query or comma-separated URLs}"; TAG="${2:?tag}"; NV="${3:-3}"
SRV="${SRV:-haiyuez@10.136.20.188}"; SRVDIR="${SRVDIR:-/data/haiyuez/visteon_cabin_vlm}"
PW="${PW:-haiyuefortis}"; SCENE="${SCENE:-0.08}"; START="${START:-0:30}"; DUR="${DUR:-300}"
WORK="$(mktemp -d)/harvest_$TAG"; mkdir -p "$WORK/frames"
ssh_() { sshpass -p "$PW" ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20 "$SRV" "$@"; }
scp_() { sshpass -p "$PW" scp -o StrictHostKeyChecking=no -o ConnectTimeout=20 "$@"; }

# 1. resolve URLs (explicit list or YouTube search)
URLS=()
if [[ "$IN" == *"http"* ]]; then
  IFS=',' read -ra URLS <<< "$IN"
else
  echo "[search] ytsearch$NV: $IN"
  while IFS= read -r id; do
    [ -n "$id" ] && URLS+=("https://www.youtube.com/watch?v=$id")
  done < <(yt-dlp --get-id "ytsearch${NV}:${IN}" 2>/dev/null)
fi
echo "[urls] ${#URLS[@]} videos"

# 2. download + scene-change extract (robust: skip failures, never abort the batch)
i=0
for u in "${URLS[@]}"; do
  i=$((i+1)); n="${TAG}_${i}"
  yt-dlp -f "best[height<=720]" --download-sections "*${START}-$(python3 -c "import datetime,sys;p=[int(x) for x in '${START}'.split(':')];s=p[0]*60+p[1] if len(p)==2 else p[0];print(datetime.timedelta(seconds=s+${DUR}))")" \
         -o "$WORK/${n}.%(ext)s" "$u" >/dev/null 2>&1 || { echo "  [skip] $u (download failed)"; continue; }
  src=$(ls "$WORK/${n}".* 2>/dev/null | grep -vE '\.jpg$' | head -1)
  [ -z "$src" ] && { echo "  [skip] $u (no file)"; continue; }
  ffmpeg -i "$src" -vf "select='gt(scene,${SCENE})',scale=1280:-1" -vsync vfr -q:v 3 "$WORK/frames/${n}_%04d.jpg" >/dev/null 2>&1
  rm -f "$src"
  echo "  [ok] $n: $(ls "$WORK/frames/${n}_"*.jpg 2>/dev/null | wc -l | tr -d ' ') frames"
done
NF=$(ls "$WORK/frames/"*.jpg 2>/dev/null | wc -l | tr -d ' ')
echo "[frames] $NF diverse frames"
[ "$NF" -eq 0 ] && { echo "no frames harvested"; exit 1; }

# 3. push to server + run multi_extract (resumable)
tar czf "$WORK/frames.tar.gz" -C "$WORK" frames 2>/dev/null
DST="$SRVDIR/data/harvest_$TAG"
until ssh_ "mkdir -p $DST"; do sleep 30; done
until scp_ "$WORK/frames.tar.gz" "$SRV:$DST/"; do echo "  retry push"; sleep 30; done
ssh_ "cd $DST && tar xzf frames.tar.gz && python3 -c \"import os,json;d='$DST/frames';open('$DST/input.jsonl','w').write('\n'.join(json.dumps({'image':os.path.join(d,f)}) for f in os.listdir(d) if f.endswith('.jpg')))\""
echo "[server] launching multi_extract on $DST"
ssh_ "source /home/haiyuez/miniconda3/etc/profile.d/conda.sh; conda activate $SRVDIR/envs/cabin-vlm 2>/dev/null; cd $SRVDIR; VULTR_KEYS=${KEYFILE:-$SRVDIR/.vultr_keys.env} WORKERS=2 VISION_MAX_TOKENS=1500 VISION_MODEL='nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16' setsid bash -c 'python -m caption_pipeline.multi_extract --inp $DST/input.jsonl --out $DST' > $SRVDIR/logs/harvest_$TAG.log 2>&1 </dev/null & disown; echo LAUNCHED"
echo "[done] harvest_$TAG launched; per-use-case sharegpt will appear in $DST/harvest_*_sharegpt.json"
