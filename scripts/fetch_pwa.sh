#!/usr/bin/env bash
set -euo pipefail
url="$1"                      # copy from the archive page
name="$(basename "$url")"
mkdir -p traces data/processed
curl -L "$url" -o "traces/$name"
case "$name" in
  *.gz) gunzip -f "traces/$name"; swf="${name%.gz}" ;;
  *)    swf="$name" ;;
esac
python scripts/swf_to_csv.py --swf "traces/$swf" --out data/processed/llnl_jobs.csv
echo "[OK] Converted -> data/processed/llnl_jobs.csv"
