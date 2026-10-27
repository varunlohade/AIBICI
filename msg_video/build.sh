#!/usr/bin/env bash
# Full build: narration -> scene videos -> sfx events -> audio mix -> final mp4.
# Narration WAVs are committed; set KOKORO_DIR and pass --tts to regenerate them.
set -euo pipefail
cd "$(dirname "$0")"
if [[ "${1:-}" == "--tts" ]]; then python3 build_audio.py; fi
python3 script.py
[[ -d audio/sfx ]] || python3 audio/make_sfx.py
[[ -d audio/music ]] || python3 audio/make_music.py
node render.js --video all --jobs 4
node render.js --sfx
python3 mix_audio.py
python3 - <<'PY'
import json
ids = list(json.load(open("cues.json")))
open("out/concat.txt", "w").write("".join(f"file 'scenes/{i}.mp4'\n" for i in ids))
PY
ffmpeg -y -loglevel error -f concat -safe 0 -i out/concat.txt -i out/final_audio.wav \
  -map 0:v -map 1:a -c:v copy -c:a aac -b:a 192k -shortest -movflags +faststart out/msg_explainer.mp4
echo "Done: $(pwd)/out/msg_explainer.mp4"
