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
# One Instagram Reel per part (1080x1920), plus the full-length cut.
python3 - <<'PY'
from script import PARTS, SCENES
groups = {p["id"]: p["scenes"] for p in PARTS}
groups["full"] = [s["id"] for s in SCENES]
for name, ids in groups.items():
    open(f"out/concat_{name}.txt", "w").write("".join(f"file 'scenes/{i}.mp4'\n" for i in ids))
PY
for name in part1 part2 part3 full; do
  audio="out/${name}_audio.wav"; [[ $name == full ]] && audio="out/final_audio.wav"
  ffmpeg -y -loglevel error -f concat -safe 0 -i "out/concat_${name}.txt" -i "$audio" \
    -map 0:v -map 1:a -c:v copy -c:a aac -b:a 192k -shortest -movflags +faststart "out/msg_${name}.mp4"
  echo "Done: $(pwd)/out/msg_${name}.mp4"
done
