# Handoff: switching the narrator to ElevenLabs

For a new Claude session working in `varunlohade/video`, branch `claude/msg-explainer-video-2t6cao`.
The project root is the repo root, which mirrors `msg_video/` in AIBICI.

## Where things stand

- **Video:** one ~2:18 vertical Instagram Reel about MSG, in the client-approved "Kombu Paper" style (`design/APPROVED.md`).
- **Script:** the single source of truth is `script.py`. Narration WAVs and `cues.json` (line and word timings) are generated from it.
- **Rendering:** scenes in `scenes/*.js` read every timing from `cues.json`, so re-recording the narration re-times the whole video.
- **Client rules:** see the top of `BRIEF.md`.
  - At most one short label on screen besides captions.
  - Vivid colours only.
  - 9:16 safe zones.
- **Pending from the client:** real interview clips for the cold open. When they arrive, use `prep_clips.py`, `clips/clips.json` and `lib/clips.js`. The clip lines in `script.py` use the voice `"CLIP:<id>"`.

## Narrator via ElevenLabs (the client is paying, so keep credit use low)

The environment must have `ELEVENLABS_API_KEY` set and `api.elevenlabs.io` allowed. Check both first:

    [ -n "$ELEVENLABS_API_KEY" ] && curl -s -o /dev/null -w '%{http_code}\n' https://api.elevenlabs.io/v1/models

Then, from the repo root:

1. **Audition** about 4 voices on one short passage (~720 characters in total, budget-checked, cached):

       python3 audio/audition_voices.py 4

   Send `out/preview/elevenlabs_audition.m4a` to the client and let them pick a number. The voice IDs are in `out/preview/elevenlabs_audition.json`.
   - The client asked for a voice "like Kurzgesagt's narrator". Do not clone or imitate that real person.
   - Pick stock or library voices with similar qualities: calm, warm, articulate, a little playful.
2. **Full narration** for the chosen voice (~1,900 characters; every line is cached in `audio/tts_cache/`, which is committed):

       TTS_ENGINE=elevenlabs ELEVENLABS_VOICE_ID=<id> KOKORO_DIR=<dir with kokoro-v1.0.onnx + voices-v1.0.bin> python3 build_audio.py

   Kokoro is still used for the cold-open talking heads, until the real clips replace them. It is a ~350 MB download from https://github.com/thewh1teagle/kokoro-onnx/releases/tag/model-files-v1.0, and needs `pip install kokoro-onnx soundfile`.
3. **Re-time the shot list:** `python3 design/gen_shotlist.py`. If a word anchor moved, fix the anchor.
4. **Render and mix:** `./build.sh`. The output is `out/msg_reel.mp4`.

The full Reel must stay under 3:00. Commit `audio/tts_cache/`, `audio/narration/` and `cues.json` after rendering.

## Real interview clips for the cold open (client request)

1. Candidates are researched in `clips/CANDIDATES.md`, with URL, timestamps, verbatim quote and confidence. The client picks which ones to use.
   - Only use VERIFIED or LIKELY quotes.
   - The on-screen credit must name the speaker, the show or outlet, and the year.
2. Add the chosen clips to `clips/clips.json`. Each one needs `id`, `url`, `in`/`out` (seconds in the original video), `speaker`, `credit` and `quote` (exact words).
3. Download only those segments. This needs network access to youtube.com, *.googlevideo.com, youtubei.googleapis.com and i.ytimg.com, added under the environment's Allowed domains:

       pip install yt-dlp && python3 fetch_clips.py && python3 prep_clips.py

4. In `script.py` s01, replace the TVDOC/CHEF/GURU lines with `("CLIP:<id>", "<exact quote>", pause)`, then run build_audio.py.
5. Adapt `scenes/s01_coldopen.js`:
   - The hero TV screens play `CLIPS.draw(...)` instead of the puppets.
   - The lower third shows the real name and credit, with no DRAMATIZATION line.
   - `heroDoctor`, `heroChef` and `heroGuru` are the swap points.
