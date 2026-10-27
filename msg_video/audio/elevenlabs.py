"""ElevenLabs TTS with a persistent per-line cache, so each line is only ever paid for once.

The API key is read from $ELEVENLABS_API_KEY or /root/.secrets/elevenlabs_key and never written to the repo.
Cached MP3s live in audio/tts_cache/ (committed: re-running the build costs nothing for unchanged lines).
"""
import hashlib, json, os, subprocess, urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "tts_cache")
API = "https://api.elevenlabs.io/v1"
MODEL = "eleven_multilingual_v2"
SETTINGS = {"stability": 0.45, "similarity_boost": 0.8, "style": 0.2, "use_speaker_boost": True}


def api_key():
    k = os.environ.get("ELEVENLABS_API_KEY")
    if not k and os.path.exists("/root/.secrets/elevenlabs_key"):
        k = open("/root/.secrets/elevenlabs_key").read().strip()
    if not k:
        raise SystemExit("No ElevenLabs key: set ELEVENLABS_API_KEY or write /root/.secrets/elevenlabs_key")
    return k


def request(path, body=None, accept="application/json"):
    req = urllib.request.Request(API + path, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"xi-api-key": api_key(), "Content-Type": "application/json", "Accept": accept})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def synth(text, voice_id, speed=1.0, prev_text=None, next_text=None, sr=24000):
    """Return mono float32 audio at `sr`. Neighbouring lines are passed as context for natural prosody
    (they are not spoken and not billed)."""
    settings = dict(SETTINGS, speed=round(max(0.7, min(1.2, speed)), 3))
    body = {"text": text, "model_id": MODEL, "voice_settings": settings}
    if prev_text:
        body["previous_text"] = prev_text
    if next_text:
        body["next_text"] = next_text
    key = hashlib.sha256(json.dumps([voice_id, body], sort_keys=True).encode()).hexdigest()[:20]
    os.makedirs(CACHE, exist_ok=True)
    mp3 = os.path.join(CACHE, key + ".mp3")
    if not os.path.exists(mp3):
        data = request(f"/text-to-speech/{voice_id}?output_format=mp3_44100_128", body, accept="audio/mpeg")
        with open(mp3, "wb") as f:
            f.write(data)
        print(f"  [elevenlabs] paid {len(text)} chars: {text[:50]}")
    pcm = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", mp3, "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(pcm, np.float32).copy()


def characters_left():
    s = json.loads(request("/user/subscription"))
    return s["character_limit"] - s["character_count"], s["character_limit"]
