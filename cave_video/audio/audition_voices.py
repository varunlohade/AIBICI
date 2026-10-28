"""Audition ElevenLabs narrator voices on one short passage (budget-capped).

Shortlists voices whose labels describe a calm, warm documentary/narration delivery, renders the same
~180-character sample with each (cached), and writes out/preview/elevenlabs_audition.m4a with a spoken
number before each sample. Usage: python3 audio/audition_voices.py [max_voices=4]
"""
import json, os, subprocess, sys, urllib.error

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import elevenlabs as el

SAMPLE = ("Your tongue tastes it, and tells your brain: protein! Then it reaches your gut... and gets eaten. "
          "Which sounds alarmingly chemical. But it's in tomatoes, parmesan, even breast milk.")
WANT = ("narrat", "documentar", "calm", "warm", "educational", "storytell", "british", "deep", "soft")


def score(v):
    lab = " ".join(str(x) for x in (v.get("labels") or {}).values()).lower() + " " + (v.get("description") or "").lower()
    s = sum(w in lab for w in WANT)
    return s + (1 if (v.get("labels") or {}).get("gender") == "male" else 0)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    left, limit = el.characters_left()
    cost = n * len(SAMPLE)
    print(f"credits left: {left}/{limit}; this audition costs at most {cost} chars (cached samples are free)")
    if cost > left:
        raise SystemExit("not enough credits")
    voices = json.loads(el.request("/voices"))["voices"]
    ranked = sorted(voices, key=score, reverse=True)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out, sr, pick = [], 24000, []
    for v in ranked:
        if len(pick) == n:
            break
        try:
            sample = el.synth(SAMPLE, v["voice_id"])
        except urllib.error.HTTPError as e:
            if e.code in (401, 402, 403):   # voice not usable on this plan via the API; try the next one
                print(f"   skip {v['name']}: HTTP {e.code}")
                continue
            raise
        pick.append(v)
        i = len(pick)
        print(f"{i}. {v['name']}  {v['voice_id']}  {v.get('labels')}")
        num = el.synth(f"Voice {i}.", v["voice_id"])
        out += [num * 0.7, np.zeros(int(.4 * sr), np.float32), sample, np.zeros(int(1.2 * sr), np.float32)]
    os.makedirs(os.path.join(root, "out", "preview"), exist_ok=True)
    wav = os.path.join(root, "out", "preview", "elevenlabs_audition.wav")
    sf.write(wav, np.concatenate(out), sr)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", wav, "-c:a", "aac", "-b:a", "160k", wav[:-4] + ".m4a"], check=True)
    json.dump([{"n": i, "name": v["name"], "voice_id": v["voice_id"]} for i, v in enumerate(pick, 1)],
              open(os.path.join(root, "out", "preview", "elevenlabs_audition.json"), "w"), indent=1)
    print("wrote", wav[:-4] + ".m4a")


if __name__ == "__main__":
    main()
