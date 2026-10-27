"""Audition ElevenLabs narrator voices on one short passage (budget-capped).

Shortlists voices whose labels describe a calm, warm documentary/narration delivery, renders the same
~180-character sample with each (cached), and writes out/preview/elevenlabs_audition.m4a with a spoken
number before each sample. Usage: python3 audio/audition_voices.py [max_voices=4]
"""
import json, os, subprocess, sys

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
    pick = sorted(voices, key=score, reverse=True)[:n]
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out, sr = [], 24000
    for i, v in enumerate(pick, 1):
        print(f"{i}. {v['name']}  {v['voice_id']}  {v.get('labels')}")
        num = el.synth(f"Voice {i}.", v["voice_id"])
        out += [num * 0.7, np.zeros(int(.4 * sr), np.float32), el.synth(SAMPLE, v["voice_id"]),
                np.zeros(int(1.2 * sr), np.float32)]
    os.makedirs(os.path.join(root, "out", "preview"), exist_ok=True)
    wav = os.path.join(root, "out", "preview", "elevenlabs_audition.wav")
    sf.write(wav, np.concatenate(out), sr)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", wav, "-c:a", "aac", "-b:a", "160k", wav[:-4] + ".m4a"], check=True)
    json.dump([{"n": i, "name": v["name"], "voice_id": v["voice_id"]} for i, v in enumerate(pick, 1)],
              open(os.path.join(root, "out", "preview", "elevenlabs_audition.json"), "w"), indent=1)
    print("wrote", wav[:-4] + ".m4a")


if __name__ == "__main__":
    main()
