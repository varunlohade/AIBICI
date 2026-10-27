"""Synthesize narration per scene with Kokoro TTS and write cues.json.

Usage: KOKORO_DIR=/path/with/kokoro-v1.0.onnx+voices-v1.0.bin python3 build_audio.py
"""
import json, os, re
import numpy as np
import soundfile as sf
from kokoro_onnx import Kokoro
from script import SCENES, VOICES

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "audio", "narration")
LEAD_IN = 0.6   # silence before the first line of every scene
SR = 24000

def syllables(w):
    core = re.sub(r"[^A-Za-z0-9]", "", w)
    if re.fullmatch(r"\d{4}s?", core):            # years: "nineteen sixty-eight"
        return 5
    if core.isdigit():
        return 2 * len(core)
    if len(core) > 1 and core.isupper():           # acronyms are spelled out: M-S-G
        return sum(3 if ch == "W" else 1 for ch in core)
    w = core.lower()
    if not w:
        return 1
    n = len(re.findall(r"[aeiouy]+", w))
    if w.endswith("e") and not w.endswith(("le", "ee")) and n > 1:
        n -= 1
    return max(1, n)


def word_times(text, audio, t0):
    """Estimate per-word timings for captions: trim silence, then spread speech time
    over words by syllable count, with extra weight for punctuation pauses."""
    env = np.abs(audio)
    win = int(0.02 * SR)
    env = np.convolve(env, np.ones(win) / win, mode="same")
    voiced = np.where(env > 0.02 * env.max())[0]
    a, b = (voiced[0], voiced[-1]) if len(voiced) else (0, len(audio))
    words = text.split()
    weights, pauses = [], []
    for w in words:
        weights.append(syllables(w) + (0.15 if len(w) > 7 else 0))
        pauses.append(2.2 if w[-1] in ".?!" else 1.2 if w[-1] in ",:;" else 0.0)
    pauses[-1] = 0.0
    total = sum(weights) + sum(pauses)
    unit = (b - a) / SR / total
    t, out = t0 + a / SR, []
    for w, wt, pz in zip(words, weights, pauses):
        out.append({"w": w, "start": round(t, 3), "end": round(t + wt * unit, 3)})
        t += (wt + pz) * unit
    return out


def retime_words():
    """Recompute caption word timings from the existing narration WAVs (no TTS needed)."""
    path = os.path.join(HERE, "cues.json")
    cues = json.load(open(path))
    for sid, c in cues.items():
        audio, sr = sf.read(os.path.join(OUT, sid + ".wav"), dtype="float32")
        for ln in c["lines"]:
            seg = audio[int(ln["start"] * sr):int(ln["end"] * sr)]
            ln["words"] = word_times(ln["text"], seg, ln["start"])
    with open(path, "w") as f:
        json.dump(cues, f, indent=1)


def main():
    kd = os.environ["KOKORO_DIR"]
    k = Kokoro(f"{kd}/kokoro-v1.0.onnx", f"{kd}/voices-v1.0.bin")
    os.makedirs(OUT, exist_ok=True)
    cues = {}
    for sc in SCENES:
        chunks, lines, t = [np.zeros(int(LEAD_IN * SR), np.float32)], [], LEAD_IN
        for voice, text, pause in sc["lines"]:
            speed = 1.0 if voice == "NARR" else 1.08
            a, sr = k.create(text, voice=VOICES[voice], speed=speed, lang="en-us")
            assert sr == SR
            dur = len(a) / SR
            lines.append({"voice": voice, "text": text, "start": round(t, 3), "end": round(t + dur, 3),
                          "words": word_times(text, a, t)})
            chunks += [a.astype(np.float32), np.zeros(int(pause * SR), np.float32)]
            t += dur + pause
        audio = np.concatenate(chunks)
        sf.write(os.path.join(OUT, sc["id"] + ".wav"), audio, SR)
        cues[sc["id"]] = {"title": sc["title"], "duration": round(len(audio) / SR, 3), "lines": lines}
        print(sc["id"], cues[sc["id"]]["duration"])
    with open(os.path.join(HERE, "cues.json"), "w") as f:
        json.dump(cues, f, indent=1)
    print("total", round(sum(c["duration"] for c in cues.values()), 1))
    from script import PARTS
    for p in PARTS:
        print(p["id"], round(sum(cues[i]["duration"] for i in p["scenes"]), 1), "s")

if __name__ == "__main__":
    import sys
    retime_words() if "--words-only" in sys.argv else main()
