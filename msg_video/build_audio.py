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
LEAD_IN = 0.3   # silence before the first line of every scene
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


BASE_SPEED = {"NARR": 1.03}   # natural documentary pace; talking heads run a little faster
OTHER_SPEED = 1.12


def voice_style(k, spec):
    """'name' or a weighted blend like 'bm_george*0.5+am_michael*0.5'."""
    if "+" not in spec:
        return spec
    return sum(float(w) * k.get_voice_style(n) for n, w in (p.split("*") for p in spec.split("+")))


def polish(a):
    """Gentle voice-over chain: rumble high-pass, low-mid warmth, presence lift,
    de-ess, soft compression and a very short room."""
    n = len(a)
    F = np.fft.rfftfreq(n, 1 / SR)
    g = 1 / np.sqrt(1 + (70 / np.maximum(F, 1)) ** 4)
    g *= 1 + 0.25 * np.exp(-((F - 180) / 90) ** 2)
    g *= 1 + 0.18 * np.exp(-((F - 3500) / 1500) ** 2)
    g *= 1 - 0.20 * np.exp(-((F - 7000) / 1200) ** 2)
    a = np.fft.irfft(np.fft.rfft(a) * g, n)
    env = np.convolve(np.abs(a), np.ones(240) / 240, "same")
    thr = 0.12
    gain = np.where(env > thr, (thr + (env - thr) / 3) / np.maximum(env, 1e-9), 1.0)
    a = a * np.convolve(gain, np.ones(480) / 480, "same")
    rng = np.random.default_rng(1)
    ir = rng.standard_normal(int(0.25 * SR)) * np.exp(-np.arange(int(0.25 * SR)) / (0.04 * SR)) * 0.02
    ir[0] = 1
    return np.convolve(a, ir)[:n].astype(np.float32)


def main():
    kd = os.environ["KOKORO_DIR"]
    k = Kokoro(f"{kd}/kokoro-v1.0.onnx", f"{kd}/voices-v1.0.bin")
    os.makedirs(OUT, exist_ok=True)
    cues = {}
    for sc in SCENES:
        chunks, lines, t = [np.zeros(int(LEAD_IN * SR), np.float32)], [], LEAD_IN
        for voice, text, pause, *rest in sc["lines"]:
            speed = BASE_SPEED.get(voice, OTHER_SPEED) * (rest[0] if rest else 1.0)
            a, sr = k.create(text, voice=voice_style(k, VOICES[voice]), speed=speed, lang="en-us")
            assert sr == SR
            if voice == "NARR":
                a = polish(a)
            dur = len(a) / SR
            lines.append({"voice": voice, "text": text, "start": round(t, 3), "end": round(t + dur, 3),
                          "words": word_times(text, a, t)})
            chunks += [a.astype(np.float32), np.zeros(int(pause * SR), np.float32)]
            t += dur + pause
        audio = np.concatenate(chunks)
        audio = audio / max(1e-9, np.max(np.abs(audio))) * 0.89
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
