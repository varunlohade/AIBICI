"""Synthesize narration per scene with Kokoro TTS and write cues.json.

Usage: KOKORO_DIR=/path/with/kokoro-v1.0.onnx+voices-v1.0.bin python3 build_audio.py
"""
import json, os
import numpy as np
import soundfile as sf
from kokoro_onnx import Kokoro
from script import SCENES, VOICES

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "audio", "narration")
LEAD_IN = 0.6   # silence before the first line of every scene
SR = 24000

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
            lines.append({"voice": voice, "text": text, "start": round(t, 3), "end": round(t + dur, 3)})
            chunks += [a.astype(np.float32), np.zeros(int(pause * SR), np.float32)]
            t += dur + pause
        audio = np.concatenate(chunks)
        sf.write(os.path.join(OUT, sc["id"] + ".wav"), audio, SR)
        cues[sc["id"]] = {"title": sc["title"], "duration": round(len(audio) / SR, 3), "lines": lines}
        print(sc["id"], cues[sc["id"]]["duration"])
    with open(os.path.join(HERE, "cues.json"), "w") as f:
        json.dump(cues, f, indent=1)
    print("total", round(sum(c["duration"] for c in cues.values()), 1))

if __name__ == "__main__":
    main()
