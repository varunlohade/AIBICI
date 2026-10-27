"""Prepare real interview clips listed in clips/clips.json for rendering.

For each clip: trim [in, out] from clips/src/<file>, export frames at 30 fps as JPEGs
(clips/prepared/<id>/f0001.jpg ...) for the canvas renderer, and a loudness-normalized
24 kHz mono WAV (clips/prepared/<id>.wav) that build_audio.py splices into the narration.
"""
import json, os, shutil, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
FPS, WIDTH = 30, 640


def main():
    spec = json.load(open(os.path.join(HERE, "clips", "clips.json")))
    out_root = os.path.join(HERE, "clips", "prepared")
    os.makedirs(out_root, exist_ok=True)
    index = {}
    for c in spec["clips"]:
        src = os.path.join(HERE, "clips", "src", c["file"])
        d = os.path.join(out_root, c["id"])
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d)
        cut = ["-ss", str(c["in"]), "-to", str(c["out"]), "-i", src]
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *cut, "-an", "-vf",
                        f"fps={FPS},scale={WIDTH}:-2", "-q:v", "3", os.path.join(d, "f%04d.jpg")], check=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *cut, "-vn", "-ac", "1", "-ar", "24000",
                        "-af", "loudnorm=I=-18:TP=-2:LRA=7", os.path.join(out_root, c["id"] + ".wav")], check=True)
        frames = len([f for f in os.listdir(d) if f.endswith(".jpg")])
        w, h = map(int, subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                        "stream=width,height", "-of", "csv=p=0", os.path.join(d, "f0001.jpg")],
                                       capture_output=True, text=True, check=True).stdout.strip().split(","))
        index[c["id"]] = {"frames": frames, "fps": FPS, "w": w, "h": h,
                          "speaker": c["speaker"], "credit": c["credit"], "quote": c["quote"]}
        print(f'{c["id"]}: {frames} frames, {w}x{h}, {c["speaker"]} ({c["credit"]})')
    with open(os.path.join(out_root, "index.json"), "w") as f:
        json.dump(index, f, indent=1)


if __name__ == "__main__":
    main()
