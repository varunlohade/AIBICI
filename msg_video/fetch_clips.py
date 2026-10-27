"""Download the source segments for the real interview clips listed in clips/clips.json.

Each clip entry may have "url" plus "in"/"out" (seconds in the ORIGINAL video). Only that segment
(with 1 s of padding) is downloaded, to clips/src/<id>.mp4, and the entry's "file"/"in"/"out" are
rewritten to point at the local file. Then run prep_clips.py. Needs network access to the video host
(for YouTube: youtube.com, *.googlevideo.com, youtubei.googleapis.com, i.ytimg.com).
"""
import json, os, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
PAD = 1.0


def main():
    path = os.path.join(HERE, "clips", "clips.json")
    spec = json.load(open(path))
    os.makedirs(os.path.join(HERE, "clips", "src"), exist_ok=True)
    for c in spec["clips"]:
        if not c.get("url") or c.get("file", "").startswith(c["id"] + "."):
            continue
        start, end = max(0.0, c["in"] - PAD), c["out"] + PAD
        out = os.path.join(HERE, "clips", "src", c["id"] + ".mp4")
        subprocess.run(["yt-dlp", "-q", "--no-playlist", "-f", "bv*[height<=1080]+ba/b[height<=1080]/b",
                        "--merge-output-format", "mp4", "--download-sections", f"*{start}-{end}",
                        "--force-keyframes-at-cuts", "-o", out, c["url"]], check=True)
        c.update({"source_in": c["in"], "source_out": c["out"], "file": c["id"] + ".mp4",
                  "in": round(c["in"] - start, 3), "out": round(c["out"] - start, 3)})
        print(f'{c["id"]}: downloaded {c["url"]} [{start}-{end}] -> clips/src/{c["id"]}.mp4')
    json.dump(spec, open(path, "w"), indent=1)


if __name__ == "__main__":
    main()
