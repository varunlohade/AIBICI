"""Technical audit of a finished Reel: format, length, loudness, blank or black frames,
silent gaps and the loop seam. Usage: python3 audit_tech.py [out/msg_reel.mp4]. Exits 1 on any FAIL."""
import json, re, subprocess, sys

import numpy as np

f = sys.argv[1] if len(sys.argv) > 1 else "out/msg_reel.mp4"
results = []


def check(name, ok, detail):
    results.append((name, "PASS" if ok else "FAIL", detail))


def run(*a):
    return subprocess.run(a, capture_output=True, text=True).stderr + subprocess.run(a, capture_output=True, text=True).stdout


probe = json.loads(subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", f],
                                  capture_output=True, text=True).stdout)
v = next(s for s in probe["streams"] if s["codec_type"] == "video")
a = next(s for s in probe["streams"] if s["codec_type"] == "audio")
dur = float(probe["format"]["duration"])
check("Length under 3:00 (Instagram Reel limit)", dur < 180, f"{dur:.1f} s")
check("Resolution 1080x1920 (9:16)", (v["width"], v["height"]) == (1080, 1920), f'{v["width"]}x{v["height"]}')
check("Frame rate 30 fps", v["r_frame_rate"] == "30/1", v["r_frame_rate"])
check("Codecs H.264 + AAC, yuv420p", v["codec_name"] == "h264" and a["codec_name"] == "aac" and v["pix_fmt"] == "yuv420p",
      f'{v["codec_name"]}/{v["pix_fmt"]} + {a["codec_name"]}')
check("Audio and video lengths match", abs(float(v["duration"]) - float(a["duration"])) < 0.1,
      f'video {float(v["duration"]):.2f} s, audio {float(a["duration"]):.2f} s')
size = int(probe["format"]["size"]) / 1e6
check("File size reasonable for upload", size < 250, f"{size:.0f} MB")

loud = subprocess.run(["ffmpeg", "-nostats", "-i", f, "-af", "ebur128=peak=true", "-f", "null", "-"],
                      capture_output=True, text=True).stderr
I = float(re.findall(r"I:\s+(-?[\d.]+) LUFS", loud)[-1])
tp = float(re.findall(r"Peak:\s+(-?[\d.]+) dBFS", loud)[-1])
check("Integrated loudness about -14 LUFS (Instagram/Reels norm)", -16 <= I <= -12, f"{I:.1f} LUFS")
check("True peak below -1 dBTP", tp <= -1.0, f"{tp:.1f} dBTP")

sil = subprocess.run(["ffmpeg", "-nostats", "-i", f, "-af", "silencedetect=n=-45dB:d=1.5", "-f", "null", "-"],
                     capture_output=True, text=True).stderr
gaps = re.findall(r"silence_duration: ([\d.]+)", sil)
check("No silent gaps over 1.5 s", not gaps, f"{len(gaps)} gaps" + (f": {gaps}" if gaps else ""))

blk = subprocess.run(["ffmpeg", "-nostats", "-i", f, "-vf", "blackdetect=d=0.05:pix_th=0.08", "-an", "-f", "null", "-"],
                     capture_output=True, text=True).stderr
blacks = re.findall(r"black_start:([\d.]+) black_end:([\d.]+)", blk)
check("No black frames", not blacks, f"{len(blacks)} black runs" + (f": {blacks[:5]}" if blacks else ""))

# blank frames: sample 10 fps at low res; flag frames where one flat colour covers > 92% of the picture
W, H = 54, 96
raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", f, "-vf", f"fps=10,scale={W}:{H}", "-f", "rawvideo",
                      "-pix_fmt", "rgb24", "-"], capture_output=True).stdout
frames = np.frombuffer(raw, np.uint8).reshape(-1, H, W, 3)
blank = []
for i, fr in enumerate(frames):
    q = (fr // 24).reshape(-1, 3)
    _, counts = np.unique(q, axis=0, return_counts=True)
    if counts.max() / len(q) > 0.92:
        blank.append(round(i / 10, 1))
check("No blank single-colour frames (sampled at 10 fps)", not blank, f"{len(blank)} flagged" + (f" at {blank[:10]} s" if blank else ""))

first, last = frames[0].astype(int), frames[-1].astype(int)
diff = np.abs(first - last).mean()
check("Seamless loop (last frame matches first)", diff < 6, f"mean pixel difference {diff:.1f}/255 at 54x96")

w = max(len(r[0]) for r in results)
for name, status, detail in results:
    print(f"{status}  {name:<{w}}  {detail}")
sys.exit(any(s == "FAIL" for _, s, _ in results))
