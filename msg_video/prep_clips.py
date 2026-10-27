"""Prepare real interview clips listed in clips/clips.json for rendering.

For each clip: trim [in, out] from clips/src/<file>, export frames at 30 fps as JPEGs
(clips/prepared/<id>/f0001.jpg ...) for the canvas renderer, and a loudness-normalized
24 kHz mono WAV (clips/prepared/<id>.wav) that build_audio.py splices into the narration.
"""
import json, os, shutil, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
FPS, WIDTH = 30, 640


def track_face(d, frames, zoom, blur=0.0):
    """Follow the speaker's face: optionally blur everything except the speaker (portrait-mode style,
    so lecture-slide text behind them is unreadable), then crop to a smoothed window around them.
    The output keeps the original frame size."""
    import cv2
    import numpy as np
    det = [cv2.CascadeClassifier(cv2.data.haarcascades + n)
           for n in ("haarcascade_frontalface_default.xml", "haarcascade_profileface.xml")]
    paths = [os.path.join(d, f"f{i:04d}.jpg") for i in range(1, frames + 1)]
    imgs = [cv2.imread(p) for p in paths]
    H, W = imgs[0].shape[:2]
    faces = []
    for im in imgs:
        g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        found = []
        for dt in det:
            for flip in (False, True):
                gg = cv2.flip(g, 1) if flip else g
                for (x, y, fw, fh) in dt.detectMultiScale(gg, 1.1, 4, minSize=(24, 24)):
                    found.append(((W - x - fw / 2) if flip else (x + fw / 2), y + fh / 2, fw))
        faces.append(max(found, key=lambda f: f[2]) if found else None)
    known = [i for i, f in enumerate(faces) if f]
    if not known:
        print("  track_face: no face found, leaving frames untouched")
        return
    k = np.ones(15) / 15                                       # ~0.5 s smoothing so the crop glides
    smooth = lambda a: np.convolve(np.pad(a, 7, mode="edge"), k, "valid")
    xs, ys, fs = (smooth(np.interp(range(frames), known, [faces[i][j] for i in known])) for j in range(3))
    cw, ch = int(W / zoom), int(H / zoom)
    yy, xx = np.mgrid[0:H, 0:W]
    for im, p, cx, cy, fw in zip(imgs, paths, xs, ys, fs):
        if blur:
            head = ((xx - cx) / (fw * 0.7)) ** 2 + ((yy - cy + fw * 0.1) / (fw * 0.95)) ** 2 <= 1
            body = ((xx - cx) / (fw * 1.55)) ** 2 + ((yy - (cy + fw * 2.3)) / (fw * 1.75)) ** 2 <= 1
            m = cv2.GaussianBlur((head | body).astype(np.float32), (0, 0), fw * 0.12)[..., None]
            bg = cv2.GaussianBlur(im, (0, 0), blur) * 0.82          # blurred and slightly dimmed
            im = (im * m + bg * (1 - m)).astype(np.uint8)
        x0 = int(min(max(cx - cw / 2, 0), W - cw))
        y0 = int(min(max(cy - ch * 0.38, 0), H - ch))            # face about 38% down, shoulders in frame
        cv2.imwrite(p, cv2.resize(im[y0:y0 + ch, x0:x0 + cw], (W, H), interpolation=cv2.INTER_CUBIC),
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"  track_face: face found in {len(known)}/{frames} frames, zoom {zoom}, background blur {blur}")


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
        if "video_hold_after" in c:
            # the source cuts away from the speaker (fade, slide): hold the last on-camera frame
            # while the audio plays on, rather than showing the cutaway
            keep = max(1, int(c["video_hold_after"] * FPS))
            for i in range(keep + 1, frames + 1):
                shutil.copyfile(os.path.join(d, f"f{keep:04d}.jpg"), os.path.join(d, f"f{i:04d}.jpg"))
        if c.get("track_face"):
            track_face(d, frames, float(c["track_face"].get("zoom", 1.6)), float(c["track_face"].get("blur", 0)))
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
