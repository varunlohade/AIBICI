#!/usr/bin/env python3
"""Mix narration + ducked music beds + SFX, mastered to ~-14 LUFS (48 kHz stereo).

Outputs:
  out/final_audio.wav       every scene, in cues.json order
  out/<part>_audio.wav      one per entry of PARTS in script.py (e.g. part1/2/3 Instagram Reels); each starts at
                            its first scene (t=0), lasts exactly the sum of its scenes' cue durations, has its own
                            music fade-in, a 1.8 s musical "button" ending and its own loudness normalisation.

Usage:
  python3 mix_audio.py                 # re-reads cues.json, script.py PARTS and out/sfx_events.json every run
  python3 mix_audio.py --refresh-sfx   # first runs `node render.js --sfx` to re-collect SFX events
  python3 mix_audio.py --events F      # use another events JSON (testing)
  python3 mix_audio.py --stems         # also writes out/stem_<name>_{narration,music,sfx}.wav (pre-master)
  python3 mix_audio.py --only part2    # render just one output ("full" or a part id)

Missing SFX / music WAVs are generated automatically (audio/make_sfx.py, audio/make_music.py).
"""
import json
import os
import runpy
import subprocess
import sys

import numpy as np
import soundfile as sf

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "audio"))
from dsp import (SR, bp2, compressor, db, fft_filter, hp2, limiter, lufs, ns, place, ramp_in, ramp_out,  # noqa: E402
                 resample, todb)

# ------------------------------------------------------------------ mix settings
MUSIC_DB = -16.0        # music loudness relative to narration loudness, in gaps
DUCK_DB = 6.0           # extra music reduction while narration is active  (=> ~-22 dB under speech)
SFX_DB = -10.0          # SFX (gain 1) peak relative to the narration peak
STING_DB = 3.0          # ending sting relative to the music gap level
BED_LUFS = -23.0        # loudness the beds are normalised to by make_music.py
XFADE = 1.5             # seconds, crossfade between music beds
PART_FADE_IN = 1.5      # extra music fade-in at the start of each part
STING_LEN = 1.8         # sting occupies the last STING_LEN seconds of a part; the bed fades out under it
TARGET_LUFS = -14.0
CEILING_DB = -2.0       # sample-peak ceiling; AAC encoding adds inter-sample overs, so this keeps true peak under -1 dBTP
MUSIC_HP = 60.0         # mobile speakers: high-pass music (24 dB/oct) ...
SFX_HP = 50.0           # ... and SFX (12 dB/oct)
NARR_HP = 70.0          # rumble filter on narration
SCRATCH_GAP = (0.06, 0.9, 1.6)  # music dropout after record_scratch: fade-out, hold, fade-back-in (s)

BED_FOR_SCENE = {"s01": "tension", "s02": "wonder", "s03": "wonder", "s04": "wonder", "s05": "science",
                 "s06": "mystery", "s07": "resolve", "s08": "resolve", "s09": "outro"}

SFX_DIR = os.path.join(ROOT, "audio", "sfx")
MUSIC_DIR = os.path.join(ROOT, "audio", "music")
OUT = os.path.join(ROOT, "out")
_warned = set()


def warn(msg):
    if msg not in _warned:
        _warned.add(msg)
        print("WARNING:", msg, file=sys.stderr)


def load(path, want_ch):
    x, sr = sf.read(path, dtype="float64", always_2d=True)
    x = x.T
    if x.shape[0] != want_ch:
        x = np.repeat(x.mean(0, keepdims=True), want_ch, axis=0)
    if sr != SR:
        x = np.stack([resample(c, sr, SR) for c in x])
    return x


def find_audio(d, name):
    for ext in (".wav", ".flac"):
        p = os.path.join(d, name + ext)
        if os.path.exists(p):
            return p
    return None


def bed_for(sid):
    bed = BED_FOR_SCENE.get(sid[:3])
    if bed is None:
        warn(f"no music bed mapped for {sid}; using 'wonder'")
        bed = "wonder"
    return bed


def ensure_music(names):
    missing = sorted({n for n in names if not find_audio(MUSIC_DIR, n)})
    if missing:
        print("generating music:", missing)
        subprocess.run([sys.executable, os.path.join(ROOT, "audio", "make_music.py"), *missing], check=True)


# ------------------------------------------------------------------ pieces
def build_narration(sids, cues, offsets, total):
    narr = np.zeros(total)
    for sid in sids:
        p = os.path.join(ROOT, "audio", "narration", sid + ".wav")
        if not os.path.exists(p):
            warn(f"missing narration {p}; leaving silence")
            continue
        x = load(p, 1)[0]
        n = ns(cues[sid]["duration"])
        if abs(len(x) - n) > SR * 0.02:
            warn(f"{sid}: narration is {len(x) / SR:.3f}s but cues.json says {n / SR:.3f}s (stale cues?) "
                 f"- following cues.json")
        place(narr, x[:n], offsets[sid])
    return fft_filter(narr, lambda f: hp2(f, NARR_HP))


def duck_envelope(narr, peak):
    """1 while narration is active (with look-ahead, hold and smooth attack/release), else 0."""
    hop = ns(0.01)
    nf = int(np.ceil(len(narr) / hop))
    p = np.zeros(nf * hop)
    p[:len(narr)] = narr ** 2
    rms = np.sqrt(p.reshape(nf, hop).mean(1))
    act = (rms > peak * db(-38)).astype(float)
    hold, look = 25, 10                      # 250 ms hold bridges word gaps, 100 ms look-ahead
    a = act.copy()
    for k in range(1, hold + 1):
        a[k:] = np.maximum(a[k:], act[:-k])
    for k in range(1, look + 1):
        a[:-k] = np.maximum(a[:-k], act[k:])
    att, rel = np.exp(-1 / 6.0), np.exp(-1 / 45.0)   # ~60 ms attack, ~450 ms release (in 10 ms frames)
    env = np.empty(nf)
    cur = 0.0
    for i, v in enumerate(a):
        c = att if v > cur else rel
        cur = c * cur + (1 - c) * v
        env[i] = cur
    return np.interp(np.arange(len(narr)), (np.arange(nf) + 0.5) * hop, env)


def loop_to(bed, n):
    """Return n samples of bed, looping its body with 2 s equal-power crossfades if it is too short."""
    if bed.shape[1] >= n:
        return bed[:, :n]
    warn(f"music bed too short ({bed.shape[1] / SR:.1f}s < {n / SR:.1f}s); looping")
    xf = ns(2.0)
    body = bed[:, ns(2.0):bed.shape[1] - ns(3.0)]
    out = bed[:, :bed.shape[1] - ns(3.0)].copy()
    fo = np.cos(0.5 * np.pi * np.arange(xf) / xf)
    fi = np.sin(0.5 * np.pi * np.arange(xf) / xf)
    while out.shape[1] < n:
        out[:, -xf:] = out[:, -xf:] * fo + body[:, :xf] * fi
        out = np.concatenate([out, body[:, xf:]], axis=1)
    return out[:, :n]


def build_music(sids, cues, offsets, total, events_abs, part, label):
    runs = []
    for sid in sids:
        bed = bed_for(sid)
        start, end = offsets[sid], offsets[sid] + ns(cues[sid]["duration"])
        if runs and runs[-1][0] == bed:
            runs[-1][2] = end
        else:
            runs.append([bed, start, end])
    ensure_music([r[0] for r in runs])
    music = np.zeros((2, total))
    h = ns(XFADE / 2)
    for i, (name, s, e) in enumerate(runs):
        first, last = i == 0, i == len(runs) - 1
        s0 = s if first else s - h
        e0 = e if last else e + h
        seg = loop_to(load(find_audio(MUSIC_DIR, name), 2), e0 - s0)
        seg = fft_filter(seg, lambda f: hp2(f, MUSIC_HP) ** 2 * (1 - 0.3 * bp2(f, 2500, 0.8)))  # + room for voice
        n = seg.shape[1]
        g = np.ones(n)
        if first and part:
            g *= ramp_in(n, ns(PART_FADE_IN))
        if not first:
            g[:2 * h] *= np.sin(0.5 * np.pi * np.arange(2 * h) / (2 * h))
        if not last:
            g[n - 2 * h:] *= np.cos(0.5 * np.pi * np.arange(2 * h) / (2 * h))
        elif part:   # bed fades out under the sting
            g *= ramp_out(n, ns(STING_LEN + 0.5))
        else:
            g *= ramp_out(n, ns(2.5))
        place(music, seg * g[None], s0)
        print(f"  [{label}] music {name:8s} {s / SR:7.2f}s -> {e / SR:7.2f}s")
    # brief dropout for the record scratch(es)
    drop = np.ones(total)
    fo, hold, fi = SCRATCH_GAP
    for t, name, _ in events_abs:
        if name == "record_scratch":
            a = ns(t + 0.04)
            seg = np.concatenate([np.cos(0.5 * np.pi * np.arange(ns(fo)) / ns(fo)) ** 2, np.zeros(ns(hold)),
                                  np.sin(0.5 * np.pi * np.arange(ns(fi)) / ns(fi)) ** 2])
            tmp = np.ones(total)
            e = min(total, a + len(seg))
            if a < total:
                tmp[a:e] = seg[:e - a]
            drop = np.minimum(drop, tmp)
    sting = None
    if part:
        nm = "sting_" + runs[-1][0]
        ensure_music([nm])
        # starts STING_LEN before the end, or just after the last narration line if that ends later
        last = sids[-1]
        lines = cues[last].get("lines") or [{"end": 0.0}]
        start = max(total - ns(STING_LEN), offsets[last] + ns(lines[-1]["end"] + 0.05))
        start = min(start, total - ns(0.8))
        st = load(find_audio(MUSIC_DIR, nm), 2)[:, :total - start]
        st = st * ramp_out(st.shape[1], ns(0.25))[None]
        sting = np.zeros((2, total))
        place(sting, fft_filter(st, lambda f: hp2(f, MUSIC_HP) ** 2), start)
    return music * drop[None], sting


def read_events(refresh, path=None):
    if path:
        with open(path) as f:
            return json.load(f)
    path = os.path.join(OUT, "sfx_events.json")
    if refresh or not os.path.exists(path):
        print("running: node render.js --sfx")
        subprocess.run(["node", os.path.join(ROOT, "render.js"), "--sfx"], cwd=ROOT, check=True)
    with open(path) as f:
        return json.load(f)


def build_sfx(sids, ev, cues, offsets, total, level):
    from make_sfx import ALLOWED  # noqa: E402
    allowed = {os.path.splitext(f)[0] for f in os.listdir(SFX_DIR)} if os.path.isdir(SFX_DIR) else set()
    names = {e.get("name") for sid in sids for e in ev.get(sid, [])}
    need = sorted((names & set(ALLOWED)) - allowed)
    if need:
        print("generating sfx:", need)
        subprocess.run([sys.executable, os.path.join(ROOT, "audio", "make_sfx.py"), *need], check=True)
        allowed |= set(need)
    cache = {}
    sfx = np.zeros(total)
    placed, events_abs = 0, []
    for sid in sids:
        for e in ev.get(sid, []):
            name = e.get("name")
            t = float(e.get("t", 0))
            gain = e.get("gain")
            gain = 1.0 if gain is None else float(gain)
            if name not in allowed:
                warn(f"{sid} t={t:.2f}: unknown sfx '{name}', skipped")
                continue
            if t < 0 or t > cues[sid]["duration"]:
                warn(f"{sid} t={t:.2f}: '{name}' outside scene (0..{cues[sid]['duration']}); clamped")
                t = min(max(t, 0.0), cues[sid]["duration"])
            if name not in cache:
                cache[name] = load(os.path.join(SFX_DIR, name + ".wav"), 1)[0]
            place(sfx, cache[name], offsets[sid] + ns(t), gain * level)
            placed += 1
            events_abs.append(((offsets[sid] + ns(t)) / SR, name, gain))
    return fft_filter(sfx, lambda f: hp2(f, SFX_HP)), placed, events_abs


# ------------------------------------------------------------------ one output
def render(label, sids, cues, ev, part, stems):
    offsets, acc = {}, 0.0
    for sid in sids:
        offsets[sid] = ns(acc)
        acc += cues[sid]["duration"]
    total = ns(acc)

    narr = build_narration(sids, cues, offsets, total)
    npk = np.abs(narr).max()
    nl = lufs(np.stack([narr, narr]))
    sfx, placed, events_abs = build_sfx(sids, ev, cues, offsets, total, npk * db(SFX_DB) / db(-3.0))
    music, sting = build_music(sids, cues, offsets, total, events_abs, part, label)
    env = duck_envelope(narr, npk)
    mgain = db(nl + MUSIC_DB - BED_LUFS)
    music *= mgain * db(-DUCK_DB * env)[None]
    if sting is not None:
        music += sting * mgain * db(STING_DB)

    mix = np.stack([narr, narr]) + music + sfx[None]
    if stems:
        for nm, x in (("narration", narr), ("music", music), ("sfx", sfx)):
            sf.write(os.path.join(OUT, f"stem_{label}_{nm}.wav"), np.atleast_2d(x).T, SR, subtype="PCM_24")

    # ---- master: level, gentle glue compression, loudness, limiting
    mix *= db(-18.0 - lufs(mix))
    mix = compressor(mix, thresh_db=-21.0, ratio=2.0, attack=0.015, release=0.2)
    for _ in range(4):
        mix *= db(TARGET_LUFS - lufs(mix))
        mix = limiter(mix, CEILING_DB, release=0.08)
        if abs(lufs(mix) - TARGET_LUFS) < 0.2:
            break
    mix *= (ramp_in(total, ns(0.01)) * ramp_out(total, ns(0.05 if part else 0.3)))[None]
    assert np.isfinite(mix).all(), "NaN/inf in mix"
    return mix, placed, acc


def load_parts():
    try:
        return runpy.run_path(os.path.join(ROOT, "script.py"), run_name="script_parts").get("PARTS", [])
    except Exception as e:  # noqa: BLE001
        warn(f"could not read PARTS from script.py ({e}); rendering the full mix only")
        return []


def main(argv):
    with open(os.path.join(ROOT, "cues.json")) as f:
        cues = json.load(f)
    ev_path = argv[argv.index("--events") + 1] if "--events" in argv else None
    only = argv[argv.index("--only") + 1] if "--only" in argv else None
    ev = read_events("--refresh-sfx" in argv, ev_path)
    for sid in ev:
        if sid not in cues:
            warn(f"sfx_events has unknown scene {sid}; ignored")

    jobs = [("full", "final_audio.wav", list(cues), False)]
    used = set()
    for p in load_parts():
        sids = [s for s in p["scenes"] if s in cues]
        for s in p["scenes"]:
            if s not in cues:
                warn(f"{p['id']}: scene {s} not in cues.json; skipped")
        used |= set(sids)
        if sids:
            jobs.append((p["id"], f"{p['id']}_audio.wav", sids, True))
    if len(jobs) > 1 and set(cues) - used:
        warn(f"scenes not in any part: {sorted(set(cues) - used)}")

    os.makedirs(OUT, exist_ok=True)
    for label, fname, sids, part in jobs:
        if only and label != only:
            continue
        mix, placed, dur = render(label, sids, cues, ev, part, "--stems" in argv)
        out = os.path.join(OUT, fname)
        sf.write(out, mix.T, SR, subtype="PCM_16")
        print(f"wrote {out}: {mix.shape[1] / SR:.3f}s (cues {dur:.3f}s) | {' '.join(sids)} | {placed} sfx | "
              f"{lufs(mix):.1f} LUFS, peak {todb(np.abs(mix).max()):.2f} dBFS")


if __name__ == "__main__":
    main(sys.argv[1:])
