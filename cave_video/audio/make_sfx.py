#!/usr/bin/env python3
"""Procedurally synthesize every SFX allowed by BRIEF.md into audio/sfx/<name>.wav (48 kHz mono, -3 dBFS peak).

Usage: python3 audio/make_sfx.py [name ...]      (no names = all)
Deterministic: same output every run.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dsp import (SR, bell_curve, bp2, db, dc_block, env_ad, env_asr, fft_filter, freeverb_ir, convolve, hp1, hp2,
                 lin, logsweep, lp1, lp2, mtof, ns, ramp_in, ramp_out, rng, saw, sine, stats, stft_filter, tri, tt,
                 write, place, phase)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sfx")
SFX = {}


def sfx(fn):
    SFX[fn.__name__] = fn
    return fn


# ------------------------------------------------------------- helpers
def white(n, seed):
    return rng(seed).standard_normal(n)


def nb(n, fc, q, seed):
    """Band-passed noise (static)."""
    return fft_filter(white(n, seed), lambda f: bp2(f, fc, q))


def sweep(x, fc_t, q=1.0, kind="bp", nfft=512):
    """Time-varying resonant filter. fc_t(t) gives cutoff/centre in Hz for times t (seconds)."""
    resp = {"bp": bp2, "lp": lp2, "hp": hp2}[kind]
    return stft_filter(x, lambda t, f: resp(f, fc_t(t), q), nfft=nfft)


def ping(freq, dur, decay, attack=0.002, partials=((1, 1.0),), ph=0.0):
    n = ns(dur)
    y = np.zeros(n)
    for r, a in partials:
        y += a * sine(np.asarray(freq) * r, n, ph)
    return y * env_ad(dur, attack, decay)


def bell(freq, dur=1.2, decay=0.45, bright=1.0):
    """Soft UI bell: near-harmonic partials, the upper ones dying faster."""
    n = ns(dur)
    t = tt(dur)
    y = np.zeros(n)
    for r, a, d in ((1, 1.0, 1.0), (2.0, 0.35 * bright, 0.55), (3.01, 0.12 * bright, 0.35), (4.2, 0.08 * bright, 0.2),
                    (5.43, 0.04 * bright, 0.12)):
        y += a * np.sin(2 * np.pi * freq * r * t) * np.exp(-t / (decay * d))
    return y * ramp_in(n, ns(0.002))


def small_room(x, wet=0.25, seconds=0.9):
    ir = freeverb_ir(seconds=seconds, room=0.7, damp=0.5, stereo=False, predelay=0.008)[0]
    w = convolve(x, ir)[0]
    out = np.zeros(len(w))
    out[:len(x)] += x
    return out + wet * w * np.sqrt(np.mean(x ** 2)) / max(np.sqrt(np.mean(w ** 2)), 1e-9)


def mix(n, *parts):
    """parts: (signal, start_seconds, gain)."""
    out = np.zeros(n)
    for s, t0, g in parts:
        place(out, s, ns(t0), g)
    return out


def soft_clip(x, drive=1.5):
    return np.tanh(x * drive) / np.tanh(drive)


def crackle(n, rate, seed, fc=3000, q=0.8, decay=0.0015):
    """Sparse random filtered clicks (vinyl/neon/sizzle crackle)."""
    r = rng(seed)
    imp = np.zeros(n)
    k = r.poisson(rate * n / SR)
    pos = r.integers(0, n, k)
    imp[pos] = r.uniform(-1, 1, k) * r.uniform(0.2, 1, k) ** 2
    kern = np.exp(-np.arange(ns(decay * 6)) / SR / decay)
    y = np.convolve(imp, kern)[:n]
    return fft_filter(y, lambda f: bp2(f, fc, q))


# ------------------------------------------------------------- whooshes
def _whoosh(dur, f0, fpk, f1, peak, q, seed, tonal=0.0):
    n = ns(dur)
    x = white(n, seed)
    fc = lambda t: np.interp(t / dur, [0, peak, 1], [f0, fpk, f1])
    a = sweep(x, fc, q)
    b = sweep(white(n, seed + 1), lambda t: fc(t) * 0.5, q * 0.8)
    y = a + 0.7 * b
    if tonal:
        y += tonal * sweep(white(n, seed + 2), fc, 12.0)
    y = fft_filter(y, lambda f: lp1(f, 7000) * hp1(f, 120))
    return y * bell_curve(n, peak, 1.6)


@sfx
def whoosh():
    return _whoosh(0.75, 350, 2200, 500, 0.5, 1.3, 11, tonal=0.25)


@sfx
def whoosh_long():
    return _whoosh(1.7, 220, 1900, 300, 0.62, 1.1, 12, tonal=0.35)


@sfx
def swoosh_up():
    dur = 0.65
    n = ns(dur)
    y = _whoosh(dur, 300, 4500, 5000, 0.85, 1.4, 13, tonal=0.3)
    f = logsweep(n, 220, 1100, 1.3)
    y += 0.12 * np.sin(phase(f, n)) * bell_curve(n, 0.85, 1.5)
    return y * ramp_out(n, ns(0.05))


@sfx
def swoosh_down():
    dur = 0.65
    n = ns(dur)
    y = _whoosh(dur, 5000, 4200, 280, 0.18, 1.4, 14, tonal=0.3)
    f = logsweep(n, 1100, 180, 0.8)
    y += 0.12 * np.sin(phase(f, n)) * bell_curve(n, 0.2, 1.5)
    return y


@sfx
def riser():
    dur = 2.6
    n = ns(dur)
    t = tt(dur)
    x = sweep(white(n, 15), lambda tt_: 200 * (35 ** (tt_ / dur) ** 1.4), 1.6)
    f = logsweep(n, 110, 880, 1.6)
    trem = 0.75 + 0.25 * np.sin(2 * np.pi * np.cumsum(3 + 14 * (t / dur) ** 2) / SR)
    tone = (saw(f, n) * 0.5 + saw(f * 1.006, n) * 0.5)
    tone = sweep(tone, lambda tt_: 400 + 5000 * (tt_ / dur) ** 2, 0.9, "lp", 1024)
    y = (x + 0.35 * tone) * trem * (t / dur) ** 2.2
    return y * ramp_out(n, ns(0.03))


# ------------------------------------------------------------- bubbly UI
@sfx
def pop():
    dur = 0.22
    n = ns(dur)
    f = 380 + 600 * (1 - np.exp(-tt(dur) / 0.018))
    y = np.sin(phase(f, n)) * env_ad(dur, 0.002, 0.045)
    y += 0.25 * np.sin(phase(f * 2, n)) * env_ad(dur, 0.001, 0.018)
    y += 0.15 * nb(n, 2500, 1.0, 21) * env_ad(dur, 0.0005, 0.004)
    return y


@sfx
def pop_small():
    dur = 0.14
    n = ns(dur)
    f = 750 + 900 * (1 - np.exp(-tt(dur) / 0.012))
    y = np.sin(phase(f, n)) * env_ad(dur, 0.0015, 0.03)
    y += 0.15 * nb(n, 4000, 1.0, 22) * env_ad(dur, 0.0005, 0.003)
    return y


def _bubble(f0, f1, dur, decay, seed=0):
    n = ns(dur)
    t = tt(dur)
    f = f0 * (f1 / f0) ** np.clip(t / (dur * 0.6), 0, 1)
    return np.sin(phase(f, n)) * env_ad(dur, 0.003, decay)


@sfx
def bubble():
    n = ns(0.35)
    return mix(n, (_bubble(320, 900, 0.18, 0.05), 0, 1.0), (_bubble(480, 1300, 0.14, 0.035), 0.09, 0.55))


@sfx
def blip():
    n = ns(0.16)
    a = ping(988, 0.07, 0.025, 0.002, ((1, 1), (3, 0.08)))
    b = ping(1480, 0.09, 0.03, 0.002, ((1, 1), (3, 0.08)))
    return mix(n, (a, 0, 0.8), (b, 0.055, 1.0))


@sfx
def click():
    dur = 0.05
    n = ns(dur)
    y = nb(n, 3500, 1.5, 31) * env_ad(dur, 0.0003, 0.0018)
    y += 0.5 * ping(1900, dur, 0.006, 0.0005)
    y += 0.3 * ping(600, dur, 0.008, 0.001)
    return y


@sfx
def tick():
    dur = 0.06
    n = ns(dur)
    y = nb(n, 4200, 3.0, 32) * env_ad(dur, 0.0002, 0.0015)
    y += 0.6 * ping(2350, dur, 0.007, 0.0004, ((1, 1), (2.7, 0.3)))
    return small_room(y, 0.12, 0.4)[:ns(0.12)]


@sfx
def ding():
    y = bell(1568, 1.4, 0.5)
    return small_room(y, 0.22)[:ns(1.6)]


@sfx
def chime():
    n = ns(1.8)
    notes = [72, 76, 79, 84]
    y = mix(n, *[(bell(mtof(m + 12), 1.4, 0.45, 0.8), i * 0.075, 0.8 - 0.08 * i) for i, m in enumerate(notes)])
    return small_room(y, 0.3)[:n]


@sfx
def sparkle():
    dur = 1.1
    n = ns(dur)
    r = rng(41)
    y = np.zeros(n)
    scale = [0, 2, 4, 7, 9]
    t = 0.0
    while t < 0.75:
        m = 84 + r.choice(scale) + 12 * r.integers(0, 2)
        g = 0.4 + 0.6 * (1 - t / 0.8)
        place(y, ping(mtof(m), 0.35, 0.06 + 0.04 * r.random(), 0.001, ((1, 1), (2.01, 0.2))), ns(t), g * r.uniform(0.5, 1))
        t += r.exponential(0.035) + 0.012
    y += 0.12 * sweep(white(n, 42), lambda tt_: 6000 + 0 * tt_, 2.0) * bell_curve(n, 0.2, 2)
    return small_room(y, 0.35)[:n]


@sfx
def success():
    n = ns(1.3)
    notes = [72, 76, 79, 84]
    parts = []
    for i, m in enumerate(notes):
        p = ping(mtof(m), 0.9, 0.18 if i < 3 else 0.45, 0.003, ((1, 1), (2, 0.25), (3, 0.08)))
        parts.append((p, i * 0.085, 0.7 + 0.1 * i))
    y = mix(n, *parts)
    return small_room(y, 0.25)[:n]


@sfx
def error():
    n = ns(0.5)

    def buzz(m, d):
        k = ns(d)
        x = saw(mtof(m), k) * 0.5 + saw(mtof(m) * 1.01, k) * 0.5
        x = fft_filter(x, lambda f: lp2(f, 1400, 0.9))
        return x * env_asr(k, 0.006, 0.05)

    return mix(n, (buzz(62, 0.16), 0, 1.0), (buzz(57, 0.24), 0.17, 1.0))


@sfx
def alarm():
    """Gentle two-tone alarm (soft, rounded square-ish beeps)."""
    n = ns(1.6)
    parts = []
    for i in range(6):
        f = 880 if i % 2 == 0 else 698
        k = ns(0.2)
        x = np.sin(phase(f, k)) + 0.25 * np.sin(phase(3 * f, k)) + 0.1 * np.sin(phase(5 * f, k))
        x *= env_asr(k, 0.01, 0.06)
        parts.append((x, i * 0.24, 1.0))
    return small_room(mix(n, *parts), 0.2)[:n] * ramp_out(n, ns(0.05))


@sfx
def boing():
    dur = 0.8
    n = ns(dur)
    t = tt(dur)
    f = 180 * (1 + 0.9 * (1 - np.exp(-t / 0.25))) * (1 + 0.35 * np.exp(-t / 0.18) * np.sin(2 * np.pi * 14 * t))
    y = np.sin(phase(f, n)) + 0.3 * tri(f * 2, n)
    y = fft_filter(y, lambda f_: lp2(f_, 2500))
    y *= env_ad(dur, 0.004, 0.22)
    y += 0.4 * ping(140, dur, 0.05, 0.002)
    return y


# ------------------------------------------------------------- impacts
def _thump(f0, f1, dur, decay, glide=0.04):
    n = ns(dur)
    t = tt(dur)
    f = f1 + (f0 - f1) * np.exp(-t / glide)
    return np.sin(phase(f, n)) * env_ad(dur, 0.002, decay)


@sfx
def impact_soft():
    dur = 0.9
    n = ns(dur)
    y = _thump(150, 72, dur, 0.18)
    y += 0.45 * fft_filter(white(n, 51), lambda f: lp2(f, 600)) * env_ad(dur, 0.001, 0.03)
    y += 0.25 * _thump(240, 150, dur, 0.04)
    return small_room(soft_clip(y, 3.0), 0.15)[:n]


@sfx
def impact_big():
    dur = 2.4
    n = ns(dur)
    y = _thump(140, 68, dur, 0.5, 0.06)
    y += 0.6 * sweep(white(n, 52), lambda t: 300 + 2500 * np.exp(-t / 0.08), 0.8, "lp") * env_ad(dur, 0.001, 0.12)
    y += 0.3 * _thump(180, 90, dur, 0.09)
    y = soft_clip(y, 2.4)
    ir = freeverb_ir(seconds=2.2, room=0.88, damp=0.5, stereo=False)[0]
    w = convolve(y, ir)[0][:n]
    w = fft_filter(w, lambda f: lp1(f, 2500))
    return y + 0.35 * w * np.sqrt(np.mean(y ** 2)) / np.sqrt(np.mean(w ** 2))


@sfx
def thud():
    dur = 0.45
    n = ns(dur)
    y = _thump(170, 85, dur, 0.09, 0.025)
    y += 0.3 * fft_filter(white(n, 53), lambda f: lp2(f, 400)) * env_ad(dur, 0.001, 0.02)
    return soft_clip(y, 2.0)


@sfx
def stamp():
    dur = 0.5
    n = ns(dur)
    y = 0.9 * _thump(200, 100, dur, 0.06, 0.02)
    y += 1.4 * nb(n, 1000, 0.7, 54) * env_ad(dur, 0.0005, 0.025)
    y += 0.25 * nb(n, 3200, 2.0, 55) * env_ad(dur, 0.0005, 0.006)
    rattle = nb(n, 2200, 4.0, 56) * env_ad(dur, 0.002, 0.05) * (0.5 + 0.5 * np.sin(2 * np.pi * 70 * tt(dur)))
    place(y, 0.12 * rattle, ns(0.012))
    return small_room(soft_clip(y, 1.4), 0.12)[:n]


@sfx
def heartbeat():
    n = ns(1.0)
    lub = _thump(100, 62, 0.4, 0.07, 0.03)
    dub = _thump(110, 68, 0.4, 0.06, 0.03)
    y = mix(n, (lub, 0, 1.0), (dub, 0.27, 0.75))
    return fft_filter(soft_clip(y, 2.2), lambda f: lp2(f, 450))


# ------------------------------------------------------------- vintage / electrical
def _vinyl_source(seconds, seed=61):
    """A chord-y 'record' to scratch: tones + noise, as a function we can read at arbitrary positions."""
    n = ns(seconds)
    t = tt(seconds)
    src = np.zeros(n)
    for m in (57, 60, 64, 69, 45):
        src += saw(mtof(m), n) * 0.2
    src = fft_filter(src, lambda f: lp2(f, 2500))
    src += 0.4 * fft_filter(white(n, seed), lambda f: bp2(f, 1500, 0.7)) * (0.6 + 0.4 * np.sin(2 * np.pi * 4 * t))
    return src


@sfx
def record_scratch():
    dur = 0.75
    n = ns(dur)
    t = tt(dur)
    src = _vinyl_source(3.0)
    # platter velocity (x normal speed): yank back, push forward, yank back, then stop
    v = lin(t, [(0, 1.0), (0.04, -3.2), (0.17, -2.6), (0.24, 2.8), (0.38, 3.2), (0.46, -2.4), (0.58, -1.6),
                (0.66, 0.0), (dur, 0.0)])
    pos = 1.5 * SR + np.cumsum(v)
    y = np.interp(pos, np.arange(len(src)), src)
    # friction 'zip' emphasises the fast parts
    speed = np.abs(v)
    y = y * (0.3 + 0.7 * np.clip(speed / 3, 0, 1))
    y += 0.25 * crackle(n, 300, 62, 2500) + 0.05 * nb(n, 4500, 1.2, 63) * np.clip(speed / 3, 0, 1)
    y = fft_filter(y, lambda f: hp2(f, 150) * lp1(f, 6000))
    return y * env_asr(n, 0.003, 0.08)


def _crt_whine(n, seed=0):
    t = np.arange(n) / SR
    return np.sin(2 * np.pi * 15734 * t) * 0.6 + np.sin(2 * np.pi * 7867 * t) * 0.25


@sfx
def tv_on():
    dur = 1.3
    n = ns(dur)
    t = tt(dur)
    relay = nb(n, 2500, 1.2, 71) * env_ad(dur, 0.0003, 0.003)
    thump = soft_clip(_thump(120, 65, dur, 0.12, 0.03), 2.0)
    # degauss 'bwoom': 60 Hz family with a quick wobble, decaying
    hum = sum(a * np.sin(2 * np.pi * h * 60 * t) for h, a in ((1, 1), (2, 0.6), (3, 0.3), (5, 0.15)))
    hum *= env_ad(dur, 0.02, 0.25) * (1 + 0.3 * np.sin(2 * np.pi * 9 * t))
    stat = fft_filter(white(n, 72), lambda f: hp2(f, 900)) * np.exp(-t / 0.18) * ramp_in(n, ns(0.03))
    whine = _crt_whine(n) * ramp_in(n, ns(0.35)) * env_asr(n, 0.001, 0.25)
    y = 0.9 * relay + 0.9 * thump + 0.35 * hum + 0.22 * stat + 0.05 * whine
    return y * ramp_out(n, ns(0.05))


@sfx
def tv_off():
    dur = 0.9
    n = ns(dur)
    t = tt(dur)
    f = logsweep(n, 2600, 90, 0.5)
    pew = np.sin(phase(f, n)) * env_ad(dur, 0.002, 0.12)
    thump = soft_clip(_thump(110, 60, dur, 0.1, 0.02), 2.0)
    whine = _crt_whine(n) * (t < 0.06) * ramp_in(n, 1)
    whine *= ramp_out(n, 1)
    whine[:ns(0.06)] *= ramp_out(ns(0.06), ns(0.01))
    cr = crackle(n, 220, 73, 3000) * np.exp(-t / 0.15)
    y = 0.5 * pew + 0.8 * thump + 0.05 * whine + 0.25 * cr
    place(y, 0.6 * nb(ns(0.05), 2200, 1.0, 74) * env_ad(0.05, 0.0003, 0.004), 0)
    return y


@sfx
def static():
    dur = 1.0
    n = ns(dur)
    t = tt(dur)
    r = rng(81)
    x = fft_filter(white(n, 82), lambda f: hp2(f, 350) * lp2(f, 9000))
    # grainy amplitude jitter (frame-rate-ish flutter + random grains)
    steps = np.repeat(r.uniform(0.45, 1, int(np.ceil(n / ns(0.012)))), ns(0.012))[:n]
    steps = fft_filter(steps, lambda f: lp1(f, 200))
    y = x * steps * (1 + 0.15 * np.sin(2 * np.pi * 59.94 * t)) + 0.3 * crackle(n, 400, 83, 3500, 0.6)
    return y * env_asr(n, 0.02, 0.12)


@sfx
def channel_switch():
    dur = 0.5
    n = ns(dur)
    t = tt(dur)
    r = rng(91)
    burst = fft_filter(white(n, 92), lambda f: hp2(f, 600) * lp2(f, 8000))
    gates = np.repeat(r.uniform(0.2, 1, int(np.ceil(n / ns(0.02)))), ns(0.02))[:n]
    burst *= fft_filter(gates, lambda f: lp1(f, 300)) * env_asr(n, 0.003, 0.18) * (t < 0.36)
    tone = np.sin(2 * np.pi * 1050 * t) * ((t > 0.05) & (t < 0.11)) * 0.25
    clk = nb(n, 2000, 1.0, 93) * env_ad(dur, 0.0003, 0.003)
    y = 0.7 * clk + 0.6 * burst + tone
    place(y, 0.6 * nb(ns(0.04), 1800, 1.0, 94) * env_ad(0.04, 0.0003, 0.003), ns(0.33))
    return fft_filter(y, lambda f: lp1(f, 12000)) * ramp_out(n, ns(0.06))


@sfx
def glitch():
    dur = 0.55
    n = ns(dur)
    r = rng(101)
    y = np.zeros(n)
    pos = 0
    while pos < n - ns(0.03):
        seg = ns(r.uniform(0.012, 0.05))
        kind = r.integers(0, 4)
        if kind == 0:   # square blip
            s = np.sign(np.sin(phase(r.uniform(300, 2500), seg))) * 0.5
        elif kind == 1:  # crushed noise
            s = np.round(white(seg, 102 + pos) * 2) / 4
        elif kind == 2:  # stutter of the previous chunk
            prev = y[max(0, pos - seg):pos]
            s = np.tile(prev if len(prev) else np.zeros(1), 3)[:seg] * 0.9
        else:           # downward zap
            s = saw(logsweep(seg, r.uniform(1500, 4000), 200), seg) * 0.5
        s = s * env_asr(len(s), 0.001, 0.002)
        y[pos:pos + len(s)] += s
        pos += seg + (ns(r.uniform(0, 0.02)) if r.random() < 0.4 else 0)
    y = np.round(y * 12) / 12          # bit-crush
    y = fft_filter(y, lambda f: lp2(f, 9000) * hp1(f, 120))
    return y * env_asr(n, 0.002, 0.04)


def _neon(dur, seed, flicker_until=0.35):
    n = ns(dur)
    t = tt(dur)
    r = rng(seed)
    # mains buzz: 60 Hz fundamental with strong even (120 Hz) component and rich harmonics
    hum = np.zeros(n)
    for h in range(1, 24):
        a = (1.0 if h == 2 else 0.5) / h ** 0.8 * (1.3 if h % 2 == 0 else 1.0)
        hum += a * np.sin(2 * np.pi * 60 * h * t + r.uniform(0, 6.28))
    hum = fft_filter(hum, lambda f: bp2(f, 900, 0.5) + 0.12 * lp2(f, 150))
    hum = soft_clip(hum / np.abs(hum).max(), 2.0)
    # flicker gating at ignition
    g = np.ones(n)
    tpos = 0.0
    while tpos < flicker_until:
        on = r.uniform(0.015, 0.06)
        off = r.uniform(0.02, 0.07)
        g[ns(tpos + on):ns(tpos + on + off)] = 0.05
        tpos += on + off
    g = fft_filter(g, lambda f: lp1(f, 400))
    zap = crackle(n, 30, seed + 1, 3500, 0.7, 0.002) * (0.4 + 2.0 * (t < flicker_until + 0.05))
    return hum, g, zap


@sfx
def neon_buzz():
    dur = 1.6
    n = ns(dur)
    hum, g, zap = _neon(dur, 111)
    y = 0.7 * hum * g + 0.35 * zap
    place(y, 0.6 * nb(ns(0.03), 2800, 1.0, 112) * env_ad(0.03, 0.0003, 0.004), 0)
    return y * env_asr(n, 0.003, 0.25)


@sfx
def neon_flicker():
    dur = 0.75
    n = ns(dur)
    hum, g, zap = _neon(dur, 121, flicker_until=0.7)
    y = 0.7 * hum * g + 0.45 * zap
    return y * env_asr(n, 0.003, 0.06)


# ------------------------------------------------------------- foley-ish
def _key(seed, strong=1.0):
    r = rng(seed)
    d = 0.09
    n = ns(d)
    hit = nb(n, r.uniform(2500, 3800), 1.2, seed) * env_ad(d, 0.0004, 0.005)
    body = nb(n, r.uniform(450, 700), 1.2, seed + 1) * env_ad(d, 0.0008, 0.02)
    ring = ping(r.uniform(3000, 3400), d, 0.02, 0.0005) * 0.12
    bottom = nb(n, 1200, 1.0, seed + 2) * env_ad(d, 0.0003, 0.004)
    y = strong * soft_clip(1.0 * hit + 1.2 * body + ring, 2.0)
    place(y, 0.45 * bottom, ns(0.028))  # key bottoming out / typebar return
    return y


@sfx
def typewriter():
    dur = 1.25
    n = ns(dur)
    r = rng(131)
    y = np.zeros(n)
    t = 0.01
    i = 0
    while t < dur - 0.12:
        place(y, _key(140 + i, r.uniform(0.6, 1.0)), ns(t))
        t += r.uniform(0.07, 0.16) + (0.12 if r.random() < 0.12 else 0)
        i += 1
    return small_room(y, 0.12, 0.5)[:n]


@sfx
def paper():
    dur = 0.65
    n = ns(dur)
    t = tt(dur)
    base = sweep(white(n, 151), lambda tt_: 1800 + 2500 * np.sin(np.pi * tt_ / dur), 0.6)
    am = np.abs(fft_filter(rng(152).standard_normal(n), lambda f: lp1(f, 35)))
    am /= am.max()
    cr = crackle(n, 260, 153, 3500, 0.7, 0.0012) + 0.6 * crackle(n, 180, 154, 1800, 0.9, 0.002)
    y = 0.6 * base * am + 0.8 * cr * (0.3 + am)
    return fft_filter(y, lambda f: hp2(f, 400) * lp2(f, 6500)) * env_asr(n, 0.02, 0.12) * bell_curve(n, 0.4, 0.5)


@sfx
def crunch():
    dur = 0.4
    n = ns(dur)
    t = tt(dur)
    dens = np.exp(-t / 0.08)
    cr = crackle(n, 1500, 161, 2600, 0.6, 0.0012) * dens
    cr += 0.7 * crackle(n, 700, 162, 1200, 0.8, 0.002) * dens
    body = fft_filter(white(n, 163), lambda f: bp2(f, 700, 0.8)) * env_ad(dur, 0.002, 0.04)
    return soft_clip(cr + 0.5 * body, 1.3)


@sfx
def chomp():
    dur = 0.38
    n = ns(dur)
    y = 0.8 * crunch()[:n] * env_ad(dur, 0.001, 0.06)
    y += 0.8 * _thump(170, 90, dur, 0.05, 0.02)
    mouth = fft_filter(white(n, 171), lambda f: bp2(f, 350, 1.5)) * env_ad(dur, 0.01, 0.06)
    place(y, 0.5 * mouth, ns(0.04))
    return y


@sfx
def slurp():
    dur = 0.7
    n = ns(dur)
    t = tt(dur)
    fc = lambda tt_: 450 * (4.5 ** (tt_ / dur))
    x = sweep(white(n, 181), fc, 3.0)
    gurg = np.abs(fft_filter(rng(182).standard_normal(n), lambda f: bp2(f, 25, 1.0)))
    gurg /= gurg.max()
    y = x * (0.3 + gurg)
    r = rng(183)
    for k in range(7):
        t0 = r.uniform(0.05, dur - 0.15)
        place(y, 0.5 * _bubble(r.uniform(250, 500), r.uniform(700, 1400), 0.08, 0.02), ns(t0))
    return y * env_asr(n, 0.05, 0.15)


@sfx
def sizzle():
    dur = 2.0
    n = ns(dur)
    t = tt(dur)
    hiss = fft_filter(white(n, 191), lambda f: hp2(f, 2200) * lp2(f, 7500))
    flutter = 0.6 + 0.4 * np.abs(fft_filter(rng(192).standard_normal(n), lambda f: lp1(f, 12)))
    flutter /= flutter.max()
    pops = crackle(n, 120, 193, 3000, 0.6, 0.0008) + 0.6 * crackle(n, 40, 194, 1800, 1.0, 0.002)
    y = 0.35 * hiss * flutter + pops
    return y * env_asr(n, 0.15, 0.4)


@sfx
def bubbling_pot():
    dur = 2.6
    n = ns(dur)
    r = rng(201)
    y = 0.15 * fft_filter(white(n, 202), lambda f: lp2(f, 180))
    t = 0.0
    while t < dur - 0.15:
        f0 = r.uniform(140, 380)
        place(y, _bubble(f0, f0 * r.uniform(1.8, 3.0), r.uniform(0.06, 0.12), r.uniform(0.02, 0.04)), ns(t),
              r.uniform(0.3, 1.0))
        t += r.exponential(0.07) + 0.015
    y = fft_filter(y, lambda f: lp2(f, 2500))
    return small_room(y, 0.15, 0.5)[:n] * env_asr(n, 0.12, 0.35)


@sfx
def cash():
    dur = 1.3
    n = ns(dur)
    # drawer/mechanism clunk
    clunk = 0.8 * _thump(150, 80, 0.3, 0.05, 0.02) + 0.6 * nb(ns(0.3), 900, 0.8, 211) * env_ad(0.3, 0.0005, 0.02)
    rattle = nb(ns(0.25), 2600, 3.0, 212) * env_ad(0.25, 0.002, 0.05) * (0.5 + 0.5 * np.sin(2 * np.pi * 55 * tt(0.25)))
    # the 'ka-ching' bell: bright inharmonic partials, two quick strikes
    def ching(f):
        t = tt(1.0)
        s = sum(a * np.sin(2 * np.pi * f * r_ * t) * np.exp(-t / d) for r_, a, d in
                ((1, 1, 0.5), (2.32, 0.6, 0.3), (4.25, 0.35, 0.15), (6.1, 0.2, 0.08)))
        return s * ramp_in(len(t), ns(0.001))
    y = mix(n, (clunk, 0, 0.8), (rattle, 0.03, 0.25), (ching(2093), 0.12, 0.55), (ching(2637), 0.2, 0.7))
    return small_room(y, 0.18)[:n]


# ------------------------------------------------------------- abyss / cave diving (Boesmansgat reel)
def _rms(x):
    return float(np.sqrt(np.mean(np.square(x)) + 1e-18))


def _uw(y, fc=3500.0):
    """Underwater muffling: gentle low-pass."""
    return fft_filter(y, lambda f: lp2(f, fc, 0.6))


def _wobble(n, rate_hz, seed):
    """Smooth random modulation, roughly in [-1, 1], bandwidth ~rate_hz."""
    x = fft_filter(white(n, seed), lambda f: lp2(f, rate_hz, 0.7))
    return x / (np.abs(x).max() + 1e-9)


def _inhale(dur, seed, strain=0.0):
    """Rebreather inhale: airflow hiss through the mouthpiece; strain adds flutter and a throaty gasp."""
    n = ns(dur)
    a = sweep(white(n, seed), lambda t: 1300 + 900 * np.clip(t / dur, 0, 1) + 700 * strain, 0.9)
    b = fft_filter(white(n, seed + 1), lambda f: hp2(f, 2600) * lp2(f, 7000))
    y = a / _rms(a) + 0.35 * b / _rms(b)
    if strain:
        g = fft_filter(white(n, seed + 2), lambda f: bp2(f, 750, 5.0) + 0.7 * bp2(f, 1250, 6.0))
        y += 0.5 * strain * g / _rms(g)
    y = fft_filter(y, lambda f: hp2(f, 350))
    flutter = 1 + (0.12 + 0.3 * strain) * _wobble(n, 9 + 6 * strain, seed + 3)
    return y * bell_curve(n, 0.35 - 0.1 * strain, 1.4) * flutter


def _exhale(dur, seed, strain=0.0):
    """Exhale into the loop: muffled breath through hose/counterlung (no bubbles - it's closed circuit)."""
    n = ns(dur)
    a = fft_filter(white(n, seed), lambda f: lp2(f, 700 + 300 * strain, 0.7) * hp2(f, 120))
    c = fft_filter(white(n, seed + 1), lambda f: bp2(f, 420, 2.5) + 0.6 * bp2(f, 1100, 3.0))
    y = a / _rms(a) + 0.6 * c / _rms(c)
    y *= bell_curve(n, 0.22, 1.2) * (1 + (0.15 + 0.2 * strain) * _wobble(n, 6 + 6 * strain, seed + 2))
    bag = fft_filter(white(n, seed + 3), lambda f: bp2(f, 170, 1.2))      # counterlung filling
    return y + 0.45 * bag / _rms(bag) * bell_curve(n, 0.55, 2.0)


def _valve(seed, g=1.0):
    """Tiny mushroom-valve tick in the mouthpiece."""
    d = 0.04
    return g * (nb(ns(d), 2200, 3.0, seed) * env_ad(d, 0.0003, 0.003) + 0.3 * ping(1400, d, 0.004, 0.0004))


def _breaths(cycles, n, strain=0.0, seed=400):
    """cycles: (start, inhale_dur, exhale_gap, exhale_dur)."""
    parts = []
    for i, (t0, di, gap, de) in enumerate(cycles):
        inh = _inhale(di, seed + 10 * i, strain)
        exh = _exhale(de, seed + 10 * i + 5, strain)
        parts += [(inh / _rms(inh), t0, 1.0), (_valve(seed + 10 * i + 7), t0, 0.5),
                  (exh / _rms(exh), t0 + di + gap, 0.85), (_valve(seed + 10 * i + 8, 0.6), t0 + di + gap, 0.4)]
    return _uw(mix(n, *parts), 6500)


@sfx
def regulator_breath():
    """Two calm rebreather breaths, 3.0 s per cycle (6.0 s file; place every 6 s for continuous breathing)."""
    return _breaths([(0.0, 1.15, 0.2, 1.35), (3.0, 1.2, 0.18, 1.3)], ns(6.0))


@sfx
def regulator_breath_fast():
    """Three strained, quicker breaths, ~1.55 s per cycle (~4.65 s)."""
    return _breaths([(0.0, 0.55, 0.07, 0.68), (1.55, 0.5, 0.06, 0.7), (3.1, 0.58, 0.06, 0.66)], ns(4.65),
                    strain=1.0, seed=430)


@sfx
def regulator_breath_stop():
    """Inhale, exhale, then the next inhale cuts off dead at ~3.5 s (with a faint valve tick)."""
    n = ns(3.56)
    y = _breaths([(0.0, 1.15, 0.2, 1.35)], n, seed=460)
    cut = _inhale(1.3, 470) / _rms(_inhale(1.3, 470))
    k = ns(0.5)
    cut = cut[:k] * ramp_out(k, ns(0.018))
    y += _uw(mix(n, (cut, 3.0, 1.0), (_valve(471), 3.0, 0.5), (_valve(472, 0.8), 3.5, 0.5)), 6500)
    return y


@sfx
def sonar_ping():
    """Deep, eerie ping (~700 Hz + 350 Hz undertone, slow beating) with two distant returns and a long tail."""
    dur = 7.0
    n = ns(dur)
    t = tt(dur)
    f = 700 * (1 - 0.015 * (1 - np.exp(-t / 0.6)))
    y = np.sin(phase(f, n)) * env_ad(dur, 0.004, 0.45)
    y += 0.6 * np.sin(phase(f * 1.004, n)) * env_ad(dur, 0.004, 0.5)          # slow beating -> eerie
    y += 0.45 * np.sin(phase(f * 0.5, n)) * env_ad(dur, 0.01, 0.7)            # deep undertone
    y += 0.1 * np.sin(phase(f * 2.01, n)) * env_ad(dur, 0.003, 0.15)
    y += 0.08 * nb(n, 2500, 1.0, 301) * env_ad(dur, 0.0005, 0.004)            # transducer tick
    dry = fft_filter(y, lambda fr: lp1(fr, 1400) * hp1(fr, 220))
    for dt, g in ((1.3, 0.3), (2.7, 0.12)):
        place(y, dry, ns(dt), g)
    ir = freeverb_ir(seconds=5.5, room=0.95, damp=0.3, stereo=False, predelay=0.03, lo_cut=150, hi_cut=5000)[0]
    w = convolve(y, ir)[0][:n]
    return y + 0.9 * w * _rms(y) / _rms(w)


@sfx
def deep_rumble():
    """5 s low rumble: energy 60-200 Hz, saturated so its harmonics (200-600 Hz) carry on phone speakers."""
    dur = 5.0
    n = ns(dur)
    t = tt(dur)
    nz = fft_filter(white(n, 311), lambda f: bp2(f, 95, 0.8) * lp2(f, 220) * hp2(f, 55))
    drift = 1 + 0.01 * _wobble(n, 0.8, 312)
    tone = sine(65.4 * drift, n) + 0.6 * sine(98.0 * drift, n) + 0.45 * sine(131.0 * drift, n)
    y = nz / _rms(nz) + 0.8 * tone / _rms(tone)
    y *= 1 + 0.35 * _wobble(n, 1.5, 313)
    y = soft_clip(y / np.abs(y).max(), 2.6)                                    # harmonics
    grit = nb(n, 320, 0.9, 314) * (1 + 0.5 * _wobble(n, 4, 315))
    y = y / _rms(y) + 0.2 * grit / _rms(grit)
    y = fft_filter(y, lambda f: hp2(f, 55) * lp2(f, 1100))
    return y * env_asr(n, 1.2, 1.8)


def _bubble_cluster(dur, seed, until, gap, fmin, fmax, thin=1.0):
    n = ns(dur)
    r = rng(seed)
    y = np.zeros(n)
    t = 0.0
    while t < until:
        size = r.random()                                     # big bubbles: low and longer
        f0 = fmin * (fmax / fmin) ** (1 - size)
        bd = 0.03 + 0.09 * size
        b = _bubble(f0, f0 * r.uniform(1.4, 2.3), bd * 2, bd * 0.45)
        place(y, b, ns(t), r.uniform(0.3, 1.0) * (1 - 0.5 * t / until))
        t += r.exponential(gap * (1 + thin * t)) + 0.008
    return y


@sfx
def bubbles_rise():
    """A cluster of rising bubbles, dense then thinning out (~2.4 s)."""
    dur = 2.6
    n = ns(dur)
    y = _bubble_cluster(dur, 321, 1.9, 0.045, 260, 1800)
    fizz = fft_filter(white(n, 322), lambda f: bp2(f, 2500, 0.8)) * bell_curve(n, 0.15, 2.0)
    y = y + 0.06 * fizz / _rms(fizz) * _rms(y)
    return small_room(_uw(y, 4200), 0.3, 0.8)[:n]


@sfx
def line_creak():
    """Guide line / rope under tension: stick-slip creak plus fibrous scraping (~2.4 s)."""
    dur = 2.4
    n = ns(dur)
    t = tt(dur)
    r = rng(331)
    rate = np.clip(32 + 22 * np.sin(2 * np.pi * 0.45 * t + 0.5) + 12 * _wobble(n, 3, 332), 12, 80)
    ph = np.cumsum(rate) / SR
    idx = np.nonzero(np.diff(np.floor(ph)) > 0)[0]
    imp = np.zeros(n)
    imp[idx] = r.uniform(0.4, 1.0, len(idx))
    tension = bell_curve(n, 0.45, 1.0)
    imp *= tension * np.clip(0.6 + 0.6 * _wobble(n, 5, 333), 0, 1)
    kt = tt(0.06)
    body = sum(a * np.sin(2 * np.pi * f * kt) * np.exp(-kt / d)
               for f, a, d in ((380, 1.0, 0.012), (830, 0.7, 0.009), (1650, 0.5, 0.006), (2900, 0.25, 0.004)))
    creak = convolve(imp, body)[0][:n]
    scrape = fft_filter(white(n, 334), lambda f: bp2(f, 2200, 0.9)) * (0.3 + 0.7 * np.abs(_wobble(n, 8, 335)))
    y = creak / _rms(creak) + 0.3 * scrape * tension / _rms(scrape)
    return small_room(_uw(y, 5000), 0.15, 0.5)[:n]


@sfx
def alarm_beep():
    """Dive-computer / rebreather warning: three short piezo-ish beeps (~0.75 s)."""
    n = ns(0.9)
    parts = []
    for i in range(3):
        k = ns(0.11)
        f = 2400
        x = np.sin(phase(f, k)) + 0.3 * np.sin(phase(3 * f, k)) + 0.1 * np.sin(phase(5 * f, k))
        parts.append((x * env_asr(k, 0.004, 0.015), i * 0.19, 1.0))
    y = fft_filter(mix(n, *parts), lambda f: lp2(f, 7000))
    return small_room(y, 0.12, 0.5)[:n]


def _heart(period, count, dub_at, seed):
    n = ns(period * count + 0.4)
    parts = []
    for i in range(count):
        k = ns(0.35)
        lub = _thump(110, 68, 0.35, 0.075, 0.03) + 0.4 * _thump(210, 150, 0.35, 0.03, 0.01)
        lub += 0.9 * nb(k, 260, 1.0, seed + i) * env_ad(0.35, 0.002, 0.025)          # chest 'knock'
        dub = _thump(125, 76, 0.35, 0.06, 0.025) + 0.4 * _thump(240, 170, 0.35, 0.025, 0.01)
        dub += 0.8 * nb(k, 300, 1.0, seed + 50 + i) * env_ad(0.35, 0.002, 0.02)
        parts += [(lub, i * period, 1.0), (dub, i * period + dub_at, 0.72)]
    y = soft_clip(mix(n, *parts) * 2.2, 3.0)                 # harmonics so phones hear it
    return fft_filter(y, lambda f: lp2(f, 900) * hp2(f, 60) * (1 + 1.5 * bp2(f, 280, 1.2)))


@sfx
def heartbeat_slow():
    """Two slow beats, ~52 bpm (1.15 s period, ~2.3 s file)."""
    return _heart(1.15, 2, 0.30, 351)


@sfx
def heartbeat_fast():
    """Four fast beats, ~125 bpm (0.48 s period, ~1.95 s file)."""
    return _heart(0.48, 4, 0.19, 361)


@sfx
def torch_click():
    """Dive torch push-button: press + latch, slightly muffled."""
    dur = 0.25
    n = ns(dur)
    press = nb(n, 2800, 1.8, 371) * env_ad(dur, 0.0003, 0.0025) + 0.5 * ping(1300, dur, 0.006, 0.0005) \
        + 0.5 * _thump(420, 260, dur, 0.012, 0.005)
    latch = nb(n, 3600, 2.5, 372) * env_ad(dur, 0.0002, 0.0015) + 0.4 * ping(2100, dur, 0.004, 0.0003)
    y = mix(n, (press, 0, 1.0), (latch, 0.045, 0.7))
    return small_room(_uw(y, 6000), 0.1, 0.3)[:n]


@sfx
def water_muffle_hit():
    """Underwater impact: muffled low body, pressure wave, a burst of bubbles, dark tail (~3 s)."""
    dur = 3.0
    n = ns(dur)
    y = _thump(130, 72, dur, 0.3, 0.05) + 0.6 * _thump(280, 150, dur, 0.09, 0.02)
    press = fft_filter(white(n, 381), lambda f: lp2(f, 600) * hp2(f, 80)) * env_ad(dur, 0.003, 0.15)
    y = soft_clip(y + 0.7 * press / np.abs(press).max(), 3.5)
    y = fft_filter(y, lambda f: lp2(f, 1000, 0.6) * hp2(f, 65) * (1 + bp2(f, 300, 1.0)))
    bub = _bubble_cluster(1.2, 382, 0.6, 0.03, 220, 1100)
    place(y, _uw(bub, 2500), ns(0.04), 0.35)
    ir = freeverb_ir(seconds=2.6, room=0.9, damp=0.6, stereo=False, predelay=0.012, hi_cut=1800)[0]
    w = convolve(y, ir)[0][:n]
    return y + 0.45 * w * _rms(y) / _rms(w)


@sfx
def drone_swell():
    """4 s rising dread swell (minor-second cluster, filter opening, slight upward bend); ends at its peak -
    place it so it ends on the hit (t = hit - 4.0)."""
    dur = 4.0
    n = ns(dur)
    t = tt(dur)
    bend = 2 ** (0.8 * (t / dur) ** 2 / 12)
    r = rng(391)
    y = np.zeros(n)
    for m, a in ((38, 0.8), (50, 1.0), (51, 0.7), (57, 0.6), (62, 0.45), (63, 0.3)):
        for v in (-1, 0, 1):
            y += a * saw(mtof(m) * bend * 2 ** (v * 6 / 1200), n, r.random())
    y = stft_filter(y, lambda tq, f: lp2(f, 160 * (3200 / 160) ** ((tq / dur) ** 1.3), 1.3) * hp2(f, 60), nfft=2048)
    nz = sweep(white(n, 392), lambda tq: 250 * (2800 / 250) ** (tq / dur), 0.9)
    y = fft_filter(y / _rms(y) + 0.22 * nz / _rms(nz), lambda f: lp2(f, 4500))
    return y * (t / dur) ** 2.2 * ramp_out(n, ns(0.04))


@sfx
def silence_drop():
    """Reverse swell (reversed reverb of a low hit + minor chord) that stops dead at its loudest point.
    Place it so it ENDS at the cut: t = cut - 2.5, and add a music_mute at the cut."""
    dur = 2.5
    m = ns(1.0)
    src = _thump(110, 55, 1.0, 0.25, 0.04)
    for k, mm in enumerate((50, 57, 62, 65)):
        src += 0.25 * np.sin(phase(mtof(mm), m)) * env_ad(1.0, 0.002, 0.3)
    src += 0.3 * nb(m, 1500, 0.7, 401) * env_ad(1.0, 0.0005, 0.03)
    ir = freeverb_ir(seconds=3.2, room=0.93, damp=0.35, stereo=False, predelay=0.0)[0]
    w = convolve(soft_clip(src, 2.0), ir)[0]
    rev = w[::-1][-ns(dur):]
    nz = sweep(white(ns(dur), 402), lambda tq: 300 * (3000 / 300) ** (tq / dur), 0.8)
    y = rev / _rms(rev) + 0.25 * nz / _rms(nz) * (tt(dur) / dur) ** 3
    return y * ramp_in(len(y), ns(0.05)) * ramp_out(len(y), ns(0.008))


# ------------------------------------------------------------- finalize / main
def finalize(y, peak_db=-3.0, max_rms_db=-14.0):
    y = np.asarray(y, dtype=np.float64)
    y = dc_block(y)
    y = fft_filter(y, lambda f: hp2(f, 45))   # mobile speakers: nothing useful below ~50 Hz
    pk = np.abs(y).max()
    # trim trailing silence below -60 dB rel. peak (keep 25 ms)
    above = np.nonzero(np.abs(y) > pk * db(-60))[0]
    if len(above):
        y = y[:min(len(y), above[-1] + ns(0.025))]
    y = y * ramp_in(len(y), ns(0.001)) * ramp_out(len(y), ns(0.012))
    y = y * db(peak_db) / np.abs(y).max()
    rms = np.sqrt(np.mean(y ** 2))
    if rms > db(max_rms_db):   # dense/sustained sounds: cap loudness so they sit level with the transients
        y *= db(max_rms_db) / rms
    return y


def main(names):
    os.makedirs(OUT, exist_ok=True)
    names = names or list(SFX)
    print(f"{'name':22s} {'dur':>5s} {'peak':>6s} {'rms':>6s} {'centroid':>8s}")
    for name in names:
        y = finalize(SFX[name]())
        assert np.isfinite(y).all(), name
        write(os.path.join(OUT, name + ".wav"), y)
        s = stats(y)
        print(f"{name:22s} {s['dur']:5.2f} {s['peak_db']:6.1f} {s['rms_db']:6.1f} {s['centroid']:8.0f}")


ALLOWED = ("whoosh whoosh_long swoosh_up swoosh_down pop pop_small bubble blip click tick ding chime sparkle "
           "impact_soft impact_big thud stamp boing riser record_scratch tv_on tv_off channel_switch static glitch "
           "neon_buzz neon_flicker typewriter paper heartbeat alarm crunch slurp chomp sizzle bubbling_pot cash "
           "error success "
           # abyss / cave diving
           "regulator_breath regulator_breath_fast regulator_breath_stop sonar_ping deep_rumble bubbles_rise "
           "line_creak alarm_beep heartbeat_slow heartbeat_fast torch_click water_muffle_hit drone_swell "
           "silence_drop").split()

if __name__ == "__main__":
    missing = set(ALLOWED) - set(SFX)
    assert not missing, missing
    main(sys.argv[1:])
