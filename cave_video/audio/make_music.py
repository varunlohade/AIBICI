#!/usr/bin/env python3
"""Original ambient synth beds (Epic-Mountain-ish: warm detuned pads, soft plucks, sub bass, big reverb).

Usage: python3 audio/make_music.py [bed|sting_<bed> ...] [--flac]
Writes audio/music/<bed>.wav and sting_<bed>.wav (1.8 s endings) (48 kHz stereo, 16-bit). Beds are loudness-normalised to about -23 LUFS (peaks < -1 dBFS),
fade in over ~2 s and out over the last ~3 s; mix_audio.py trims/loops/crossfades them.
Deterministic: same output every run.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dsp import (SR, lufs, convolve, db, echo, env_ad, env_asr, fft_filter, freeverb_ir, hp1, hp2, bp2, limiter, lp1, lp2,
                 mtof, ns, pan2, phase, place, ramp_in, ramp_out, rng, saw, sine, stats, stft_filter, tt, write)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "music")
TARGET_LUFS = -23.0
_cache = {}


# ============================================================= instruments (return mono or stereo arrays)
def pad_chord(notes, dur, cutoff=1800.0, detune=9.0, attack=1.2, release=1.8, voices=3, seed=0, cut_fn=None,
              sub_oct=0.0):
    """Warm detuned saw pad, stereo-spread voices, low-passed (optionally time-varying cutoff)."""
    n = ns(dur + release)
    r = rng(seed)
    out = np.zeros((2, n))
    for m in notes:
        f = mtof(m)
        for v in range(voices):
            cents = (v - (voices - 1) / 2) * detune + r.uniform(-2, 2)
            fv = f * 2 ** (cents / 1200)
            # slow random vibrato for life
            vib = 1 + 0.0015 * np.sin(2 * np.pi * r.uniform(0.15, 0.4) * np.arange(n) / SR + r.uniform(0, 6))
            x = saw(fv * vib, n, r.uniform(0, 1))
            p = (v - (voices - 1) / 2) / max(1, (voices - 1) / 2) * 0.7
            out += pan2(x, p) / voices
        if sub_oct:
            out += sub_oct * sine(f / 2, n)[None] * 0.5
    if cut_fn is None:
        out = fft_filter(out, lambda fr: lp2(fr, cutoff, 0.8) * hp2(fr, 110))
    else:
        out = stft_filter(out, lambda t, fr: lp2(fr, cut_fn(t), 0.8) * hp2(fr, 110), nfft=2048)
    env = np.ones(n)
    env *= env_asr(n, attack, release, shape=1.5)
    return out * env[None] / np.sqrt(len(notes))


def pluck(m, bright=1.0, decay=0.35, dur=None):
    """Soft FM pluck (kalimba / mallet-ish)."""
    key = ("pl", m, round(bright, 2), round(decay, 3))
    if key in _cache:
        return _cache[key]
    dur = dur or min(2.5, decay * 6)
    n = ns(dur)
    t = tt(dur)
    f = mtof(m)
    idx = bright * 2.2 * np.exp(-t / 0.06) + 0.25 * bright
    y = np.sin(phase(f, n) + idx * np.sin(phase(2 * f, n)))
    y += 0.25 * np.sin(phase(f * 4.0, n)) * np.exp(-t / 0.03) * bright
    y *= env_ad(dur, 0.003, decay)
    y = fft_filter(y, lambda fr: lp2(fr, 1500 + 5000 * bright), pad=4096)
    y *= ramp_out(n, ns(0.05))
    _cache[key] = y * 1.4
    return _cache[key]


def bell(m, decay=1.4, dur=None):
    key = ("bell", m, decay)
    if key in _cache:
        return _cache[key]
    dur = dur or decay * 4
    n = ns(dur)
    t = tt(dur)
    f = mtof(m)
    y = np.zeros(n)
    for r_, a, d in ((1, 1.0, 1.0), (2.0, 0.4, 0.6), (3.0, 0.15, 0.4), (4.07, 0.12, 0.25), (5.2, 0.05, 0.15)):
        y += a * np.sin(2 * np.pi * f * r_ * t * (1 + 0.0007)) * np.exp(-t / (decay * d))
        y += 0.3 * a * np.sin(2 * np.pi * f * r_ * t) * np.exp(-t / (decay * d))
    y *= ramp_in(n, ns(0.004)) * ramp_out(n, ns(0.1))
    _cache[key] = y * 1.3
    return _cache[key]


def sub(m, dur, attack=0.03, release=0.3):
    n = ns(dur + release)
    f = mtof(m)
    y = np.sin(phase(f, n)) + 0.12 * np.sin(phase(2 * f, n))
    return 0.5 * np.tanh(1.3 * y) * env_asr(n, attack, release)


def kick(soft=1.0):
    key = ("kick", soft)
    if key in _cache:
        return _cache[key]
    d = 0.5
    t = tt(d)
    f = 45 + 80 * np.exp(-t / 0.035)
    y = np.sin(phase(f, len(t))) * env_ad(d, 0.002, 0.16)
    click = fft_filter(rng(7).standard_normal(len(t)), lambda fr: bp2(fr, 1500, 0.7)) * env_ad(d, 0.0005, 0.003)
    y = y + 0.15 * soft * click
    _cache[key] = 0.7 * y
    return _cache[key]


def noise_hit(fc, q, decay, seed, dur=0.3, hp=None):
    key = ("nh", fc, q, decay, seed, hp)
    if key in _cache:
        return _cache[key]
    n = ns(dur)
    x = rng(seed).standard_normal(n)
    x = fft_filter(x, lambda fr: bp2(fr, fc, q) * (hp2(fr, hp) if hp else 1), pad=4096)
    y = x * env_ad(dur, 0.001, decay)
    _cache[key] = y
    return y


def shaker(i):
    return 1.6 * noise_hit(7500, 1.2, 0.025, 30 + i % 4, 0.15, 4000)


def clap():
    n = ns(0.35)
    y = np.zeros(n)
    for k, dt in enumerate((0, 0.009, 0.019)):
        place(y, noise_hit(1300, 1.0, 0.006, 40 + k, 0.05), ns(dt))
    place(y, noise_hit(1200, 0.8, 0.07, 44, 0.3), ns(0.025), 0.6)
    return y


def tick(hi=True):
    return noise_hit(3800 if hi else 2600, 4.0, 0.004, 50 + hi, 0.08) + \
        0.5 * np.sin(2 * np.pi * (2100 if hi else 1500) * tt(0.08)) * env_ad(0.08, 0.0005, 0.008)


def bubble_note(m):
    d = 0.18
    n = ns(d)
    t = tt(d)
    f = mtof(m) * (0.6 + 0.4 * (1 - np.exp(-t / 0.03)))
    return np.sin(phase(f, n)) * env_ad(d, 0.003, 0.05)


def swell(dur, f0=300, f1=5000, seed=9):
    """Filtered-noise swell (reverse-cymbal-ish) to lead into a section."""
    n = ns(dur)
    x = rng(seed).standard_normal(n)
    x = stft_filter(x, lambda t, fr: bp2(fr, f0 * (f1 / f0) ** (t / dur), 0.9), nfft=2048)
    return x * (np.arange(n) / n) ** 2.5 * ramp_out(n, ns(0.04))


# ============================================================= arrangement buses
class Bed:
    def __init__(self, dur, bpm, seed=1):
        self.dur = dur
        self.bpm = bpm
        self.beat = 60.0 / bpm
        self.bar = 4 * self.beat
        self.n = ns(dur + 6)
        self.dry = np.zeros((2, self.n))
        self.verb = np.zeros((2, self.n))
        self.dly = np.zeros((2, self.n))
        self.r = rng(seed)

    def add(self, sig, t, gain=1.0, pan=0.0, verb=0.25, delay=0.0):
        sig = np.asarray(sig, dtype=np.float64)
        st = pan2(sig, pan) if sig.ndim == 1 else sig
        s = ns(t)
        place(self.dry, st, s, gain)
        if verb:
            place(self.verb, st, s, gain * verb)
        if delay:
            place(self.dly, st, s, gain * delay)

    def b(self, bar, beat=0.0):
        return bar * self.bar + beat * self.beat

    def render(self, ir_seconds=3.5, room=0.88, verb_level=1.0, delay_div=0.75, pump=None, fade_in=2.0, fade_out=3.0):
        ir = freeverb_ir(seconds=ir_seconds, room=room, damp=0.4)
        d = echo(self.dly, delay_div * self.beat, feedback=0.42, taps=6, damp_hz=3500)[:, :self.n]
        dl = np.zeros_like(self.dry)
        dl[:, :d.shape[1]] = d
        wet = convolve(self.verb + 0.4 * dl, ir)[:, :self.n]
        wet *= np.sqrt(np.mean(self.verb ** 2) + 1e-12) / np.sqrt(np.mean(wet ** 2) + 1e-12) * 1.1
        mix = self.dry + dl + verb_level * wet
        if pump is not None:
            mix *= pump[None, :self.n]
        mix = fft_filter(mix, lambda fr: hp2(fr, 28))
        n = ns(self.dur)
        mix = mix[:, :n]
        mix *= (ramp_in(n, ns(fade_in)) * ramp_out(n, ns(fade_out)))[None]
        mix *= db(TARGET_LUFS - lufs(mix[:, ns(fade_in):n - ns(fade_out)]))
        mix = np.tanh(mix * 1.2) / 1.2 if np.abs(mix).max() > 0.7 else mix
        return limiter(mix, -1.0)

    def pump_curve(self, depth=0.35, start=0.0, end=None, every=1.0):
        """Sidechain-style gain pumping on every beat (for the upbeat beds)."""
        t = np.arange(self.n) / SR
        ph = ((t - start) / (self.beat * every)) % 1.0
        g = 1 - depth * np.exp(-ph * self.beat * every / 0.09)
        end = end if end is not None else self.dur
        g = np.where((t >= start) & (t < end), g, 1.0)
        return fft_filter(g, lambda fr: lp1(fr, 60))


def arp_pattern(chord, pattern, octave_up=12):
    tones = sorted(chord)[-4:]
    tones = [x + octave_up for x in tones] + [tones[0] + octave_up + 12]
    return [tones[i % len(tones)] for i in pattern]


# ============================================================= the beds
def bed_tension(dur=52.0):
    """Cold open: D minor, ticking clock, low ostinato, eerie bells, slowly tightening."""
    B = Bed(dur, 84, seed=11)
    prog = [([50, 57, 64, 65], 38), ([46, 53, 57, 62], 34), ([43, 50, 58, 64], 31), ([45, 52, 57, 62], 33),
            ([50, 57, 64, 65], 38), ([46, 53, 57, 62], 34), ([43, 50, 58, 62], 31), ([45, 52, 57, 61], 33)]
    bars_per = 2
    nbars = int(np.ceil(dur / B.bar))
    for bar in range(0, nbars, bars_per):
        notes, root = prog[(bar // bars_per) % len(prog)]
        t0 = B.b(bar)
        inten = min(1.0, 0.45 + bar / nbars)
        cd = B.bar * bars_per
        pad = pad_chord(notes, cd + 0.4, cutoff=700 + 900 * inten, detune=11, attack=1.0, release=1.6, seed=bar)
        B.add(pad, t0, 0.55, verb=0.45)
        B.add(sub(root, cd - 0.1, 0.4, 0.6), t0, 0.5 * inten, verb=0)
        # low muted ostinato (8ths) from bar 4
        if bar >= 4:
            for k in range(8 * bars_per):
                m = root + 12 + (0 if k % 4 != 3 else 7)
                B.add(pluck(m, 0.25, 0.12), t0 + k * B.beat / 2, 0.32 * inten * (1.2 if k % 2 == 0 else 0.8),
                      pan=-0.2, verb=0.15, delay=0.12)
        # eerie bell pair (minor second) every 4 bars
        if bar % 4 == 2:
            B.add(bell(81, 1.6), t0 + B.beat, 0.13, pan=0.4, verb=0.8, delay=0.3)
            B.add(bell(82, 1.6), t0 + 2.5 * B.beat, 0.10, pan=-0.4, verb=0.8, delay=0.3)
    # clock: tick-tock on every beat
    nbeats = int(dur / B.beat)
    for k in range(2, nbeats):
        g = 0.12 + 0.12 * k / nbeats
        B.add(tick(k % 2 == 0), k * B.beat, g, pan=0.25 if k % 2 else -0.25, verb=0.2)
    # heartbeat-ish sub pulse in the second half
    for bar in range(nbars // 2, nbars):
        B.add(kick(0.2), B.b(bar), 0.28, verb=0.1)
        B.add(kick(0.2), B.b(bar, 0.6), 0.18, verb=0.1)
    B.add(swell(4.0, 200, 4000), dur - 6.0, 0.08, verb=0.5)
    return B.render(ir_seconds=4.0, room=0.9)


def bed_wonder(dur=102.0):
    """Discovery: D major/lydian, 16th-note pluck arps, bells, sub bass, gentle shaker."""
    B = Bed(dur, 96, seed=21)
    prog = [([50, 57, 61, 64, 66], 38), ([47, 54, 57, 62, 64], 35), ([43, 50, 54, 59, 61], 31),
            ([45, 52, 57, 59, 61], 33)]
    nbars = int(np.ceil(dur / B.bar))
    melody = [78, 76, 73, 74, 81, 78, 76, 73]  # F#5 E5 C#5 D5 A5 F#5 E5 C#5
    pat16 = [0, 2, 1, 3, 2, 4, 3, 1, 0, 2, 1, 3, 4, 3, 2, 1]
    for bar in range(0, nbars, 2):
        notes, root = prog[(bar // 2) % len(prog)]
        t0 = B.b(bar)
        sect = bar / nbars
        busy = 0 if bar < 4 else (1 if bar < 10 else (2 if bar < 30 else 1))
        cut = lambda t, b=bar: 900 + 1400 * np.minimum(1, (b + t / B.bar) / 12)
        pad = pad_chord(notes, 2 * B.bar + 0.3, detune=10, attack=1.4, release=2.0, seed=bar, cut_fn=cut)
        B.add(pad, t0, 0.5, verb=0.5)
        if bar >= 6:
            B.add(sub(root, 2 * B.bar - 0.2, 0.08, 0.4), t0, 0.45, verb=0)
        for half in range(2):
            tb = t0 + half * B.bar
            if busy >= 1:
                step = 0.25 if busy == 2 else 0.5
                seq = arp_pattern(notes, pat16, 12)
                for k in range(int(4 / step)):
                    m = seq[k % len(seq)]
                    acc = 1.0 if k % (4 if step == 0.25 else 2) == 0 else 0.7
                    B.add(pluck(m, 0.75, 0.22), tb + k * step * B.beat, 0.20 * acc,
                          pan=0.35 * np.sin(k * 0.8), verb=0.3, delay=0.25)
            if busy == 2:
                for k in range(8):
                    B.add(shaker(k), tb + k * 0.5 * B.beat + 0.25 * B.beat, 0.05 * (1.2 if k % 2 else 0.8),
                          pan=0.3, verb=0.1)
                B.add(kick(0.3), tb, 0.22, verb=0.05)
        if 10 <= bar < 34 and bar % 2 == 0:
            m = melody[(bar // 2) % len(melody)]
            B.add(bell(m, 1.2), t0 + B.beat * 0.0, 0.16, pan=-0.3, verb=0.6, delay=0.35)
            B.add(bell(m - 5 if m - 5 > 70 else m + 2, 1.0), t0 + B.beat * 3, 0.10, pan=0.3, verb=0.6, delay=0.35)
    B.add(swell(3.5, 300, 5000), B.b(10) - 3.5, 0.06, verb=0.5)
    return B.render(ir_seconds=3.5, room=0.88)


def bed_science(dur=56.0):
    """Biology: F major, bouncy mallets, bubble blips, light kick/snap/shaker."""
    B = Bed(dur, 112, seed=31)
    prog = [([53, 57, 60, 65], 41), ([50, 57, 62, 65], 38), ([46, 53, 58, 62], 34), ([48, 55, 60, 64], 36)]
    nbars = int(np.ceil(dur / B.bar))
    pent = [72, 74, 77, 79, 81, 84, 86]
    rhythm = [0, 0.75, 1.5, 2.0, 2.75, 3.5]
    for bar in range(nbars):
        notes, root = prog[bar % len(prog)]
        t0 = B.b(bar)
        pad = pad_chord(notes, B.bar + 0.2, cutoff=1400, detune=8, attack=0.4, release=1.0, seed=bar)
        B.add(pad, t0, 0.32, verb=0.4)
        tones = sorted(notes)
        for i, bt in enumerate(rhythm):
            m = tones[(i * 2 + bar) % len(tones)] + 12
            B.add(pluck(m, 0.55, 0.13), t0 + bt * B.beat, 0.24, pan=-0.3 if i % 2 else 0.3, verb=0.25, delay=0.15)
        for bt in (0, 1.5, 2.5):
            B.add(sub(root, 0.35, 0.01, 0.12), t0 + bt * B.beat, 0.45, verb=0)
        if bar >= 2:
            B.add(kick(0.5), t0, 0.32, verb=0.05)
            B.add(kick(0.5), t0 + 2 * B.beat, 0.26, verb=0.05)
            for k in range(8):
                B.add(shaker(k), t0 + k * 0.5 * B.beat, 0.05 * (1.3 if k % 2 else 0.7), pan=0.35, verb=0.05)
        if bar >= 4:
            for bt in (1, 3):
                B.add(clap(), t0 + bt * B.beat, 0.12, pan=0.1, verb=0.35)
        # bubbles
        for k in range(16):
            if B.r.random() < 0.22:
                B.add(bubble_note(int(B.r.choice(pent))), t0 + k * 0.25 * B.beat, 0.10,
                      pan=B.r.uniform(-0.7, 0.7), verb=0.4, delay=0.2)
    return B.render(ir_seconds=2.8, room=0.84, pump=B.pump_curve(0.2, B.b(2)))


def bed_mystery(dur=90.0):
    """Investigation: E minor/phrygian, muted plucks, ticking, glassy bells, low pulses."""
    B = Bed(dur, 76, seed=41)
    prog = [([52, 59, 62, 66], 40), ([48, 55, 59, 64], 36), ([45, 52, 55, 60], 33), ([47, 54, 57, 64], 35),
            ([52, 59, 62, 66], 40), ([48, 55, 59, 64], 36), ([41, 48, 52, 57], 41), ([47, 54, 57, 63], 35)]
    nbars = int(np.ceil(dur / B.bar))
    glass = [83, 79, 78, 76, 83, 81, 79, 78]
    ost = [0, 7, 10, 7, 12, 7, 10, 7]
    for bar in range(0, nbars, 2):
        i = (bar // 2) % len(prog)
        notes, root = prog[i]
        t0 = B.b(bar)
        thin = bar >= nbars - 4
        busy = bar >= 4 and not thin
        pad = pad_chord(notes, 2 * B.bar + 0.3, cutoff=850, detune=12, attack=1.5, release=2.0, seed=bar)
        B.add(pad, t0, 0.55, verb=0.5)
        B.add(sub(root, 2 * B.bar - 0.3, 0.3, 0.6), t0, 0.42, verb=0)
        for half in range(2):
            tb = t0 + half * B.bar
            if busy:
                for k in range(8):
                    m = root + 12 + ost[k]
                    B.add(pluck(m, 0.22, 0.11), tb + k * 0.5 * B.beat, 0.26 * (1.15 if k % 4 == 0 else 0.85),
                          pan=-0.25, verb=0.15, delay=0.3)
            for k in range(0, 4, 2 if not busy else 1):
                B.add(tick(k % 2 == 0), tb + k * B.beat, 0.09, pan=0.3, verb=0.3)
            if 8 <= bar < nbars - 4:
                B.add(kick(0.1), tb, 0.26, verb=0.1)
                B.add(kick(0.1), tb + 2.5 * B.beat, 0.16, verb=0.1)
        if bar % 4 == 2 or (bar >= 12 and bar % 2 == 0):
            B.add(bell(glass[(bar // 2) % len(glass)], 1.8), t0 + B.beat, 0.10, pan=0.4, verb=0.9, delay=0.4)
    return B.render(ir_seconds=4.5, room=0.9, verb_level=1.1)


def bed_resolve(dur=80.0):
    """Bright & confident: G major, driving 8th plucks, four-on-the-floor soft kick, claps, bells, pumping pad."""
    B = Bed(dur, 118, seed=51)
    prog = [([55, 59, 62, 67], 43), ([54, 57, 62, 66], 42), ([52, 59, 62, 67], 40), ([48, 55, 59, 64], 36)]
    nbars = int(np.ceil(dur / B.bar))
    melody = [[79, 81, 83, 81], [78, 81, 86, 83], [79, 83, 86, 88], [84, 83, 79, 76]]
    pat = [0, 1, 2, 3, 4, 3, 2, 1]
    for bar in range(nbars):
        notes, root = prog[bar % len(prog)]
        t0 = B.b(bar)
        brk = 18 <= bar < 22
        end = bar >= nbars - 2
        if end:
            notes, root = [55, 59, 62, 67, 71], 43
        pad = pad_chord(notes, B.bar + 0.2, cutoff=2200 if not brk else 1200, detune=9, attack=0.25,
                        release=1.0 if not end else 3.0, seed=bar)
        B.add(pad, t0, 0.45, verb=0.4)
        if end:
            if bar == nbars - 2:
                B.add(bell(79, 2.0), t0, 0.18, verb=0.7, delay=0.3)
                B.add(sub(root, 2 * B.bar, 0.02, 1.0), t0, 0.5, verb=0)
                B.add(kick(0.6), t0, 0.35)
            continue
        seq = arp_pattern(notes, pat, 12)
        for k in range(8):
            if brk and k % 2:
                continue
            B.add(pluck(seq[k], 0.9, 0.16), t0 + k * 0.5 * B.beat, 0.2 * (1.2 if k % 2 == 0 else 0.85),
                  pan=0.3 * np.sin(k), verb=0.25, delay=0.2)
        if not brk:
            for bt in (0, 1.5, 2, 3.5):
                B.add(sub(root, 0.4, 0.01, 0.15), t0 + bt * B.beat, 0.45, verb=0)
        if bar >= 2 and not brk:
            for bt in range(4):
                B.add(kick(0.7), t0 + bt * B.beat, 0.36, verb=0.03)
            for k in range(16):
                B.add(shaker(k), t0 + k * 0.25 * B.beat, 0.035 * (1.4 if k % 2 else 0.7), pan=0.35, verb=0.05)
        if bar >= 6 and not brk:
            for bt in (1, 3):
                B.add(clap(), t0 + bt * B.beat, 0.16, verb=0.3)
        if bar >= 8 and not brk:
            mel = melody[bar % 4]
            for k, m in enumerate(mel):
                B.add(bell(m, 0.9), t0 + k * B.beat, 0.12, pan=-0.25, verb=0.5, delay=0.3)
        if bar == 21:
            B.add(swell(B.bar, 300, 6000), t0, 0.1, verb=0.4)
    pump = B.pump_curve(0.3, B.b(2), B.b(nbars - 2))
    return B.render(ir_seconds=2.8, room=0.86, pump=pump)


def bed_outro(dur=42.0):
    """Emotional close: Am - F - C/E - Gsus/G - Am - F - G - C(hold). Swell to bar 7, resolve, long tail."""
    B = Bed(dur, 70, seed=61)
    chords = [([45, 52, 59, 60, 64], 33), ([41, 48, 55, 57, 64], 29), ([40, 48, 55, 60, 67], 36),
              ([43, 50, 55, 60, 62], 31), ([45, 52, 57, 60, 64], 33), ([41, 48, 53, 57, 65], 29),
              ([43, 50, 55, 59, 62], 31), ([48, 55, 60, 64, 67, 72], 36)]
    melody = [(76, 0), (72, 2), (69, 0), (72, 2), (79, 0), (76, 2), (74, 0), (72, 0), (76, 2), (81, 0), (79, 2),
              (77, 0), (74, 2), (76, 0), (72, 2)]
    nb_ = len(chords)
    mel_i = 0
    for i, (notes, root) in enumerate(chords):
        t0 = B.b(i)
        last = i == nb_ - 1
        cd = B.bar if not last else dur - t0 - 1.0
        inten = [0.45, 0.55, 0.65, 0.8, 0.85, 0.95, 1.0, 0.9][i]
        pad = pad_chord(notes, cd + 0.3, detune=10, attack=1.2 if i else 2.5, release=2.5 if not last else 4.0,
                        seed=i, cut_fn=lambda t, a=inten, l=last: (700 + 2200 * a) * (1 if not l else
                                                                                         np.maximum(0.4, 1 - t / 14)))
        B.add(pad, t0, 0.6 * inten, verb=0.55)
        B.add(sub(root, cd - 0.2, 0.2, 1.0 if not last else 4.0), t0, 0.45 * inten, verb=0)
        # slow 8th-note arps
        steps = 8 if not last else 12
        seq = arp_pattern(notes, [0, 1, 2, 3, 4, 3, 2, 1, 0, 1, 2, 4], 12)
        for k in range(steps):
            g = 0.18 * inten * (1 if not last else max(0.0, 1 - k / 12))
            B.add(pluck(seq[k % len(seq)], 0.5, 0.35), t0 + k * 0.5 * B.beat, g, pan=0.4 * np.sin(k * 0.9),
                  verb=0.45, delay=0.3)
        # melody: 2 notes per bar (beat 0 and 2) from bar 1
        if i >= 1:
            for bt in (0, 2):
                if mel_i < len(melody):
                    m, _ = melody[mel_i]
                    mel_i += 1
                    B.add(bell(m, 1.6), t0 + bt * B.beat, 0.17 * inten, pan=-0.15, verb=0.7, delay=0.3)
    B.add(swell(B.bar * 1.0, 250, 5000), B.b(nb_ - 1) - B.bar, 0.09, verb=0.6)
    B.add(kick(0.0), B.b(nb_ - 1), 0.3, verb=0.4)
    B.add(bell(84, 3.0), B.b(nb_ - 1), 0.14, verb=0.9, delay=0.35)
    return B.render(ir_seconds=5.0, room=0.92, verb_level=1.2, fade_out=4.0)


BEDS = dict(tension=bed_tension, wonder=bed_wonder, science=bed_science, mystery=bed_mystery,
            resolve=bed_resolve, outro=bed_outro)

# ------------------------------------------------------------- stings: 1.8 s "button" endings in each bed's key
STING_LEN = 1.8
STING_CHORDS = dict(tension=[50, 62, 65, 69, 74], wonder=[50, 62, 66, 69, 74, 78], science=[53, 65, 69, 72, 77],
                    mystery=[52, 64, 67, 71, 76], resolve=[55, 67, 71, 74, 79, 83], outro=[48, 60, 64, 67, 72, 76])


def sting(name):
    """Short 'button' ending: strummed bell+pluck chord, a soft (not sub-heavy) hit, quick reverb bloom,
    fully decayed by STING_LEN so it can sit on the very last frames of a part."""
    notes = STING_CHORDS[name]
    B = Bed(STING_LEN, 100, seed=99)
    for i, m in enumerate(notes[1:]):
        B.add(pluck(m, 0.8, 0.3), 0.018 * i, 0.30, pan=-0.4 + 0.2 * i, verb=0.5)
        B.add(bell(m + 12, 0.7), 0.018 * i + 0.01, 0.12, pan=0.4 - 0.2 * i, verb=0.6)
    B.add(sub(notes[0] + 12, 0.25, 0.005, 0.5), 0.0, 0.8, verb=0)        # body ~100-150 Hz, not sub
    B.add(kick(0.8), 0.0, 0.35, verb=0.1)
    B.add(clap(), 0.0, 0.12, verb=0.4)
    ir = freeverb_ir(seconds=1.5, room=0.8, damp=0.45)
    n = ns(STING_LEN)
    wet = convolve(B.verb, ir)[:, :n]
    wet *= np.sqrt(np.mean(B.verb[:, :n] ** 2)) / np.sqrt(np.mean(wet ** 2) + 1e-12)
    x = B.dry[:, :n] + 0.9 * wet
    x = fft_filter(x, lambda fr: hp2(fr, 70) ** 2)
    x *= np.exp(-np.arange(n) / SR / 0.9)[None] * ramp_out(n, ns(0.25))[None] * ramp_in(n, ns(0.002))[None]
    return x * db(-6.0) / np.abs(x).max()


def main(argv):
    flac = "--flac" in argv
    names = [a for a in argv if not a.startswith("--")] or (list(BEDS) + ["sting_" + k for k in BEDS])
    os.makedirs(OUT, exist_ok=True)
    for name in names:
        x = sting(name[6:]) if name.startswith("sting_") else BEDS[name]()
        assert np.isfinite(x).all(), name
        ext = ".flac" if flac else ".wav"
        write(os.path.join(OUT, name + ext), x)
        s = stats(x)
        print(f"{name:8s} {s['dur']:6.1f}s peak {s['peak_db']:6.1f} dBFS  rms {s['rms_db']:6.1f}  "
              f"centroid {s['centroid']:6.0f} Hz", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
