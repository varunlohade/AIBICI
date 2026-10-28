"""Tiny numpy-only DSP toolkit shared by make_sfx.py, make_music.py and ../mix_audio.py.

Everything is deterministic (seeded RNGs) so generated audio is reproducible.
"""
import numpy as np
import soundfile as sf

SR = 48000


# ---------------------------------------------------------------- basics
def db(x):
    return 10.0 ** (np.asarray(x, dtype=np.float64) / 20.0)


def todb(x):
    return 20.0 * np.log10(np.maximum(np.abs(x), 1e-12))


def ns(sec):
    return int(round(sec * SR))


def tt(dur):
    return np.arange(ns(dur)) / SR


def rng(seed):
    return np.random.default_rng(seed)


def mtof(m):
    return 440.0 * 2.0 ** ((np.asarray(m, dtype=np.float64) - 69.0) / 12.0)


def phase(freq, n):
    """Running phase (radians) for a scalar or per-sample frequency, starting at 0."""
    f = np.broadcast_to(np.asarray(freq, dtype=np.float64), (n,))
    p = np.cumsum(f) / SR
    return 2 * np.pi * np.concatenate([[0.0], p[:-1]])


def sine(freq, n, ph0=0.0):
    return np.sin(phase(freq, n) + ph0)


def saw(freq, n, ph0=0.0):
    """Band-limited (PolyBLEP) sawtooth, freq scalar or per-sample."""
    inc = np.broadcast_to(np.asarray(freq, dtype=np.float64), (n,)) / SR
    ph = (ph0 + np.concatenate([[0.0], np.cumsum(inc)[:-1]])) % 1.0
    y = 2.0 * ph - 1.0
    m = ph < inc
    x = ph[m] / inc[m]
    y[m] -= x + x - x * x - 1.0
    m = ph > 1.0 - inc
    x = (ph[m] - 1.0) / inc[m]
    y[m] -= x * x + x + x + 1.0
    return y


def tri(freq, n, ph0=0.0):
    ph = (phase(freq, n) / (2 * np.pi) + ph0) % 1.0
    return 4.0 * np.abs(ph - 0.5) - 1.0


# ---------------------------------------------------------------- envelopes
def ramp_in(n_total, n_ramp):
    e = np.ones(n_total)
    n_ramp = min(n_ramp, n_total)
    if n_ramp > 0:
        e[:n_ramp] = np.sin(0.5 * np.pi * np.arange(n_ramp) / n_ramp) ** 2
    return e


def ramp_out(n_total, n_ramp):
    return ramp_in(n_total, n_ramp)[::-1]


def env_ad(dur, attack=0.005, decay=0.2, curve="exp"):
    """Attack (raised cosine) then exponential decay with time constant `decay`."""
    t = tt(dur)
    e = np.exp(-np.maximum(t - attack, 0) / decay) if curve == "exp" else np.maximum(0, 1 - (t - attack) / decay)
    na = ns(attack)
    if na > 0:
        e[:na] = np.sin(0.5 * np.pi * np.arange(na) / na) ** 2
    return e


def env_asr(n, attack, release, shape=2.0):
    """Smooth attack / sustain / release envelope over n samples (attack/release in seconds)."""
    e = np.ones(n)
    na, nr = min(ns(attack), n), min(ns(release), n)
    if na > 0:
        e[:na] = np.sin(0.5 * np.pi * np.arange(na) / na) ** shape
    if nr > 0:
        e[n - nr:] *= np.cos(0.5 * np.pi * np.arange(nr) / nr) ** shape
    return e


def bell_curve(n, peak=0.5, sharp=2.0):
    """Asymmetric smooth hump 0->1->0, peak position as a fraction."""
    x = np.linspace(0, 1, n)
    a = np.where(x < peak, x / max(peak, 1e-6), (1 - x) / max(1 - peak, 1e-6))
    return np.sin(0.5 * np.pi * np.clip(a, 0, 1)) ** sharp


def lin(t, pts):
    """Piecewise-linear automation: pts = [(t0, v0), (t1, v1), ...]."""
    xs, ys = zip(*pts)
    return np.interp(t, xs, ys)


def logsweep(n, f0, f1, curve=1.0):
    x = np.linspace(0, 1, n) ** curve
    return f0 * (f1 / f0) ** x


# ---------------------------------------------------------------- filters (magnitude responses)
def lp2(f, fc, q=0.707):
    r = f / fc
    return 1.0 / np.sqrt((1 - r * r) ** 2 + (r / q) ** 2)


def hp2(f, fc, q=0.707):
    r = f / fc
    return r * r / np.sqrt((1 - r * r) ** 2 + (r / q) ** 2)


def bp2(f, fc, q=1.0):
    r = f / fc
    return (r / q) / np.sqrt((1 - r * r) ** 2 + (r / q) ** 2)


def lp1(f, fc):
    return 1.0 / np.sqrt(1 + (f / fc) ** 2)


def hp1(f, fc):
    r = f / fc
    return r / np.sqrt(1 + r * r)


def fft_filter(x, gain_fn, pad=None):
    """Zero-phase static filter: multiply the spectrum by gain_fn(freqs). Works on (n,) or (ch, n)."""
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[-1]
    pad = SR // 4 if pad is None else pad
    nfft = 1 << int(np.ceil(np.log2(n + pad)))
    X = np.fft.rfft(x, nfft, axis=-1)
    f = np.fft.rfftfreq(nfft, 1 / SR)
    g = gain_fn(np.maximum(f, 1e-3))
    return np.fft.irfft(X * g, nfft, axis=-1)[..., :n]


def stft_filter(x, gain_fn, nfft=1024):
    """Time-varying zero-phase filter. gain_fn(t_col, f_row) -> (frames, bins) gains.
    sqrt-Hann analysis/synthesis at 50% overlap (perfect reconstruction when gain==1)."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 2:
        return np.stack([stft_filter(c, gain_fn, nfft) for c in x])
    n = len(x)
    hop = nfft // 2
    nfr = int(np.ceil((n + nfft) / hop)) + 1
    if nfr % 2:
        nfr += 1
    buf = np.zeros(hop * (nfr + 1))
    buf[nfft:nfft + n] = x
    win = np.sqrt(0.5 - 0.5 * np.cos(2 * np.pi * np.arange(nfft) / nfft))
    idx = np.arange(nfr)[:, None] * hop + np.arange(nfft)[None, :]
    frames = buf[idx] * win
    X = np.fft.rfft(frames, axis=1)
    f = np.maximum(np.fft.rfftfreq(nfft, 1 / SR), 1e-3)
    tcent = (np.arange(nfr) * hop + nfft / 2 - nfft) / SR
    G = gain_fn(np.clip(tcent, 0, n / SR)[:, None], f[None, :])
    Y = np.fft.irfft(X * G, nfft, axis=1) * win
    out = np.zeros_like(buf)
    ev = Y[0::2].reshape(-1)
    od = Y[1::2].reshape(-1)
    out[:len(ev)] += ev
    out[hop:hop + len(od)] += od[:len(out) - hop]
    return out[nfft:nfft + n]


def dc_block(x):
    return fft_filter(x, lambda f: hp1(f, 18.0) ** 2)


# ---------------------------------------------------------------- reverb (Freeverb-style comb network)
def freeverb_ir(seconds=3.0, room=0.86, damp=0.35, spread=23, predelay=0.018, lo_cut=180.0, hi_cut=9000.0,
                stereo=True):
    """Impulse response of a Schroeder/Freeverb network (8 damped feedback combs in parallel, 4 allpasses in
    series). The recursive network is evaluated exactly in the frequency domain (its transfer function on an
    FFT grid) instead of a per-sample loop; the result is then used for fast FFT convolution."""
    combs = [1116, 1188, 1277, 1356, 1422, 1491, 1557, 1617]
    alls = [556, 441, 341, 225]
    k = SR / 44100.0
    nfft = 1 << int(np.ceil(np.log2(seconds * SR * 1.6)))
    w = 2 * np.pi * np.fft.rfftfreq(nfft, 1 / SR) / SR
    z1 = np.exp(-1j * w)
    chans = []
    for c in range(2 if stereo else 1):
        off = spread * c
        H = np.zeros_like(w, dtype=np.complex128)
        for d in combs:
            D = int((d + off) * k)
            L = (1 - damp) / (1 - damp * z1)          # one-pole lowpass inside the feedback loop
            H += 1.0 / (1 - room * L * np.exp(-1j * w * D))
        for d in alls:
            M = int((d + off) * k)
            zm = np.exp(-1j * w * M)
            H *= (-1 + 1.5 * zm) / (1 - 0.5 * zm)      # Freeverb allpass, g = 0.5
        ir = np.fft.irfft(H, nfft)[:ns(seconds)]
        chans.append(ir)
    ir = np.stack(chans)
    ir = fft_filter(ir, lambda f: hp2(f, lo_cut) * lp2(f, hi_cut))
    ir *= np.exp(-np.arange(ir.shape[1]) / SR / (seconds / 7.0))[None] * ramp_out(ir.shape[1], ns(0.3))[None]
    ir = np.concatenate([np.zeros((ir.shape[0], ns(predelay))), ir], axis=1)
    ir /= np.sqrt(np.sum(ir ** 2) / ir.shape[0])
    return ir


def convolve(x, ir):
    """FFT convolution, x (n,) or (ch,n); ir (ch,m) or (m,). Output length n + m - 1."""
    x = np.atleast_2d(x)
    ir = np.atleast_2d(ir)
    if x.shape[0] == 1 and ir.shape[0] > 1:
        x = np.repeat(x, ir.shape[0], axis=0)
    if ir.shape[0] == 1 and x.shape[0] > 1:
        ir = np.repeat(ir, x.shape[0], axis=0)
    n = x.shape[1] + ir.shape[1] - 1
    nfft = 1 << int(np.ceil(np.log2(n)))
    y = np.fft.irfft(np.fft.rfft(x, nfft, axis=1) * np.fft.rfft(ir, nfft, axis=1), nfft, axis=1)[:, :n]
    return y


def echo(x, delay, feedback=0.4, taps=6, damp_hz=4000.0, pingpong=True):
    """Feedback delay unrolled into taps (each repeat progressively darker). x mono -> stereo out."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 2:
        x = x.mean(0)
    d = ns(delay)
    out = np.zeros((2, len(x) + d * taps))
    y = x.copy()
    for i in range(1, taps + 1):
        y = fft_filter(y, lambda f: lp1(f, damp_hz), pad=4096) * feedback
        ch = (i % 2) if pingpong else None
        if ch is None:
            out[:, d * i:d * i + len(y)] += y
        else:
            out[ch, d * i:d * i + len(y)] += y
    return out


# ---------------------------------------------------------------- dynamics
def _frame_peak(x, hop):
    x = np.atleast_2d(x)
    n = x.shape[1]
    nf = int(np.ceil(n / hop))
    a = np.zeros(nf * hop)
    a[:n] = np.abs(x).max(0)
    return a.reshape(nf, hop).max(1)


def limiter(x, ceiling_db=-1.0, release=0.12, hop=128):
    """Look-ahead (one frame) peak limiter, frame-domain gain computer + linear interpolation."""
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    n = x.shape[1]
    ceil = db(ceiling_db)
    pk = _frame_peak(x, hop)
    g = np.minimum(1.0, ceil / np.maximum(pk, 1e-9))
    nodes = np.minimum(np.concatenate([[1.0], g]), np.concatenate([g, [1.0]]))  # boundary nodes
    a = np.exp(-hop / SR / release)
    s = np.empty_like(nodes)
    cur = 1.0
    for i, v in enumerate(nodes):
        cur = min(v, a * cur + (1 - a) * 1.0)
        s[i] = cur
    gain = np.interp(np.arange(n), np.arange(len(s)) * hop, s)
    y = x * gain[None]
    return np.clip(y, -ceil, ceil)


def compressor(x, thresh_db=-20.0, ratio=2.0, attack=0.01, release=0.18, knee=6.0, hop=240):
    """Gentle RMS-detecting bus compressor (frame domain), stereo-linked."""
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    n = x.shape[1]
    nf = int(np.ceil(n / hop))
    p = np.zeros(nf * hop)
    p[:n] = np.mean(x ** 2, 0)
    lvl = 10 * np.log10(p.reshape(nf, hop).mean(1) + 1e-12)
    over = lvl - thresh_db
    gr = np.where(over <= -knee / 2, 0.0,
                  np.where(over >= knee / 2, over * (1 - 1 / ratio),
                           (1 - 1 / ratio) * (over + knee / 2) ** 2 / (2 * knee)))
    aa, ar = np.exp(-hop / SR / attack), np.exp(-hop / SR / release)
    s = np.empty_like(gr)
    cur = 0.0
    for i, v in enumerate(gr):
        c = aa if v > cur else ar
        cur = c * cur + (1 - c) * v
        s[i] = cur
    gain = db(-np.interp(np.arange(n), (np.arange(nf) + 0.5) * hop, s))
    return x * gain[None]


# ---------------------------------------------------------------- loudness (BS.1770-style)
def _k_weight_mag(f):
    w = 2 * np.pi * f / SR
    z = np.exp(-1j * w)

    def bq(b, a):
        return (b[0] + b[1] * z + b[2] * z * z) / (a[0] + a[1] * z + a[2] * z * z)

    h = bq([1.53512485958697, -2.69169618940638, 1.19839281085285], [1, -1.69065929318241, 0.73248077421585])
    h *= bq([1.0, -2.0, 1.0], [1, -1.99004745483398, 0.99007225036621])
    return np.abs(h)


def lufs(x):
    """Integrated loudness approximation: K-weighted power per 400 ms block (via Parseval), 75% overlap,
    absolute (-70) and relative (-10 LU) gating."""
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    N, hop = ns(0.4), ns(0.1)
    n = x.shape[1]
    if n < N:
        return -70.0
    nb = 1 + (n - N) // hop
    kw = _k_weight_mag(np.fft.rfftfreq(N, 1 / SR)) ** 2
    wts = np.full(len(kw), 2.0)
    wts[0] = 1.0
    if N % 2 == 0:
        wts[-1] = 1.0
    pw = np.zeros(nb)
    idx = np.arange(nb)[:, None] * hop + np.arange(N)[None, :]
    for c in x:
        for s in range(0, nb, 256):
            X = np.fft.rfft(c[idx[s:s + 256]], axis=1)
            pw[s:s + 256] += (np.abs(X) ** 2 * kw * wts).sum(1) / N ** 2
    L = -0.691 + 10 * np.log10(pw + 1e-15)
    g = L > -70
    if not g.any():
        return -70.0
    rel = -0.691 + 10 * np.log10(pw[g].mean()) - 10
    g &= L > rel
    return float(-0.691 + 10 * np.log10(pw[g].mean()))


# ---------------------------------------------------------------- io / misc
def resample(x, sr_in, sr_out=SR):
    """FFT resampling (good for integer-ish ratios on whole clips)."""
    x = np.asarray(x, dtype=np.float64)
    if sr_in == sr_out:
        return x
    n_out = int(round(len(x) * sr_out / sr_in))
    pad = 2048
    xp = np.concatenate([x, np.zeros(pad)])
    X = np.fft.rfft(xp)
    m = int(round(len(xp) * sr_out / sr_in))
    Y = np.zeros(m // 2 + 1, dtype=np.complex128)
    k = min(len(X), len(Y))
    Y[:k] = X[:k]
    # gentle taper near the old Nyquist to avoid ringing
    taper = max(1, k // 40)
    Y[k - taper:k] *= np.cos(0.5 * np.pi * np.arange(taper) / taper) ** 2
    y = np.fft.irfft(Y, m) * (m / len(xp))
    return y[:n_out]


def place(buf, sig, start, gain=1.0):
    """Add sig ((n,) or (ch,n)) into buf ((n,) or (ch,n)) at sample index start (clipped to buffer)."""
    sig = np.asarray(sig)
    n = buf.shape[-1]
    s0 = max(0, start)
    a = s0 - start
    e = min(n, start + sig.shape[-1])
    if e <= s0:
        return
    buf[..., s0:e] += gain * sig[..., a:a + (e - s0)]


def pan2(x, p):
    """Equal-power pan of a mono signal, p in [-1, 1]."""
    a = (p + 1) * np.pi / 4
    return np.stack([x * np.cos(a), x * np.sin(a)])


def write(path, x, subtype="PCM_16"):
    x = np.asarray(x, dtype=np.float64)
    data = x.T if x.ndim == 2 else x
    sf.write(path, data, SR, subtype=subtype)


def stats(x):
    x = np.atleast_2d(x)
    m = x.mean(0)
    pk = np.abs(x).max()
    rms = np.sqrt(np.mean(m ** 2))
    S = np.abs(np.fft.rfft(m * np.hanning(len(m)))) if len(m) > 16 else np.zeros(2)
    f = np.fft.rfftfreq(len(m), 1 / SR)
    cent = float((S * f).sum() / max(S.sum(), 1e-12))
    return dict(peak_db=float(todb(pk)), rms_db=float(todb(rms)), centroid=cent, dur=x.shape[1] / SR,
                nan=bool(np.isnan(x).any()))
