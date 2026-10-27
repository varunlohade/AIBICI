// Afterglow: style frames A, B, C + the reusable neon kit (window.AG).
// Contract: see design/render_styleframes.js. Everything is deterministic (AG.rng), no Math.random/Date.
// The kit (AG.*) is written so the animator can lift it into the scene library as-is:
//   - every drawable is a pre-sampled polyline ([[x,y],...]) => one code path for glow, draw-on (AG.P.trim) and morph (AG.P.morph)
//   - AG.tube = layered neon stroke (halo / mid / body / hot core), additive ('lighter')
//   - AG.neonText = Righteous rendered as a tube, AG.zig = the glutamate mascot, AG.strike = seeded neon flicker
(function () {
  'use strict';
  const W = 1080, H = 1920; // vertical 9:16 Reel
  const AG = window.AG = { W, H };

  // ---------------------------------------------------------------- palette
  const C = AG.C = {
    soy: '#17100F', aubergine: '#2A1730', chalk: '#F5EDE3', core: '#FFF6E0',
    amber: '#FFB547', cyan: '#3FE0C5', pink: '#FF6FB5', violet: '#9C7BFF', red: '#FF4F4F',
    // client rule: vivid deep grounds instead of near-black (one per act family)
    inkViolet: '#22074F', inkTeal: '#023D48', inkIndigo: '#150A4C',
    skin1: '#F7C6A3', skin2: '#C68A62', skin3: '#A86A48', skin3fill: '#8A5236',
  };
  const hex = h => [1, 3, 5].map(i => parseInt(h.slice(i, i + 2), 16));
  AG.rgba = (h, a) => { const p = hex(h); return `rgba(${p[0]},${p[1]},${p[2]},${a})`; };
  AG.mix = (a, b, t) => { const A = hex(a), B = hex(b); return '#' + A.map((v, i) => Math.round(v + (B[i] - v) * t).toString(16).padStart(2, '0')).join(''); };
  const rgba = AG.rgba;
  AG.unlit = { color: '#F5EDE3', a: .25 }; // unlit glass picks up the scene's ambient hue (set per scene; never grey)
  AG.coreMix = .45; // how much of the tube colour tints the hot core (raise for more saturated scenes)

  // ---------------------------------------------------------------- math / rng / timing
  AG.rng = seed => { let a = (seed >>> 0) || 1; return () => { a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; };
  AG.clamp = (x, a = 0, b = 1) => Math.max(a, Math.min(b, x));
  AG.lerp = (a, b, t) => a + (b - a) * t;
  AG.prog = (t, a, b) => AG.clamp((t - a) / (b - a));
  AG.expoOut = x => x >= 1 ? 1 : 1 - Math.pow(2, -10 * x);
  // Neon strike: 0 before t0; seeded on/off chatter for 0.25s; then lit with a 1-2% hum.
  AG.strike = (t, t0 = 0, seed = 1, dur = 0.25) => {
    if (t < t0) return 0;
    const k = (t - t0) / dur;
    if (k < 1) { const r = AG.rng(seed * 7919 + Math.floor(k * 8)); r(); const v = r(); return Math.floor(k * 8) % 2 ? (v < .35 ? .25 : 0) : .55 + v * .45; }
    return 0.985 + 0.015 * Math.sin(t * 52 + seed) * Math.sin(t * 3.1 + seed * 2);
  };
  // breathing / idle drift
  AG.breathe = (t, seed = 0, amp = 1, speed = 1) => amp * (Math.sin(t * 1.3 * speed + seed) * .6 + Math.sin(t * 2.1 * speed + seed * 1.7) * .4);

  // ---------------------------------------------------------------- polylines
  const P = AG.P = {};
  P.line = (x0, y0, x1, y1, n = 2) => { const o = []; for (let i = 0; i < n; i++) { const k = i / (n - 1); o.push([x0 + (x1 - x0) * k, y0 + (y1 - y0) * k]); } return o; };
  P.arc = (cx, cy, r, a0, a1, n = 32, ry = r) => { const o = []; for (let i = 0; i <= n; i++) { const a = a0 + (a1 - a0) * i / n; o.push([cx + Math.cos(a) * r, cy + Math.sin(a) * ry]); } return o; };
  P.circle = (cx, cy, r, n = 64, ry = r) => P.arc(cx, cy, r, 0, Math.PI * 2, n, ry);
  P.cubic = (p0, p1, p2, p3, n = 24) => { const o = []; for (let i = 0; i <= n; i++) { const t = i / n, u = 1 - t; o.push([u * u * u * p0[0] + 3 * u * u * t * p1[0] + 3 * u * t * t * p2[0] + t * t * t * p3[0], u * u * u * p0[1] + 3 * u * u * t * p1[1] + 3 * u * t * t * p2[1] + t * t * t * p3[1]]); } return o; };
  // Catmull-Rom through points (smooth hand-placed curves)
  P.spline = (pts, closed = false, n = 12) => {
    const o = [], m = pts.length, g = i => closed ? pts[(i + m) % m] : pts[AG.clamp(i, 0, m - 1)];
    const segs = closed ? m : m - 1;
    for (let i = 0; i < segs; i++) {
      const p0 = g(i - 1), p1 = g(i), p2 = g(i + 1), p3 = g(i + 2);
      for (let j = 0; j < n; j++) {
        const t = j / n, t2 = t * t, t3 = t2 * t;
        o.push([0, 1].map(d => .5 * ((2 * p1[d]) + (-p0[d] + p2[d]) * t + (2 * p0[d] - 5 * p1[d] + 4 * p2[d] - p3[d]) * t2 + (-p0[d] + 3 * p1[d] - 3 * p2[d] + p3[d]) * t3)));
      }
    }
    o.push(closed ? [...pts[0]] : [...pts[m - 1]]);
    return o;
  };
  // Fillet corners so a tube "bends" (min radius r) instead of making a sharp angle.
  P.round = (pts, r = 14, closed = false, seg = 8) => {
    const n = closed ? pts.length - (dist(pts[0], pts[pts.length - 1]) < .01 ? 1 : 0) : pts.length, o = [];
    for (let i = 0; i < n; i++) {
      const p = pts[i];
      if (!closed && (i === 0 || i === n - 1)) { o.push(p); continue; }
      const a = pts[(i - 1 + n) % n], b = pts[(i + 1) % n];
      const l1 = dist(a, p), l2 = dist(b, p);
      const v1 = [(a[0] - p[0]) / l1, (a[1] - p[1]) / l1], v2 = [(b[0] - p[0]) / l2, (b[1] - p[1]) / l2];
      const th = Math.acos(AG.clamp(v1[0] * v2[0] + v1[1] * v2[1], -1, 1));
      const d = Math.min(r / Math.tan(th / 2 || 1e-3), l1 * .5, l2 * .5);
      const p1 = [p[0] + v1[0] * d, p[1] + v1[1] * d], p2 = [p[0] + v2[0] * d, p[1] + v2[1] * d];
      for (let j = 0; j <= seg; j++) { const t = j / seg, u = 1 - t; o.push([u * u * p1[0] + 2 * u * t * p[0] + t * t * p2[0], u * u * p1[1] + 2 * u * t * p[1] + t * t * p2[1]]); }
    }
    if (closed) o.push([...o[0]]);
    return o;
  };
  P.rrect = (x, y, w, h, r) => P.round([[x, y], [x + w, y], [x + w, y + h], [x, y + h], [x, y]], r, true, 10);
  // CRT glass: a rounded rectangle whose edges bow outward a little (barrel)
  P.crt = (cx, cy, w, h, bulge = 10, r = 36) => P.densify(P.rrect(cx - w / 2, cy - h / 2, w, h, r), 6).map(p => {
    const u = AG.clamp((p[0] - cx) / (w / 2), -1, 1), v = AG.clamp((p[1] - cy) / (h / 2), -1, 1);
    return [p[0] + Math.sign(u) * bulge * (1 - v * v) * Math.abs(u), p[1] + Math.sign(v) * bulge * (1 - u * u) * Math.abs(v)];
  });
  function walk(pts, i, dir, d) { const n = pts.length; let acc = 0, j = i; while (acc < d) { const k = (j + dir + n) % n; const s = dist(pts[j], pts[k]); if (acc + s >= d) { const f = (d - acc) / s; return [pts[j][0] + (pts[k][0] - pts[j][0]) * f, pts[j][1] + (pts[k][1] - pts[j][1]) * f]; } acc += s; j = k; } return pts[j]; }
  function dist(a, b) { return Math.hypot(a[0] - b[0], a[1] - b[1]); }
  P.dist = dist;
  P.len = pts => { let s = 0; for (let i = 1; i < pts.length; i++) s += dist(pts[i - 1], pts[i]); return s; };
  // point + tangent at arc-length fraction f
  P.at = (pts, f) => {
    const L = P.len(pts) * AG.clamp(f); let acc = 0;
    for (let i = 1; i < pts.length; i++) { const s = dist(pts[i - 1], pts[i]); if (acc + s >= L || i === pts.length - 1) { const k = s ? (L - acc) / s : 0; const a = pts[i - 1], b = pts[i]; return { x: a[0] + (b[0] - a[0]) * k, y: a[1] + (b[1] - a[1]) * k, tx: (b[0] - a[0]) / (s || 1), ty: (b[1] - a[1]) / (s || 1) }; } acc += s; }
    return { x: pts[0][0], y: pts[0][1], tx: 1, ty: 0 };
  };
  // sub-path between arc-length fractions a..b (draw-on / un-draw)
  P.trim = (pts, a, b) => {
    const L = P.len(pts), A = L * AG.clamp(a), B = L * AG.clamp(b), o = []; let acc = 0;
    if (B <= A) return o;
    for (let i = 1; i < pts.length; i++) {
      const p = pts[i - 1], q = pts[i], s = dist(p, q), e = acc + s;
      const lerpP = d => { const k = s ? (d - acc) / s : 0; return [p[0] + (q[0] - p[0]) * k, p[1] + (q[1] - p[1]) * k]; };
      if (e >= A && acc <= B) { if (!o.length) o.push(lerpP(Math.max(A, acc))); o.push(e <= B ? q : lerpP(B)); }
      acc = e;
    }
    return o;
  };
  P.resample = (pts, n = 200) => { const o = []; for (let i = 0; i < n; i++) { const a = P.at(pts, i / (n - 1)); o.push([a.x, a.y]); } return o; };
  // One-Line Morph: both shapes resampled to the same count, then lerped. Resample once at load in production.
  P.morph = (A, B, t, n = 200) => { const a = A.length === n ? A : P.resample(A, n), b = B.length === n ? B : P.resample(B, n); return a.map((p, i) => [p[0] + (b[i][0] - p[0]) * t, p[1] + (b[i][1] - p[1]) * t]); };
  P.xf = (pts, { x = 0, y = 0, s = 1, sx = s, sy = s, rot = 0 } = {}) => { const c = Math.cos(rot), si = Math.sin(rot); return pts.map(p => [x + (p[0] * sx) * c - (p[1] * sy) * si, y + (p[0] * sx) * si + (p[1] * sy) * c]); };
  P.map = (pts, fn) => pts.map(fn);
  P.densify = (pts, step = 8) => { const o = [pts[0]]; for (let i = 1; i < pts.length; i++) { const a = pts[i - 1], b = pts[i], n = Math.max(1, Math.ceil(dist(a, b) / step)); for (let j = 1; j <= n; j++) o.push([a[0] + (b[0] - a[0]) * j / n, a[1] + (b[1] - a[1]) * j / n]); } return o; };
  // jagged bolt between two points
  P.bolt = (rng, x0, y0, x1, y1, n = 5, jag = 12) => { const o = [[x0, y0]], dx = x1 - x0, dy = y1 - y0, L = Math.hypot(dx, dy), nx = -dy / L, ny = dx / L; for (let i = 1; i < n; i++) { const k = i / n, j = (i % 2 ? 1 : -1) * jag * (.6 + rng() * .6); o.push([x0 + dx * k + nx * j, y0 + dy * k + ny * j]); } o.push([x1, y1]); return o; };
  // question mark as one tube + dot
  P.qmark = (x, y, s) => ({ hook: P.spline([[x - .32 * s, y - .55 * s], [x - .2 * s, y - .85 * s], [x + .08 * s, y - .92 * s], [x + .32 * s, y - .72 * s], [x + .28 * s, y - .42 * s], [x, y - .22 * s], [x, y + .05 * s]], false, 10), dot: [x, y + .32 * s] });

  // ---------------------------------------------------------------- drawing primitives
  AG.path = (ctx, pts, closed = false, keep = false) => { if (!keep) ctx.beginPath(); if (!pts.length) return; ctx.moveTo(pts[0][0], pts[0][1]); for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1]); if (closed) ctx.closePath(); };
  // THE neon tube. o: color, w (8 hero / 5 secondary / 3 detail), lit (0..1 brightness; 0 = unlit glass), glow (halo multiplier), alpha
  AG.tube = (ctx, pts, o = {}) => {
    if (!pts || pts.length < 2) return;
    const col = o.color || C.amber, w = o.w || 8, lit = o.lit == null ? 1 : o.lit, glow = o.glow == null ? 1 : o.glow, A = o.alpha == null ? 1 : o.alpha;
    ctx.save(); ctx.lineCap = 'round'; ctx.lineJoin = 'round';
    AG.path(ctx, pts, o.closed);
    if (lit < 1 && o.unlit !== false) { ctx.strokeStyle = rgba(o.unlitColor || AG.unlit.color, AG.unlit.a * (1 - lit) * A); ctx.lineWidth = w; ctx.stroke(); }
    if (lit > 0) {
      ctx.globalCompositeOperation = 'lighter';
      const L = lit * A;
      if (o.bloom) { ctx.strokeStyle = rgba(col, .05 * L * glow); ctx.lineWidth = w * 7; ctx.stroke(); }
      ctx.strokeStyle = rgba(col, .10 * L * glow); ctx.lineWidth = w * 3; ctx.stroke();
      ctx.strokeStyle = rgba(col, .30 * L * glow); ctx.lineWidth = w * 1.5; ctx.stroke();
      // body + core in normal blending so the palette hue stays exact on coloured grounds (additive would wash it out)
      ctx.globalCompositeOperation = 'source-over';
      ctx.strokeStyle = rgba(col, .96 * L); ctx.lineWidth = w; ctx.stroke();
      if (o.core !== false) { ctx.strokeStyle = rgba(o.coreColor || AG.mix(C.core, col, o.coreMix == null ? AG.coreMix : o.coreMix), .95 * L); ctx.lineWidth = Math.max(1.2, w * .36); ctx.stroke(); }
    }
    ctx.restore();
  };
  AG.tubes = (ctx, list, o) => list.forEach(p => AG.tube(ctx, p, o));
  // a glowing dot (tube seen end-on / electrode / spark)
  AG.dot = (ctx, x, y, r, col, lit = 1) => AG.tube(ctx, [[x - .01, y], [x + .01, y]], { color: col, w: r * 2, lit });
  // tinted glass fill (15-25%) for mass: faces, bodies, organs
  AG.glass = (ctx, pts, col, a = .2) => { ctx.save(); AG.path(ctx, pts, true); ctx.fillStyle = rgba(col, a); ctx.fill(); ctx.restore(); };
  // emitted light on the wall behind a lit drawing
  AG.pool = (ctx, x, y, r, col, a = .3, sy = 1) => {
    ctx.save(); ctx.translate(x, y); ctx.scale(1, sy);
    const g = ctx.createRadialGradient(0, 0, 0, 0, 0, r); g.addColorStop(0, rgba(col, a)); g.addColorStop(.35, rgba(col, a * .45)); g.addColorStop(1, rgba(col, 0));
    ctx.globalCompositeOperation = 'lighter'; ctx.fillStyle = g; ctx.fillRect(-r, -r, 2 * r, 2 * r); ctx.restore();
  };
  // Neon sign text (Righteous): glass fill + layered tube stroke. Max 3 words per sign.
  AG.neonText = (ctx, text, x, y, o = {}) => {
    const size = o.size || 110, col = o.color || C.amber, lit = o.lit == null ? 1 : o.lit, w = o.w || Math.max(3, size * .05);
    ctx.save(); ctx.font = `${size}px ${o.font || 'Righteous'}`; ctx.textAlign = o.align || 'center'; ctx.textBaseline = o.baseline || 'middle';
    if (o.tracking) ctx.letterSpacing = o.tracking + 'px';
    ctx.lineJoin = 'round'; ctx.lineCap = 'round';
    ctx.fillStyle = rgba(col, (o.glass == null ? .12 : o.glass) * Math.max(lit, .4)); ctx.fillText(text, x, y);
    if (lit < 1) { ctx.strokeStyle = rgba(AG.unlit.color, AG.unlit.a * (1 - lit)); ctx.lineWidth = w; ctx.strokeText(text, x, y); }
    if (lit > 0) {
      ctx.globalCompositeOperation = 'lighter';
      [[w * 3.2, .10], [w * 1.6, .30]].forEach(([lw, a]) => { ctx.strokeStyle = rgba(col, a * lit); ctx.lineWidth = lw; ctx.strokeText(text, x, y); });
      ctx.globalCompositeOperation = 'source-over'; ctx.strokeStyle = rgba(col, .96 * lit); ctx.lineWidth = w; ctx.strokeText(text, x, y);
      ctx.strokeStyle = rgba(AG.mix(C.core, col, .2), .95 * lit); ctx.lineWidth = Math.max(1.2, w * .34); ctx.strokeText(text, x, y);
    }
    ctx.restore();
  };
  // small legible text: filled, never glowing
  AG.label = (ctx, text, x, y, o = {}) => {
    ctx.save(); ctx.font = `${o.weight || 700} ${o.size || 44}px ${o.font || 'Nunito'}`; ctx.textAlign = o.align || 'left'; ctx.textBaseline = o.baseline || 'alphabetic';
    if (o.tracking) ctx.letterSpacing = o.tracking + 'px';
    ctx.fillStyle = o.color || C.chalk; ctx.globalAlpha = o.alpha == null ? 1 : o.alpha; ctx.fillText(text, x, y); ctx.restore();
  };

  // ---------------------------------------------------------------- cached textures (built once from fixed seeds)
  const _cache = {};
  AG.canvas = (w, h) => { const c = document.createElement('canvas'); c.width = w; c.height = h; return c; };
  AG.cache = (key, w, h, fn) => { if (!_cache[key]) { const c = AG.canvas(w, h); fn(c.getContext('2d'), w, h); _cache[key] = c; } return _cache[key]; };
  AG.scanlines = (ctx, a = .08, period = 2, x = 0, y = 0, w = W, h = H) => {
    const pat = AG.cache('scan' + period, 4, period * 2, (g) => { g.fillStyle = '#000'; g.fillRect(0, 0, 4, period); g.fillRect(0, period * 2 - period, 0, 0); });
    const tile = AG.cache('scanT' + period, 8, period * 2, (g) => { g.fillStyle = '#000'; g.fillRect(0, 0, 8, period); });
    ctx.save(); ctx.globalAlpha = a; ctx.fillStyle = ctx.createPattern(tile, 'repeat'); ctx.fillRect(x, y, w, h); ctx.restore(); void pat;
  };
  // faint brick wall on Aubergine (behind signs / TV wall)
  AG.brickWall = (ctx, o = {}) => {
    const base = o.base || '#2A0A5C', hi = o.hi || '#36107A', lo = o.lo || '#1A0545', mortar = o.mortar || '#12033A';
    const tex = AG.cache('brick' + base, W, H, (g) => {
      const r = AG.rng(1908);
      g.fillStyle = mortar; g.fillRect(0, 0, W, H);
      const bh = 46, bw = 132;
      for (let row = 0, y = 0; y < H; row++, y += bh) {
        for (let x = -(row % 2) * bw / 2; x < W; x += bw) {
          g.fillStyle = AG.mix(base, r() > .5 ? hi : lo, r() * .6);
          g.fillRect(x + 3, y + 3, bw - 6, bh - 6);
          if (r() < .35) { g.fillStyle = rgba(hi, .8); g.fillRect(x + 7, y + 4, bw * (.3 + r() * .5), 2); }
        }
      }
    });
    ctx.save(); ctx.globalAlpha = o.alpha == null ? 1 : o.alpha; ctx.drawImage(tex, 0, 0); ctx.restore();
  };
  // CRT static (violet / pink / a little red), 640x360 grain texture drawn at 2x, scrolled per screen
  AG.staticTex = (seed = 5) => AG.cache('static' + seed, 640, 720, (g, w, h) => {
    const img = g.createImageData(w, h), d = img.data, r = AG.rng(seed);
    const V = hex('#8A5BFF'), Pk = hex('#FF4FB0'), R = hex(C.red);
    let band = 0;
    for (let y = 0; y < h; y++) {
      if (y % 6 === 0) band = r();
      for (let x = 0; x < w; x++) {
        const i = (y * w + x) * 4, v = Math.min(1, Math.pow(r(), 1.9) * (.7 + band * .7)), m = r();
        const c = m < .04 ? R : (m < .5 + band * .3 ? V : Pk);
        d[i] = c[0] * v; d[i + 1] = c[1] * v; d[i + 2] = c[2] * v; d[i + 3] = 255;
      }
    }
    g.putImageData(img, 0, 0);
  });
  AG.drawStatic = (ctx, x, y, w, h, t, seed = 1, a = .7) => {
    const tex = AG.staticTex(5), r = AG.rng(seed);
    const off = Math.floor((t * 977 + r() * 400) % 360);
    ctx.save(); ctx.imageSmoothingEnabled = false; ctx.globalAlpha = a; ctx.globalCompositeOperation = 'lighter';
    ctx.drawImage(tex, Math.floor(r() * 200), off, w / 2, h / 2, x, y, w, h);
    // rolling hum bar
    const by = y + ((t * .35 + r()) % 1) * h;
    const g = ctx.createLinearGradient(0, by - 60, 0, by + 60); g.addColorStop(0, rgba(C.violet, 0)); g.addColorStop(.5, rgba(C.violet, .18)); g.addColorStop(1, rgba(C.violet, 0));
    ctx.globalAlpha = 1; ctx.fillStyle = g; ctx.fillRect(x, by - 60, w, 120);
    ctx.restore();
  };
  // Draw fn into an offscreen layer, then composite with a red/cyan chromatic split (broadcast feel).
  AG.chroma = (ctx, fn, dx = 2, a = .55) => {
    const L = AG.cache('chromaL', W, H, () => { }), T = AG.cache('chromaT', W, H, () => { });
    const l = L.getContext('2d'), tt = T.getContext('2d');
    l.setTransform(1, 0, 0, 1, 0, 0); l.globalCompositeOperation = 'source-over'; l.clearRect(0, 0, W, H); fn(l);
    const tint = col => { tt.setTransform(1, 0, 0, 1, 0, 0); tt.globalCompositeOperation = 'source-over'; tt.clearRect(0, 0, W, H); tt.drawImage(L, 0, 0); tt.globalCompositeOperation = 'source-in'; tt.fillStyle = col; tt.fillRect(0, 0, W, H); };
    ctx.save(); ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.globalCompositeOperation = 'lighter';
    tint('#ff3b3b'); ctx.globalAlpha = a * .5; ctx.drawImage(T, -dx, 0);
    tint('#2fd8ff'); ctx.drawImage(T, dx, 0);
    ctx.globalAlpha = 1; ctx.drawImage(L, 0, 0); ctx.restore();
  };
  // soft vignette
  AG.vignette = (ctx, a = .6, inner = .45, col = '#000000') => {
    const D = Math.max(W, H) * .62, g = ctx.createRadialGradient(W / 2, H / 2, D * inner, W / 2, H / 2, D * 1.05);
    g.addColorStop(0, rgba(col, 0)); g.addColorStop(1, rgba(col, a)); ctx.save(); ctx.fillStyle = g; ctx.fillRect(0, 0, W, H); ctx.restore();
  };
  // CRT glitch tear: copy a horizontal band and shove it sideways, with colour fringes
  AG.tear = (ctx, y, h, dx) => {
    const B = AG.cache('tearB', W, 400, () => { }), b = B.getContext('2d');
    b.clearRect(0, 0, W, 400); b.drawImage(ctx.canvas, 0, y, W, h, 0, 0, W, h);
    ctx.save(); ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.drawImage(B, 0, 0, W, h, dx, y, W, h);
    ctx.drawImage(B, W - Math.abs(dx), 0, Math.abs(dx), h, dx > 0 ? 0 : W - Math.abs(dx), y, Math.abs(dx), h);
    ctx.globalCompositeOperation = 'lighter';
    ctx.fillStyle = rgba(C.red, .16); ctx.fillRect(0, y, W, 2); ctx.fillStyle = rgba(C.chalk, .10); ctx.fillRect(0, y + h - 2, W, 2);
    ctx.restore();
  };

  // ---------------------------------------------------------------- characters
  // Zig: glutamate's skeletal formula (HOOC-CH(NH2)-CH2-CH2-COOH) bent from one amber tube.
  // Chain C1..C5 at 120 deg, C=O "ears" up at both carboxyl ends, OH legs down, NH2 a short nub under C2.
  // Eyes sit under the middle vertex (C3). Units: L = bond length in px. phase drives the walk.
  AG.zigShape = (L = 30, o = {}) => {
    const c = Math.cos(Math.PI / 6) * L, s = L / 2, ph = o.phase || 0, lift = o.walk ? Math.sin(ph) : 0, lift2 = o.walk ? Math.sin(ph + Math.PI) : 0;
    const C1 = [0, 0], C2 = [c, s], C3 = [2 * c, 0], C4 = [3 * c, s], C5 = [4 * c, 0];
    const cx = 2 * c, cy = s * .6;
    const sh = p => [p[0] - cx, p[1] - cy];
    const legL = [[-c * .7 + lift * L * .15, s * 1.1 - Math.max(0, lift) * L * .15], [-c * .78 + lift * L * .2, L * 1.25 - Math.max(0, lift) * L * .25]];
    const legR = [[4 * c + c * .7 + lift2 * L * .15, s * 1.1 - Math.max(0, lift2) * L * .15], [4 * c + c * .78 + lift2 * L * .2, L * 1.25 - Math.max(0, lift2) * L * .25]];
    const chain = P.round([legL[1], legL[0], C1, C2, C3, C4, C5, legR[0], legR[1]].map(sh), Math.max(3, L * .2), false, 6);
    const gap = L * .14;
    const ear = (C, dx) => [[C[0] + dx, C[1] - gap], [C[0] + dx, C[1] - L * .9]].map(sh);
    return {
      chain,
      ears: [ear(C1, -L * .09), ear(C1, L * .09), ear(C5, -L * .09), ear(C5, L * .09)],
      nh2: [[C2[0], C2[1] + gap], [C2[0], C2[1] + L * .62]].map(sh),
      nub: sh([C2[0], C2[1] + L * .62]),
      eyes: [sh([cx - L * .2, s * .42]), sh([cx + L * .2, s * .42])],
      eyeLen: L * .22,
    };
  };
  AG.zig = (ctx, x, y, L = 30, o = {}) => {
    const S = AG.zigShape(L, o), tf = p => P.xf(p, { x, y, rot: o.rot || 0, sx: o.flip ? -1 : 1, sy: 1 });
    const col = o.color || C.amber, w = o.w || Math.max(2.5, L * .17), lit = o.lit == null ? 1 : o.lit;
    const tube = o.trim ? p => P.trim(p, o.trim[0], o.trim[1]) : p => p;
    AG.tube(ctx, tube(tf(S.chain)), { color: col, w, lit, glow: o.glow, bloom: o.bloom });
    if (!o.trim || o.trim[1] > .95) {
      S.ears.forEach(e => AG.tube(ctx, tf(e), { color: col, w: w * .7, lit, glow: o.glow }));
      AG.tube(ctx, tf(S.nh2), { color: col, w: w * .8, lit, glow: o.glow });
      const n = tf([S.nub])[0]; AG.dot(ctx, n[0], n[1], w * .75, col, lit);
    }
    if (o.eyes !== false) {
      const bl = o.blink ? .15 : 1;
      S.eyes.forEach(e => { const [ex, ey] = tf([e])[0]; const d = S.eyeLen * .5 * bl; AG.tube(ctx, [[ex, ey - d], [ex, ey + d]], { color: C.core, w: w * .62, lit: lit, glow: .4, core: false }); });
    }
  };
  // round cell/face: dash eyes, optional mouth line, blush dash. Drawn in local frame (rot).
  AG.face = (ctx, x, y, s, o = {}) => {
    const col = o.color || C.pink, rot = o.rot || 0, lk = o.look || [0, 0];
    const tf = p => P.xf(p, { x, y, rot });
    const eh = s * .16 * (o.blink ? .15 : 1), ex = s * .2, ey = -s * .08;
    [-1, 1].forEach(k => AG.tube(ctx, tf([[k * ex + lk[0], ey - eh + lk[1]], [k * ex + lk[0], ey + eh + lk[1]]]), { color: C.core, w: Math.max(3, s * .07), glow: .5, core: false }));
    if (o.blush !== false) [-1, 1].forEach(k => AG.tube(ctx, tf([[k * ex * 1.75 - s * .06 + lk[0], s * .12 + lk[1]], [k * ex * 1.75 + s * .06 + lk[0], s * .12 + lk[1]]]), { color: C.pink, w: Math.max(2.5, s * .045), glow: .6 }));
    if (o.mouth === 'smile') AG.tube(ctx, tf(P.arc(lk[0], s * .1 + lk[1], s * .13, .2 * Math.PI, .8 * Math.PI, 12)), { color: col, w: Math.max(3, s * .055), glow: .6 });
    if (o.mouth === 'open') { const m = tf(P.circle(lk[0] + (o.mouthDx || 0), s * .24 + lk[1], s * (o.mouthR || .17), 28, s * (o.mouthRy || .2))); AG.glass(ctx, m, C.soy, .7); AG.tube(ctx, m, { color: col, w: Math.max(3, s * .06), glow: .7 }); }
  };
  // little amber flame tube ("fuel")
  AG.flame = (ctx, x, y, s, t = 0, col = C.amber, seed = 3) => {
    const f = 1 + .06 * Math.sin(t * 17 + seed) + .04 * Math.sin(t * 29 + seed * 2), sw = .05 * Math.sin(t * 11 + seed);
    const outer = P.spline([[0, 0], [-.42, -.28], [-.32, -.78], [-.05 + sw, -1.2 * f], [.12, -.85], [.38, -.62], [.4, -.22], [0, 0]].map(p => [x + p[0] * s, y + p[1] * s]), false, 10);
    const inner = P.spline([[0, -.12], [-.16, -.32], [-.06 + sw, -.66 * f], [.14, -.36], [0, -.12]].map(p => [x + p[0] * s, y + p[1] * s]), false, 10);
    AG.pool(ctx, x, y - s * .5, s * 1.6, col, .22);
    AG.tube(ctx, outer, { color: col, w: Math.max(3, s * .07) });
    AG.tube(ctx, inner, { color: AG.mix(col, C.core, .35), w: Math.max(2.5, s * .05) });
  };
  // a burst of short tube dashes (sparks / breaking apart)
  AG.sparks = (ctx, x, y, r, n, col, seed = 1, o = {}) => {
    const R = AG.rng(seed);
    for (let i = 0; i < n; i++) {
      const a = R() * Math.PI * 2, d0 = r * (.15 + R() * .55), d1 = d0 + r * (.12 + R() * .25);
      const p = [[x + Math.cos(a) * d0, y + Math.sin(a) * d0], [x + Math.cos(a) * d1, y + Math.sin(a) * d1]];
      if (R() < .35) AG.dot(ctx, p[1][0], p[1][1], (o.w || 4) * .6, col); else AG.tube(ctx, p, { color: col, w: o.w || 4, glow: .9 });
    }
  };

  // ---------------------------------------------------------------- Reel layout (9:16, Instagram safe zones)
  // top 0-220: Reel UI (background only) | 220-1300: story | 1300-1480: caption band | 1480-1920: UI (background only)
  // right 140px kept clear of key content => story column centre ~ x 500.
  AG.SAFE = { top: 220, bottom: 1480, right: W - 140, capY0: 1300, capY1: 1480, cx: 500 };

  // Burned-in caption, "the line lights up": spoken words Chalk, upcoming words a deep tint of the accent,
  // the current word is lit in the accent with a tiny glutamate-zigzag tube under it. Plate = deep scene ink.
  AG.zigLine = (x0, x1, y, amp = 6, period = 24) => {
    const n = Math.max(2, Math.round((x1 - x0) / (period / 2))), pts = [];
    for (let i = 0; i <= n; i++) pts.push([x0 + (x1 - x0) * i / n, y + (i % 2 ? amp : -amp)]);
    return P.round(pts, 3, false, 4);
  };
  AG.caption = (ctx, text, cur, o = {}) => {
    const size = o.size || 62, acc = o.accent || C.amber, cx = o.cx || AG.SAFE.cx, maxW = o.maxW || 820, plate = o.plate || C.inkViolet;
    const y0 = o.y0 || AG.SAFE.capY0, y1 = o.y1 || AG.SAFE.capY1;
    ctx.save(); ctx.font = `900 ${size}px Nunito`; ctx.textBaseline = 'alphabetic';
    const words = text.split(' '), sp = ctx.measureText(' ').width, lines = [[]];
    const total = words.reduce((a, w) => a + ctx.measureText(w).width + sp, -sp), nl = Math.ceil(total / maxW);
    const wrapW = nl > 1 ? total / nl * 1.12 : maxW;
    let lw = 0;
    words.forEach((w, i) => { const ww = ctx.measureText(w).width; if (lw + ww > wrapW && lines[lines.length - 1].length) { lines.push([]); lw = 0; } lines[lines.length - 1].push({ w, i, ww }); lw += ww + sp; });
    const lh = size * 1.18, blockH = lines.length * lh, top = (y0 + y1) / 2 - blockH / 2;
    const widest = Math.max(...lines.map(l => l.reduce((a, b) => a + b.ww, 0) + sp * (l.length - 1)));
    const px = cx - widest / 2 - 40, py = top - 24, pw = widest + 80, ph = blockH + 50;
    ctx.fillStyle = rgba(AG.mix(plate, '#000000', .35), .9); ctx.beginPath(); ctx.roundRect(px, py, pw, ph, 28); ctx.fill();
    ctx.restore();
    AG.tube(ctx, P.rrect(px, py, pw, ph, 28), { color: acc, w: 2.5, lit: .35, core: false, glow: .5, closed: true });
    lines.forEach((l, li) => {
      const lwid = l.reduce((a, b) => a + b.ww, 0) + sp * (l.length - 1); let x = cx - lwid / 2;
      const by = top + lh * (li + 1) - size * .3;
      l.forEach(w => {
        ctx.save(); ctx.font = `900 ${size}px Nunito`; ctx.textBaseline = 'alphabetic';
        if (w.i < cur) { ctx.fillStyle = C.chalk; ctx.fillText(w.w, x, by); }
        else if (w.i > cur) { ctx.fillStyle = AG.mix(acc, plate, .5); ctx.fillText(w.w, x, by); }
        else {
          ctx.shadowColor = rgba(acc, .9); ctx.shadowBlur = 18; ctx.fillStyle = AG.mix(acc, C.core, .15); ctx.fillText(w.w, x, by);
          ctx.restore(); ctx.save();
          AG.tube(ctx, AG.zigLine(x + 4, x + w.ww - 4, by + size * .2, size * .07, size * .34), { color: acc, w: 4.5, glow: .8 });
        }
        ctx.restore(); x += w.ww + sp;
      });
    });
  };
  // vignette into the scene's own deep ink (never to grey)
  AG.inkVignette = (ctx, col, a = .8, inner = .45) => AG.vignette(ctx, a, inner, col);

  // ---------------------------------------------------------------- FRAME A: TV wall, the (fictional) Daytime TV Doctor
  function frameA(ctx, t) {
    const T = 7.5 + (t - 1);
    const cam = { x: AG.breathe(T, 1, 5, .4), y: AG.breathe(T, 2, 4, .4) };
    const CX = 505, CY = 700, BW = 830, BH = 640, GAP = 44;
    AG.unlit = { color: '#B48CFF', a: .5 };
    ctx.fillStyle = C.inkViolet; ctx.fillRect(0, 0, W, H);
    ctx.save(); ctx.translate(cam.x, cam.y);
    AG.brickWall(ctx, { base: '#2C0A63', hi: '#3D1290', lo: '#1D0548', mortar: '#140339' });
    const lit = AG.strike(T, 5.05, 11);
    AG.pool(ctx, CX, CY, 1000, C.violet, .3 * lit, 1.1);
    AG.pool(ctx, CX, CY, 520, '#B07CFF', .22 * lit, 1);

    // concave wall: screens bend toward camera away from the centre; mild tilt
    const warp = p => { const nx = (p[0] - CX) / 800, ny = (p[1] - CY) / 1100; return [CX + (p[0] - CX) * (1 + .06 * nx * nx + .05 * ny * ny), CY + (p[1] - CY) * (1 + .12 * nx * nx + .04 * ny * ny) + (p[0] - CX) * .02]; };
    const screens = [];
    for (let row = -1; row <= 2; row++) for (let col = -1; col <= 1; col++) screens.push({ row, col, cx: CX + col * (BW + GAP), cy: CY + row * (BH + GAP) });
    let centre;
    screens.forEach((s, i) => {
      const isC = !s.row && !s.col;
      const bez = P.map(P.densify(P.rrect(s.cx - BW / 2, s.cy - BH / 2, BW, BH, 46), 10), warp);
      const scr = P.map(P.crt(s.cx, s.cy - 20, BW - 116, BH - 150, 10, 48), warp);
      AG.glass(ctx, bez, '#120330', .96);
      if (!isC) {
        ctx.save(); AG.path(ctx, scr, true); ctx.clip();
        ctx.fillStyle = '#16043A'; ctx.fillRect(s.cx - BW, s.cy - BH, BW * 2, BH * 2);
        AG.drawStatic(ctx, s.cx - BW / 2 - 30, s.cy - BH / 2 - 30, BW + 60, BH, T, 31 + i * 7, s.row === 2 ? .7 : .95);
        AG.scanlines(ctx, .3, 2, s.cx - BW, s.cy - BH, BW * 2, BH * 2);
        const vg = ctx.createRadialGradient(s.cx, s.cy, 150, s.cx, s.cy, 520); vg.addColorStop(0, 'rgba(22,4,58,0)'); vg.addColorStop(1, 'rgba(22,4,58,.7)'); ctx.fillStyle = vg; ctx.fillRect(s.cx - BW, s.cy - BH, BW * 2, BH * 2);
        ctx.restore();
        AG.pool(ctx, s.cx, s.cy - 20, 460, i % 2 ? C.pink : C.violet, .07);
      }
      AG.tube(ctx, bez, { lit: 0, w: 5, closed: true });
      AG.tube(ctx, scr, { lit: 0, w: 3, closed: true });
      const yb = s.cy + BH / 2 - 42;
      [s.cx + BW / 2 - 90, s.cx + BW / 2 - 146].forEach(kx => { const k = warp([kx, yb]); AG.tube(ctx, P.circle(k[0], k[1], 13, 24), { lit: 0, w: 3 }); });
      for (let g = 0; g < 5; g++) { const a = warp([s.cx - BW / 2 + 66 + g * 18, yb - 10]), b = warp([s.cx - BW / 2 + 66 + g * 18, yb + 10]); AG.tube(ctx, [a, b], { lit: 0, w: 3 }); }
      const pip = warp([s.cx + BW / 2 - 200, yb]);
      AG.dot(ctx, pip[0], pip[1], 4, isC ? C.violet : C.red, isC ? 1 : .8);
      if (isC) centre = scr;
    });

    // ---- centre screen: lit Static Violet broadcast
    ctx.save(); AG.path(ctx, centre, true); ctx.clip();
    ctx.fillStyle = '#1B0547'; ctx.fillRect(0, 0, W, H);
    const g0 = ctx.createRadialGradient(CX - 40, CY - 60, 30, CX, CY - 20, 440); g0.addColorStop(0, rgba('#7B4DFF', .45)); g0.addColorStop(1, rgba('#7B4DFF', .04));
    ctx.fillStyle = g0; ctx.fillRect(0, 0, W, H);
    for (let k = 0; k < 7; k++) AG.tube(ctx, [[CX - 420 + k * 140, CY + 270], [CX - 300 + k * 100, CY + 20]], { color: C.violet, w: 2, lit: .18, core: false });
    AG.coreMix = .7;
    AG.chroma(ctx, (l) => { l.translate(cam.x, cam.y); l.translate(CX - 10, CY - 12); l.scale(1.12, 1.12); l.translate(-975, -455); drawDoctor(l, T); drawNeuronSign(l, T); }, 2.5, .45);
    AG.coreMix = .45;
    // on-air bug: a tiny rabbit-ears TV glyph + red pip (no text)
    const bx = CX + 300, by = CY - 210;
    AG.tube(ctx, P.rrect(bx - 26, by - 14, 52, 38, 9), { color: C.chalk, w: 3, lit: .8, glow: .4 });
    AG.tube(ctx, [[bx - 4, by - 14], [bx - 18, by - 34]], { color: C.chalk, w: 3, lit: .8, glow: .4 }); AG.tube(ctx, [[bx + 4, by - 14], [bx + 18, by - 34]], { color: C.chalk, w: 3, lit: .8, glow: .4 });
    AG.dot(ctx, bx - 52, by + 5, 7, C.red, Math.sin(T * 6) > 0 ? 1 : .35);
    AG.scanlines(ctx, .25, 2);
    const vg = ctx.createRadialGradient(CX, CY - 20, 230, CX, CY - 20, 490); vg.addColorStop(0, 'rgba(18,3,48,0)'); vg.addColorStop(1, 'rgba(18,3,48,.6)');
    ctx.fillStyle = vg; ctx.fillRect(0, 0, W, H);
    ctx.restore();
    AG.tube(ctx, centre, { color: C.violet, w: 3.5, lit: .55 * lit, closed: true, core: false });

    // foreground: a pink neon tube passing near camera, unlit + defocused
    const fg = P.spline([[-80, 1560], [160, 1640], [330, 1800], [420, 2000]], false, 16);
    AG.tube(ctx, fg, { lit: 0, w: 52, alpha: .35, unlitColor: C.pink }); AG.tube(ctx, fg, { lit: 0, w: 24, alpha: .45, unlitColor: C.pink });
    ctx.restore();

    lowerThird(ctx, 'TV DOCTOR', 'DRAMATIZATION - fictional character', C.violet, T, 60, 1086);
    AG.scanlines(ctx, .07, 2);
    AG.inkVignette(ctx, '#0E0228', .75, .45);
    AG.caption(ctx, 'It overexcites your brain cells.', 1, { accent: C.violet, plate: C.inkViolet });
  }

  function lowerThird(ctx, role, tag, col, T, x, y) {
    ctx.save(); ctx.font = '400 30px "Space Mono"'; const w = ctx.measureText(tag).width + 130, h = 150; ctx.restore();
    ctx.save(); ctx.fillStyle = rgba('#12032E', .94); ctx.beginPath(); ctx.roundRect(x, y, w, h, 18); ctx.fill(); ctx.restore();
    AG.tube(ctx, [[x + 30, y + 30], [x + 30, y + h - 30]], { color: col, w: 6, lit: AG.strike(T, 5.1, 3) });
    AG.label(ctx, role, x + 62, y + 72, { font: 'Space Mono', size: 52, weight: 700, tracking: 2 });
    ctx.save(); ctx.fillStyle = C.red; ctx.fillRect(x + 64, y + 96, 12, 30); ctx.restore();
    AG.label(ctx, tag, x + 90, y + 121, { font: 'Space Mono', size: 30, weight: 400, color: '#E6D9FF' });
  }

  function drawDoctor(ctx, T) {
    const V = C.violet, nod = AG.breathe(T, 4, 3, 1.2), talk = (Math.sin(T * 19) > -.2) ? 1 : .35;
    ctx.save(); ctx.translate(0, nod * .6);
    // head (3/4 view, facing right toward the neuron), ear, side-part. No eyes: no likeness.
    const head = P.spline([[884, 262], [940, 280], [966, 330], [962, 380], [940, 420], [898, 444], [852, 434], [818, 400], [804, 350], [812, 298], [842, 268]], true, 10);
    AG.glass(ctx, head, V, .14);
    AG.tube(ctx, head, { color: V, w: 7 });
    AG.tube(ctx, P.spline([[812, 340], [796, 348], [798, 372], [814, 382]], false, 8), { color: V, w: 4 });
    AG.tube(ctx, P.spline([[830, 300], [858, 268], [912, 262], [950, 284], [962, 318]], false, 10), { color: V, w: 5 });
    AG.tube(ctx, P.spline([[858, 268], [880, 286], [902, 292]], false, 8), { color: V, w: 3 });
    // mouth: open, mid-claim
    const mo = 22 + 14 * talk;
    const mouth = P.spline([[898, 382], [928, 374], [958, 380], [952, 380 + mo * .75], [930, 384 + mo], [906, 382 + mo * .6]], true, 8);
    AG.glass(ctx, mouth, '#0A0220', .85); AG.tube(ctx, mouth, { color: V, w: 4 });
    [0, 1, 2].forEach(k => AG.tube(ctx, P.arc(974, 396, 26 + k * 18, -.5, .5, 10), { color: V, w: 3, lit: .8 - k * .2, core: false }));
    // neck, collar, tie, lab coat
    AG.tube(ctx, [[866, 440], [864, 482]], { color: V, w: 5 }); AG.tube(ctx, [[922, 430], [926, 482]], { color: V, w: 5 });
    AG.tube(ctx, P.round([[848, 478], [896, 520], [944, 478]], 10), { color: V, w: 5 });
    AG.tube(ctx, P.round([[890, 528], [882, 600], [896, 650], [910, 600], [902, 528]], 8), { color: C.pink, w: 4 });
    AG.tube(ctx, P.spline([[846, 476], [770, 492], [716, 540], [692, 640], [688, 700]], false, 10), { color: V, w: 7 });
    AG.tube(ctx, P.spline([[946, 476], [1022, 492], [1076, 540], [1100, 640], [1104, 700]], false, 10), { color: V, w: 7 });
    AG.tube(ctx, P.spline([[846, 488], [812, 560], [826, 700]], false, 8), { color: V, w: 5 });
    AG.tube(ctx, P.spline([[946, 488], [980, 560], [966, 700]], false, 8), { color: V, w: 5 });
    // stethoscope (Kelp Cyan)
    const K = C.cyan;
    AG.tube(ctx, P.spline([[818, 590], [826, 520], [850, 470], [872, 452]], false, 10), { color: K, w: 6 });
    AG.tube(ctx, P.spline([[918, 452], [946, 470], [966, 520], [968, 584], [948, 620]], false, 10), { color: K, w: 6 });
    AG.tube(ctx, P.round([[818, 590], [800, 626], [812, 640]], 8), { color: K, w: 4 }); AG.tube(ctx, P.round([[818, 590], [836, 626], [826, 640]], 8), { color: K, w: 4 });
    AG.tube(ctx, P.circle(940, 638, 17, 24), { color: K, w: 5 });
    ctx.restore();
  }

  function drawNeuronSign(ctx, T) {
    const x = 1150, y = 440, Pk = C.pink;
    const lit = AG.strike(T, 6.2, 5);
    const soma = P.spline([[x - 4, y - 40], [x + 30, y - 30], [x + 44, y + 4], [x + 26, y + 36], [x - 10, y + 42], [x - 40, y + 18], [x - 38, y - 22]], true, 8);
    AG.pool(ctx, x, y, 170, Pk, .08 * lit);
    AG.glass(ctx, soma, Pk, .2);
    AG.tube(ctx, soma, { color: Pk, w: 6, lit });
    const dendrite = (a, len) => {
      const p0 = [x + Math.cos(a) * 44, y + Math.sin(a) * 44], p1 = [x + Math.cos(a) * (44 + len), y + Math.sin(a) * (44 + len)];
      AG.tube(ctx, P.spline([p0, [(p0[0] + p1[0]) / 2 + Math.sin(a) * 8, (p0[1] + p1[1]) / 2 - Math.cos(a) * 8], p1], false, 8), { color: Pk, w: 4, lit });
      [-.55, .5].forEach(da => { const l2 = len * .45; AG.tube(ctx, [p1, [p1[0] + Math.cos(a + da) * l2, p1[1] + Math.sin(a + da) * l2]], { color: Pk, w: 3, lit }); });
    };
    [-2.2, -1.35, -.4, 2.6, 3.4].forEach((a, i) => dendrite(a, 40 + (i % 2) * 18));
    const ax = P.spline([[x + 30, y + 34], [x + 58, y + 90], [x + 64, y + 140], [x + 84, y + 180]], false, 10);
    AG.tube(ctx, ax, { color: Pk, w: 4, lit });
    [[-.6, 30], [.2, 34], [.9, 28]].forEach(([da, l]) => { const e = [x + 84 + Math.cos(1 + da) * l, y + 180 + Math.sin(1 + da) * l]; AG.tube(ctx, [[x + 84, y + 180], e], { color: Pk, w: 3, lit }); AG.dot(ctx, e[0], e[1], 5, Pk, lit); });
    // overexcited: Alarm Red crackle, re-seeded every ~3 frames
    const RR = AG.rng(400 + Math.floor(T * 10));
    for (let i = 0; i < 7; i++) {
      const a = -2.9 + i * .78 + RR() * .3, r0 = 62 + RR() * 14, r1 = r0 + 34 + RR() * 30;
      AG.tube(ctx, P.bolt(RR, x + Math.cos(a) * r0, y + Math.sin(a) * r0, x + Math.cos(a) * r1, y + Math.sin(a) * r1, 4, 9), { color: C.red, w: 3.5, lit: RR() < .85 ? 1 : .3 });
    }
  }

  // ---------------------------------------------------------------- FRAME B: gut-lining cells eating glutamate
  function frameB(ctx, t) {
    const T = 18.5 + (t - 1);
    AG.unlit = { color: '#7FF5E0', a: .4 };
    const bg = ctx.createRadialGradient(420, 760, 80, 500, 900, 1300); bg.addColorStop(0, '#07606E'); bg.addColorStop(.55, C.inkTeal); bg.addColorStop(1, '#012631');
    ctx.fillStyle = bg; ctx.fillRect(0, 0, W, H);

    // gut lining: lower-left (near, big) up to upper-right (far, small)
    const base = P.resample(P.spline([[120, 2150], [500, 1660], [770, 1220], [960, 880], [1290, 330]], false, 20), 700);
    const kAt = f => AG.lerp(1.3, .45, f);
    const target = [800, 1170];
    let best = 0, bd = 1e9;
    for (let f = 0; f < 1; f += .002) { const a = P.at(base, f), d = Math.hypot(a.x - target[0], a.y - target[1]); if (d < bd) { bd = d; best = f; } }
    const vs = [{ f: best, hero: true }];
    for (let f = best, k; f > -.05;) { k = kAt(f); f -= .074 * k * (f === best ? 1.45 : 1); vs.unshift({ f }); }
    for (let f = best, k; f < 1.05;) { k = kAt(f); f += .074 * k * (f === best ? 1.45 : 1); vs.push({ f }); }
    const outline = [], cells = [];
    vs.forEach((v, i) => {
      const a = P.at(base, AG.clamp(v.f)), tx = a.tx, ty = a.ty, nx = ty, ny = -tx;
      const ext = v.f < 0 ? v.f : v.f > 1 ? v.f - 1 : 0, bx = a.x + tx * ext * 1600, byy = a.y + ty * ext * 1600;
      const k = kAt(v.f) * (v.hero ? 1.6 : 1), h = 270 * k, R = 60 * k, nw = 30 * k, bw = 48 * k;
      const sway = AG.breathe(T, i * 1.7, 7 * k, 1.1);
      const L = (u, w) => [bx + tx * u + nx * w + tx * sway * (w / h), byy + ty * u + ny * w + ty * sway * (w / h)];
      const pts = [L(-bw * 1.5, -4 * k), L(-bw, 6 * k), L(-nw, h * .45), L(-nw * 1.05, h - R * 1.55)];
      for (let j = 0; j <= 14; j++) { const ang = Math.PI + .5 - (Math.PI + 1) * j / 14; pts.push(L(Math.cos(ang) * R, h - R + Math.sin(ang) * R)); }
      pts.push(L(nw * 1.05, h - R * 1.55), L(nw, h * .45), L(bw, 6 * k), L(bw * 1.5, -4 * k));
      outline.push(...pts);
      const bc = L(0, h - R);
      cells.push({ x: bc[0], y: bc[1], R, k, hero: !!v.hero, i, nx, ny });
    });
    const lining = P.spline(outline, false, 5);

    // tissue: a solid deep magenta "glass" (mixed as an opaque tint so pink-over-teal never goes muddy)
    AG.pool(ctx, 700, 1000, 800, C.pink, .12, 1.1);
    ctx.save(); AG.path(ctx, [...lining, [1400, 2100], [200, 2300]], true);
    const tg = ctx.createLinearGradient(500, 900, 1100, 1500); tg.addColorStop(0, '#5E0F63'); tg.addColorStop(1, '#3A0848');
    ctx.fillStyle = tg; ctx.fill(); ctx.restore();
    // capillary inside the tissue (Kelp Cyan = body), dim
    AG.tube(ctx, base.map(p => [p[0] + 130, p[1] + 60]), { color: C.pink, w: 3, lit: .45, core: false, glow: .5 });
    AG.tube(ctx, base.map(p => [p[0] + 240, p[1] + 130]), { color: C.cyan, w: 3, lit: .4, core: false, glow: .5 });
    cells.forEach(c => { if (c.hero) AG.glass(ctx, P.circle(c.x, c.y, c.R, 40), '#8A1C86', .9); });

    const DOF = 1330;
    ctx.save(); ctx.beginPath(); ctx.rect(0, 0, W, DOF); ctx.clip(); AG.tube(ctx, lining, { color: C.pink, w: 8 }); ctx.restore();
    ctx.save(); ctx.beginPath(); ctx.rect(0, DOF, W, H); ctx.clip();
    AG.tube(ctx, lining, { color: C.pink, w: 40, lit: .16, glow: 1.4, core: false }); AG.tube(ctx, lining, { color: C.pink, w: 18, lit: .3, core: false, glow: .4 });
    ctx.restore();

    // glutamate stream (the only amber): from the far channel down toward the hero
    const hero = cells.find(c => c.hero);
    const lane = P.resample(P.spline([[960, 140], [840, 400], [730, 640], [hero.x + hero.R * 1.6, hero.y - hero.R * 1.3]], false, 16), 300);
    for (let j = 0; j < 11; j++) {
      const R = AG.rng(900 + j), ff = ((j * .091 + T * .045) % 1);
      if (ff > .82) continue;
      const a = P.at(lane, ff), s = AG.lerp(16, 38, Math.pow(ff, 1.1)), off = (R() - .5) * 300 * (ff * .8 + .3);
      AG.zig(ctx, a.x - a.ty * off, a.y + a.tx * off, s, { rot: .5 + (R() - .5) * 1.0 + Math.sin(T * 1.7 + j) * .15, phase: T * 6 + j, walk: true, glow: .9 });
    }
    cells.forEach(c => {
      if (c.hero || c.y > DOF - 20 || c.y < -50) return;
      AG.face(ctx, c.x, c.y, c.R * 1.7, { rot: -.2, mouth: c.i % 3 === 0 ? 'open' : 'smile', mouthR: .09, mouthRy: .08, look: [c.R * .1, -c.R * .05], blink: c.i % 5 === 1 });
    });

    // HERO cell: swallowing a Zig, which breaks into amber sparks inside; flame above = fuel
    const fx = hero.x, fy = hero.y, FR = hero.R;
    AG.pool(ctx, fx, fy, FR * 2.6, C.amber, .2);
    // sparks inside the body (lower bulb + neck): the Zig breaking down
    const fcx = fx + hero.nx * FR * .1, fcy = fy + hero.ny * FR * .1 - FR * .08;
    AG.sparks(ctx, fx - hero.nx * FR * .85, fy - hero.ny * FR * .85, FR * .5, 14, C.amber, 21, { w: 4 });
    AG.face(ctx, fcx, fcy, FR * 1.6, { rot: -.2, mouth: 'open', mouthR: .15, mouthRy: .13, mouthDx: FR * .3, look: [FR * .14, -FR * .1] });
    // the Zig going in, feet first (its first 15% is already inside the mouth)
    const ZL = 30, mxp = fcx + FR * .49, myp = fcy + FR * .19;
    AG.zig(ctx, mxp + .7 * 2.1 * ZL, myp - .7 * 2.1 * ZL, ZL, { rot: -.78, trim: [.14, 1], glow: 1.1 });
    AG.flame(ctx, fx - FR * .1, fy - FR * 1.3, FR * .85, T, C.amber, 4);

    // FUEL sign (Kelp Cyan) upper-left of the channel (right edge belongs to the Reel UI)
    const sl = AG.strike(T, 17.6, 9);
    AG.pool(ctx, 360, 440, 380, C.cyan, .16 * sl, .6);
    AG.neonText(ctx, 'FUEL', 360, 440, { size: 170, color: '#5CFFE4', lit: sl, tracking: 12, glass: .18 });
    AG.inkVignette(ctx, '#011C26', .7, .5);
    AG.caption(ctx, 'your intestines love glutamate.', 3, { accent: C.amber, plate: '#04303A' });
  }

  // ---------------------------------------------------------------- FRAME C: the 1968 letter + headline stamp (amber phosphor CRT)
  const PHOS = '#FFC870';
  const LET = { x: 60, y: 245, w: 620, h: 1020 };
  // the letter is pictures, not words: squiggle "type", plus three glowing symptom glyphs where the symptoms are
  function letterLayer() {
    return AG.cache('letter1968', W, H, (g) => {
      const { x: x0, y: y0, w, h } = LET, R = AG.rng(1968);
      g.fillStyle = '#21125F'; g.beginPath(); g.roundRect(x0, y0, w, h, 8); g.fill();
      g.strokeStyle = PHOS; g.lineWidth = 3; g.stroke();
      g.lineCap = 'round'; g.lineJoin = 'round';
      const squig = (y, x1, x2, col = '#E89A3A', lw = 3) => {
        g.strokeStyle = col; g.lineWidth = lw; let x = x1;
        while (x < x2 - 24) {
          const ww = Math.min(x2 - x, 26 + R() * 90); g.beginPath();
          for (let k = 0; k <= ww; k += 3) { const yy = y + Math.sin((x + k) * .38) * 4 + Math.sin((x + k) * .11) * 2; k ? g.lineTo(x + k, yy) : g.moveTo(x + k, yy); }
          g.stroke(); x += ww + 18;
        }
      };
      squig(y0 + 70, x0 + w - 200, x0 + w - 50, PHOS, 4);        // date line
      squig(y0 + 140, x0 + 50, x0 + 300, PHOS, 4);               // salutation
      let y = y0 + 210;
      squig(y, x0 + 100, x0 + w - 50); y += 52;
      squig(y, x0 + 50, x0 + w - 60); y += 52;
      for (let i = 0; i < 3; i++) { squig(y, x0 + 50, x0 + 200 + i * 60); y += 66; }
      for (let i = 0; i < 9; i++) { squig(y, x0 + 50, x0 + w - 50 - R() * 80); y += 50; }
    });
  }
  // symptom glyphs inside the letter: numbness (tingling hand), weakness (drooping arm), palpitations (heart + beat)
  function symptomGlyphs(ctx, T) {
    const o = { color: PHOS, w: 4, glow: .8 }, y0 = LET.y + 210 + 104, xs = [LET.x + 290, LET.x + 350, LET.x + 410];
    // numbness: open hand outline + tingle dots
    const hx = xs[0] + 40, hy = y0;
    AG.tube(ctx, P.round([[hx - 18, hy + 14], [hx - 18, hy - 10], [hx - 10, hy - 22], [hx - 10, hy - 4], [hx - 2, hy - 26], [hx - 2, hy - 4], [hx + 6, hy - 24], [hx + 6, hy - 2], [hx + 14, hy - 16], [hx + 14, hy + 14], [hx - 18, hy + 14]], 3), o);
    [[hx - 32, hy - 26], [hx + 30, hy - 30], [hx + 34, hy + 2]].forEach(([x, y]) => AG.dot(ctx, x, y, 3.5, PHOS));
    // weakness: a battery running low
    const wx = xs[1] + 80, wy = y0 + 66;
    AG.tube(ctx, P.rrect(wx - 30, wy - 16, 54, 32, 6), o); AG.tube(ctx, [[wx + 30, wy - 6], [wx + 30, wy + 6]], o); AG.tube(ctx, [[wx - 20, wy - 6], [wx - 20, wy + 6]], { ...o, color: C.red });
    // palpitations: heart + fast beat line (the zigzag again)
    const px = xs[2] + 110, py = y0 + 132;
    AG.tube(ctx, P.spline([[px, py + 20], [px - 24, py], [px - 22, py - 18], [px - 8, py - 22], [px, py - 10], [px + 8, py - 22], [px + 22, py - 18], [px + 24, py], [px, py + 20]], false, 6), { ...o, color: C.red });
    AG.tube(ctx, P.round([[px + 36, py], [px + 48, py], [px + 56, py - 18], [px + 66, py + 16], [px + 76, py - 12], [px + 84, py], [px + 100, py]], 3), { ...o, w: 3, color: C.red });
  }
  const sketch = {
    wine(ctx, x, y, s, o) { const p = q => q.map(([a, b]) => [x + a * s, y + b * s]); AG.tube(ctx, P.round(p([[-14, -210], [-14, -160], [-46, -110], [-46, 0], [46, 0], [46, -110], [14, -160], [14, -210]]), 16 * s), o); AG.tube(ctx, p([[-46, -70], [46, -70]]), { ...o, w: 3 }); AG.tube(ctx, p([[-46, -24], [46, -24]]), { ...o, w: 3 }); AG.tube(ctx, p([[-18, -224], [18, -224]]), { ...o, w: 3 }); },
    salt(ctx, x, y, s, o) { const p = q => q.map(([a, b]) => [x + a * s, y + b * s]); AG.tube(ctx, P.round(p([[-40, 0], [-44, -130], [-30, -176], [30, -176], [44, -130], [40, 0], [-40, 0]]), 14 * s), o); AG.tube(ctx, P.spline(p([[-30, -176], [-20, -206], [20, -206], [30, -176]]), false, 6), o); [[-12, -192], [0, -196], [12, -192]].forEach(([dx, dy]) => AG.dot(ctx, x + dx * s, y + dy * s, 2.5, PHOS)); [[-14, -70, .3], [14, -84, -.4], [0, -46, .9]].forEach(([cx, cy, r]) => AG.tube(ctx, P.xf(P.rrect(-11, -11, 22, 22, 4), { x: x + cx * s, y: y + cy * s, rot: r, s }), { ...o, w: 3 })); },
    shaker(ctx, x, y, s, o) { const p = q => q.map(([a, b]) => [x + a * s, y + b * s]); AG.tube(ctx, P.round(p([[-42, 0], [-46, -150], [-32, -196], [32, -196], [46, -150], [42, 0], [-42, 0]]), 14 * s), o); AG.tube(ctx, p([[-46, -150], [46, -150]]), { ...o, w: 3 }); [[-14, -176], [0, -182], [14, -176], [-7, -164], [7, -164]].forEach(([dx, dy]) => AG.dot(ctx, x + dx * s, y + dy * s, 2.5, PHOS)); AG.zig(ctx, x, y - 76 * s, 17 * s, { glow: .5, w: 3 }); },
  };
  function frameC(ctx, t) {
    const T = 29.0 + (t - 1), slam = 28.86; // headline landed ~4 frames ago
    const since = T - slam, flash = since < 0 ? 0 : Math.exp(-since * 10), kick = since < 0 ? 0 : Math.exp(-since * 14) * 10;
    AG.unlit = { color: PHOS, a: .4 };
    const g = ctx.createRadialGradient(420, 760, 60, 520, 900, 1300); g.addColorStop(0, '#2A1478'); g.addColorStop(.5, C.inkIndigo); g.addColorStop(1, '#0B0533');
    ctx.fillStyle = g; ctx.fillRect(0, 0, W, H);
    ctx.save(); ctx.translate(kick * .6, -kick * .4);
    const L = letterLayer(), Lb = AG.cache('letter1968blur', W, H, (b) => { b.filter = 'blur(8px)'; b.drawImage(L, 0, 0); });
    ctx.save(); ctx.globalCompositeOperation = 'lighter'; ctx.globalAlpha = .45; ctx.drawImage(Lb, 0, 0); ctx.restore();
    ctx.drawImage(L, 0, 0);
    symptomGlyphs(ctx, T);

    // margin: question marks hanging over wine, salt, MSG (no labels; the pictures carry it)
    const sk = { color: PHOS, w: 4, glow: .6 }, mx = 820;
    [['wine', .54], ['salt', .62], ['shaker', .56]].forEach(([fn, s], i) => {
      const top = 236 + i * 222, base = top + 190;
      const bob = AG.breathe(T, i * 2, 4, 2), q = P.qmark(mx + 2, top + 26 + bob, 50);
      AG.tube(ctx, q.hook, { color: C.amber, w: 6 }); AG.dot(ctx, q.dot[0], q.dot[1], 6, C.amber);
      sketch[fn](ctx, mx, base, s, sk);
    });

    // the stamp: the 1968 headline, quoted, shown as the thing being critiqued
    const sx = 490, sy = 1105, rot = -6 * Math.PI / 180, sc = 1 + kick * .012, sw = 850, sh = 330;
    ctx.save(); ctx.translate(sx, sy); ctx.rotate(rot); ctx.scale(sc, sc);
    AG.pool(ctx, 0, 0, 760, C.red, .24 * (.4 + flash), .7);
    ctx.save(); ctx.fillStyle = rgba('#2A0630', .9); ctx.beginPath(); ctx.roundRect(-sw / 2, -sh / 2, sw, sh, 22); ctx.fill(); ctx.restore();
    ctx.save(); ctx.shadowColor = C.red; ctx.shadowBlur = 40; // the frame's single shadowBlur hero
    AG.tube(ctx, P.rrect(-sw / 2, -sh / 2, sw, sh, 22), { color: C.red, w: 9, closed: true, bloom: true });
    ctx.restore();
    AG.tube(ctx, P.rrect(-sw / 2 + 20, -sh / 2 + 20, sw - 40, sh - 40, 12), { color: C.red, w: 3.5, closed: true, glow: .6 });
    ['\u201CCHINESE-RESTAURANT', 'SYNDROME\u201D'].forEach((s, i) => AG.neonText(ctx, s, 0, -56 + i * 112, { size: i ? 118 : 66, color: C.red, tracking: i ? 8 : 0 }));
    [[-sw / 2 - 26, -sh / 2 - 18, -1, -1], [sw / 2 + 26, sh / 2 + 18, 1, 1]].forEach(([x, y, dx, dy]) => {
      [[0, 0], [1, .45], [.45, 1]].forEach(([a, bb], i) => AG.tube(ctx, [[x, y], [x + dx * (24 + a * 30) * (i ? 1 : .8), y + dy * (24 + bb * 30) * (i ? 1 : .8)]], { color: C.red, w: 4 }));
    });
    ctx.restore();
    ctx.restore();

    ctx.save(); ctx.globalCompositeOperation = 'lighter'; ctx.fillStyle = rgba(C.red, .14 * flash + .03); ctx.fillRect(0, 0, W, H); ctx.restore();
    // glitch tear across the CRT (through the letter, not the headline)
    AG.tear(ctx, 392, 26, 30); AG.tear(ctx, 1010, 6, -18); AG.tear(ctx, 1268, 6, 14);
    AG.scanlines(ctx, .22, 2);
    const v = ctx.createRadialGradient(540, 900, 420, 540, 960, 1250); v.addColorStop(0, 'rgba(11,5,51,0)'); v.addColorStop(.72, 'rgba(11,5,51,.35)'); v.addColorStop(1, 'rgba(8,3,40,.92)');
    ctx.fillStyle = v; ctx.fillRect(0, 0, W, H);
    const edge = P.crt(540, 960, W - 36, H - 36, 6, 70);
    ctx.save(); ctx.beginPath(); ctx.rect(0, 0, W, H); AG.path(ctx, edge, true, true); ctx.fillStyle = '#080228'; ctx.fill('evenodd'); ctx.restore();
    AG.tube(ctx, edge, { color: C.amber, w: 2, lit: .3, core: false, glow: .4, closed: true });
    AG.caption(ctx, 'under a catchy headline:', 3, { accent: C.amber, plate: C.inkIndigo });
  }

  window.STYLEFRAMES = {
    size: [W, H],
    fonts: [
      { family: 'Righteous', url: '/assets/Righteous.ttf', weight: '400' },
      { family: 'Nunito', url: '/assets/Nunito.ttf', weight: '200 1000' },
      { family: 'Space Mono', url: '/assets/SpaceMono-Regular.ttf', weight: '400' },
      { family: 'Space Mono', url: '/assets/SpaceMono-Bold.ttf', weight: '700' },
      { family: 'Courier Prime', url: '/assets/CourierPrime-Regular.ttf', weight: '400' },
      { family: 'Courier Prime', url: '/assets/CourierPrime-Bold.ttf', weight: '700' },
    ],
    frames: { A: frameA, B: frameB, C: frameC },
  };
})();
