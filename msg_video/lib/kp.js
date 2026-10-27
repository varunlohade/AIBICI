// KP: the Kombu Paper scene kit (approved style, 1080x1920 vertical Reel).
// Built from the approved style-frame helpers in design/frames/kombu-paper/frames.js (window.KP there).
// Cut paper + two-colour risograph: no outlines, no glows, no gradients. Depth = stacked paper + hard offset shadows.
//
// Animation rules baked in here:
//   - Deterministic: no Math.random / Date. Everything is seeded (KP.rng) and a pure function of t.
//   - Stop-motion BOIL: edge jitter and tiny position wobble re-roll on held steps of KP.BOIL_FPS (floor(t*8)).
//     Camera moves stay smooth at 30 fps. Use KP.boil(t) as the jitter seed, KP.held(t) for stepped idle motion.
//   - Expensive full-frame textures (riso grain, paper fibre, flecks, newsprint halftone) are baked ONCE into
//     offscreen canvases and reused (KP.texture). Puppet cut-out layers use bounded offscreen canvases.
//   - Motion language: quick ease-out with a small back-overshoot (1.3), then a 'paper settle' squash
//     (scaleY .94 -> 1 over 4 frames) as if pressed onto the table: KP.arrive / KP.pose.
//
// Index:
//   KP.W/H, KP.P palette, KP.mix/rgba, KP.rng
//   time:    KP.BOIL_FPS, KP.boil(t), KP.held(t), KP.wob(t, seed, amp), KP.clamp/lerp/prog/ease/ep/backOut
//   motion:  KP.arrive(t, t0, d) -> {k, p, sx, sy, a, on}; KP.leave(t, t1, d); KP.pose(ctx, ar, px, py, mode, dist)
//            KP.camera(ctx, cx, cy, s, fn), KP.speak(t, words) jaw 0..1, KP.wordAt(words, t)
//   shapes:  KP.rr/egg/ell/drop/curve/tornRect/torn/move/rot/jitter/path/arcPts/bolt
//   paper:   KP.paper(ctx, pts, fill, {shadow, shadowA, mis, misOff, misA}), KP.brush, KP.squiggle, KP.SH
//   text:    KP.font, KP.text, KP.measure, KP.slip (torn paper label)
//   texture: KP.tile(name), KP.texture(ctx, name, alpha, ox, oy), KP.grainIn(ctx, path, alpha)
//   chars:   KP.face, KP.goldie, KP.cell, KP.arm, KP.fastener, KP.tv, KP.puppet(ctx, kind, ...) (doctor|chef|guru),
//            KP.puppetDoctor (style-frame alias), KP.ripple, KP.stamp, KP.noMsgSign, KP.static (channel switch)
//   captions: KP.caption (strip renderer), KP.captionLayer(ctx, t, id, lines) (drawn by player.html on top of scenes)
(function () {
  const W = 1080, H = 1920, TAU = Math.PI * 2;
  const KP = { W, H };
  window.KP = KP;
  if (window.KZ) KP.scene = (id, def) => KZ.scene(id, def);

  // ---------------------------------------------------------------- colour
  const hex = h => [1, 3, 5].map(i => parseInt(h.slice(i, i + 2), 16));
  const mix = (a, b, t) => { const A = hex(a), B = hex(b); return '#' + A.map((v, i) => Math.round(v + (B[i] - v) * t).toString(16).padStart(2, '0')).join(''); };
  const rgba = (h, a) => { const p = hex(h); return `rgba(${p[0]},${p[1]},${p[2]},${a})`; };
  KP.mix = mix; KP.rgba = rgba;
  // Vivid riso stock (client rule: no greys/beige/cream/muddy). 'rice' = clean white paper for slips/captions.
  const P = KP.P = {
    rice: '#FFFFFF', ink: '#1E1033', plum: '#4422CC', gold: '#FFB000', lemon: '#FFE135',
    pink: '#FF3D8B', bubble: '#FF7AB6', magenta: '#D0157A', coral: '#FF5A4E', tomato: '#F2301B',
    blue: '#1F5BFF', turq: '#00C2C7', kombu: '#00B26B', grape: '#7B3FF2', brass: '#FFC21A',
    lavender: '#A58BFF',
  };
  P.kraft = P.brass;
  P.pink60 = P.bubble; P.pink30 = '#FFA8D2';
  P.pinkDeep = P.magenta;
  P.mouth = mix(P.magenta, P.ink, 0.45);
  P.goldLight = P.lemon;
  P.ink70 = mix(P.ink, P.grape, 0.35);
  P.screenDim = '#2A13A0';
  P.wood = '#FF8A1F';
  P.shadow = rgba(P.ink, 0.22);

  // ---------------------------------------------------------------- rng / time / easing
  KP.rng = seed => { let s = seed >>> 0 || 1; return () => ((s = Math.imul(s ^ (s >>> 15), 1 | s), s = s + Math.imul(s ^ (s >>> 7), 61 | s) ^ s, ((s ^ (s >>> 14)) >>> 0) / 4294967296)); };
  KP.hash = (a, b = 0) => { const r = KP.rng((a * 73856093) ^ (b * 19349663) ^ 0x9e3779b9); r(); return r(); };
  KP.BOIL_FPS = 8;                                   // stop-motion boil: new jitter 8x per second
  KP.boil = t => Math.floor(t * KP.BOIL_FPS + 1e-6);  // integer step index (use as jitter seed offset)
  KP.held = (t, fps = KP.BOIL_FPS) => Math.floor(t * fps + 1e-6) / fps; // time held on stop-motion steps
  // stepped hand-placed wobble: re-rolls every boil step. returns [dx, dy]
  KP.wob = (t, seed = 0, amp = 2) => { const b = KP.boil(t), r = KP.rng(seed * 7717 + b * 131 + 17); r(); return [(r() * 2 - 1) * amp, (r() * 2 - 1) * amp]; };
  KP.wobA = (t, seed = 0, amp = .01) => { const r = KP.rng(seed * 9973 + KP.boil(t) * 37 + 5); r(); return (r() * 2 - 1) * amp; };

  KP.clamp = (x, a = 0, b = 1) => Math.max(a, Math.min(b, x));
  KP.lerp = (a, b, t) => a + (b - a) * t;
  KP.prog = (t, a, b) => KP.clamp((t - a) / (b - a));
  KP.backOut = (x, s = 1.3) => { x = KP.clamp(x); const c3 = s + 1; return 1 + c3 * Math.pow(x - 1, 3) + s * Math.pow(x - 1, 2); };
  KP.ease = {
    lin: x => x,
    out: x => 1 - Math.pow(1 - x, 3),
    in: x => x * x * x,
    inOut: x => x < .5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2,
    back: x => KP.backOut(x, 1.3),
    snap: x => 1 - Math.pow(1 - x, 5),
  };
  KP.ep = (t, a, b, e = 'inOut') => KP.ease[e](KP.prog(t, a, b));
  KP.F = 1 / 30;
  // 'placed by hand' arrival: back-out (1.3) over d, then paper-settle squash (scaleY .94 -> 1 over 4 frames)
  KP.arrive = (t, t0, d = .3) => {
    const k = KP.prog(t, t0, t0 + d), p = t < t0 ? 0 : KP.backOut(k);
    const u = KP.prog(t, t0 + d, t0 + d + 4 * KP.F), sy = t < t0 + d ? 1 : .94 + .06 * KP.ease.out(u);
    return { k, p, sx: 1 + (1 - sy) * .6, sy, a: KP.clamp(k * 4), on: t >= t0 };
  };
  // exit: 0 -> 1 eased-in over d ending at t1
  KP.leave = (t, t1, d = .2) => KP.ease.in(KP.prog(t, t1 - d, t1));
  // apply an arrival around pivot (px, py) (usually the bottom-centre of the piece)
  //   mode 'pop' scales from 0, 'drop' falls from above, 'up' rises from below, 'left'/'right' slide in
  KP.pose = (ctx, ar, px, py, mode = 'pop', dist = 160) => {
    ctx.translate(px, py);
    if (mode === 'pop') ctx.scale(ar.p * ar.sx, ar.p * ar.sy);
    else {
      const o = (1 - ar.p) * dist;
      if (mode === 'drop') ctx.translate(0, -o); else if (mode === 'up') ctx.translate(0, o);
      else if (mode === 'left') ctx.translate(-o, 0); else if (mode === 'right') ctx.translate(o, 0);
      ctx.scale(ar.sx, ar.sy);
    }
    ctx.translate(-px, -py);
  };
  // smooth multiplane camera: looks at (cx, cy) at scale s, frame centre (490, 860) = middle of the key-content box
  KP.CAM = [490, 860];
  KP.camera = (ctx, cx, cy, s, fn) => { ctx.save(); ctx.translate(KP.CAM[0], KP.CAM[1]); ctx.scale(s, s); ctx.translate(-cx, -cy); fn(); ctx.restore(); };
  // current word index in a cues line's words (-1 before the first)
  KP.wordAt = (words, t) => { let k = -1; for (let i = 0; i < words.length; i++) if (words[i].start <= t) k = i; return k; };
  // puppet jaw 0..1 from word timings: flaps ~ every syllable while a word is spoken, held on twos (15 fps)
  KP.speak = (t, words) => {
    const th = Math.floor(t * 15) / 15;
    for (const w of words || []) {
      if (th >= w.start && th < w.end) {
        const syl = Math.max(1, Math.round(w.w.replace(/[^a-z]/gi, '').length / 3.2)), u = (th - w.start) / (w.end - w.start);
        return .25 + .75 * Math.abs(Math.sin(Math.PI * u * syl));
      }
    }
    return 0;
  };

  // ---------------------------------------------------------------- shape builders (closed point lists)
  KP.rr = (x, y, w, h, r, step = 16) => {
    const pts = [], cs = 7;
    const corner = (cx, cy, a0) => { for (let i = 0; i <= cs; i++) { const a = a0 + i / cs * Math.PI / 2; pts.push([cx + Math.cos(a) * r, cy + Math.sin(a) * r]); } };
    const edge = (x0, y0, x1, y1) => { const n = Math.max(1, Math.round(Math.hypot(x1 - x0, y1 - y0) / step)); for (let i = 1; i < n; i++) pts.push([x0 + (x1 - x0) * i / n, y0 + (y1 - y0) * i / n]); };
    corner(x + w - r, y + r, -Math.PI / 2); edge(x + w, y + r, x + w, y + h - r);
    corner(x + w - r, y + h - r, 0); edge(x + w - r, y + h, x + r, y + h);
    corner(x + r, y + h - r, Math.PI / 2); edge(x, y + h - r, x, y + r);
    corner(x + r, y + r, Math.PI); edge(x + r, y, x + w - r, y);
    return pts;
  };
  KP.egg = (cx, cy, rx, ry, egg = 0.06, n = 56) => {
    const pts = [];
    for (let i = 0; i < n; i++) { const a = i / n * TAU, s = Math.sin(a); pts.push([cx + Math.cos(a) * rx * (1 + egg * s), cy + s * ry]); }
    return pts;
  };
  KP.ell = (cx, cy, rx, ry, n = 48) => KP.egg(cx, cy, rx, ry, 0, n);
  KP.drop = (cx, cy, h, n = 44) => {
    const pts = [], w = h / 1.2;
    for (let i = 0; i < n; i++) {
      const a = i / n * TAU;
      pts.push([cx + Math.sin(a) * Math.pow(Math.sin(a / 2), 1.25) * w * 0.56, cy - Math.cos(a) * h * 0.5 + h * 0.04]);
    }
    return pts;
  };
  KP.curve = (ctrl, closed = true, seg = 10) => {
    const n = ctrl.length, out = [], g = i => closed ? ctrl[(i + n) % n] : ctrl[Math.max(0, Math.min(n - 1, i))];
    const last = closed ? n : n - 1;
    for (let i = 0; i < last; i++) {
      const p0 = g(i - 1), p1 = g(i), p2 = g(i + 1), p3 = g(i + 2);
      for (let j = 0; j < seg; j++) {
        const t = j / seg, t2 = t * t, t3 = t2 * t;
        out.push([0, 1].map(k => 0.5 * ((2 * p1[k]) + (-p0[k] + p2[k]) * t + (2 * p0[k] - 5 * p1[k] + 4 * p2[k] - p3[k]) * t2 + (-p0[k] + 3 * p1[k] - 3 * p2[k] + p3[k]) * t3)));
      }
    }
    if (!closed) out.push(ctrl[n - 1].slice());
    return out;
  };
  KP.tornRect = (x, y, w, h, seed, tear = 5) => {
    const r = KP.rng(seed), pts = [];
    for (let i = 0; i <= 12; i++) pts.push([x + w * i / 12, y + (r() - .5) * 1.6]);
    for (let yy = 6; yy < h; yy += 6) pts.push([x + w + (r() - .5) * 2 * tear, y + yy]);
    for (let i = 12; i >= 0; i--) pts.push([x + w * i / 12, y + h + (r() - .5) * 1.6]);
    for (let yy = h - 6; yy > 0; yy -= 6) pts.push([x + (r() - .5) * 2 * tear, y + yy]);
    return pts;
  };
  KP.move = (pts, dx, dy) => pts.map(([x, y]) => [x + dx, y + dy]);
  KP.rot = (pts, cx, cy, a) => { const c = Math.cos(a), s = Math.sin(a); return pts.map(([x, y]) => [cx + (x - cx) * c - (y - cy) * s, cy + (x - cx) * s + (y - cy) * c]); };
  // hand-cut edge jitter along the outline normal (re-seed with + KP.boil(t) to make it boil)
  KP.jitter = (pts, seed, amp = 1.5) => {
    const n = pts.length, r = KP.rng(seed * 7919 + n), ph = [r() * TAU, r() * TAU, r() * TAU], f = [2 + (r() * 2 | 0), 5 + (r() * 3 | 0), 11];
    const out = new Array(n);
    for (let i = 0; i < n; i++) {
      const p = pts[i], a = pts[(i - 1 + n) % n], b = pts[(i + 1) % n];
      let nx = b[1] - a[1], ny = a[0] - b[0]; const l = Math.hypot(nx, ny) || 1; nx /= l; ny /= l;
      const u = i / n;
      const v = .45 * Math.sin(TAU * f[0] * u + ph[0]) + .3 * Math.sin(TAU * f[1] * u + ph[1]) + .15 * Math.sin(TAU * f[2] * u + ph[2]) + .25 * (r() * 2 - 1);
      out[i] = [p[0] + nx * v * amp, p[1] + ny * v * amp];
    }
    return out;
  };
  KP.torn = (pts, seed, amp = 4, step = 7) => {
    const r = KP.rng(seed), out = [];
    for (let i = 0; i < pts.length; i++) {
      const a = pts[i], b = pts[(i + 1) % pts.length], n = Math.max(1, Math.round(Math.hypot(b[0] - a[0], b[1] - a[1]) / step));
      let nx = b[1] - a[1], ny = a[0] - b[0]; const l = Math.hypot(nx, ny) || 1; nx /= l; ny /= l;
      for (let k = 0; k < n; k++) { const u = k / n, o = (r() - .5) * 2 * amp; out.push([a[0] + (b[0] - a[0]) * u + nx * o, a[1] + (b[1] - a[1]) * u + ny * o]); }
    }
    return out;
  };
  KP.path = pts => { const p = new Path2D(); pts.forEach(([x, y], i) => i ? p.lineTo(x, y) : p.moveTo(x, y)); p.closePath(); return p; };
  KP.arcPts = (cx, cy, rx, ry, a0, a1, n = 14) => { const o = []; for (let i = 0; i <= n; i++) { const a = a0 + (a1 - a0) * i / n; o.push([cx + Math.cos(a) * rx, cy + Math.sin(a) * ry]); } return o; };
  // jagged 'zap' bolt polygon from (x,y) along angle a
  KP.bolt = (x, y, len, a, w, seed) => {
    const r = KP.rng(seed), n = 4, L = [], R = [];
    for (let i = 0; i <= n; i++) {
      const u = i / n, d = len * u, side = (i % 2 ? 1 : -1) * w * (.9 + r() * .5), ww = w * (1 - u * .85) * .55;
      const px = x + Math.cos(a) * d - Math.sin(a) * side * (i && i < n ? 1 : 0), py = y + Math.sin(a) * d + Math.cos(a) * side * (i && i < n ? 1 : 0);
      L.push([px - Math.sin(a) * ww, py + Math.cos(a) * ww]); R.push([px + Math.sin(a) * ww, py - Math.cos(a) * ww]);
    }
    return L.concat([[x + Math.cos(a) * (len + w), y + Math.sin(a) * (len + w)]], R.reverse());
  };

  // ---------------------------------------------------------------- paper + brush
  KP.SH = { near: [6, 8], far: [3, 4] };
  KP.paper = (ctx, pts, fill, o = {}) => {
    const path = KP.path(pts);
    if (o.shadow !== false) {
      const [dx, dy] = o.shadow || KP.SH.near;
      ctx.save(); ctx.translate(dx, dy); ctx.fillStyle = rgba(P.ink, o.shadowA ?? .22); ctx.fill(path); ctx.restore();
    }
    if (o.mis) {
      const [mx, my] = o.misOff || [-2.5, 2];
      ctx.save(); ctx.globalCompositeOperation = 'multiply'; ctx.globalAlpha *= o.misA ?? .35;
      ctx.translate(mx, my); ctx.fillStyle = o.mis; ctx.fill(path); ctx.restore();
    }
    if (fill) { ctx.fillStyle = fill; ctx.fill(path); }
    return path;
  };
  KP.brush = (ctx, pts, w, color, o = {}) => {
    if (pts.length < 2) return;
    const taper = o.taper ?? .45, n = pts.length, L = [], R = [], ws = [];
    for (let i = 0; i < n; i++) {
      const a = pts[Math.max(0, i - 1)], b = pts[Math.min(n - 1, i + 1)];
      let nx = -(b[1] - a[1]), ny = b[0] - a[0]; const l = Math.hypot(nx, ny) || 1; nx /= l; ny /= l;
      const u = i / (n - 1), ww = w * (1 - taper + taper * Math.pow(Math.sin(Math.PI * u), .5)) / 2;
      ws.push(ww); L.push([pts[i][0] + nx * ww, pts[i][1] + ny * ww]); R.push([pts[i][0] - nx * ww, pts[i][1] - ny * ww]);
    }
    const poly = L.concat(R.reverse());
    let area = 0; for (let i = 0; i < poly.length; i++) { const p = poly[i], q = poly[(i + 1) % poly.length]; area += p[0] * q[1] - q[0] * p[1]; }
    const path = KP.path(poly), ccw = area < 0;
    path.moveTo(pts[0][0] + ws[0], pts[0][1]); path.arc(pts[0][0], pts[0][1], ws[0], 0, TAU, ccw);
    path.moveTo(pts[n - 1][0] + ws[n - 1], pts[n - 1][1]); path.arc(pts[n - 1][0], pts[n - 1][1], ws[n - 1], 0, TAU, ccw);
    ctx.fillStyle = color; ctx.fill(path);
  };
  // wavy 'handwriting' line (no readable words)
  KP.squiggle = (ctx, x0, x1, y, seed, o = {}) => {
    const r = KP.rng(seed), pts = [], amp = o.amp ?? 6, wl = o.wl ?? 26, ph = r() * TAU;
    for (let x = x0; x <= x1; x += 4) pts.push([x, y + Math.sin((x - x0) / wl * TAU + ph) * amp * (.7 + .3 * Math.sin(x * .013 + ph))]);
    KP.brush(ctx, pts, o.w || 7, o.color || P.ink, { taper: .5 });
  };
  // motion marks: little ink dashes radiating from (x,y)
  KP.marks = (ctx, x, y, r, n, seed, o = {}) => {
    const rr = KP.rng(seed);
    for (let i = 0; i < n; i++) {
      const a = (o.a0 ?? 0) + (o.span ?? TAU) * (i + .5) / n + (rr() - .5) * .3, l = (o.len || 26) * (.8 + rr() * .4);
      KP.brush(ctx, [[x + Math.cos(a) * r, y + Math.sin(a) * r], [x + Math.cos(a) * (r + l), y + Math.sin(a) * (r + l)]], o.w || 6, o.color || P.ink, { taper: .5 });
    }
  };

  // ---------------------------------------------------------------- text
  KP.font = (size, fam = 'Nunito', weight = 800) => `${weight} ${size}px "${fam}"`;
  const mctx = document.createElement('canvas').getContext('2d'), mcache = new Map();
  KP.measure = (ctx, s, o = {}) => {
    const f = KP.font(o.size || 46, o.font || 'Nunito', o.weight ?? 800), key = f + '|' + (o.ls || 0) + '|' + s;
    let w = mcache.get(key);
    if (w === undefined) { mctx.font = f; mctx.letterSpacing = (o.ls || 0) + 'px'; w = mctx.measureText(s).width; mcache.set(key, w); }
    return w;
  };
  KP.text = (ctx, s, x, y, o = {}) => {
    ctx.save();
    ctx.font = KP.font(o.size || 46, o.font || 'Nunito', o.weight ?? 800);
    ctx.letterSpacing = (o.ls || 0) + 'px';
    ctx.textAlign = o.align || 'left'; ctx.textBaseline = o.baseline || 'alphabetic';
    ctx.globalAlpha *= o.alpha ?? 1; ctx.fillStyle = o.color || P.ink; ctx.fillText(s, x, y);
    ctx.restore();
  };
  // text printed on a torn paper slip. lines = [{s, size, font, weight, color, ls}], (x,y) = top-left before rotation
  KP.slip = (ctx, lines, x, y, o = {}) => {
    const padX = o.padX ?? 28, padY = o.padY ?? 18, gap = o.gap ?? 8;
    const ws = lines.map(l => KP.measure(ctx, l.s, l));
    const w = Math.max(...ws) + padX * 2, h = lines.reduce((a, l) => a + l.size, 0) + gap * (lines.length - 1) + padY * 2;
    ctx.save(); ctx.translate(x + w / 2, y + h / 2); ctx.rotate(o.rot || 0); ctx.translate(-w / 2, -h / 2);
    const pts = KP.jitter(KP.tornRect(0, 0, w, h, o.seed || 3, o.tear ?? 5), (o.seed || 3) + (o.boil || 0), 1);
    KP.paper(ctx, pts, o.fill || P.rice, { shadow: o.shadow ?? KP.SH.near, shadowA: o.shadowA });
    if (o.tape) {
      const tp = KP.jitter(KP.tornRect(-22, -10, 90, 34, 77, 3), 5 + (o.boil || 0), .8);
      ctx.save(); ctx.rotate(-.5); KP.paper(ctx, tp, rgba(o.tape, .85), { shadow: false }); ctx.restore();
    }
    let yy = padY;
    lines.forEach((l, i) => {
      yy += l.size;
      const lx = o.align === 'center' ? (w - ws[i]) / 2 : padX;
      KP.text(ctx, l.s, lx, yy - l.size * .2, l);
      yy += gap;
    });
    ctx.restore();
    return { w, h };
  };

  // ---------------------------------------------------------------- textures (baked once)
  const tiles = {};
  const mkCanvas = (w, h) => { const c = document.createElement('canvas'); c.width = w; c.height = h; return c; };
  KP.tile = name => {
    if (tiles[name]) return tiles[name];
    const S = 512, c = mkCanvas(S, S), g = c.getContext('2d'), r = KP.rng({ grain: 101, news: 202, speck: 303 }[name]);
    if (name === 'speck') {
      for (let i = 0; i < 2600; i++) { g.fillStyle = `rgba(243,234,215,${.2 + r() * .5})`; const s = r() < .85 ? 1 : 2; g.fillRect(r() * S | 0, r() * S | 0, s, s); }
      return (tiles[name] = c);
    }
    g.fillStyle = '#fff'; g.fillRect(0, 0, S, S);
    if (name === 'news') {
      const ph = [r() * TAU, r() * TAU, r() * TAU];
      g.fillStyle = '#000';
      const sp = 8;
      for (let i = -2; i < S / sp + 2; i++) for (let j = -2; j < S / sp + 2; j++) {
        const x = (i + (j % 2) * .5) * sp, y = j * sp * .866;
        const d = .5 + .5 * (.5 * Math.sin(TAU * 2 * x / S + ph[0]) * Math.sin(TAU * 3 * y / S + ph[1]) + .5 * Math.sin(TAU * (x + y) / S + ph[2]));
        g.beginPath(); g.arc((x + S) % S, (y + S) % S, .5 + d * 1.9, 0, TAU); g.fill();
      }
    }
    const img = g.getImageData(0, 0, S, S), d = img.data;
    for (let i = 0; i < d.length; i += 4) {
      const n = r(); let v = 0;
      if (n < .22) v = 30 + r() * 110; else if (n < .5) v = r() * 30;
      d[i] = Math.max(0, d[i] - v); d[i + 1] = Math.max(0, d[i + 1] - v); d[i + 2] = Math.max(0, d[i + 2] - v);
    }
    g.putImageData(img, 0, 0);
    for (let i = 0; i < 1400; i++) { g.fillStyle = `rgba(0,0,0,${.15 + r() * .35})`; const x = r() * S, y = r() * S, s = 1 + r() * 2.2; g.beginPath(); g.arc(x, y, s, 0, TAU); g.fill(); }
    g.lineCap = 'round';
    for (let i = 0; i < 110; i++) {
      const x = r() * S, y = r() * S, a = r() * TAU, len = 60 + r() * 180, bend = (r() - .5) * 60;
      g.strokeStyle = `rgba(60,40,20,${.25 + r() * .25})`; g.lineWidth = .6 + r() * .9;
      for (const ox of [-S, 0, S]) for (const oy of [-S, 0, S]) {
        g.beginPath(); g.moveTo(x + ox, y + oy);
        g.quadraticCurveTo(x + ox + Math.cos(a) * len / 2 - Math.sin(a) * bend, y + oy + Math.sin(a) * len / 2 + Math.cos(a) * bend, x + ox + Math.cos(a) * len, y + oy + Math.sin(a) * len);
        g.stroke();
      }
    }
    return (tiles[name] = c);
  };
  // full-frame texture canvases, baked once per (name, offset) and reused every frame
  const fulls = {};
  KP.full = (name, ox = 0, oy = 0) => {
    const key = name + ':' + ox + ':' + oy;
    if (fulls[key]) return fulls[key];
    const c = mkCanvas(W, H), g = c.getContext('2d');
    const pat = g.createPattern(KP.tile(name), 'repeat'); pat.setTransform(new DOMMatrix([1, 0, 0, 1, ox, oy]));
    g.fillStyle = pat; g.fillRect(0, 0, W, H);
    return (fulls[key] = c);
  };
  // one full-frame pass: 'grain'/'news' multiply; 'speck' normal (light flecks on dark grounds)
  KP.texture = (ctx, name, a, ox = 0, oy = 0) => {
    ctx.save(); ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.globalCompositeOperation = name === 'speck' ? 'source-over' : 'multiply'; ctx.globalAlpha = a;
    ctx.drawImage(KP.full(name, ox, oy), 0, 0);
    ctx.restore();
  };
  // grain only inside a path (in current transform); bounded by the path's device bbox to stay cheap
  KP.grainIn = (ctx, path, a = .1, bbox = null) => {
    ctx.save(); ctx.clip(path);
    ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.globalCompositeOperation = 'multiply'; ctx.globalAlpha = a;
    const src = KP.full('grain', 101, 57);
    if (bbox) { const [x, y, w, h] = bbox.map(Math.round); ctx.drawImage(src, x, y, w, h, x, y, w, h); }
    else ctx.drawImage(src, 0, 0);
    ctx.restore();
  };
  // device-space bounding box [x, y, w, h] of a user-space rect under the current transform (clamped to the frame)
  KP.devBox = (ctx, x0, y0, x1, y1) => {
    const m = ctx.getTransform(), cs = [[x0, y0], [x1, y0], [x0, y1], [x1, y1]].map(([x, y]) => [m.a * x + m.c * y + m.e, m.b * x + m.d * y + m.f]);
    const ax = KP.clamp(Math.floor(Math.min(...cs.map(p => p[0]))), 0, W), ay = KP.clamp(Math.floor(Math.min(...cs.map(p => p[1]))), 0, H);
    const bx = KP.clamp(Math.ceil(Math.max(...cs.map(p => p[0]))), 0, W), by = KP.clamp(Math.ceil(Math.max(...cs.map(p => p[1]))), 0, H);
    return [ax, ay, Math.max(1, bx - ax), Math.max(1, by - ay)];
  };
  // bounded offscreen layer for cut-outs (destination-out). bbox = [x0, y0, x1, y1] in the CURRENT user space.
  const layers = {};
  KP.layer = (ctx, key, bbox) => {
    const m = ctx.getTransform();
    const cs = [[bbox[0], bbox[1]], [bbox[2], bbox[1]], [bbox[0], bbox[3]], [bbox[2], bbox[3]]].map(([x, y]) => [m.a * x + m.c * y + m.e, m.b * x + m.d * y + m.f]);
    const bx = Math.floor(Math.min(...cs.map(p => p[0]))), by = Math.floor(Math.min(...cs.map(p => p[1])));
    const bw = Math.ceil(Math.max(...cs.map(p => p[0]))) - bx + 2, bh = Math.ceil(Math.max(...cs.map(p => p[1]))) - by + 2;
    let c = layers[key];
    if (!c || c.width < bw || c.height < bh) c = layers[key] = mkCanvas(Math.max(bw, c ? c.width : 0), Math.max(bh, c ? c.height : 0));
    const g = c.getContext('2d');
    g.setTransform(1, 0, 0, 1, 0, 0); g.globalCompositeOperation = 'source-over'; g.globalAlpha = 1;
    g.clearRect(0, 0, c.width, c.height);   // whole canvas: fractional-offset draws sample 1px past the source rect
    g.setTransform(m.a, m.b, m.c, m.d, m.e - bx, m.f - by);
    return {
      c, g,
      draw(alpha = 1, dx = 0, dy = 0, op = 'source-over') {
        ctx.save(); ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.globalAlpha *= alpha; ctx.globalCompositeOperation = op;
        ctx.drawImage(c, 0, 0, bw, bh, bx + dx, by + dy, bw, bh); ctx.restore();
      },
    };
  };

  // ---------------------------------------------------------------- characters & props
  KP.fastener = (ctx, x, y, r = 6) => {
    ctx.fillStyle = rgba(P.ink, .3); ctx.beginPath(); ctx.arc(x + 1.5, y + 2, r, 0, TAU); ctx.fill();
    ctx.fillStyle = P.kraft; ctx.beginPath(); ctx.arc(x, y, r, 0, TAU); ctx.fill();
    ctx.fillStyle = mix(P.kraft, P.ink, .35); ctx.beginPath(); ctx.arc(x - r * .15, y - r * .15, r * .32, 0, TAU); ctx.fill();
  };
  // face: s ~ head width. eyes 'dot'|'happy'|'closed'|'worried'; mouth 'smile'|'open'|'chomp'|'o'|'wobble'|'none'
  KP.face = (ctx, x, y, s, o = {}) => {
    const ink = o.ink || P.ink, ex = s * .17, er = Math.max(3, s * .045), lw = Math.max(5, s * .045);
    const lx = o.look || 0;
    if (o.cheeks !== false) {
      ctx.save(); ctx.globalCompositeOperation = 'multiply'; ctx.globalAlpha *= .4; ctx.fillStyle = o.cheek || P.pink;
      for (const sx of [-1, 1]) { ctx.beginPath(); ctx.ellipse(x + lx + sx * s * .27, y + s * .1, s * .085, s * .065, 0, 0, TAU); ctx.fill(); }
      ctx.restore();
    }
    for (const sx of [-1, 1]) {
      const cx = x + lx + sx * ex;
      if (o.eyes === 'happy') KP.brush(ctx, KP.arcPts(cx, y + er, er * 1.9, er * 1.7, Math.PI * 1.1, Math.PI * 1.9, 8), lw * .9, ink, { taper: .3 });
      else if (o.eyes === 'closed') KP.brush(ctx, KP.arcPts(cx, y - er * .6, er * 1.8, er * 1.4, Math.PI * .15, Math.PI * .85, 8), lw * .9, ink, { taper: .3 });
      else {
        ctx.fillStyle = ink; ctx.beginPath(); ctx.ellipse(cx, y, er * (o.eyeScale || 1), er * 1.12 * (o.eyeScale || 1), 0, 0, TAU); ctx.fill();
        if (o.eyes === 'worried') KP.brush(ctx, [[cx - sx * er * 2.2, y - er * 3.4], [cx + sx * er * 1.2, y - er * 2.4]], lw * .8, ink, { taper: .3 });
      }
    }
    const mx = x + lx + (o.mouthDx || 0) * s, my = y + s * .14;
    const m = o.mouth || 'smile';
    if (m === 'smile') KP.brush(ctx, KP.arcPts(mx, my - s * .04, s * .08, s * .06, Math.PI * .15, Math.PI * .85, 10), lw, ink, { taper: .5 });
    else if (m === 'wobble') KP.brush(ctx, [[mx - s * .08, my + 2], [mx - s * .04, my - 3], [mx, my + 2], [mx + s * .04, my - 3], [mx + s * .08, my + 2]], lw * .9, ink, { taper: .4 });
    else if (m === 'o') { ctx.fillStyle = ink; ctx.beginPath(); ctx.ellipse(mx, my + 2, s * .035, s * .045, 0, 0, TAU); ctx.fill(); }
    else if (m === 'open' || m === 'chomp') {
      const mw = s * (m === 'chomp' ? .25 : .1), mh = s * (m === 'chomp' ? .21 : .08);
      ctx.fillStyle = o.mouthFill || P.mouth; ctx.beginPath(); ctx.ellipse(mx, my + mh * .3, mw, mh, 0, 0, TAU); ctx.fill();
      return { mx, my: my + mh * .3, mw, mh };
    }
  };
  // Goldie: Dashi Gold paper droplet mascot. (x,y) = belly centre, h = height
  KP.goldie = (ctx, x, y, h, o = {}) => {
    const seed = (o.seed || 1) * 31 + (o.boil || 0);
    ctx.save(); ctx.translate(x, y); ctx.rotate(o.rot || 0);
    if (o.squash) ctx.scale(1 / o.squash, o.squash);
    const pts = KP.jitter(KP.drop(0, 0, h), seed, Math.min(1.5, h * .02));
    KP.paper(ctx, pts, P.gold, { shadow: o.shadow ?? (h > 80 ? KP.SH.near : KP.SH.far), mis: P.tomato, misOff: [-2, 2], misA: .4 });
    KP.paper(ctx, KP.jitter(KP.egg(-h * .12, h * .2, h * .2, h * .13, 0, 24), seed + 3, .8), P.lemon, { shadow: false });
    if (o.steam !== false) {
      const s = h * .5, b = -h * .56;
      KP.brush(ctx, KP.curve([[0, b], [s * .14, b - s * .18], [-s * .06, b - s * .34], [s * .1, b - s * .52]], false, 6), Math.max(4, h * .055), P.goldLight, { taper: .7 });
    }
    KP.face(ctx, o.look || 0, h * .1, h * .62, { eyes: o.eyes || 'dot', mouth: o.mouth || 'smile' });
    ctx.restore();
  };
  // gut cell character: (x, y) = body centre, w,h = body size. returns face info
  KP.cell = (ctx, x, y, w, h, o = {}) => {
    const seed = (o.seed || 1) * 13 + (o.boil || 0);
    const body = KP.jitter(KP.egg(x, y, w / 2, h / 2, .1, 60), seed, 1.5);
    KP.paper(ctx, body, o.fill || P.pink60, { shadow: o.shadow || KP.SH.near, mis: P.blue, misOff: [2.5, -2], misA: .3 });
    const bw = w * .52, by = y + h * .1, bh = h * .26;
    const bibC = KP.jitter(KP.curve([[x - bw * .22, by - bh * .55], [x + bw * .22, by - bh * .55], [x + bw * .5, by - bh * .05], [x + bw * .3, by + bh * .55], [x, by + bh * .4], [x - bw * .3, by + bh * .55], [x - bw * .5, by - bh * .05]], true, 8), seed + 9, 1.2);
    KP.paper(ctx, bibC, o.bib || P.pink30, { shadow: KP.SH.far, shadowA: .16 });
    KP.fastener(ctx, x, by - bh * .42, Math.max(4, w * .022));
    return KP.face(ctx, x + (o.look || 0) * w, y - h * .17, w * .78, { eyes: o.eyes, mouth: o.mouth, mouthDx: o.mouthDx, cheek: P.pink });
  };
  // mitten arm cut as a separate piece pivoting at a fastener
  KP.arm = (ctx, sx, sy, len, ang, thick, fill, o = {}) => {
    ctx.save(); ctx.translate(sx, sy); ctx.rotate(ang);
    const p = KP.jitter(KP.curve([[-thick * .4, -thick * .45], [len * .6, -thick * .5], [len + thick * .25, -thick * .6], [len + thick * .7, 0], [len + thick * .25, thick * .6], [len * .6, thick * .5], [-thick * .4, thick * .45], [-thick * .6, 0]], true, 6), (o.seed || 2) + (o.boil || 0), 1);
    KP.paper(ctx, p, fill, { shadow: o.shadow || KP.SH.far, mis: o.mis, misOff: [2, -2], misA: .3 });
    if (o.thumb) KP.paper(ctx, KP.jitter(KP.egg(len + thick * .2, -thick * .55, thick * .22, thick * .3, 0, 20), 4, .6), fill, { shadow: false });
    KP.fastener(ctx, 0, 0, Math.max(4, thick * .2));
    ctx.restore();
  };
  // the Fifth Ripple: 4 paper rings + wide gold 5th. o.p in [0,1] = spread
  KP.ripple = (ctx, x, y, R, o = {}) => {
    const sq = o.squash ?? .35, p = o.p ?? 1;
    for (let i = 0; i < 5; i++) {
      const rr = R * (i + 1) / 5 * p, gold = i === 4;
      if (rr < 2) continue;
      const pts = KP.jitter(KP.ell(x, y, rr, rr * sq, 56), 40 + i + (o.boil || 0), 1.2);
      pts.push(pts[0]);
      KP.brush(ctx, pts, gold ? 14 : 6, gold ? P.gold : (o.ring || P.rice), { taper: .1 });
    }
  };

  // ---------------------------------------------------------------- TV set (cardboard cut-out). returns screen rect
  KP.tvScreen = (x, y, w, h, o = {}) => {
    const pad = o.pad || Math.min(w, h) * .07, panel = o.panel || 'right';
    const pw = panel === 'right' ? Math.min(w * .22, 150) : 0, ph = panel === 'bottom' ? Math.min(h * .14, 96) : 0;
    return { x: x + pad, y: y + pad, w: w - pad * 2 - pw, h: h - pad * 2 - ph, pad, pw, ph, panel };
  };
  KP.tv = (ctx, x, y, w, h, o = {}) => {
    const seed = (o.seed || 1) + (o.boil || 0), body = o.body || P.kraft;
    const sh = o.shadow || KP.SH.near;
    if (o.antenna) {
      const ax = x + w * (o.antenna), ay = y + 4, ac = o.antColor || mix(body, P.ink, .45);
      KP.brush(ctx, [[ax, ay], [ax - w * .12, ay - h * .3]], 6, ac, { taper: .2 });
      KP.brush(ctx, [[ax, ay], [ax + w * .1, ay - h * .34]], 6, ac, { taper: .2 });
      ctx.fillStyle = ac; ctx.beginPath(); ctx.arc(ax - w * .12, ay - h * .3, 7, 0, TAU); ctx.arc(ax + w * .1, ay - h * .34, 7, 0, TAU); ctx.fill();
      KP.paper(ctx, KP.jitter(KP.egg(ax, ay + 2, 26, 14, 0, 24), seed + 5, 1), mix(body, P.ink, .2), { shadow: sh });
    }
    if (o.feet) for (const fx of [.18, .82]) KP.paper(ctx, KP.jitter(KP.rr(x + w * fx - 16, y + h - 6, 32, 26, 8, 10), seed + fx * 10, 1), mix(body, P.ink, .3), { shadow: sh });
    KP.paper(ctx, KP.jitter(KP.rr(x, y, w, h, 24), seed, 1.5), body, { shadow: sh, shadowA: o.shadowA });
    const S = KP.tvScreen(x, y, w, h, o), { pw, ph } = S;
    KP.paper(ctx, KP.jitter(KP.rr(S.x - 10, S.y - 10, S.w + 20, S.h + 20, 26), seed + 1, 1.2), mix(body, P.ink, .18), { shadow: false });
    KP.paper(ctx, KP.jitter(KP.rr(S.x, S.y, S.w, S.h, 20), seed + 2, 1.2), o.screen || P.screenDim, { shadow: false });
    const knob = mix(body, P.ink, .32), knobHi = mix(body, P.rice, .35);
    if (S.panel === 'right') {
      const cx = x + w - S.pad - pw / 2;
      for (const k of [0, 1]) { const ky = S.y + S.h * (.2 + k * .28), ka = o.knob && k === 0 ? o.knob : 0; KP.paper(ctx, KP.jitter(KP.egg(cx, ky, pw * .24, pw * .24, .05, 28), seed + 7 + k, 1), knob, { shadow: KP.SH.far }); KP.brush(ctx, [[cx, ky], [cx + pw * .17 * Math.cos(-.6 + ka), ky + pw * .17 * Math.sin(-.6 + ka)]], 5, knobHi, { taper: .1 }); }
      for (let i = 0; i < 4; i++) KP.brush(ctx, [[cx - pw * .26, S.y + S.h * (.68 + i * .08)], [cx + pw * .26, S.y + S.h * (.68 + i * .08)]], 5, knob, { taper: .3 });
    } else {
      const cy = y + h - S.pad - ph / 2;
      for (const k of [0, 1]) { const kx = x + w - S.pad - ph * (.5 + k * 1.1), ka = o.knob && k === 1 ? o.knob : 0; KP.paper(ctx, KP.jitter(KP.egg(kx, cy, ph * .3, ph * .3, .05, 28), seed + 7 + k, 1), knob, { shadow: KP.SH.far }); KP.brush(ctx, [[kx, cy], [kx + ph * .21 * Math.cos(-.65 + ka), cy + ph * .21 * Math.sin(-.65 + ka)]], 5, knobHi, { taper: .1 }); }
      for (let i = 0; i < 5; i++) KP.brush(ctx, [[S.x + 10 + i * 22, cy - ph * .22], [S.x + 10 + i * 22, cy + ph * .22]], 6, knob, { taper: .3 });
    }
    if (!o.lit) {
      ctx.save(); ctx.clip(KP.path(KP.rr(S.x, S.y, S.w, S.h, 20)));
      const gx = S.x + S.w * .62;
      KP.paper(ctx, [[gx, S.y - 10], [gx + S.w * .12, S.y - 10], [gx - S.w * .1, S.y + S.h + 10], [gx - S.w * .22, S.y + S.h + 10]], mix(o.screen || P.screenDim, P.rice, .06), { shadow: false });
      ctx.restore();
    }
    return { x: S.x, y: S.y, w: S.w, h: S.h };
  };
  // channel-switch 'static' in cut paper: misregistered riso strips + flecks inside a rect. f = frame index (30fps)
  KP.static = (ctx, S, f, o = {}) => {
    const r = KP.rng(9001 + f * 31), cols = o.colors || [P.pink, P.lemon, P.turq, P.grape, P.rice, P.blue];
    ctx.save(); ctx.clip(KP.path(KP.rr(S.x, S.y, S.w, S.h, 20)));
    ctx.fillStyle = o.ground || P.ink; ctx.fillRect(S.x, S.y, S.w, S.h);
    let y = S.y;
    while (y < S.y + S.h) {
      const hh = 8 + r() * 46, c = cols[(r() * cols.length) | 0], dx = (r() - .5) * 60;
      if (r() < .7) {
        ctx.globalAlpha = .55 + r() * .45; ctx.fillStyle = c;
        ctx.fillRect(S.x + dx - 20, y, S.w + 40, hh);
        ctx.globalCompositeOperation = 'multiply'; ctx.fillStyle = cols[(r() * cols.length) | 0]; ctx.globalAlpha = .4;
        ctx.fillRect(S.x + dx + 6, y + 3, S.w * (.2 + r() * .6), hh * .6);
        ctx.globalCompositeOperation = 'source-over';
      }
      y += hh;
    }
    ctx.globalAlpha = 1; ctx.fillStyle = P.rice;
    for (let i = 0; i < 60; i++) ctx.fillRect(S.x + r() * S.w, S.y + r() * S.h, 3 + r() * 10, 2 + r() * 3);
    ctx.restore();
  };

  // ---------------------------------------------------------------- shadow puppets (fictional cold-open speakers)
  // Ink silhouette on a stick, profile facing right. (x,y) = neck, s = scale.
  //   kind: 'doctor' (head mirror + stethoscope, pointing) | 'chef' (toque, raised ladle) | 'guru' (topknot, selfie phone)
  //   o.jaw 0..1 (hinged wedge), o.gesture = forearm rotation (rad) about the elbow, o.boil, o.stickTo (y of stick end)
  KP.puppet = (ctx, kind, x, y, s, o = {}) => {
    const b = o.boil || 0, top = kind === 'chef' ? -330 : -230, st = o.stickTo ?? y + 400 * s;
    const L = KP.layer(ctx, 'puppet', [x - 140 * s, y + top * s, x + 280 * s, y + 240 * s]), g = L.g;
    const T = pts => pts.map(([px, py]) => [x + px * s, y + py * s]);
    const J = (pts, k) => KP.jitter(T(pts), k + b, 1.4);
    const F = p => g.fill(KP.path(p));
    const sh = [64, -8], el = [146, -44], ga = o.gesture || 0;
    const Hd = pts => KP.rot(pts, el[0], el[1], ga);              // forearm/hand space -> body space
    const hd = Hd([[176, -128]])[0];
    g.fillStyle = P.ink;
    // torso
    if (kind === 'guru') F(J(KP.curve([[-24, -30], [-70, -18], [-92, 30], [-100, 200], [0, 206], [100, 200], [92, 30], [70, -18], [24, -30]], true, 6), 11));
    else F(J(KP.curve([[-26, -30], [-74, -20], [-98, 20], [-104, 120], [-108, 200], [0, 206], [108, 200], [102, 120], [96, 20], [72, -20], [26, -30]], true, 6), 11));
    F(J(KP.rr(-22, -60, 44, 50, 12, 10), 12));                     // neck
    F(J(KP.egg(0, -102, 60, 64, -.04, 52), 13));                   // head
    F(J(KP.egg(54, -98, 15, 13, 0, 18), 14));                      // nose
    if (kind === 'doctor') {
      F(J([[30, -150], [44, -168], [52, -162], [40, -146]], 15));
      F(J(KP.rot(KP.ell(54, -176, 32, 28, 36), 54, -176, -.35), 16));
    } else if (kind === 'chef') {
      F(J(KP.rr(-52, -186, 96, 40, 8, 10), 15));                   // hat band
      F(J(KP.curve([[-56, -176], [-78, -226], [-50, -282], [-6, -300], [40, -290], [70, -244], [48, -176]], true, 6), 16)); // puff
    } else if (kind === 'guru') {
      F(J(KP.egg(-26, -176, 30, 26, 0, 28), 15));                  // topknot
      F(J(KP.curve([[-58, -108], [-74, -40], [-60, 10], [-36, -20], [-30, -90]], true, 6), 16)); // hair down the back
    }
    // arm (separate piece) + mitten hand
    g.lineCap = 'round'; g.lineJoin = 'round'; g.strokeStyle = P.ink;
    g.lineWidth = 36 * s; g.beginPath(); g.moveTo(x + sh[0] * s, y + sh[1] * s); g.lineTo(x + el[0] * s, y + el[1] * s); g.stroke();
    g.lineWidth = 30 * s; g.beginPath(); g.moveTo(x + el[0] * s, y + el[1] * s); g.lineTo(x + hd[0] * s, y + hd[1] * s); g.stroke();
    F(J(Hd(KP.egg(180, -140, 24, 22, 0, 24)), 17));
    if (kind === 'doctor') F(J(Hd(KP.rot(KP.rr(176, -192, 16, 50, 8, 8), 184, -160, .32)), 18));     // pointing finger
    if (kind === 'chef') {                                          // ladle: handle + bowl
      g.lineWidth = 12 * s; const a = T(Hd([[184, -146], [222, -250]])); g.beginPath(); g.moveTo(...a[0]); g.lineTo(...a[1]); g.stroke();
      F(J(Hd(KP.curve([[196, -252], [250, -252], [244, -224], [222, -212], [200, -224]], true, 6)), 18));
    }
    if (kind === 'guru' && o.phone) F(J(Hd(KP.rot(KP.rr(170, -232, 52, 92, 10, 10), 196, -186, .18)), 18));     // phone
    // ---- cut-outs (light shows through)
    g.save(); g.globalCompositeOperation = 'destination-out';
    const jo = KP.clamp(o.jaw ?? .5), hinge = [-4, -78], a0 = -.06, a1 = a0 + .04 + jo * .5;
    F(T([hinge, [hinge[0] + 120 * Math.cos(a0), hinge[1] + 120 * Math.sin(a0)], [hinge[0] + 120 * Math.cos(a1), hinge[1] + 120 * Math.sin(a1)]]));
    g.beginPath(); g.arc(x + 30 * s, y - 118 * s, 7.5 * s, 0, TAU); g.fill();       // eye
    const line = (pts, w) => { g.lineWidth = w * s; g.beginPath(); pts.forEach(([px, py], i) => i ? g.lineTo(x + px * s, y + py * s) : g.moveTo(x + px * s, y + py * s)); g.stroke(); };
    if (kind === 'doctor') {
      line(KP.curve([[-58, -128], [-10, -152], [34, -154]], false, 8), 4.5);
      g.beginPath(); g.arc(x + 55 * s, y - 176 * s, 8 * s, 0, TAU); g.fill();
      g.lineWidth = 3.5 * s; g.beginPath(); g.ellipse(x + 54 * s, y - 176 * s, 21 * s, 18 * s, -.35, 0, TAU); g.stroke();
      line([[-24, -26], [4, 74], [26, -26]], 4.5);
      for (const by of [104, 146]) { g.beginPath(); g.arc(x + 6 * s, y + by * s, 5 * s, 0, TAU); g.fill(); }
      line(KP.curve([[-30, -26], [-46, 24], [-34, 70], [-16, 92]], false, 8).concat(KP.curve([[-16, 92], [-24, 128], [-46, 150]], false, 8)), 5.5);
      line(KP.curve([[30, -26], [40, 26], [8, 72], [-16, 92]], false, 8), 5.5);
      g.lineWidth = 5.5 * s; g.beginPath(); g.arc(x - 50 * s, y + 164 * s, 14 * s, 0, TAU); g.stroke();
      g.beginPath(); g.arc(x - 50 * s, y + 164 * s, 4 * s, 0, TAU); g.fill();
    } else if (kind === 'chef') {
      line([[-50, -172], [44, -172]], 4.5);                          // hat band seam
      line(KP.curve([[-30, -186], [-36, -240], [-20, -280]], false, 6), 4);  // pleats
      line(KP.curve([[8, -186], [10, -250], [20, -288]], false, 6), 4);
      line([[-30, -22], [0, 18], [30, -22]], 6);                     // neckerchief
      for (const [bx, by] of [[-30, 60], [30, 60], [-30, 110], [30, 110], [-30, 160], [30, 160]]) { g.beginPath(); g.arc(x + bx * s, y + by * s, 6 * s, 0, TAU); g.fill(); }
    } else if (kind === 'guru') {
      line(KP.curve([[-44, -148], [-20, -162], [10, -164]], false, 6), 4);   // hair parting
      g.lineWidth = 4 * s; g.beginPath(); g.arc(x + 6 * s, y - 64 * s, 9 * s, 0, TAU); g.stroke();  // hoop earring
      line(KP.curve([[-40, -24], [-10, 30], [24, 34], [44, -20]], false, 8), 5);  // necklace
      g.beginPath(); g.arc(x + 8 * s, y + 38 * s, 8 * s, 0, TAU); g.fill();
      if (o.phone) {
      const ph = T(Hd(KP.rot(KP.rr(178, -222, 36, 72, 6, 10), 196, -186, .18)));
      g.lineWidth = 3.5 * s; g.stroke(KP.path(ph));
      }
    }
    line(KP.arcPts(sh[0], sh[1], 20, 20, -1.4, 1.9, 10), 3);       // arm piece separation slit
    g.restore();
    // sticks (under the silhouette)
    if (st != null) {
      KP.brush(ctx, [[x + 6 * s, y + 180 * s], [x + 2 * s, st]], 16 * s, P.wood, { taper: .05 });
      KP.brush(ctx, [[x + hd[0] * s, y + (hd[1] - 4) * s], [x + 128 * s, st]], 6 * s, mix(P.wood, P.ink, .15), { taper: .05 });
    }
    L.draw(o.shadowA ?? .22, 8 * s, 10 * s);
    L.draw(1);
    for (const [fx, fy] of [sh, el, hinge]) KP.fastener(ctx, x + fx * s, y + fy * s, 6.5 * s);
    if (o.strap !== false) KP.paper(ctx, KP.jitter(KP.rr(x - 18 * s, y + 170 * s, 46 * s, 22 * s, 4, 8), 31 + b, 1), P.lemon, { shadow: KP.SH.far });
    return { hand: [x + hd[0] * s, y + hd[1] * s], mouth: [x + 120 * s, y - 78 * s], head: [x, y - 102 * s] };
  };
  KP.puppetDoctor = (ctx, x, y, s, o = {}) => KP.puppet(ctx, 'doctor', x, y, s, o);

  // ---------------------------------------------------------------- 'NO MSG' sign (cut-paper marquee board)
  // (x, y) = centre, w = width. o.bulbs 0..1 = how many bulbs are lit (chase order), o.no 0..1 = 'NO' lit,
  // o.msg 0..1 = 'MSG' lit, o.boil. Unlit letters print dim; no glows anywhere (flat riso).
  KP.noMsgSign = (ctx, x, y, w, o = {}) => {
    const h = w * .58, b = o.boil || 0, x0 = x - w / 2, y0 = y - h / 2;
    // hanging chains
    for (const cx of [x0 + w * .2, x0 + w * .8]) KP.brush(ctx, [[cx, y0 - 600], [cx, y0 + 10]], 7, P.ink, { taper: .05 });
    KP.paper(ctx, KP.jitter(KP.rr(x0, y0, w, h, 26, 20), 501 + b, 1.6), P.tomato, { shadow: [12, 16], shadowA: .4, mis: P.pink, misOff: [-4, 3], misA: .5 });
    KP.paper(ctx, KP.jitter(KP.rr(x0 + 28, y0 + 28, w - 56, h - 56, 16, 20), 502 + b, 1.2), P.magenta, { shadow: false });
    // bulbs around the frame
    const nb = 22, per = 2 * (w + h), lit = o.bulbs ?? 1;
    for (let i = 0; i < nb; i++) {
      let d = (i + .5) / nb * per, bx, by;
      if (d < w) { bx = x0 + d; by = y0 + 14; } else if ((d -= w) < h) { bx = x0 + w - 14; by = y0 + d; }
      else if ((d -= h) < w) { bx = x0 + w - d; by = y0 + h - 14; } else { d -= w; bx = x0 + 14; by = y0 + h - d; }
      const on = i / nb < lit && !(o.flick && KP.hash(i, o.flick) < .3);
      ctx.fillStyle = on ? P.lemon : P.magenta;
      ctx.beginPath(); ctx.arc(bx, by, w * .016, 0, TAU); ctx.fill();
      if (on) { ctx.fillStyle = P.rice; ctx.beginPath(); ctx.arc(bx - w * .005, by - w * .005, w * .006, 0, TAU); ctx.fill(); }
    }
    // the letters: 'NO' on top (small), 'MSG' big
    const sz1 = w * .12, sz2 = w * .25;
    const word = (s, cx, cy, size, on, dimC) => {
      ctx.save(); ctx.font = KP.font(size, 'Young Serif', 400); ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.letterSpacing = (size * .04) + 'px';
      const [jx, jy] = KP.wob(b / KP.BOIL_FPS, cx | 0, 1.2);
      if (on) { ctx.fillStyle = rgba(P.ink, .45); ctx.fillText(s, cx + jx + 5, cy + jy + 7); ctx.globalCompositeOperation = 'multiply'; ctx.globalAlpha = .6; ctx.fillStyle = P.pink; ctx.fillText(s, cx + jx - 3, cy + jy + 3); ctx.globalCompositeOperation = 'source-over'; ctx.globalAlpha = 1; }
      ctx.fillStyle = on ? (o.letter || P.lemon) : dimC; ctx.fillText(s, cx + jx, cy + jy);
      ctx.restore();
    };
    const dim = P.pink;
    word('NO', x, y0 + h * .29, sz1, (o.no ?? 1) > .5, dim);
    word('MSG', x, y0 + h * .64, sz2, (o.msg ?? 1) > .5, dim);
    return { x: x0, y: y0, w, h };
  };

  // ---------------------------------------------------------------- SHAKER (canonical MSG shaker prop, no lettering)
  // (x, y) = bottom centre, h = height. White glass body, gold crystals (glutamate) inside, grape dome cap.
  KP.shaker = (ctx, x, y, h, o = {}) => {
    const w = h * .44, b = o.boil || 0, sd = o.seed || 7, top = -h * .7;
    ctx.save(); ctx.translate(x, y); ctx.rotate(o.rot || 0); if (o.sx || o.sy) ctx.scale(o.sx || 1, o.sy || 1);
    // faceted glass body: narrow waist, flat base
    const body = KP.jitter(KP.curve([[-w * .42, top], [-w * .5, top * .9], [-w * .5, top * .1], [-w * .44, 0], [w * .44, 0], [w * .5, top * .1], [w * .5, top * .9], [w * .42, top]], true, 6), sd + b, 1.2);
    KP.paper(ctx, body, P.rice, { shadow: o.shadow || KP.SH.near, shadowA: o.shadowA ?? .35, mis: P.turq, misOff: [-3, 2], misA: .45 });
    const r = KP.rng(sd * 3 + 1);
    ctx.save(); ctx.clip(KP.path(body));
    for (let i = 0; i < 46; i++) { const cx = (r() - .5) * w * .95, cy = top * Math.pow(r(), 1.6) * .5, cs = h * (.016 + r() * .014); KP.paper(ctx, KP.rot([[cx - cs, cy - cs], [cx + cs, cy - cs], [cx + cs, cy + cs], [cx - cs, cy + cs]], cx, cy, r() * 3), P.gold, { shadow: false }); }
    ctx.restore();
    // two short slanted glints (cut turquoise slivers)
    for (const [gx, gy, gl] of [[-w * .3, top * .86, h * .07], [-w * .3, top * .7, h * .03]]) KP.brush(ctx, [[gx, gy], [gx + gl * .5, gy + gl]], h * .026, P.turq, { taper: .5 });
    // cap: short dome with a rim
    const cap = KP.jitter(KP.curve([[-w * .5, top + 2], [-w * .46, top - h * .14], [-w * .22, top - h * .25], [w * .22, top - h * .25], [w * .46, top - h * .14], [w * .5, top + 2]], true, 6), sd + 5 + b, 1.2);
    KP.paper(ctx, cap, o.cap || P.pink, { shadow: KP.SH.far, shadowA: .3, mis: P.tomato, misOff: [-3, 2], misA: .45 });
    KP.paper(ctx, KP.jitter(KP.rr(-w * .54, top - h * .04, w * 1.08, h * .07, 6, 10), sd + 6 + b, 1), KP.mix(o.cap || P.pink, P.ink, .25), { shadow: false });
    ctx.fillStyle = P.ink; for (const [hx, hy] of [[-w * .18, top - h * .16], [0, top - h * .2], [w * .18, top - h * .16]]) { ctx.beginPath(); ctx.arc(hx, hy, h * .02, 0, TAU); ctx.fill(); }
    ctx.restore();
  };
  // big '?' cut from ink paper with a hard shadow and a misregistered under-print
  KP.qmark = (ctx, x, y, size, o = {}) => {
    ctx.save(); ctx.translate(x, y); ctx.rotate(o.rot || 0);
    ctx.font = KP.font(size, 'Young Serif', 400); ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillStyle = rgba(P.ink, .35); ctx.fillText('?', size * .06, size * .08);
    ctx.save(); ctx.globalCompositeOperation = 'multiply'; ctx.globalAlpha *= .6; ctx.fillStyle = o.mis || P.pink; ctx.fillText('?', -size * .025, size * .02); ctx.restore();
    ctx.fillStyle = o.color || P.ink; ctx.fillText('?', 0, 0);
    ctx.restore();
  };
  // torn lower third for the fictional speakers: role (the one label) + the required DRAMATIZATION line
  KP.lowerThird = (ctx, role, x, y, o = {}) => {
    const lines = [];
    if (role) lines.push({ s: role, size: 44, ls: 2, color: P.rice });
    lines.push({ s: 'DRAMATIZATION - fictional character', size: 28, weight: 700, color: P.lemon });
    return KP.slip(ctx, lines, x, y, { rot: o.rot ?? -.025, seed: o.seed || 12, boil: o.boil, padX: 34, padY: 18, tape: P.pink, fill: P.blue, shadowA: .3 });
  };

  // ---------------------------------------------------------------- distressed rubber stamp (cached canvas)
  const stamps = {};
  KP.stamp = (key, lines, w, h, o = {}) => {
    if (stamps[key]) return stamps[key];
    const c = mkCanvas(w, h), g = c.getContext('2d'), r = KP.rng(o.seed || 55), ink = o.ink || P.tomato;
    g.fillStyle = ink; g.strokeStyle = ink;
    const outer = KP.jitter(KP.rr(10, 10, w - 20, h - 20, 18, 12), 3, 2.2), inner = KP.jitter(KP.rr(30, 30, w - 60, h - 60, 10, 12), 4, 1.6);
    g.fill(KP.path(outer)); g.globalCompositeOperation = 'destination-out'; g.fill(KP.path(KP.jitter(KP.rr(22, 22, w - 44, h - 44, 13, 12), 5, 1.6))); g.globalCompositeOperation = 'source-over';
    g.lineWidth = 5; g.stroke(KP.path(inner));
    g.textAlign = 'center'; g.textBaseline = 'alphabetic';
    const fs = o.size || 80, lh = fs * 1.12, top = h / 2 - (lines.length * lh) / 2 + fs * .82;
    g.font = `400 ${fs}px "Young Serif"`; g.letterSpacing = (o.ls || 2) + 'px';
    lines.forEach((s, i) => { g.fillText(s, w / 2, top + i * lh, w - 130); });
    g.globalCompositeOperation = 'destination-out';
    for (let i = 0; i < 2600; i++) { const x = r() * w, y = r() * h, fall = Math.max(0, (x / w) * .6 + (1 - y / h) * .5 - .35); g.fillStyle = `rgba(0,0,0,${Math.min(1, .25 + fall * 1.4)})`; g.beginPath(); g.arc(x, y, .6 + r() * (1.2 + fall * 3), 0, TAU); g.fill(); }
    for (let i = 0; i < 26; i++) { const x = w * (.55 + r() * .45), y = h * r() * .55; g.fillStyle = `rgba(0,0,0,${.35 + r() * .5})`; g.beginPath(); g.ellipse(x, y, 6 + r() * 26, 3 + r() * 10, r() * 3, 0, TAU); g.fill(); }
    for (let i = 0; i < 9; i++) { const y = r() * h; g.fillStyle = `rgba(0,0,0,${.15 + r() * .25})`; g.fillRect(r() * w * .5, y, w * (.3 + r() * .5), 1.5 + r() * 2.5); }
    g.globalCompositeOperation = 'source-over';
    return (stamps[key] = c);
  };

  // ================================================================= CAPTIONS
  // Torn white paper strip, Nunito Black 58px (shrunk from the frames' 64px so the strip stays inside y 1300-1480). Spoken words ink, upcoming lavender, current word on a raised chip
  // with a hard shadow + misregistered under-print. Chip colour by act (see KP.CAP.accent).
  // words = [{s, state: 'done'|'on'|'later'}], lines = [[wordIndex...], ...]
  KP.CAP = {
    size: 58, cx: 470, cy: 1390, lh: 1.18, padY: 14, maxW: 780, maxWords: 7, maxChars: 34,
    lavender: P.lavender,
    // per-scene chip colour; [fill, under-print ink, text colour]. Approved: pink cold open, gold glutamate, red 1960s.
    accent: {
      s01_coldopen: [P.pink, P.tomato, P.ink], s02_title: [P.pink, P.tomato, P.ink],
      s06_origin: [P.tomato, P.pink, P.rice],
      // not yet specified by the client (proposed, see report): discovery green, body turquoise, tests blue, outro pink
      s03_ikeda: [P.kombu, P.blue, P.ink], s04_everywhere: [P.kombu, P.blue, P.ink], s05_body: [P.turq, P.blue, P.ink],
      s07_tests: [P.blue, P.grape, P.rice], s08_twist: [P.blue, P.grape, P.rice], s09_outro: [P.pink, P.tomato, P.ink],
    },
    gold: [P.gold, P.tomato, P.ink],
    goldWord: /glutam/i,
  };
  // draw one strip. chip = {i (word index), from (prev index), u (hop progress 0..1), land (0..1 after landing)}
  KP.caption = (ctx, words, lines, cx, cy, o = {}) => {
    const size = o.size || KP.CAP.size, lh = size * KP.CAP.lh, padX = 38, padY = KP.CAP.padY, sp = size * .28;
    const fnt = { size, font: 'Nunito', weight: 900 };
    const ww = words.map(w => KP.measure(ctx, w.s, fnt));
    const lw = lines.map(l => l.reduce((a, i, k) => a + ww[i] + (k ? sp : 0), 0));
    const w = Math.max(...lw) + padX * 2, h = lines.length * lh + padY * 2 - (lh - size) * .5;
    const x0 = cx - w / 2, y = cy - h / 2;
    // word boxes
    const box = [];
    lines.forEach((l, li) => { let xx = cx - lw[li] / 2; const by = y + padY + li * lh + size * .82; l.forEach(i => { box[i] = [xx, by, ww[i]]; xx += ww[i] + sp; }); });
    ctx.save();
    ctx.translate(cx, cy); ctx.rotate(o.rot ?? -.012); ctx.translate(-cx, -cy);
    const strip = KP.paper(ctx, KP.jitter(KP.tornRect(x0, y, w, h, o.seed || 9, 6), (o.seed || 9) + (o.boil || 0), 1), o.fill || P.rice, { shadow: KP.SH.near, shadowA: o.shadowA ?? .4 });
    if (o.grain !== false) KP.grainIn(ctx, strip, .1, KP.devBox(ctx, x0 - 12, y - 6, x0 + w + 12, y + h + 6));
    const ch = o.chip, [fill, mis, txt] = o.accent || KP.CAP.accent.s01_coldopen;
    let lift = null;
    if (ch && ch.i >= 0) {
      // the raised chip hops from the previous word, then settles (squash); drawn under the words
      const A = box[ch.i], B = ch.from != null && ch.from >= 0 && ch.u < 1 ? box[ch.from] : A, u = KP.ease.out(ch.u ?? 1);
      const bx = KP.lerp(B[0], A[0], u), by = KP.lerp(B[1], A[1], u), bw = KP.lerp(B[2], A[2], u);
      const hop = B === A ? 0 : Math.sin(Math.PI * u) * 16;
      const sq = ch.land == null ? 1 : .9 + .1 * KP.ease.out(ch.land);
      lift = [bx + bw / 2, by + size * .28, sq, hop];
      ctx.save();
      ctx.translate(lift[0], lift[1]); ctx.scale(1 + (1 - sq) * .5, sq); ctx.translate(-lift[0], -lift[1]);
      ctx.translate(-3, -6 - hop);
      const cp = KP.jitter(KP.rr(bx - 14, by - size * .9, bw + 28, size * 1.18, 10, 10), 21 + (o.boil || 0), 1);
      KP.paper(ctx, cp, fill, { shadow: [7, 10 + hop * .5], shadowA: .3, mis, misOff: [-3, 2], misA: .45 });
      ctx.restore();
    }
    words.forEach((wd, i) => {
      if (wd.state !== 'on') return KP.text(ctx, wd.s, box[i][0], box[i][1], { ...fnt, color: wd.state === 'done' ? P.ink : KP.CAP.lavender });
      ctx.save();
      if (lift) { ctx.translate(lift[0], lift[1]); ctx.scale(1 + (1 - lift[2]) * .5, lift[2]); ctx.translate(-lift[0], -lift[1]); ctx.translate(-3, -6); }
      KP.text(ctx, wd.s, box[i][0], box[i][1], { ...fnt, color: txt });
      ctx.restore();
    });
    ctx.restore();
    return { w, h };
  };

  // split a cues line into phrase chunks (<= maxWords, <= maxChars), preferring breaks after punctuation
  const chunkLine = words => {
    const n = words.length, C = KP.CAP, best = new Array(n + 1).fill(Infinity), prev = new Array(n + 1).fill(0);
    best[0] = 0;
    const len = (i, j) => words.slice(i, j).reduce((a, w) => a + w.w.length, 0) + (j - i - 1);
    for (let j = 1; j <= n; j++) for (let i = Math.max(0, j - C.maxWords); i < j; i++) {
      const k = j - i, ch = len(i, j);
      if (ch > C.maxChars && k > 1) continue;
      const last = words[j - 1].w, punct = /[.!?]["')]*$|\.\.\.$/.test(last) ? 0 : /[,;:]$/.test(last) ? 1 : (j === n ? 0 : 5);
      const c = best[i] + punct + (k < 3 ? (3 - k) * 2.5 : 0) + Math.pow(Math.max(0, ch - 24) / 4, 2) + 1.5;
      if (c < best[j]) { best[j] = c; prev[j] = i; }
    }
    const out = []; let j = n; while (j > 0) { out.unshift([prev[j], j]); j = prev[j]; }
    return out;
  };
  // lay a chunk out in 1-2 lines (balanced, top line longer); shrink the font if it still does not fit
  const layoutChunk = ws => {
    const C = KP.CAP, sp = C.size * .28, fnt = { size: C.size, font: 'Nunito', weight: 900 };
    const w = ws.map(s => KP.measure(null, s, fnt)), tot = w.reduce((a, b) => a + b, 0) + sp * (ws.length - 1);
    const idx = ws.map((_, i) => i);
    if (tot <= 560 || ws.length < 2) return { lines: [idx], size: tot > C.maxW ? Math.floor(C.size * C.maxW / tot) : C.size };
    let bestK = 1, bestV = Infinity;
    for (let k = 1; k < ws.length; k++) {
      const a = w.slice(0, k).reduce((x, y) => x + y, 0) + sp * (k - 1), b = w.slice(k).reduce((x, y) => x + y, 0) + sp * (ws.length - k - 1);
      const v = Math.max(a, b) + (b > a ? 30 : 0);
      if (v < bestV) { bestV = v; bestK = k; }
    }
    return { lines: [idx.slice(0, bestK), idx.slice(bestK)], size: bestV > C.maxW ? Math.floor(C.size * C.maxW / bestV) : C.size };
  };
  const capCache = {};
  KP.captionChunks = (id, cueLines) => {
    if (capCache[id]) return capCache[id];
    const chunks = [];
    cueLines.forEach((L, li) => {
      const words = L.words || [];
      chunkLine(words).forEach(([a, b]) => {
        const ws = words.slice(a, b);
        chunks.push({ line: li, words: ws, t0: ws[0].start - .1, wEnd: ws[ws.length - 1].end, lineEnd: L.end, ...layoutChunk(ws.map(w => w.w)) });
      });
    });
    chunks.forEach((c, i) => {
      const nx = chunks[i + 1];
      const hold = c.wEnd + .45;                       // a strip lingers briefly after its last word
      c.t1 = nx ? (nx.t0 - hold < .5 ? nx.t0 : hold) : hold;
      c.joined = !!(nx && c.t1 === nx.t0);              // next strip replaces this one directly
      c.seed = 9 + i * 13;
      c.rot = (KP.hash(i, 77) - .5) * .03 - .006;
    });
    return (capCache[id] = chunks);
  };
  // the caption layer, drawn by player.html on top of every scene (unless the scene sets captions: false)
  KP.captionLayer = (ctx, t, id, cueLines) => {
    if (!cueLines || !cueLines.length) return;
    const chunks = KP.captionChunks(id, cueLines);
    let ci = -1;
    for (let i = 0; i < chunks.length; i++) if (t >= chunks[i].t0 && t < chunks[i].t1) ci = i;
    if (ci < 0) return;
    const c = chunks[ci], prevJoined = ci > 0 && chunks[ci - 1].joined;
    const k = KP.wordAt(c.words, t);
    const words = c.words.map((w, i) => ({ s: w.w, state: i < k ? 'done' : i === k ? 'on' : 'later' }));
    const acc = KP.CAP.accent[id] || KP.CAP.accent.s01_coldopen;
    const accent = k >= 0 && KP.CAP.goldWord.test(c.words[k].w) ? KP.CAP.gold : acc;
    let chip = null;
    if (k >= 0) {
      const ts = c.words[k].start, hopD = .09;
      chip = { i: k, from: k - 1, u: k > 0 ? KP.prog(t, ts, ts + hopD) : 1, land: KP.prog(t, ts + (k > 0 ? hopD : 0), ts + (k > 0 ? hopD : 0) + 4 * KP.F) };
    }
    // entrance: pop + slide up (a fresh strip being slapped down), quicker when it replaces a strip
    const ar = KP.arrive(t, c.t0, prevJoined ? .16 : .24);
    const out = c.joined ? 0 : KP.leave(t, c.t1, .16);
    ctx.save();
    ctx.globalAlpha = 1 - out;
    const cy = KP.CAP.cy + (1 - ar.p) * 18 + out * 14, s0 = prevJoined ? .94 : .82;
    ctx.translate(KP.CAP.cx, cy + 60); ctx.scale((s0 + (1 - s0) * ar.p) * ar.sx * (1 - out * .06), (s0 + (1 - s0) * ar.p) * ar.sy * (1 - out * .06)); ctx.translate(-KP.CAP.cx, -cy - 60);
    ctx.globalAlpha *= KP.clamp(ar.k * 3);
    KP.caption(ctx, words, c.lines, KP.CAP.cx, cy, { size: c.size, boil: KP.boil(t), seed: c.seed, rot: c.rot, accent, chip });
    ctx.restore();
  };
})();
