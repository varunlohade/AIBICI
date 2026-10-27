// Direction 3 — LUNCHBOX LAB — style frames A, B, C (vertical 9:16 Reel, 1080x1920).
// The first half of this file is a reusable kit ("LB") meant to become the scene kit if this
// direction is chosen. Everything is deterministic (seeded RNG, no Math.random, no Date).
//
// Kit overview
//   LB.C, LB.F             palette + font stacks (from DIRECTIONS.md)
//   LB.SAFE, LB.CAP, LB.col  9:16 Reel safe area, caption band, 6-column grid
//   paths:   LB.rr, LB.ell, LB.circ, LB.capsule, LB.poly, LB.hexRod, LB.star, LB.xf
//   render:  LB.draw (ink outline + off-register fill + hard ink shadow + halftone shade)
//            LB.blob (several parts merged under one outline), LB.hose (rubber-hose limb)
//            LB.shade (halftone crescent on the shadow side), LB.halftone (graded dot field)
//            LB.dots (cached dot CanvasPattern)
//   layout:  LB.compartment (recessed tray cell), LB.lid (Tray Grey lid), LB.sticker (plate)
//   type:    LB.text (outlined display type), LB.fit, LB.caption (Reel caption sticker)
//   faces:   LB.eye, LB.brow, LB.mouth, LB.cheek
//   chars:   LB.hand, LB.miniUmi, LB.gutCell, LB.tvDoc, LB.brain, LB.zap
(function () {
  'use strict';
  const LB = (window.LB = window.LB || {});
  const TAU = Math.PI * 2;

  // ---------------------------------------------------------------- palette & type
  const C = (LB.C = {
    cream: '#FFF4E2', ink: '#1E1A1D', butter: '#FFD95A', tomato: '#FF5A3C', pool: '#4FB3E8',
    pea: '#7BC96F', gum: '#FF9EC4', lilac: '#B9A4F2', grey: '#D8D2C8',
    skin1: '#FFD1B0', skin2: '#D99A6C', skin3: '#8C5636', white: '#FFFFFF',
  });
  LB.F = {
    display: "'Dela Gothic One'", label: "'Atkinson Hyperlegible'",
    mono: "'Space Mono'", doc: "'Courier Prime'",
  };
  LB.W = 1080; LB.H = 1920;
  // Reel-safe: top 220 (UI), bottom 440 (UI + captions), right 140 (buttons).
  LB.SAFE = { x: 60, y: 220, w: 880, h: 1060 };
  LB.CAP = { y0: 1300, y1: 1480, cx: 500 };
  // 6-column grid inside the safe width, 24px gutters.
  LB.col = (i, span = 1, cols = 6, gut = 24, S = LB.SAFE) => {
    const cw = (S.w - gut * (cols - 1)) / cols;
    return { x: S.x + i * (cw + gut), w: span * cw + (span - 1) * gut };
  };

  // ---------------------------------------------------------------- rng / math
  LB.rng = seed => {
    if (window.KZ && KZ.rng) return KZ.rng(seed);
    let s = seed >>> 0 || 1;
    return () => { s = (s + 0x6D2B79F5) >>> 0; let t = s; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  };
  LB.lerp = (a, b, k) => a + (b - a) * k;
  LB.clamp = (x, a = 0, b = 1) => Math.max(a, Math.min(b, x));
  // idle "2-beat bounce": returns 0..1..0 twice per period
  LB.bounce = (t, period = 1.2, phase = 0) => Math.abs(Math.sin((t / period + phase) * Math.PI * 2));
  LB.deg = d => d * Math.PI / 180;

  // ---------------------------------------------------------------- paths (Path2D)
  LB.rr = (x, y, w, h, r = 24) => { const p = new Path2D(); p.roundRect(x, y, w, h, r); return p; };
  LB.ell = (cx, cy, rx, ry, rot = 0) => { const p = new Path2D(); p.ellipse(cx, cy, Math.max(.1, rx), Math.max(.1, ry), rot, 0, TAU); return p; };
  LB.circ = (cx, cy, r) => LB.ell(cx, cy, r, r);
  LB.capsule = (x1, y1, x2, y2, r) => {
    const a = Math.atan2(y2 - y1, x2 - x1), p = new Path2D();
    p.arc(x1, y1, r, a + Math.PI / 2, a + Math.PI * 1.5);
    p.arc(x2, y2, r, a - Math.PI / 2, a + Math.PI / 2);
    p.closePath(); return p;
  };
  // polygon with rounded corners (r) — pts [[x,y],...]
  LB.poly = (pts, r = 0) => {
    const p = new Path2D(), n = pts.length;
    if (!r) { pts.forEach(([x, y], i) => i ? p.lineTo(x, y) : p.moveTo(x, y)); p.closePath(); return p; }
    const m = [(pts[0][0] + pts[n - 1][0]) / 2, (pts[0][1] + pts[n - 1][1]) / 2];
    p.moveTo(m[0], m[1]);
    for (let i = 0; i < n; i++) { const a = pts[i], b = pts[(i + 1) % n]; p.arcTo(a[0], a[1], b[0], b[1], r); }
    p.closePath(); return p;
  };
  // MSG rod crystal: hexagonal prism seen side-on (vertical), rounded corners.
  LB.hexRod = (cx, cy, w, h, r = 6) => {
    const k = w * 0.5, hw = w / 2, hh = h / 2;
    return LB.poly([[cx, cy - hh], [cx + hw, cy - hh + k * 0.55], [cx + hw, cy + hh - k * 0.55], [cx, cy + hh], [cx - hw, cy + hh - k * 0.55], [cx - hw, cy - hh + k * 0.55]], r);
  };
  LB.star = (cx, cy, n, rIn, rOut, rot = 0, r = 3) => {
    const pts = [];
    for (let i = 0; i < n * 2; i++) { const a = rot + i * Math.PI / n, rr = i % 2 ? rIn : rOut; pts.push([cx + Math.cos(a) * rr, cy + Math.sin(a) * rr]); }
    return LB.poly(pts, r);
  };
  // transform a path: LB.xf(p, {x,y,rot,sx,sy}) (rotation/scale about origin, then translate)
  LB.xf = (p, o = {}) => {
    const m = new DOMMatrix().translate(o.x || 0, o.y || 0).rotate((o.rot || 0) * 180 / Math.PI).scale(o.sx ?? 1, o.sy ?? o.sx ?? 1);
    const q = new Path2D(); q.addPath(p, m); return q;
  };

  // ---------------------------------------------------------------- halftone
  const patCache = {};
  // diagonal-lattice dot pattern, cached. grid = cell size, r = dot radius
  LB.dots = (ctx, grid = 8, r = 1.9, color = C.ink) => {
    const key = grid + '|' + r + '|' + color;
    if (!patCache[key]) {
      const c = document.createElement('canvas'); c.width = c.height = grid;
      const g = c.getContext('2d'); g.fillStyle = color;
      for (const [x, y] of [[grid / 2, grid / 2], [0, 0], [grid, 0], [0, grid], [grid, grid]]) { g.beginPath(); g.arc(x, y, r, 0, TAU); g.fill(); }
      patCache[key] = c;
    }
    return ctx.createPattern(patCache[key], 'repeat');
  };
  // halftone crescent on the shadow side. Light is top-left, so the lit copy is shifted up-left.
  // Works on any path (incl. unions of overlapping sub-paths) via a shared offscreen layer.
  LB.shade = (ctx, p, o = {}) => {
    const d = o.d ?? 16, dx = o.dx ?? -d, dy = o.dy ?? -d, cv = ctx.canvas;
    let L = LB._layer;
    if (!L || L.width !== cv.width || L.height !== cv.height) { L = LB._layer = document.createElement('canvas'); L.width = cv.width; L.height = cv.height; }
    const g = L.getContext('2d');
    g.setTransform(1, 0, 0, 1, 0, 0); g.globalCompositeOperation = 'source-over'; g.clearRect(0, 0, L.width, L.height);
    g.setTransform(ctx.getTransform());
    g.save(); g.clip(p); g.fillStyle = LB.dots(g, o.grid || 8, o.r || 1.9, o.color || C.ink); g.fillRect(-5000, -5000, 12000, 12000); g.restore();
    g.globalCompositeOperation = 'destination-out'; g.translate(dx, dy); g.fill(p); g.globalCompositeOperation = 'source-over';
    ctx.save(); ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.globalAlpha *= (o.alpha ?? 1); ctx.drawImage(L, 0, 0); ctx.restore();
  };
  // graded halftone field inside clip path `p` (or none). Dot radius = rMax * f(u), u along `angle`.
  // o: {x,y,w,h, grid, angle(deg, direction of increasing u), rMax, f, color, lattice(deg)}
  LB.halftone = (ctx, p, o) => {
    const g = o.grid || 14, a = LB.deg(o.angle ?? 45), la = LB.deg(o.lattice ?? 45);
    const ux = Math.cos(a), uy = Math.sin(a), lx = Math.cos(la), ly = Math.sin(la);
    const cx = o.x + o.w / 2, cy = o.y + o.h / 2, half = Math.hypot(o.w, o.h) / 2;
    const proj0 = -(Math.abs(ux) * o.w + Math.abs(uy) * o.h) / 2, projL = -2 * proj0;
    const f = o.f || (u => u), rMax = o.rMax ?? g * 0.45, path = new Path2D();
    const n = Math.ceil(half / g) + 1;
    for (let i = -n; i <= n; i++) for (let j = -n; j <= n; j++) {
      const x = cx + (i * lx - j * ly) * g, y = cy + (i * ly + j * lx) * g;
      if (x < o.x - g || x > o.x + o.w + g || y < o.y - g || y > o.y + o.h + g) continue;
      const u = LB.clamp(((x - cx) * ux + (y - cy) * uy - proj0) / projL);
      const r = rMax * f(u, x, y); if (r < 0.35) continue;
      path.moveTo(x + r, y); path.arc(x, y, r, 0, TAU);
    }
    ctx.save(); if (p) ctx.clip(p); ctx.globalAlpha = o.alpha ?? 1; ctx.fillStyle = o.color || C.ink; ctx.fill(path); ctx.restore();
  };

  // ---------------------------------------------------------------- core render
  // LB.draw(ctx, path, {fill, lw=6, reg=3, shadow=0, shade, ink, under})
  //  - shadow: solid ink sticker shadow offset (0, shadow)
  //  - reg: fill printed off-register down-right of the line (printed feel)
  //  - shade: halftone crescent (true or LB.shade options)
  LB.draw = (ctx, p, o = {}) => {
    const lw = o.lw ?? 6, reg = o.reg ?? 3, ink = o.ink || C.ink;
    ctx.save(); ctx.lineJoin = 'round'; ctx.lineCap = 'round';
    if (o.shadow) {
      ctx.save(); ctx.translate(o.shadowX || 0, o.shadow); ctx.fillStyle = ink; ctx.fill(p);
      if (lw) { ctx.lineWidth = lw; ctx.strokeStyle = ink; ctx.stroke(p); } ctx.restore();
    }
    if (o.fill) { ctx.save(); ctx.translate(reg, reg); ctx.fillStyle = o.fill; ctx.fill(p); ctx.restore(); }
    if (o.shade) LB.shade(ctx, p, o.shade === true ? {} : o.shade);
    if (o.inside) { ctx.save(); ctx.clip(p); o.inside(ctx); ctx.restore(); }
    if (lw) { ctx.lineWidth = lw; ctx.strokeStyle = ink; ctx.stroke(p); }
    ctx.restore();
  };
  // several parts sharing one silhouette outline: parts = [{p, fill, shade}]
  LB.blob = (ctx, parts, o = {}) => {
    const lw = o.lw ?? 6, ink = o.ink || C.ink;
    ctx.save(); ctx.lineJoin = 'round';
    if (o.shadow) { ctx.save(); ctx.translate(0, o.shadow); ctx.fillStyle = ink; ctx.strokeStyle = ink; ctx.lineWidth = lw * 2; for (const q of parts) { ctx.fill(q.p); ctx.stroke(q.p); } ctx.restore(); }
    ctx.strokeStyle = ink; ctx.lineWidth = lw * 2; for (const q of parts) ctx.stroke(q.p);
    for (const q of parts) { ctx.fillStyle = q.fill; ctx.fill(q.p); }
    for (const q of parts) if (q.shade) LB.shade(ctx, q.p, q.shade === true ? {} : q.shade);
    ctx.restore();
  };
  // rubber-hose limb: cubic bezier [x0,y0, c1x,c1y, c2x,c2y, x1,y1], width w
  LB.hose = (ctx, b, w, fill, lw = 6) => {
    const p = new Path2D(); p.moveTo(b[0], b[1]);
    if (b.length === 8) p.bezierCurveTo(b[2], b[3], b[4], b[5], b[6], b[7]); else p.quadraticCurveTo(b[2], b[3], b[4], b[5]);
    ctx.save(); ctx.lineCap = 'round'; ctx.lineJoin = 'round';
    ctx.strokeStyle = C.ink; ctx.lineWidth = w + lw * 2; ctx.stroke(p);
    ctx.strokeStyle = fill; ctx.lineWidth = w; ctx.stroke(p); ctx.restore();
    return p;
  };

  // ---------------------------------------------------------------- tray parts
  // recessed compartment: floor + halftone inner shadow (top-left wall) + ink rim
  LB.compartment = (ctx, x, y, w, h, o = {}) => {
    const r = o.r ?? 24, p = LB.rr(x, y, w, h, r);
    ctx.save();
    ctx.fillStyle = o.floor || C.cream; ctx.fill(p);
    if (o.inside) { ctx.save(); ctx.clip(p); o.inside(ctx, p); ctx.restore(); }
    if (o.wall !== false) LB.shade(ctx, p, { dx: o.depth ?? 18, dy: o.depth ?? 18, grid: o.grid || 8, r: o.dotR || 2.1, alpha: o.wallAlpha ?? 1, color: o.wallColor });
    ctx.lineWidth = o.lw ?? 6; ctx.strokeStyle = C.ink; ctx.lineJoin = 'round'; ctx.stroke(p);
    ctx.restore(); return p;
  };
  // Tray Grey lid. (cx,cy) centre, rot radians. o: {fill,label,sub,shadow,tab}
  LB.lid = (ctx, cx, cy, w, h, rot = 0, o = {}) => {
    ctx.save(); ctx.translate(cx, cy); ctx.rotate(rot);
    const p = LB.rr(-w / 2, -h / 2, w, h, o.r ?? 24);
    // pull tab
    const tab = LB.rr(-w * 0.12, -h / 2 - 16, w * 0.24, 34, 12);
    LB.draw(ctx, tab, { fill: o.fill || C.grey, shadow: o.shadow ?? 8, lw: 5 });
    LB.draw(ctx, p, { fill: o.fill || C.grey, shadow: o.shadow ?? 8, shade: { d: 22 } });
    // raised inner panel line
    ctx.lineWidth = 3; ctx.strokeStyle = C.ink; ctx.globalAlpha = 0.5;
    ctx.stroke(LB.rr(-w / 2 + 18, -h / 2 + 18, w - 36, h - 36, 14)); ctx.globalAlpha = 1;
    if (o.label) LB.text(ctx, o.label, o.labelDx || 0, o.sub ? -14 : 4, { font: LB.F.mono, weight: 700, size: o.size || 56, fill: C.ink, base: 'middle' });
    if (o.sub) LB.text(ctx, o.sub, 0, 44, { font: LB.F.mono, weight: 700, size: 40, fill: C.ink, base: 'middle', alpha: 0.55 });
    ctx.restore();
  };
  // sticker plate. o: {x,y (centre), w,h, rot(rad), fill, r, lw, shadow, path}. content(ctx) in local coords.
  LB.sticker = (ctx, o, content) => {
    ctx.save(); ctx.translate(o.x, o.y); ctx.rotate(o.rot || 0); if (o.sx) ctx.scale(o.sx, o.sy || o.sx);
    const p = o.path || LB.rr(-o.w / 2, -o.h / 2, o.w, o.h, o.r ?? 16);
    LB.draw(ctx, p, { fill: o.fill || C.cream, lw: o.lw ?? 6, shadow: o.shadow ?? 8, shade: o.shade, reg: o.reg });
    if (content) content(ctx);
    ctx.restore();
  };

  // ---------------------------------------------------------------- type
  LB.font = (o) => `${o.weight || 400} ${o.size || 44}px ${o.font || LB.F.label}`;
  LB.text = (ctx, s, x, y, o = {}) => {
    ctx.save(); ctx.font = LB.font(o); ctx.textAlign = o.align || 'center'; ctx.textBaseline = o.base || 'alphabetic';
    if (o.track) ctx.letterSpacing = o.track + 'px';
    if (o.alpha != null) ctx.globalAlpha = o.alpha;
    if (o.shadow) { ctx.fillStyle = C.ink; ctx.strokeStyle = C.ink; ctx.lineJoin = 'round'; ctx.lineWidth = (o.lw || 0) * 2; if (o.lw) ctx.strokeText(s, x, y + o.shadow); ctx.fillText(s, x, y + o.shadow); }
    if (o.lw) { ctx.strokeStyle = o.ink || C.ink; ctx.lineJoin = 'round'; ctx.lineWidth = o.lw * 2; ctx.strokeText(s, x, y); }
    ctx.fillStyle = o.fill || C.ink; ctx.fillText(s, x, y);
    ctx.restore();
  };
  LB.measure = (ctx, s, o) => { ctx.save(); ctx.font = LB.font(o); if (o.track) ctx.letterSpacing = o.track + 'px'; const w = ctx.measureText(s).width; ctx.restore(); return w; };
  // largest size <= o.size such that s fits width
  LB.fit = (ctx, s, maxW, o) => { let size = o.size; while (size > 10 && LB.measure(ctx, s, { ...o, size }) > maxW) size -= 2; return size; };

  // Reel caption: a Cream sticker strip, Atkinson Bold, the spoken word slapped on a Butter sticker.
  // words: array, hi: index of current word. Wraps to max 2 lines inside the caption band.
  LB.caption = (ctx, words, hi, o = {}) => {
    const size = o.size || 60, cx = o.cx ?? LB.CAP.cx, cy = o.cy ?? (LB.CAP.y0 + LB.CAP.y1) / 2;
    const maxW = o.maxW || 800, st = { font: LB.F.label, weight: 700, size }, space = size * 0.22, padX = 10;
    const ws = words.map(w => LB.measure(ctx, w, st) + padX * 2);
    const lines = [[]]; let lw = 0;
    ws.forEach((w, i) => { if (lw + w > maxW && lines[lines.length - 1].length) { lines.push([]); lw = 0; } lines[lines.length - 1].push(i); lw += w + space; });
    const lh = size * 1.32, H = lines.length * lh + 44;
    const W = Math.max(...lines.map(L => L.reduce((s, i) => s + ws[i], 0) + space * (L.length - 1))) + 56;
    LB.sticker(ctx, { x: cx, y: cy, w: W, h: H, fill: o.plate || C.cream, rot: LB.deg(o.rot ?? -1), r: 22 }, c => {
      lines.forEach((L, li) => {
        const tot = L.reduce((s, i) => s + ws[i], 0) + space * (L.length - 1);
        let x = -tot / 2; const y = -H / 2 + 22 + lh * (li + 0.5);
        for (const i of L) {
          const w = ws[i], mx = x + w / 2;
          if (i === hi) {
            LB.sticker(c, { x: mx, y: y + 2, w: w + 8, h: size * 1.18, fill: o.hiFill || C.butter, rot: LB.deg(-3), r: 12, lw: 5, shadow: 6, sx: 1.06 }, cc =>
              LB.text(cc, words[i], 0, size * 0.36, { ...st, fill: C.ink }));
          } else {
            LB.text(c, words[i], mx, y + size * 0.36, { ...st, fill: C.ink, alpha: i < hi ? 1 : 0.42 });
          }
          x += w + space;
        }
      });
    });
  };

  // ---------------------------------------------------------------- faces
  LB.eye = (ctx, x, y, s, lx = 0, ly = 0) => {
    ctx.save(); ctx.fillStyle = C.ink; ctx.fill(LB.ell(x, y, s * 0.3, s * 0.46));
    ctx.fillStyle = C.white; ctx.fill(LB.circ(x - s * 0.1 + lx * s * 0.06, y - s * 0.2 + ly * s * 0.06, s * 0.12));
    ctx.fill(LB.circ(x + s * 0.1, y + s * 0.18, s * 0.05)); ctx.restore();
  };
  LB.brow = (ctx, x, y, len, ang, w = 10) => {
    ctx.save(); ctx.fillStyle = C.ink; const dx = Math.cos(ang) * len / 2, dy = Math.sin(ang) * len / 2;
    ctx.fill(LB.capsule(x - dx, y - dy, x + dx, y + dy, w / 2)); ctx.restore();
  };
  // kind: 'open' (D-shape, tongue), 'o', 'chomp' (closed wide grin), 'happy'
  LB.mouth = (ctx, x, y, w, h, kind = 'open', lw = 5) => {
    ctx.save(); ctx.lineJoin = 'round'; ctx.lineCap = 'round';
    if (kind === 'chomp') {
      const p = new Path2D(); p.moveTo(x - w / 2, y); p.quadraticCurveTo(x, y + h * 1.4, x + w / 2, y);
      ctx.strokeStyle = C.ink; ctx.lineWidth = lw * 1.4; ctx.stroke(p);
      ctx.restore(); return;
    }
    const p = new Path2D();
    if (kind === 'o') p.ellipse(x, y, w / 2, h / 2, 0, 0, TAU);
    else { p.moveTo(x - w / 2, y - h * 0.3); p.quadraticCurveTo(x, y - h * 0.45, x + w / 2, y - h * 0.3); p.bezierCurveTo(x + w / 2, y + h * 0.5, x + w * 0.25, y + h * 0.7, x, y + h * 0.7); p.bezierCurveTo(x - w * 0.25, y + h * 0.7, x - w / 2, y + h * 0.5, x - w / 2, y - h * 0.3); p.closePath(); }
    ctx.fillStyle = C.ink; ctx.fill(p);
    ctx.save(); ctx.clip(p); ctx.fillStyle = C.tomato; ctx.fill(LB.ell(x + w * 0.05, y + h * 0.62, w * 0.3, h * 0.3)); ctx.restore();
    ctx.restore();
  };
  LB.cheek = (ctx, x, y, r, color = C.tomato, grid = 7) => {
    LB.halftone(ctx, LB.circ(x, y, r), { x: x - r, y: y - r, w: 2 * r, h: 2 * r, grid, color, rMax: grid * 0.42, f: (u, px, py) => 1 - Math.hypot(px - x, py - y) / r });
  };

  // ---------------------------------------------------------------- hands (3 fingers + thumb)
  // pose: 'point' | 'open' | 'grip'. ang: direction the hand points (radians, 0 = up). s ~ hand size
  LB.handParts = (pose, s, fill, flip = 1) => {
    const parts = [{ p: LB.circ(0, 0, s * 0.44), fill }];
    const cap = (x1, y1, x2, y2, r) => parts.push({ p: LB.capsule(x1 * flip, y1, x2 * flip, y2, r), fill });
    if (pose === 'point') {
      cap(s * 0.08, -s * 0.2, s * 0.1, -s * 1.08, s * 0.155);
      cap(-s * 0.32, -s * 0.18, -s * 0.12, -s * 0.32, s * 0.16);
      cap(-s * 0.38, s * 0.06, -s * 0.16, -s * 0.05, s * 0.16);
      cap(s * 0.36, s * 0.02, s * 0.66, -s * 0.28, s * 0.15);
    } else if (pose === 'open') {
      for (const a of [-0.42, 0, 0.42]) cap(Math.sin(a) * s * 0.2, -Math.cos(a) * s * 0.2, Math.sin(a) * s * 0.92, -Math.cos(a) * s * 0.92, s * 0.155);
      cap(s * 0.3, s * 0.05, s * 0.78, -s * 0.12, s * 0.15);
    } else { // grip: knuckles bump forward
      for (const k of [-0.3, 0, 0.3]) cap(k * s, -s * 0.28, k * s, -s * 0.5, s * 0.16);
      cap(s * 0.35, -s * 0.05, s * 0.62, -s * 0.38, s * 0.15);
    }
    return parts;
  };
  LB.hand = (ctx, x, y, ang, s, pose = 'open', fill = C.white, o = {}) => {
    ctx.save(); ctx.translate(x, y); ctx.rotate(ang);
    LB.blob(ctx, LB.handParts(pose, s, fill, o.flip || 1), { lw: o.lw ?? 5 });
    // finger separation detail for 'open'
    ctx.restore();
  };

  // ---------------------------------------------------------------- characters
  // Mini Umi: Butter rod crystal, eyes only. (x,y) centre. o: {look:[lx,ly], squash, rot, shadow}
  LB.miniUmi = (ctx, x, y, s, o = {}) => {
    ctx.save(); ctx.translate(x, y); ctx.rotate(o.rot || 0);
    const sq = o.squash || 0; ctx.scale(1 + sq * 0.15, 1 - sq * 0.15);
    const w = s * 0.62, h = s, p = LB.hexRod(0, 0, w, h, s * 0.1);
    LB.draw(ctx, p, { fill: C.butter, lw: o.lw ?? 4, reg: 2, shadow: o.shadow ?? 0, shade: { d: s * 0.18, grid: 6, r: 1.4 } });
    // facet highlight stripe
    ctx.fillStyle = C.white; ctx.fill(LB.capsule(-w * 0.26, -h * 0.2, -w * 0.26, h * 0.16, s * 0.05));
    const [lx, ly] = o.look || [0, 0];
    LB.eye(ctx, -w * 0.12 + lx * 2, -h * 0.06 + ly * 2, s * 0.3, lx, ly);
    LB.eye(ctx, w * 0.2 + lx * 2, -h * 0.06 + ly * 2, s * 0.3, lx, ly);
    ctx.restore();
  };

  // Intestinal cell character (round, Bubblegum). (x, y) = bottom contact point; r radius.
  // o: {squash 0..1, look:[lx,ly], mood:'eager'|'chomp', seed}
  LB.gutCell = (ctx, x, y, r, o = {}) => {
    ctx.save(); ctx.translate(x, y);
    const sq = o.squash || 0; ctx.scale(1 + 0.15 * sq, 1 - 0.15 * sq);
    const body = LB.ell(0, -r, r * 1.02, r);
    LB.draw(ctx, body, { fill: o.fill || C.gum, lw: 6, shade: { d: r * 0.34, grid: 8, r: 2.1 } });
    // tiny brush-border fringe on top (microvilli) — detail strokes
    ctx.save(); ctx.strokeStyle = C.ink; ctx.lineWidth = 3; ctx.lineCap = 'round';
    for (let i = -3; i <= 3; i++) { const a = -Math.PI / 2 + i * 0.17, bx = Math.cos(a) * r * 1.02, by = -r + Math.sin(a) * r; ctx.beginPath(); ctx.moveTo(bx, by); ctx.lineTo(bx * 1.12, -r + (by + r) * 1.12 - 2); ctx.stroke(); }
    ctx.restore();
    const [lx, ly] = o.look || [0, 0], ex = r * 0.34, ey = -r * 1.12;
    if (o.mood === 'chomp') {
      // squeezed happy eyes ^ ^
      ctx.save(); ctx.strokeStyle = C.ink; ctx.lineWidth = 8; ctx.lineCap = 'round';
      for (const sx of [-1, 1]) { ctx.beginPath(); ctx.moveTo(sx * ex - r * 0.15, ey + 6); ctx.lineTo(sx * ex, ey - r * 0.1); ctx.lineTo(sx * ex + r * 0.15, ey + 6); ctx.stroke(); }
      ctx.restore();
      LB.brow(ctx, -ex, ey - r * 0.36, r * 0.3, -0.25, 9); LB.brow(ctx, ex, ey - r * 0.36, r * 0.3, 0.25, 9);
      // puffed cheeks bulge + chomp grin
      LB.mouth(ctx, 0, -r * 0.62, r * 0.7, r * 0.22, 'chomp', 6);
    } else {
      LB.eye(ctx, -ex + lx * 4, ey + ly * 4, r * 0.52, lx, ly); LB.eye(ctx, ex + lx * 4, ey + ly * 4, r * 0.52, lx, ly);
      LB.brow(ctx, -ex, ey - r * 0.42, r * 0.28, -0.32, 9); LB.brow(ctx, ex, ey - r * 0.42, r * 0.28, 0.32, 9);
      LB.mouth(ctx, lx * 3, -r * 0.5, r * 0.62, r * 0.5, 'open');
    }
    LB.cheek(ctx, -r * 0.62, -r * 0.6, r * 0.2); LB.cheek(ctx, r * 0.62, -r * 0.6, r * 0.2);
    ctx.restore();
  };

  // cartoon brain built from lobes; (x,y) centre, r overall radius
  LB.brain = (ctx, x, y, r, o = {}) => {
    const lobes = [[-0.55, -0.1, 0.5], [-0.25, -0.45, 0.48], [0.22, -0.48, 0.5], [0.58, -0.12, 0.48], [0.45, 0.32, 0.45], [-0.02, 0.28, 0.5], [-0.48, 0.3, 0.42]];
    const parts = lobes.map(([a, b, k]) => ({ p: LB.circ(x + a * r, y + b * r, k * r), fill: o.fill || C.gum }));
    parts.push({ p: LB.capsule(x + r * 0.2, y + r * 0.5, x + r * 0.32, y + r * 0.85, r * 0.16), fill: o.fill || C.gum }); // stem
    LB.blob(ctx, parts, { lw: 6, shadow: o.shadow || 0 });
    // shading: one crescent over the merged silhouette
    const sil = new Path2D(); parts.forEach(q => sil.addPath(q.p));
    LB.shade(ctx, sil, { d: r * 0.28, grid: 8, r: 2.2 });
    ctx.save(); ctx.strokeStyle = C.ink; ctx.lineWidth = 5; ctx.lineCap = 'round';
    const sq = (pts) => { ctx.beginPath(); ctx.moveTo(x + pts[0] * r, y + pts[1] * r); ctx.quadraticCurveTo(x + pts[2] * r, y + pts[3] * r, x + pts[4] * r, y + pts[5] * r); ctx.stroke(); };
    sq([0.02, -0.82, -0.08, -0.3, 0.06, 0.12]);
    sq([-0.62, -0.3, -0.42, -0.12, -0.6, 0.08]); sq([-0.3, 0.12, -0.2, 0.38, -0.36, 0.55]);
    sq([0.32, -0.62, 0.48, -0.38, 0.3, -0.2]); sq([0.52, 0.0, 0.7, 0.12, 0.6, 0.34]); sq([0.22, 0.12, 0.38, 0.28, 0.2, 0.42]);
    ctx.restore();
  };
  // zig-zag zap bolt from (x,y) heading ang, length L
  LB.zap = (ctx, x, y, ang, L, o = {}) => {
    const w = o.w || L * 0.22, pts = [[0, -w * 0.5], [L * 0.45, -w * 0.15], [L * 0.38, -w * 0.62], [L, 0.0], [L * 0.52, w * 0.18], [L * 0.6, w * 0.65]];
    const p = LB.xf(LB.poly(pts, 2), { x, y, rot: ang });
    LB.draw(ctx, p, { fill: o.fill || C.tomato, lw: 5, reg: 2, shadow: o.shadow ?? 0 });
  };

  // Fictional "Daytime TV Doctor" pictogram: blank flat face, white coat, Pool stethoscope, clipboard.
  // (x, y) = centre of the head; s = head radius. t for idle breathing.
  LB.tvDoc = (ctx, x, y, s, t = 0, o = {}) => {
    const breathe = Math.sin(t * Math.PI * 1.6) * 0.012;
    ctx.save(); ctx.translate(x, y);
    const skin = o.skin || C.skin2;
    // torso (pear) — coat
    const by = s * 0.95;
    ctx.save(); ctx.translate(0, by); ctx.scale(1, 1 + breathe);
    const coat = new Path2D();
    coat.moveTo(-s * 0.55, 0); coat.bezierCurveTo(-s * 1.25, s * 0.15, -s * 1.7, s * 1.3, -s * 1.75, s * 2.8);
    coat.lineTo(s * 1.75, s * 2.8); coat.bezierCurveTo(s * 1.7, s * 1.3, s * 1.25, s * 0.15, s * 0.55, 0); coat.closePath();
    LB.draw(ctx, coat, { fill: C.white, lw: 6, shade: { d: s * 0.35 } });
    // shirt V + tie
    const v = LB.poly([[-s * 0.5, s * 0.02], [s * 0.5, s * 0.02], [s * 0.16, s * 1.25], [-s * 0.16, s * 1.25]], 4);
    LB.draw(ctx, v, { fill: C.lilac, lw: 5, reg: 2 });
    LB.draw(ctx, LB.poly([[-s * 0.13, s * 0.12], [s * 0.13, s * 0.12], [s * 0.2, s * 0.95], [0, s * 1.2], [-s * 0.2, s * 0.95]], 4), { fill: C.tomato, lw: 5, reg: 2 });
    // lapels
    ctx.lineWidth = 5; ctx.strokeStyle = C.ink; ctx.lineJoin = 'round';
    for (const k of [-1, 1]) { ctx.beginPath(); ctx.moveTo(k * s * 0.55, s * 0.02); ctx.lineTo(k * s * 0.82, s * 0.7); ctx.lineTo(k * s * 0.5, s * 0.82); ctx.lineTo(k * s * 0.22, s * 1.55); ctx.stroke(); }
    // pocket + pens
    LB.draw(ctx, LB.rr(-s * 1.15, s * 1.45, s * 0.5, s * 0.42, 6), { fill: C.white, lw: 4, reg: 0 });
    LB.draw(ctx, LB.capsule(-s * 1.0, s * 1.2, -s * 1.0, s * 1.5, s * 0.06), { fill: C.pool, lw: 3, reg: 0 });
    LB.draw(ctx, LB.capsule(-s * 0.85, s * 1.25, -s * 0.85, s * 1.5, s * 0.06), { fill: C.tomato, lw: 3, reg: 0 });
    // stethoscope (Pool tube) around the neck
    LB.hose(ctx, [-s * 0.62, s * 0.05, -s * 0.85, s * 1.0, -s * 0.2, s * 1.35, s * 0.22, s * 1.45], s * 0.12, C.pool, 4);
    LB.hose(ctx, [s * 0.62, s * 0.05, s * 0.8, s * 0.7, s * 0.6, s * 1.1, s * 0.22, s * 1.45], s * 0.12, C.pool, 4);
    LB.hose(ctx, [s * 0.22, s * 1.45, s * 0.5, s * 1.75, s * 0.68, s * 1.95], s * 0.12, C.pool, 4);
    LB.draw(ctx, LB.circ(s * 0.7, s * 2.05, s * 0.2), { fill: C.grey, lw: 5, reg: 2, shade: { d: 6, grid: 6, r: 1.4 } });
    ctx.restore();
    // neck
    LB.draw(ctx, LB.rr(-s * 0.3, s * 0.6, s * 0.6, s * 0.5, 10), { fill: skin, lw: 5, reg: 0 });
    // head: big blank oval, ears, hair cap
    LB.draw(ctx, LB.ell(-s * 0.93, s * 0.05, s * 0.17, s * 0.24), { fill: skin, lw: 5, reg: 0 });
    LB.draw(ctx, LB.ell(s * 0.93, s * 0.05, s * 0.17, s * 0.24), { fill: skin, lw: 5, reg: 0 });
    const head = LB.ell(0, 0, s * 0.92, s * 1.0);
    LB.draw(ctx, head, { fill: skin, lw: 6, shade: { d: s * 0.26 } });
    const hair = new Path2D();
    hair.moveTo(-s * 0.9, -s * 0.05); hair.bezierCurveTo(-s * 1.0, -s * 0.9, -s * 0.3, -s * 1.2, s * 0.2, -s * 1.08);
    hair.bezierCurveTo(s * 0.75, -s * 1.0, s * 1.02, -s * 0.55, s * 0.9, -s * 0.1);
    hair.bezierCurveTo(s * 0.7, -s * 0.55, s * 0.2, -s * 0.62, -s * 0.15, -s * 0.5);
    hair.bezierCurveTo(-s * 0.45, -s * 0.42, -s * 0.7, -s * 0.38, -s * 0.9, -s * 0.05); hair.closePath();
    LB.draw(ctx, hair, { fill: C.ink, lw: 6, reg: 0 });
    ctx.fillStyle = C.grey; ctx.fill(LB.capsule(-s * 0.2, -s * 0.9, s * 0.35, -s * 0.85, s * 0.05)); // hair sheen
    // only a mouth (mid-claim)
    LB.mouth(ctx, s * 0.05, s * 0.42, s * 0.52, s * 0.44, 'open', 5);
    ctx.restore();
  };
})();

// =====================================================================================
// FRAMES
// =====================================================================================
(function () {
  const { C, F } = LB;

  // ---------------------------------------------------------------- Frame A — cold-open TV wall
  function frameA(ctx, t) {
    const W = LB.W, H = LB.H;
    ctx.fillStyle = C.ink; ctx.fillRect(0, 0, W, H); // Liquorice tray
    // faint tray texture: Cream halftone, very sparse, top-left lighter
    LB.halftone(ctx, null, { x: 0, y: 0, w: W, h: H, grid: 18, color: C.cream, rMax: 2.2, angle: 60, alpha: 0.08, f: u => 1 - u });

    const g = 24, S = LB.SAFE;
    const L = LB.col(0, 3), R = LB.col(3, 3);
    // --- top row (under the Reel UI: decorative only): two closed lids
    const topY = 30, topH = 166;
    const comps = [
      { x: L.x, y: topY, w: L.w, h: topH, floor: C.pool, lid: { label: 'CH 2', sub: '7:00', rot: 0, dy: 0 } },
      { x: R.x, y: topY, w: R.w, h: topH, floor: C.pea, lid: { label: 'CH 9', sub: '7:30', rot: 0, dy: 0 } },
    ];
    // --- hero: big Lilac compartment
    const big = { x: S.x, y: S.y, w: S.w, h: 800 };
    // --- row under hero: two half-open compartments (next speakers peeking)
    const midY = big.y + big.h + g, midH = 1280 - midY;
    // --- below the caption band: more channels (decorative, under Reel UI)
    const lowY = 1500;
    const all = [
      ...comps,
      { x: L.x, y: midY, w: L.w, h: midH, floor: C.tomato, peek: 'chef', lid: { label: 'CH 4', rot: LB.deg(3), dx: 205, dy: 14, labelDx: -90 } },
      { x: R.x, y: midY, w: R.w, h: midH, floor: C.gum, peek: 'ring', lid: { label: 'CH 7', rot: LB.deg(-3), dx: -205, dy: 14, labelDx: 90 } },
      { x: L.x, y: lowY, w: L.w, h: 190, floor: C.butter, lid: { label: 'CH 11', sub: '8:00' } },
      { x: R.x, y: lowY, w: R.w, h: 190, floor: C.pool, lid: { label: 'CH 12', sub: '8:30' } },
      { x: L.x, y: lowY + 214, w: L.w, h: 190, floor: C.lilac, lid: { label: 'CH 14' } },
      { x: R.x, y: lowY + 214, w: R.w, h: 190, floor: C.tomato, lid: { label: 'CH 15' } },
    ];
    // extend the safe-width tray to the full width on the right (it's a tray, not content)
    for (const c of all) {
      LB.compartment(ctx, c.x, c.y, c.w, c.h, {
        floor: c.floor, depth: 14, inside: cx => {
          if (c.peek === 'chef') peekChef(cx, c, t);
          if (c.peek === 'ring') peekRing(cx, c, t);
        },
      });
    }
    for (const c of all) {
      const l = c.lid; if (!l) continue;
      ctx.save(); if (l.dx || l.dy) { ctx.clip(LB.rr(c.x - 3, c.y - 40, c.w + 6, c.h + 43, 24)); }
      LB.lid(ctx, c.x + c.w / 2 + (l.dx || 0), c.y + c.h / 2 + (l.dy || 0), c.w - 16, c.h - 16, l.rot || 0, { label: l.label, sub: l.sub, labelDx: l.labelDx, size: c.h < 200 ? 50 : 56 });
      ctx.restore();
    }

    // hero compartment + TV
    LB.compartment(ctx, big.x, big.y, big.w, big.h, {
      floor: C.lilac, depth: 20, inside: cx => {
        // soft burst of Lilac-on-Lilac halftone behind the TV
        LB.halftone(cx, null, { x: big.x, y: big.y, w: big.w, h: big.h, grid: 12, color: C.ink, rMax: 3.4, alpha: 0.22, angle: 90, f: u => u * u });
        drawTV(cx, big, t);
      },
    });

    // sticker lower-third: slapped over the hero compartment's bottom-left corner
    const ltW = 640, ltH = 140;
    LB.sticker(ctx, { x: S.x + ltW / 2 - 18, y: big.y + big.h - 8, w: ltW, h: ltH, rot: LB.deg(-3), fill: C.butter, r: 18 }, c => {
      LB.text(c, 'THE DAYTIME TV DOCTOR', -ltW / 2 + 34, -10, { font: F.label, weight: 700, size: 40, align: 'left', track: 1 });
      LB.text(c, 'DRAMATIZATION \u2013 fictional character', -ltW / 2 + 34, 40, { font: F.label, weight: 400, size: 28, align: 'left' });
      // sticker tab: a little red "ON AIR" dot block
      LB.draw(c, LB.circ(ltW / 2 - 52, -12, 16), { fill: C.tomato, lw: 4, reg: 0 });
    });

    LB.caption(ctx, ['overexcites', 'your', 'brain', 'cells.'], 2, { cy: 1390, size: 54, maxW: 840 });
  }

  function drawTV(ctx, big, t) {
    const tvx = big.x + 44, tvw = big.w - 88, tvy = big.y + 96, tvh = 600;
    // antenna
    ctx.save(); ctx.lineCap = 'round';
    const ax = tvx + tvw / 2, ay = tvy + 6;
    for (const [ex, ey] of [[ax - 150, big.y + 30], [ax + 120, big.y + 22]]) {
      ctx.strokeStyle = C.ink; ctx.lineWidth = 16; ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(ex, ey); ctx.stroke();
      ctx.strokeStyle = C.grey; ctx.lineWidth = 6; ctx.beginPath(); ctx.moveTo(ax, ay - 4); ctx.lineTo(ex, ey); ctx.stroke();
      LB.draw(ctx, LB.circ(ex, ey, 14), { fill: C.tomato, lw: 5, reg: 0 });
    }
    LB.draw(ctx, LB.ell(ax, ay, 56, 26), { fill: C.cream, lw: 6, reg: 0 });
    ctx.restore();
    // legs
    for (const k of [0.18, 0.82]) LB.draw(ctx, LB.poly([[tvx + tvw * k - 22, tvy + tvh - 10], [tvx + tvw * k + 22, tvy + tvh - 10], [tvx + tvw * k + (k < .5 ? -8 : 30), tvy + tvh + 52], [tvx + tvw * k + (k < .5 ? -40 : -2), tvy + tvh + 52]], 6), { fill: C.ink, lw: 6, reg: 0 });
    // body
    const body = LB.rr(tvx, tvy, tvw, tvh, 56);
    LB.draw(ctx, body, { fill: C.cream, lw: 6, shadow: 10, shade: { d: 26 } });
    // screen
    const sx = tvx + 36, sy = tvy + 34, sw = tvw - 72, sh = tvh - 150;
    const scr = LB.rr(sx, sy, sw, sh, 42);
    LB.draw(ctx, LB.rr(sx - 12, sy - 12, sw + 24, sh + 24, 52), { fill: C.ink, lw: 0, reg: 0 });
    ctx.save(); ctx.clip(scr);
    ctx.fillStyle = C.pool; ctx.fillRect(sx, sy, sw, sh);
    // studio backdrop: big halftone sunburst-ish circle + vignette dots
    LB.draw(ctx, LB.circ(sx + sw * 0.72, sy + sh * 0.42, sh * 0.48), { fill: C.cream, lw: 0, reg: 0 });
    LB.halftone(ctx, scr, { x: sx, y: sy, w: sw, h: sh, grid: 10, color: C.ink, rMax: 4, alpha: 0.5, f: (u, x, y) => { const d = Math.hypot((x - sx - sw / 2) / (sw / 2), (y - sy - sh / 2) / (sh / 2)); return LB.clamp((d - 0.72) * 2.2); } });
    // brain + zaps behind the doctor
    const bx = sx + sw * 0.72, byy = sy + sh * 0.42, br = 112;
    const jit = Math.sin(t * 18) * 2;
    LB.brain(ctx, bx + jit, byy, br);
    LB.zap(ctx, bx - br * 0.2, byy - br * 1.0, LB.deg(-115), 92, { fill: C.tomato });
    LB.zap(ctx, bx + br * 0.75, byy - br * 0.7, LB.deg(-40), 96, { fill: C.tomato });
    LB.zap(ctx, bx + br * 1.05, byy + br * 0.25, LB.deg(10), 84, { fill: C.butter });
    LB.zap(ctx, bx + br * 0.5, byy + br * 0.95, LB.deg(60), 70, { fill: C.tomato });
    // the doctor
    const hx = sx + sw * 0.3, hy = sy + sh * 0.37, hs = 74;
    const bob = Math.sin(t * Math.PI * 1.6) * 3;
    // raised arm (behind body: from his left shoulder, viewer's right? -> keep on viewer's left, finger up)
    LB.tvDoc(ctx, hx, hy + bob, hs, t);
    const shX = hx - hs * 1.3, shY = hy + bob + hs * 1.45;
    LB.hose(ctx, [shX, shY, shX - 80, shY + 10, shX - 110, shY - 70, shX - 92, shY - 168], 30, C.white, 6);
    LB.hand(ctx, shX - 92, shY - 178, LB.deg(6), 62, 'point', C.white);
    // clipboard in the other hand
    const cbx = hx + hs * 1.25, cby = hy + bob + hs * 2.45;
    ctx.save(); ctx.translate(cbx, cby); ctx.rotate(LB.deg(14));
    LB.draw(ctx, LB.rr(-62, -80, 124, 168, 12), { fill: C.pea, lw: 6, shade: { d: 12 } });
    LB.draw(ctx, LB.rr(-48, -60, 96, 136, 6), { fill: C.cream, lw: 4, reg: 0 });
    LB.draw(ctx, LB.rr(-26, -92, 52, 26, 8), { fill: C.grey, lw: 4, reg: 0 });
    ctx.fillStyle = C.ink; for (let i = 0; i < 4; i++) ctx.fill(LB.rr(-34, -36 + i * 24, i === 3 ? 40 : 68, 7, 3));
    ctx.restore();
    LB.hand(ctx, cbx - 70, cby + 22, LB.deg(-70), 46, 'grip', C.white);
    // channel bug
    LB.draw(ctx, LB.rr(sx + 26, sy + sh - 82, 168, 56, 14), { fill: C.ink, lw: 0, reg: 0 });
    LB.text(ctx, 'CH 5', sx + 110, sy + sh - 40, { font: F.mono, weight: 700, size: 40, fill: C.cream });
    LB.draw(ctx, LB.circ(sx + sw - 50, sy + 46, 14), { fill: C.tomato, lw: 4, reg: 0 });
    LB.text(ctx, 'LIVE', sx + sw - 74, sy + 60, { font: F.mono, weight: 700, size: 40, fill: C.ink, align: 'right' });
    // glass glare
    ctx.globalAlpha = 0.28; ctx.fillStyle = C.white;
    ctx.fill(LB.poly([[sx + sw * 0.08, sy], [sx + sw * 0.2, sy], [sx + sw * 0.02, sy + sh * 0.45], [sx, sy + sh * 0.45]]));
    ctx.globalAlpha = 1;
    ctx.restore();
    ctx.lineWidth = 6; ctx.strokeStyle = C.ink; ctx.stroke(scr);
    // control strip: knobs + speaker grille
    const cy = sy + sh + 58;
    for (const [kx, kr] of [[tvx + 90, 34], [tvx + 180, 26]]) {
      LB.draw(ctx, LB.circ(kx, cy, kr), { fill: C.butter, lw: 5, shade: { d: 8, grid: 6, r: 1.4 } });
      ctx.fillStyle = C.ink; ctx.fill(LB.capsule(kx, cy - kr * 0.75, kx, cy - kr * 0.1, 4));
    }
    for (let i = 0; i < 9; i++) ctx.fill(LB.capsule(tvx + tvw - 380 + i * 36, cy - 26, tvx + tvw - 380 + i * 36, cy + 26, 6));
  }

  function peekChef(ctx, c, t) {
    // toque top poking up from the bottom of an open channel (the next speaker)
    const x = c.x + 120, y = c.y + c.h + 10;
    const parts = [[0, -150, 58], [-56, -118, 46], [56, -116, 48], [0, -96, 56]].map(([a, b, r]) => ({ p: LB.circ(x + a, y + b, r), fill: C.white }));
    parts.push({ p: LB.rr(x - 74, y - 100, 148, 120, 12), fill: C.white });
    LB.blob(ctx, parts, { lw: 6 });
    ctx.strokeStyle = C.ink; ctx.lineWidth = 4; ctx.beginPath(); ctx.moveTo(x - 74, y - 60); ctx.lineTo(x + 74, y - 60); ctx.stroke();
    LB.shade(ctx, LB.rr(x - 74, y - 100, 148, 120, 12), { d: 14 });
  }
  function peekRing(ctx, c, t) {
    const x = c.x + c.w - 120, y = c.y + 120;
    LB.draw(ctx, LB.circ(x, y, 104), { fill: C.cream, lw: 6, reg: 0 });
    LB.draw(ctx, LB.circ(x, y, 64), { fill: C.gum, lw: 6, reg: 0 });
    LB.halftone(ctx, LB.circ(x, y, 64), { x: x - 64, y: y - 64, w: 128, h: 128, grid: 8, rMax: 2.6, alpha: 0.6, f: u => u });
  }

  // ---------------------------------------------------------------- Frame B — gut cells eating glutamate
  function frameB(ctx, t) {
    const W = LB.W, H = LB.H, S = LB.SAFE;
    ctx.fillStyle = C.cream; ctx.fillRect(0, 0, W, H);
    // tray body: Tray Grey slab with rim
    trayBody(ctx);

    // top row: tongue inset (top-right, 1/6) — previous beat, lid closing
    const inset = { ...LB.col(3, 3), y: 228, h: 250 };
    const lab = { ...LB.col(0, 3), y: 228, h: 250 };
    LB.compartment(ctx, lab.x, lab.y, lab.w, lab.h, { floor: C.cream, depth: 14, inside: c => {
      LB.text(c, 'SAMPLE 05', lab.x + 34, lab.y + 76, { font: F.mono, weight: 700, size: 40, align: 'left', alpha: .55 });
      LB.text(c, 'GUT', lab.x + 34, lab.y + 196, { font: F.display, size: 112, align: 'left', fill: C.gum, lw: 6 });
    } });
    LB.compartment(ctx, inset.x, inset.y, inset.w, inset.h, { floor: C.cream, depth: 14, inside: c => tongueTray(c, inset, t) });
    // the lid sliding down over the tongue
    ctx.save(); ctx.beginPath(); ctx.rect(inset.x - 30, inset.y - 60, inset.w + 60, inset.h + 60); ctx.clip();
    LB.lid(ctx, inset.x + inset.w / 2 + 8, inset.y + inset.h * -0.12, inset.w - 12, inset.h - 12, LB.deg(-4), { label: 'TONGUE', size: 40 });
    ctx.restore();

    // big Bubblegum compartment
    const big = { x: S.x, y: 502, w: S.w, h: 1276 - 502 };
    LB.compartment(ctx, big.x, big.y, big.w, big.h, { floor: C.gum, depth: 22, inside: c => gutScene(c, big, t) });

    // FUEL! sticker top-left, slapped over the corner
    LB.sticker(ctx, { x: S.x + 260, y: big.y + 120, w: 420, h: 150, rot: LB.deg(-5), fill: C.pool, r: 20 }, c => {
      const sz = LB.fit(c, 'FUEL!', 340, { font: F.display, size: 110 });
      LB.text(c, 'FUEL!', 0, sz * 0.38, { font: F.display, size: sz, fill: C.cream, lw: 6 });
    });

    LB.caption(ctx, ['your', 'intestines', 'love', 'glutamate.'], 2, { cy: 1390 });

    // below the caption: next compartments (decorative under Reel UI)
    const a = LB.col(0, 3), b = LB.col(3, 3);
    for (const [col, lab2, fl] of [[a, 'BLOOD', C.tomato], [b, 'BRAIN', C.lilac]]) {
      LB.compartment(ctx, col.x, 1504, col.w, 230, { floor: fl, depth: 14 });
      LB.lid(ctx, col.x + col.w / 2, 1504 + 115, col.w - 16, 214, 0, { label: lab2, sub: 'NEXT', size: 50 });
    }
  }

  function trayBody(ctx) {
    const p = LB.rr(24, 24, LB.W - 48, LB.H - 48, 48);
    LB.draw(ctx, p, { fill: C.grey, lw: 6, shadow: 12, reg: 0 });
    LB.halftone(ctx, p, { x: 24, y: 24, w: LB.W - 48, h: LB.H - 48, grid: 10, rMax: 2.2, alpha: 0.35, angle: 45, f: u => u });
  }

  function tongueTray(ctx, R, t) {
    const cx = R.x + R.w / 2, cy = R.y + R.h / 2 + 22;
    const tg = new Path2D(); tg.moveTo(cx - 130, cy - 90); tg.lineTo(cx + 130, cy - 90); tg.bezierCurveTo(cx + 150, cy + 40, cx + 90, cy + 120, cx, cy + 120); tg.bezierCurveTo(cx - 90, cy + 120, cx - 150, cy + 40, cx - 130, cy - 90); tg.closePath();
    LB.draw(ctx, tg, { fill: C.gum, lw: 6, shade: { d: 16 } });
    const cells = [[-70, -40, C.gum], [0, -40, C.pea], [70, -40, C.pool], [-40, 30, C.lilac], [40, 30, C.butter]];
    for (const [dx, dy, f] of cells) LB.draw(ctx, LB.rr(cx + dx - 28, cy + dy - 24, 56, 48, 12), { fill: f, lw: 4, reg: 2 });
  }

  function gutScene(ctx, big, t) {
    // lumen depth: heavier halftone near top (deep inside)
    LB.halftone(ctx, null, { x: big.x, y: big.y, w: big.w, h: big.h, grid: 12, rMax: 4.2, alpha: 0.3, angle: -90, f: u => u * u });
    // gut wall: Pool barrier line + Cream connective tissue band
    const wy = big.y + big.h - 160;
    const wall = new Path2D(); wall.moveTo(big.x - 10, wy);
    for (let i = 0; i <= 12; i++) { const x = big.x + i * (big.w / 12); wall.quadraticCurveTo(x - big.w / 24, wy + (i % 2 ? 22 : -10), x, wy); }
    wall.lineTo(big.x + big.w + 10, big.y + big.h + 20); wall.lineTo(big.x - 10, big.y + big.h + 20); wall.closePath();
    LB.draw(ctx, wall, { fill: C.cream, lw: 6, reg: 0 });
    const top = new Path2D(); top.moveTo(big.x - 10, wy + 14);
    for (let i = 0; i <= 12; i++) { const x = big.x + i * (big.w / 12); top.quadraticCurveTo(x - big.w / 24, wy + 14 + (i % 2 ? 22 : -10), x, wy + 14); }
    ctx.save(); ctx.lineCap = 'round'; ctx.strokeStyle = C.ink; ctx.lineWidth = 20; ctx.stroke(top); ctx.strokeStyle = C.pool; ctx.lineWidth = 10; ctx.stroke(top); ctx.restore();
    LB.halftone(ctx, wall, { x: big.x, y: wy, w: big.w, h: 180, grid: 10, rMax: 2.6, alpha: 0.5, angle: 90, f: u => u });
    
    // blood vessel at the bottom (mostly empty: almost nothing gets past)
    const vy = big.y + big.h - 62;
    const ves = LB.capsule(big.x - 40, vy, big.x + big.w + 40, vy, 34);
    LB.draw(ctx, ves, { fill: C.tomato, lw: 6, reg: 0, inside: c => {
      for (const bx of [0.12, 0.3, 0.52, 0.86]) LB.draw(c, LB.ell(big.x + big.w * bx, vy + 4, 26, 14), { fill: C.tomato, lw: 4, reg: 0, shade: { d: 6, grid: 6, r: 1.4 } });
      LB.miniUmi(c, big.x + big.w * 0.7, vy + 2, 36, { look: [1, 0], lw: 3 });
    } });
    ctx.fillStyle = C.cream; ctx.globalAlpha = 0.6; ctx.fill(LB.capsule(big.x + 20, vy - 18, big.x + big.w * 0.45, vy - 18, 4)); ctx.globalAlpha = 1;

    // six gut cells along the wall
    const n = 6, r = 66, step = (big.w - 40) / n, baseY = wy + 6;
    const cellX = i => big.x + 20 + step * (i + 0.5);
    // queue path (serpentine, tidy) entering from the left, ending at the front cell's mouth
    const front = 0, fx = cellX(front), fy = baseY - r * 0.85 * 1.0;
    
    // dotted queue guide
    ctx.save(); ctx.setLineDash([2, 18]); ctx.lineCap = 'round'; ctx.lineWidth = 7; ctx.strokeStyle = C.cream; ctx.globalAlpha = 0.9;
    const qp = new Path2D(); const QS = sampleQueue(big, fx, fy); QS.forEach(([x, y], i) => i ? qp.lineTo(x, y) : qp.moveTo(x, y));
    ctx.stroke(qp); ctx.restore();
    // place Umis along the queue path by sampling it
    const samples = sampleQueue(big, fx, fy);
    const spacing = 92, tot = samples.length;
    const flow = (t * 60) % spacing;
    const umis = [];
    for (let d = tot - 1 - 40 - flow; d > 0; d -= spacing) umis.push(samples[Math.floor(d)]);
    umis.forEach(([x, y], i) => LB.miniUmi(ctx, x, y - 6 + Math.sin(t * 6 + i) * 3, 60, { look: [-0.4, 0.6], shadow: 5 }));

    // cells (drawn after the queue so the front Umi tucks behind the chomp)
    for (let i = n - 1; i >= 0; i--) {
      const x = cellX(i), chomp = i === front;
      const breathe = LB.bounce(t, 1.1, i * 0.17) * 0.06;
      LB.gutCell(ctx, x, baseY, r, { squash: chomp ? 1 : breathe, mood: chomp ? 'chomp' : 'eager', look: [-0.8, -0.6] });
    }
    // crumbs + spark sticker over the chomping cell
    const sx = fx - 18, sy = baseY - r * 2 - 96;
    LB.sticker(ctx, { x: sx, y: sy, w: 0, h: 0, rot: LB.deg(8), fill: C.butter, path: LB.star(0, 0, 8, 26, 56, 0, 4), shadow: 6 }, c => {
      c.fillStyle = C.white; c.fill(LB.circ(-8, -8, 7));
    });
    for (const [dx, dy, s] of [[-72, -30, 14], [70, -12, 11], [-40, 10, 9]]) LB.draw(ctx, LB.hexRod(fx + dx, baseY - r * 1.6 + dy, s * 0.62, s, 2), { fill: C.butter, lw: 3, reg: 0 });
  }
  function sampleQueue(big, fx, fy) {
    // poly-line sampling of the same serpentine path at ~1px
    const pts = [];
    const seg = (a, b) => { const L = Math.hypot(b[0] - a[0], b[1] - a[1]); for (let i = 0; i < L; i++) pts.push([a[0] + (b[0] - a[0]) * i / L, a[1] + (b[1] - a[1]) * i / L]); };
    const arc = (cx, cy, r, a0, a1) => { const L = Math.abs(a1 - a0) * r; for (let i = 0; i < L; i++) { const a = a0 + (a1 - a0) * i / L; pts.push([cx + Math.cos(a) * r, cy + Math.sin(a) * r]); } };
    const y1 = big.y + 280, y2 = big.y + 440, xr = big.x + big.w - 155;
    seg([big.x - 60, y1], [xr, y1]);
    arc(xr, y1 + 85, 85, -Math.PI / 2, Math.PI / 2);
    seg([xr, y2], [big.x + 240, y2]);
    const qa = [big.x + 240, y2], qc = [fx + 20, y2], qb = [fx + 40, fy - 60];
    for (let i = 0; i < 260; i++) { const k = i / 260; pts.push([(1 - k) * (1 - k) * qa[0] + 2 * (1 - k) * k * qc[0] + k * k * qb[0], (1 - k) * (1 - k) * qa[1] + 2 * (1 - k) * k * qc[1] + k * k * qb[1]]); }
    return pts;
  }

  // ---------------------------------------------------------------- Frame C — the 1968 letter + headline stamp
  function frameC(ctx, t) {
    const W = LB.W, H = LB.H, S = LB.SAFE;
    ctx.fillStyle = C.cream; ctx.fillRect(0, 0, W, H);
    // newsprint: coarse 14px halftone wash
    LB.halftone(ctx, null, { x: 0, y: 0, w: W, h: H, grid: 14, rMax: 2.4, alpha: 0.35, angle: 90, f: u => 0.5 + 0.5 * u });

    // top strip (decorative, under Reel UI): newspaper rule + section header
    ctx.fillStyle = C.ink; ctx.fillRect(40, 168, W - 80, 6); ctx.fillRect(40, 180, W - 80, 2);
    LB.text(ctx, 'LETTERS', 60, 150, { font: F.display, size: 64, align: 'left' });
    

    // row of three suspects: wine, salt, shaker (columns become compartments)
    const y0 = 228, h0 = 250;
    const cols = [LB.col(0, 2), LB.col(2, 2), LB.col(4, 2)];
    const names = ['WINE?', 'SALT?', 'MSG?'];
    cols.forEach((c, i) => {
      LB.compartment(ctx, c.x, y0, c.w, h0, { floor: C.cream, depth: 14, grid: 14, dotR: 3.2, inside: cx => {
        LB.halftone(cx, null, { x: c.x, y: y0, w: c.w, h: h0, grid: 14, rMax: 3.6, alpha: 0.35, angle: 90, f: u => u });
        const ix = c.x + c.w / 2 - 22, iy = y0 + 112;
        cx.save(); cx.translate(ix, iy); cx.scale(0.72, 0.72);
        if (i === 0) wineBottle(cx, 0, 0); else if (i === 1) saltCellar(cx, 0, 10); else shaker(cx, 0, 0);
        cx.restore();
        LB.text(cx, names[i], c.x + c.w / 2, y0 + h0 - 24, { font: F.mono, weight: 700, size: 40 });
      } });
      LB.sticker(ctx, { x: c.x + c.w - 44, y: y0 + 54, w: 74, h: 84, rot: LB.deg(i % 2 ? 8 : -6), fill: C.cream, r: 14, shadow: 6 }, s =>
        LB.text(s, '?', 0, 30, { font: F.display, size: 72 }));
    });

    // the letter (page sticker)
    const px = S.x + 18, py = 506, pw = S.w - 36, ph = 770;
    LB.sticker(ctx, { x: px + pw / 2, y: py + ph / 2, w: pw, h: ph, rot: LB.deg(1.5), fill: C.white, r: 10 }, c => {
      const L = -pw / 2 + 52, T = -ph / 2;
      LB.halftone(c, LB.rr(-pw / 2, -ph / 2, pw, ph, 10), { x: -pw / 2, y: -ph / 2, w: pw, h: ph, grid: 14, rMax: 2.2, alpha: 0.18, angle: 45, f: u => u });
      const doc = { font: F.doc, size: 30 };
      LB.text(c, '1968', pw / 2 - 52, T + 86, { font: F.doc, weight: 700, size: 48, align: 'right' });
      LB.text(c, 'To the Editor:', L, T + 150, { font: F.doc, weight: 700, size: 36, align: 'left' });
      const lines = [
        ['Sir: I write about a curious'],
        ['complaint after dining out:'],
        ['', 'numbness', ', ', 'weakness', ' and'],
        ['', 'palpitations', ', some twenty'],
        ['minutes in. The cooking wine?'],
        ['The salt? The seasoning? I can\u2019t say.'],
      ];
      lines.forEach((parts, li) => {
        let x = L; const y = T + 220 + li * 50;
        parts.forEach((s, k) => {
          LB.text(c, s, x, y, { ...doc, align: 'left' });
          const w = LB.measure(c, s, doc);
          if (k % 2 === 1) { c.fillStyle = C.ink; c.fill(LB.rr(x - 2, y + 8, w + 4, 6, 3)); }
          x += w;
        });
      });
    });

    // impact lines around the stamp
    const stx = S.x + S.w / 2 - 6, sty = 1110;
    ctx.save(); ctx.translate(stx, sty); ctx.rotate(LB.deg(-7));
    ctx.lineCap = 'round'; ctx.strokeStyle = C.ink; ctx.lineWidth = 12;
    const imp = [[-445, 30, -540, 30], [-440, 130, -515, 180], [445, -110, 515, -160], [450, 10, 540, 10], [440, 120, 510, 168], [-180, 205, -210, 260]];
    for (const [a, b, c2, d] of imp) { ctx.beginPath(); ctx.moveTo(a, b); ctx.lineTo(c2, d); ctx.stroke(); }
    ctx.restore();
    // the stamp impression
    stampPlate(ctx, stx, sty, t);
    // context tag: it's a quoted 1968 headline
    LB.sticker(ctx, { x: S.x + 600, y: 1252, w: 440, h: 60, rot: LB.deg(-3), fill: C.cream, r: 12, shadow: 6 }, c =>
      LB.text(c, 'JOURNAL HEADLINE, 1968', 0, 14, { font: F.mono, weight: 700, size: 30 }));

    // the rubber stamp itself, rebounding up out of frame (decorative zone)
    rubberStamp(ctx, 690, 10, t);

    LB.caption(ctx, ['under', 'a', 'catchy', 'headline:'], 2, { cy: 1390 });

    // below caption: more newspaper columns multiplying (decorative under Reel UI)
    const rng = LB.rng(1968);
    for (let k = 0; k < 3; k++) {
      const c = LB.col(k * 2, 2), y = 1510, h = 380;
      LB.compartment(ctx, c.x, y, c.w, h, { floor: C.cream, depth: 12, grid: 14, dotR: 3, inside: cx => {
        cx.fillStyle = C.ink; cx.fillRect(c.x + 20, y + 26, c.w - 40, 26);
        for (let i = 0; i < 9; i++) { const lw = (c.w - 40) * (0.6 + rng() * 0.4); cx.globalAlpha = 0.55; cx.fillRect(c.x + 20, y + 76 + i * 30, lw, 10); }
        cx.globalAlpha = 1;
        LB.halftone(cx, LB.rr(c.x + 20, y + 76 + 9 * 30, c.w - 40, 60, 6), { x: c.x + 20, y: y + 346, w: c.w - 40, h: 60, grid: 14, rMax: 5, f: u => u });
      } });
    }
  }

  function stampPlate(ctx, x, y, t) {
    const w = 820, rng = LB.rng(7);
    const st = { font: LB.F.display, size: 84 };
    const lines = ['\u201CCHINESE-', 'RESTAURANT', 'SYNDROME\u201D'];
    const size = Math.min(84, ...lines.map(s => LB.fit(ctx, s, w - 120, st))), lh = size * 1.08, h = lh * 3 + 86;
    ctx.save(); ctx.translate(x, y); ctx.rotate(LB.deg(-7)); ctx.scale(1.03, 0.97); // squashed on impact
    // ink-splat edge: rounded rect with irregular bumps
    const pts = [], N = 90;
    for (let i = 0; i < N; i++) {
      const a = i / N * Math.PI * 2, cx = Math.cos(a), cy = Math.sin(a);
      const k = Math.pow(Math.pow(Math.abs(cx), 6) + Math.pow(Math.abs(cy), 6), -1 / 6);
      const j = 1 + (rng() - 0.5) * 0.035;
      pts.push([cx * k * w / 2 * j, cy * k * h / 2 * (1 + (rng() - 0.5) * 0.06)]);
    }
    const plate = LB.poly(pts, 6);
    LB.draw(ctx, plate, { fill: C.tomato, lw: 6, reg: 4 });
    // splat droplets
    for (let i = 0; i < 9; i++) {
      const a = rng() * Math.PI * 2, d = 1.06 + rng() * 0.12, rr = 5 + rng() * 9;
      LB.draw(ctx, LB.circ(Math.cos(a) * w / 2 * d, Math.sin(a) * h / 2 * d, rr), { fill: C.tomato, lw: 4, reg: 0 });
    }
    // inner double border knocked out
    ctx.strokeStyle = C.cream; ctx.lineWidth = 6; ctx.stroke(LB.rr(-w / 2 + 22, -h / 2 + 22, w - 44, h - 44, 18));
    ctx.lineWidth = 2; ctx.stroke(LB.rr(-w / 2 + 34, -h / 2 + 34, w - 68, h - 68, 12));
    lines.forEach((s, i) => LB.text(ctx, s, 0, -lh * 1.5 + lh * i + size * 0.88, { ...st, size, fill: C.cream }));
    // rubber-stamp speckle: knocked-out pits so it reads as an impression
    ctx.fillStyle = C.tomato;
    for (let i = 0; i < 160; i++) { const px = (rng() - 0.5) * (w - 80), py = (rng() - 0.5) * (h - 70); ctx.fill(LB.circ(px, py, 1 + rng() * 2.4)); }
    ctx.restore();
  }

  function rubberStamp(ctx, x, y, t) {
    ctx.save(); ctx.translate(x, y); ctx.rotate(LB.deg(-10));
    // motion lines (upward)
    ctx.strokeStyle = C.ink; ctx.lineWidth = 8; ctx.lineCap = 'round';
    for (const dx of [-150, -110, 110, 150]) { ctx.beginPath(); ctx.moveTo(dx, 120); ctx.lineTo(dx, 40); ctx.stroke(); }
    // knob + neck
    LB.draw(ctx, LB.circ(0, -40, 70), { fill: C.ink, lw: 6, reg: 0 });
    ctx.fillStyle = C.cream; ctx.fill(LB.ell(-22, -64, 18, 10, -0.5));
    LB.draw(ctx, LB.rr(-34, -10, 68, 70, 10), { fill: C.ink, lw: 6, reg: 0 });
    // block
    LB.draw(ctx, LB.rr(-120, 50, 240, 70, 14), { fill: C.cream, lw: 6, shade: { d: 12, grid: 14, r: 3.2 } });
    // rubber pad (Tomato)
    LB.draw(ctx, LB.rr(-112, 118, 224, 30, 8), { fill: C.tomato, lw: 6, reg: 2 });
    ctx.restore();
  }

  function wineBottle(ctx, x, y) {
    const p = new Path2D();
    p.moveTo(x - 16, y - 112); p.lineTo(x + 16, y - 112); p.lineTo(x + 16, y - 62);
    p.bezierCurveTo(x + 16, y - 40, x + 40, y - 36, x + 40, y - 6); p.lineTo(x + 40, y + 70); p.quadraticCurveTo(x + 40, y + 80, x + 30, y + 80);
    p.lineTo(x - 30, y + 80); p.quadraticCurveTo(x - 40, y + 80, x - 40, y + 70); p.lineTo(x - 40, y - 6);
    p.bezierCurveTo(x - 40, y - 36, x - 16, y - 40, x - 16, y - 62); p.closePath();
    LB.draw(ctx, p, { fill: C.ink, lw: 6, reg: 0 });
    LB.draw(ctx, LB.rr(x - 32, y + 2, 64, 48, 6), { fill: C.cream, lw: 4, reg: 0 });
    ctx.fillStyle = C.cream; ctx.fill(LB.capsule(x - 26, y - 20, x - 26, y - 2, 4));
    LB.draw(ctx, LB.rr(x - 18, y - 122, 36, 18, 4), { fill: C.cream, lw: 4, reg: 0 });
  }
  function saltCellar(ctx, x, y) {
    const body = LB.poly([[x - 40, y - 70], [x + 40, y - 70], [x + 50, y + 60], [x - 50, y + 60]], 12);
    LB.draw(ctx, body, { fill: C.cream, lw: 6, shade: { d: 14, grid: 14, r: 3.2 } });
    LB.draw(ctx, LB.ell(x, y - 74, 44, 22), { fill: C.cream, lw: 6, reg: 0 });
    ctx.fillStyle = C.ink; for (const [dx, dy] of [[-14, -4], [14, -4], [0, 6]]) ctx.fill(LB.circ(x + dx, y - 74 + dy, 4.5));
    LB.text(ctx, 'S', x, y + 34, { font: LB.F.display, size: 48 });
  }
  function shaker(ctx, x, y) {
    LB.draw(ctx, LB.rr(x - 36, y - 60, 72, 140, 18), { fill: C.cream, lw: 6, shade: { d: 14, grid: 14, r: 3.2 } });
    LB.draw(ctx, LB.rr(x - 40, y - 96, 80, 40, 16), { fill: C.ink, lw: 6, reg: 0 });
    ctx.fillStyle = C.cream; for (const dx of [-16, 0, 16]) ctx.fill(LB.circ(x + dx, y - 80, 4));
    // a few rod crystals inside (glass)
    for (const [dx, dy, r] of [[-12, 50, 0.4], [10, 42, -0.3], [0, 62, 0.1]]) LB.draw(ctx, LB.xf(LB.hexRod(0, 0, 12, 22, 2), { x: x + dx, y: y + dy, rot: r }), { fill: C.cream, lw: 3, reg: 0 });
  }

  window.STYLEFRAMES = {
    size: [1080, 1920],
    fonts: [
      { family: 'Dela Gothic One', url: '/assets/DelaGothicOne.ttf', weight: '400' },
      { family: 'Atkinson Hyperlegible', url: '/assets/AtkinsonHyperlegible-Regular.ttf', weight: '400' },
      { family: 'Atkinson Hyperlegible', url: '/assets/AtkinsonHyperlegible-Bold.ttf', weight: '700' },
      { family: 'Space Mono', url: '/assets/SpaceMono-Bold.ttf', weight: '700' },
      { family: 'Courier Prime', url: '/assets/CourierPrime-Regular.ttf', weight: '400' },
      { family: 'Courier Prime', url: '/assets/CourierPrime-Bold.ttf', weight: '700' },
    ],
    frames: { A: frameA, B: frameB, C: frameC },
  };
})();
