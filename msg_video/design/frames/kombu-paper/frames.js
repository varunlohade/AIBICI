// Direction 1: KOMBU PAPER: style frames A, B, C (vertical 9:16 Reel, 1080x1920).
// Cut paper + two-colour risograph. No outlines, no glows, no gradients.
// Depth = stacked paper + hard offset shadows. Texture = pre-baked riso grain / fibre / halftone tiles.
//
// window.KP is the reusable kit (meant to become the scene kit if this direction is chosen):
//   KP.P                      palette (+ derived tints)
//   KP.rng(seed)              deterministic RNG (same algorithm as KZ.rng)
//   KP.boil(t)                15fps boil frame index (stop-motion jitter seed)
//   KP.rr/ell/egg/drop/tornRect/curve  point-list builders (closed shapes)
//   KP.jitter(pts, seed, amp) hand-cut edge jitter (seeded, cached)
//   KP.paper(ctx, pts, fill, {shadow, shadowA, mis, misOff})  cut paper w/ hard shadow + riso misregistration
//   KP.brush(ctx, pts, w, color)  tapered Squid-Ink brush stroke
//   KP.face(...), KP.goldie(...), KP.cell(...), KP.tv(...), KP.puppetDoctor(...), KP.fastener(...)
//   KP.slip(...)              text printed on a torn paper slip
//   KP.caption(...)           burned-in Reel caption (torn strip + raised word chip)
//   KP.ripple(...)            the Fifth Ripple (4 paper rings + 1 wide gold ring)
//   KP.texture(ctx, name, a)  full-frame grain pass ('grain' | 'news' | 'speck')
//   KP.stamp(...)             distressed rubber-stamp impression (cached canvas)
(function () {
  const W = 1080, H = 1920;
  const KP = {};
  window.KP = KP;

  // ---------------------------------------------------------------- colour
  const hex = h => [1, 3, 5].map(i => parseInt(h.slice(i, i + 2), 16));
  const mix = (a, b, t) => { const A = hex(a), B = hex(b); return '#' + A.map((v, i) => Math.round(v + (B[i] - v) * t).toString(16).padStart(2, '0')).join(''); };
  const rgba = (h, a) => { const p = hex(h); return `rgba(${p[0]},${p[1]},${p[2]},${a})`; };
  KP.mix = mix; KP.rgba = rgba;
  // VIVID revision (client rule: no greys/beige/cream/muddy tones). Paper sheets are bright riso stock;
  // 'rice' is now clean white paper used only for small slips/captions.
  const P = KP.P = {
    rice: '#FFFFFF', ink: '#1E1033', plum: '#4422CC', gold: '#FFB000', lemon: '#FFE135',
    pink: '#FF3D8B', bubble: '#FF7AB6', magenta: '#D0157A', coral: '#FF5A4E', tomato: '#F2301B',
    blue: '#1F5BFF', turq: '#00C2C7', kombu: '#00B26B', grape: '#7B3FF2', brass: '#FFC21A',
  };
  P.kraft = P.brass;                 // fasteners / default cardboard
  P.pink60 = P.bubble; P.pink30 = '#FFA8D2';
  P.pinkDeep = P.magenta;
  P.mouth = mix(P.magenta, P.ink, 0.45);
  P.goldLight = P.lemon;
  P.ink70 = mix(P.ink, P.grape, 0.35);
  P.screenDim = '#2A13A0';
  P.wood = '#FF8A1F';                // puppet sticks: bright orange dowel
  P.shadow = rgba(P.ink, 0.22);

  // ---------------------------------------------------------------- rng / time
  KP.rng = seed => { let s = seed >>> 0 || 1; return () => ((s = Math.imul(s ^ (s >>> 15), 1 | s), s = s + Math.imul(s ^ (s >>> 7), 61 | s) ^ s, ((s ^ (s >>> 14)) >>> 0) / 4294967296)); };
  KP.boil = t => Math.floor(t * 15 / 2) * 2;  // on twos at 30fps -> new jitter every 2 frames
  const TAU = Math.PI * 2;

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
  // egg-ish ellipse: egg > 0 widens the bottom (circles are never perfect in this world)
  KP.egg = (cx, cy, rx, ry, egg = 0.06, n = 56) => {
    const pts = [];
    for (let i = 0; i < n; i++) { const a = i / n * TAU, s = Math.sin(a); pts.push([cx + Math.cos(a) * rx * (1 + egg * s), cy + s * ry]); }
    return pts;
  };
  KP.ell = (cx, cy, rx, ry, n = 48) => KP.egg(cx, cy, rx, ry, 0, n);
  // teardrop pointing up; (cx, cy) = centre of the round belly, total height h, width h/1.2
  KP.drop = (cx, cy, h, n = 44) => {
    const pts = [], w = h / 1.2;
    for (let i = 0; i < n; i++) {
      const a = i / n * TAU;
      const x = Math.sin(a) * Math.pow(Math.sin(a / 2), 1.25) * w * 0.56;
      const y = -Math.cos(a) * h * 0.5;
      pts.push([cx + x, cy + y + h * 0.04]);
    }
    return pts;
  };
  // Catmull-Rom through control points (closed or open)
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
  // rectangle with torn left/right ends (paper slips, lower-thirds)
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

  // hand-cut edge jitter along the outline normal: smooth periodic wobble + tiny per-vertex grit
  KP.jitter = (pts, seed, amp = 1.5) => {
    const n = pts.length, r = KP.rng(seed * 7919 + n), ph = [r() * TAU, r() * TAU, r() * TAU], f = [2 + (r() * 2 | 0), 5 + (r() * 3 | 0), 11];
    return pts.map((p, i) => {
      const a = pts[(i - 1 + n) % n], b = pts[(i + 1) % n];
      let nx = b[1] - a[1], ny = a[0] - b[0]; const l = Math.hypot(nx, ny) || 1; nx /= l; ny /= l;
      const u = i / n;
      const v = .45 * Math.sin(TAU * f[0] * u + ph[0]) + .3 * Math.sin(TAU * f[1] * u + ph[1]) + .15 * Math.sin(TAU * f[2] * u + ph[2]) + .25 * (r() * 2 - 1);
      return [p[0] + nx * v * amp, p[1] + ny * v * amp];
    });
  };
  // rough torn edge: resample densely + jagged offsets (for scraps / ripped sheets)
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

  // the core primitive: one piece of cut paper
  //   shadow: [dx,dy] | false  (near (6,8), far (3,4)); mis: second riso ink for the 2px under-print
  KP.paper = (ctx, pts, fill, o = {}) => {
    const path = KP.path(pts);
    if (o.shadow !== false) {
      const [dx, dy] = o.shadow || [6, 8];
      ctx.save(); ctx.translate(dx, dy); ctx.fillStyle = rgba(P.ink, o.shadowA ?? .22); ctx.fill(path); ctx.restore();
    }
    if (o.mis) {
      const [mx, my] = o.misOff || [-2.5, 2];
      ctx.save(); ctx.globalCompositeOperation = 'multiply'; ctx.globalAlpha = o.misA ?? .35;
      ctx.translate(mx, my); ctx.fillStyle = o.mis; ctx.fill(path); ctx.restore();
    }
    if (fill) { ctx.fillStyle = fill; ctx.fill(path); }
    return path;
  };
  KP.SH = { near: [6, 8], far: [3, 4] };

  // tapered brush stroke (opaque colour), pts = open polyline
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
  KP.arcPts = (cx, cy, rx, ry, a0, a1, n = 14) => { const o = []; for (let i = 0; i <= n; i++) { const a = a0 + (a1 - a0) * i / n; o.push([cx + Math.cos(a) * rx, cy + Math.sin(a) * ry]); } return o; };

  // ---------------------------------------------------------------- text
  KP.font = (size, fam = 'Nunito', weight = 800) => `${weight} ${size}px "${fam}"`;
  KP.text = (ctx, s, x, y, o = {}) => {
    ctx.save();
    ctx.font = KP.font(o.size || 46, o.font || 'Nunito', o.weight ?? 800);
    ctx.letterSpacing = (o.ls || 0) + 'px';
    ctx.textAlign = o.align || 'left'; ctx.textBaseline = o.baseline || 'alphabetic';
    ctx.globalAlpha *= o.alpha ?? 1; ctx.fillStyle = o.color || P.ink; ctx.fillText(s, x, y);
    ctx.restore();
  };
  KP.measure = (ctx, s, o = {}) => { ctx.save(); ctx.font = KP.font(o.size || 46, o.font || 'Nunito', o.weight ?? 800); ctx.letterSpacing = (o.ls || 0) + 'px'; const w = ctx.measureText(s).width; ctx.restore(); return w; };

  // text printed on a torn paper slip. lines = [{s, size, font, weight, color, ls}], (x,y) = top-left before rotation
  KP.slip = (ctx, lines, x, y, o = {}) => {
    const padX = o.padX ?? 28, padY = o.padY ?? 18, gap = o.gap ?? 8;
    const ws = lines.map(l => KP.measure(ctx, l.s, l));
    const w = Math.max(...ws) + padX * 2, h = lines.reduce((a, l) => a + l.size, 0) + gap * (lines.length - 1) + padY * 2;
    ctx.save(); ctx.translate(x + w / 2, y + h / 2); ctx.rotate(o.rot || 0); ctx.translate(-w / 2, -h / 2);
    const pts = KP.jitter(KP.tornRect(0, 0, w, h, o.seed || 3, o.tear ?? 5), (o.seed || 3) + (o.boil || 0), 1);
    KP.paper(ctx, pts, o.fill || P.rice, { shadow: o.shadow ?? KP.SH.near, shadowA: o.shadowA });
    if (o.tape) { // a strip of tape holding it down
      const tp = KP.jitter(KP.tornRect(-22, -10, 90, 34, 77, 3), 5, .8);
      ctx.save(); ctx.translate(0, 0); ctx.rotate(-.5); KP.paper(ctx, tp, rgba(o.tape, .85), { shadow: false }); ctx.restore();
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

  // ---------------------------------------------------------------- textures (generated once)
  const tiles = {};
  const mkCanvas = (w, h) => { const c = document.createElement('canvas'); c.width = w; c.height = h; return c; };
  KP.tile = name => {
    if (tiles[name]) return tiles[name];
    const S = 512, c = mkCanvas(S, S), g = c.getContext('2d'), r = KP.rng({ grain: 101, news: 202, speck: 303 }[name]);
    if (name === 'speck') { // light flecks: paper showing through ink on dark grounds
      for (let i = 0; i < 2600; i++) { g.fillStyle = `rgba(243,234,215,${.2 + r() * .5})`; const s = r() < .85 ? 1 : 2; g.fillRect(r() * S | 0, r() * S | 0, s, s); }
      return (tiles[name] = c);
    }
    g.fillStyle = '#fff'; g.fillRect(0, 0, S, S);
    if (name === 'news') { // coarse halftone with blotchy density (periodic in tile)
      const ph = [r() * TAU, r() * TAU, r() * TAU];
      g.fillStyle = '#000';
      const sp = 8;
      for (let i = -2; i < S / sp + 2; i++) for (let j = -2; j < S / sp + 2; j++) {
        const x = (i + (j % 2) * .5) * sp, y = j * sp * .866;
        const d = .5 + .5 * (.5 * Math.sin(TAU * 2 * x / S + ph[0]) * Math.sin(TAU * 3 * y / S + ph[1]) + .5 * Math.sin(TAU * (x + y) / S + ph[2]));
        const rad = .5 + d * 1.9;
        g.beginPath(); g.arc((x + S) % S, (y + S) % S, rad, 0, TAU); g.fill();
      }
    }
    // riso grain: per-pixel speckle + clumps
    const img = g.getImageData(0, 0, S, S), d = img.data;
    for (let i = 0; i < d.length; i += 4) {
      const n = r(); let v = 0;
      if (n < .22) v = 30 + r() * 110; else if (n < .5) v = r() * 30;
      d[i] = Math.max(0, d[i] - v); d[i + 1] = Math.max(0, d[i + 1] - v); d[i + 2] = Math.max(0, d[i + 2] - v);
    }
    g.putImageData(img, 0, 0);
    for (let i = 0; i < 1400; i++) { g.fillStyle = `rgba(0,0,0,${.15 + r() * .35})`; const x = r() * S, y = r() * S, s = 1 + r() * 2.2; g.beginPath(); g.arc(x, y, s, 0, TAU); g.fill(); }
    // paper fibre: faint long strands (baked so effective strength ~6% when tile is applied at ~12-14%)
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
  // one full-frame pass. 'grain'/'news' multiply; 'speck' normal (for dark grounds)
  KP.texture = (ctx, name, a, ox = 0, oy = 0) => {
    ctx.save(); ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.globalCompositeOperation = name === 'speck' ? 'source-over' : 'multiply'; ctx.globalAlpha = a;
    const pat = ctx.createPattern(KP.tile(name), 'repeat'); pat.setTransform(new DOMMatrix([1, 0, 0, 1, ox, oy]));
    ctx.fillStyle = pat; ctx.fillRect(0, 0, ctx.canvas.width, ctx.canvas.height);
    ctx.restore();
  };
  // offscreen layer matching the main canvas (for cut-outs with destination-out)
  const layers = {};
  KP.layer = (ctx, key) => {
    const cw = ctx.canvas.width, ch = ctx.canvas.height;
    let c = layers[key]; if (!c || c.width !== cw || c.height !== ch) c = layers[key] = mkCanvas(cw, ch);
    const g = c.getContext('2d'); g.setTransform(1, 0, 0, 1, 0, 0); g.clearRect(0, 0, cw, ch); g.setTransform(ctx.getTransform());
    return { c, g, draw(alpha = 1, dx = 0, dy = 0, op = 'source-over') { ctx.save(); ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.globalAlpha = alpha; ctx.globalCompositeOperation = op; ctx.drawImage(c, dx, dy); ctx.restore(); } };
  };

  // ---------------------------------------------------------------- characters
  KP.fastener = (ctx, x, y, r = 6) => {
    ctx.fillStyle = rgba(P.ink, .3); ctx.beginPath(); ctx.arc(x + 1.5, y + 2, r, 0, TAU); ctx.fill();
    ctx.fillStyle = P.kraft; ctx.beginPath(); ctx.arc(x, y, r, 0, TAU); ctx.fill();
    ctx.fillStyle = mix(P.kraft, P.ink, .35); ctx.beginPath(); ctx.arc(x - r * .15, y - r * .15, r * .32, 0, TAU); ctx.fill();
  };
  // face: s ~ head width. eyes 'dot' | 'happy' | 'closed'; mouth 'smile' | 'open' | 'chomp' | 'o' | 'wobble' | 'none'
  KP.face = (ctx, x, y, s, o = {}) => {
    const ink = o.ink || P.ink, ex = s * .17, er = Math.max(3, s * .045), lw = Math.max(5, s * .045);
    const lx = o.look || 0;
    // cheeks (multiply 40%)
    if (o.cheeks !== false) {
      ctx.save(); ctx.globalCompositeOperation = 'multiply'; ctx.globalAlpha = .4; ctx.fillStyle = o.cheek || P.pink;
      for (const sx of [-1, 1]) { ctx.beginPath(); ctx.ellipse(x + lx + sx * s * .27, y + s * .1, s * .085, s * .065, 0, 0, TAU); ctx.fill(); }
      ctx.restore();
    }
    for (const sx of [-1, 1]) {
      const cx = x + lx + sx * ex;
      if (o.eyes === 'happy') KP.brush(ctx, KP.arcPts(cx, y + er, er * 1.9, er * 1.7, Math.PI * 1.1, Math.PI * 1.9, 8), lw * .9, ink, { taper: .3 });
      else if (o.eyes === 'closed') KP.brush(ctx, KP.arcPts(cx, y - er * .6, er * 1.8, er * 1.4, Math.PI * .15, Math.PI * .85, 8), lw * .9, ink, { taper: .3 });
      else { ctx.fillStyle = ink; ctx.beginPath(); ctx.ellipse(cx, y, er, er * 1.12, 0, 0, TAU); ctx.fill(); }
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
    // lighter gold belly highlight cut (flat paper, not a gradient)
    KP.paper(ctx, KP.jitter(KP.egg(-h * .12, h * .2, h * .2, h * .13, 0, 24), seed + 3, .8), P.lemon, { shadow: false });
    // steam curl on top
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
    // bib (lighter pink napkin, scalloped hem)
    const bw = w * .52, by = y + h * .1, bh = h * .26;
    const bib = [];
    bib.push([x - bw * .28, by - bh * .5]); bib.push([x + bw * .28, by - bh * .5]);
    for (let i = 0; i <= 24; i++) { const u = i / 24, bx = x + bw / 2 - u * bw; bib.push([bx + (u < .5 ? 0 : 0), by + bh * .25 + Math.abs(Math.sin(u * Math.PI * 3)) * bh * .25 - (Math.abs(u - .5) * bh * .9)]); }
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

  // the Fifth Ripple: 4 paper rings + wide gold 5th. p in [0,1] = spread
  KP.ripple = (ctx, x, y, R, o = {}) => {
    const sq = o.squash ?? .35, p = o.p ?? 1;
    for (let i = 0; i < 5; i++) {
      const rr = R * (i + 1) / 5 * p, gold = i === 4;
      const pts = KP.jitter(KP.ell(x, y, rr, rr * sq, 56), 40 + i + (o.boil || 0), 1.2);
      pts.push(pts[0]);
      KP.brush(ctx, pts, gold ? 14 : 6, gold ? P.gold : (o.ring || P.rice), { taper: .1 });
    }
  };

  // ---------------------------------------------------------------- TV set (cardboard cut-out)
  // returns the screen rect {x,y,w,h}. o.panel: 'right' | 'bottom'
  KP.tv = (ctx, x, y, w, h, o = {}) => {
    const seed = (o.seed || 1) + (o.boil || 0), body = o.body || P.kraft;
    const sh = o.shadow || KP.SH.near;
    // antenna (behind)
    if (o.antenna) {
      const ax = x + w * (o.antenna), ay = y + 4;
      KP.brush(ctx, [[ax, ay], [ax - w * .12, ay - h * .3]], 6, o.antColor || mix(body, P.ink, .45), { taper: .2 });
      KP.brush(ctx, [[ax, ay], [ax + w * .1, ay - h * .34]], 6, o.antColor || mix(body, P.ink, .45), { taper: .2 });
      ctx.fillStyle = o.antColor || mix(body, P.ink, .45);
      ctx.beginPath(); ctx.arc(ax - w * .12, ay - h * .3, 7, 0, TAU); ctx.arc(ax + w * .1, ay - h * .34, 7, 0, TAU); ctx.fill();
      KP.paper(ctx, KP.jitter(KP.egg(ax, ay + 2, 26, 14, 0, 24), seed + 5, 1), mix(body, P.ink, .2), { shadow: sh });
    }
    // feet
    if (o.feet) for (const fx of [.18, .82]) KP.paper(ctx, KP.jitter(KP.rr(x + w * fx - 16, y + h - 6, 32, 26, 8, 10), seed + fx * 10, 1), mix(body, P.ink, .3), { shadow: sh });
    KP.paper(ctx, KP.jitter(KP.rr(x, y, w, h, 24), seed, 1.5), body, { shadow: sh, shadowA: o.shadowA });
    // a slightly darker cardboard lip around the screen
    const pad = o.pad || Math.min(w, h) * .07;
    const panel = o.panel || 'right', pw = panel === 'right' ? Math.min(w * .22, 150) : 0, ph = panel === 'bottom' ? Math.min(h * .14, 96) : 0;
    const sx = x + pad, sy = y + pad, sw = w - pad * 2 - pw, shh = h - pad * 2 - ph;
    KP.paper(ctx, KP.jitter(KP.rr(sx - 10, sy - 10, sw + 20, shh + 20, 26), seed + 1, 1.2), mix(body, P.ink, .18), { shadow: false });
    KP.paper(ctx, KP.jitter(KP.rr(sx, sy, sw, shh, 20), seed + 2, 1.2), o.screen || P.screenDim, { shadow: false });
    // controls
    const knob = mix(body, P.ink, .32), knobHi = mix(body, P.rice, .35);
    if (panel === 'right') {
      const cx = x + w - pad - pw / 2;
      for (const k of [0, 1]) { const ky = sy + shh * (.2 + k * .28); KP.paper(ctx, KP.jitter(KP.egg(cx, ky, pw * .24, pw * .24, .05, 28), seed + 7 + k, 1), knob, { shadow: KP.SH.far }); KP.brush(ctx, [[cx, ky], [cx + pw * .14, ky - pw * .1]], 5, knobHi, { taper: .1 }); }
      for (let i = 0; i < 4; i++) KP.brush(ctx, [[cx - pw * .26, sy + shh * (.68 + i * .08)], [cx + pw * .26, sy + shh * (.68 + i * .08)]], 5, knob, { taper: .3 });
    } else {
      const cy = y + h - pad - ph / 2;
      for (const k of [0, 1]) { const kx = x + w - pad - ph * (.5 + k * 1.1); KP.paper(ctx, KP.jitter(KP.egg(kx, cy, ph * .3, ph * .3, .05, 28), seed + 7 + k, 1), knob, { shadow: KP.SH.far }); KP.brush(ctx, [[kx, cy], [kx + ph * .17, cy - ph * .13]], 5, knobHi, { taper: .1 }); }
      for (let i = 0; i < 5; i++) KP.brush(ctx, [[sx + 10 + i * 22, cy - ph * .22], [sx + 10 + i * 22, cy + ph * .22]], 6, knob, { taper: .3 });
    }
    // dim screens get a cut-paper glare strip
    if (!o.lit) {
      ctx.save(); ctx.clip(KP.path(KP.rr(sx, sy, sw, shh, 20)));
      const gx = sx + sw * .62;
      KP.paper(ctx, [[gx, sy - 10], [gx + sw * .12, sy - 10], [gx - sw * .1, sy + shh + 10], [gx - sw * .22, sy + shh + 10]], mix(o.screen || P.screenDim, P.rice, .06), { shadow: false });
      ctx.restore();
    }
    return { x: sx, y: sy, w: sw, h: shh };
  };

  // ---------------------------------------------------------------- the fictional TV Doctor shadow puppet
  // Squid Ink silhouette, profile facing right. (x,y) = neck, s = scale. jaw = open amount 0..1
  KP.puppetDoctor = (ctx, x, y, s, o = {}) => {
    const b = o.boil || 0, L = KP.layer(ctx, 'puppet'), g = L.g;
    const T = pts => pts.map(([px, py]) => [x + px * s, y + py * s]);
    const J = (pts, k) => KP.jitter(T(pts), k + b, 1.4);
    g.fillStyle = P.ink;
    const F = p => g.fill(KP.path(p));
    // torso: lab coat
    F(J(KP.curve([[-26, -30], [-74, -20], [-98, 20], [-104, 120], [-108, 200], [0, 206], [108, 200], [102, 120], [96, 20], [72, -20], [26, -30]], true, 6), 11));
    // neck
    F(J(KP.rr(-22, -60, 44, 50, 12, 10), 12));
    // head + nose bump
    F(J(KP.egg(0, -102, 60, 64, -.04, 52), 13));
    F(J(KP.egg(54, -98, 15, 13, 0, 18), 14));
    // head mirror on its band: stem + disc
    F(J([[30, -150], [44, -168], [52, -162], [40, -146]], 15));
    F(J(KP.rot(KP.ell(54, -176, 32, 28, 36), 54, -176, -.35), 16));
    // raised arm (separate piece) + mitten with pointing finger
    const sh = [64, -8], el = [146, -44], hd = [176, -128];
    g.lineCap = 'round'; g.lineJoin = 'round'; g.strokeStyle = P.ink;
    g.lineWidth = 36 * s; g.beginPath(); g.moveTo(x + sh[0] * s, y + sh[1] * s); g.lineTo(x + el[0] * s, y + el[1] * s); g.stroke();
    g.lineWidth = 30 * s; g.beginPath(); g.moveTo(x + el[0] * s, y + el[1] * s); g.lineTo(x + hd[0] * s, y + hd[1] * s); g.stroke();
    F(J(KP.egg(180, -140, 24, 22, 0, 24), 17));
    F(J(KP.rot(KP.rr(176, -192, 16, 50, 8, 8), 184, -160, .32), 18));
    // ---- cut-outs (light shows through)
    g.save(); g.globalCompositeOperation = 'destination-out';
    // open jaw wedge: hinge behind the cheek
    const jo = o.jaw ?? .7;
    const hinge = [-4, -78], a0 = -.06, a1 = .12 + jo * .36;
    F(T([hinge, [hinge[0] + 120 * Math.cos(a0), hinge[1] + 120 * Math.sin(a0)], [hinge[0] + 120 * Math.cos(a1), hinge[1] + 120 * Math.sin(a1)]]));
    // eye
    g.beginPath(); g.arc(x + 30 * s, y - 118 * s, 7.5 * s, 0, TAU); g.fill();
    // head band slit
    g.lineWidth = 4.5 * s; g.beginPath(); g.moveTo(x - 58 * s, y - 128 * s); g.quadraticCurveTo(x - 10 * s, y - 158 * s, x + 34 * s, y - 154 * s); g.stroke();
    // mirror: punched hole + ring
    g.beginPath(); g.arc(x + 55 * s, y - 176 * s, 8 * s, 0, TAU); g.fill();
    g.lineWidth = 3.5 * s; g.beginPath(); g.ellipse(x + 54 * s, y - 176 * s, 21 * s, 18 * s, -.35, 0, TAU); g.stroke();
    // coat lapels + buttons
    g.lineWidth = 4.5 * s;
    g.beginPath(); g.moveTo(x - 24 * s, y - 26 * s); g.lineTo(x + 4 * s, y + 74 * s); g.lineTo(x + 26 * s, y - 26 * s); g.stroke();
    for (const by of [104, 146]) { g.beginPath(); g.arc(x + 6 * s, y + by * s, 5 * s, 0, TAU); g.fill(); }
    // stethoscope (cut lace line): U around the neck, Y-join, tube to chest piece
    g.lineWidth = 5.5 * s;
    const C = KP.curve([[-30, -26], [-46, 24], [-34, 70], [-16, 92]], false, 8).concat(KP.curve([[-16, 92], [-24, 128], [-46, 150]], false, 8));
    g.beginPath(); C.forEach(([px, py], i) => i ? g.lineTo(x + px * s, y + py * s) : g.moveTo(x + px * s, y + py * s)); g.stroke();
    const C2 = KP.curve([[30, -26], [40, 26], [8, 72], [-16, 92]], false, 8);
    g.beginPath(); C2.forEach(([px, py], i) => i ? g.lineTo(x + px * s, y + py * s) : g.moveTo(x + px * s, y + py * s)); g.stroke();
    g.beginPath(); g.arc(x - 50 * s, y + 164 * s, 14 * s, 0, TAU); g.stroke();
    g.beginPath(); g.arc(x - 50 * s, y + 164 * s, 4 * s, 0, TAU); g.fill();
    // separation slit between arm piece and body (separate paper piece)
    g.lineWidth = 3 * s; g.beginPath(); g.arc(x + sh[0] * s, y + sh[1] * s, 20 * s, -1.4, 1.9); g.stroke();
    g.restore();
    // sticks (drawn on the main ctx, under the silhouette)
    const st = o.stickTo || y + 400 * s;
    KP.brush(ctx, [[x + 6 * s, y + 180 * s], [x + 2 * s, st]], 16 * s, P.wood, { taper: .05 });
    KP.brush(ctx, [[x + 180 * s, y - 132 * s], [x + 128 * s, st]], 6 * s, mix(P.wood, P.ink, .15), { taper: .05 });
    // hard shadow of the puppet on the screen paper, then the puppet
    L.draw(o.shadowA ?? .22, 8, 10);
    L.draw(1);
    // paper fasteners: shoulder, elbow, jaw hinge
    for (const [fx, fy] of [sh, el, hinge]) KP.fastener(ctx, x + fx * s, y + fy * s, 6.5 * s);
    // a small kraft strap where the stick is taped on
    KP.paper(ctx, KP.jitter(KP.rr(x - 18 * s, y + 170 * s, 46 * s, 22 * s, 4, 8), 31 + b, 1), P.lemon, { shadow: KP.SH.far });
  };

  // ---------------------------------------------------------------- burned-in caption (Reel)
  // Words printed on one torn Rice Paper strip; the spoken word is a raised paper chip in the act's accent,
  // already-spoken words full ink, upcoming words 45% ink. Max 2 lines. cx = centre x, y = top.
  KP.caption = (ctx, lines, cx, y, o = {}) => {
    const size = o.size || 64, lh = size * 1.32, padX = 40, padY = 26, sp = size * .28;
    const fnt = { size, font: 'Nunito', weight: 900 };
    const lw = lines.map(l => l.reduce((a, w, i) => a + KP.measure(ctx, w.s, fnt) + (i ? sp : 0), 0));
    const w = Math.max(...lw) + padX * 2, h = lines.length * lh + padY * 2 - (lh - size) * .5;
    const x0 = cx - w / 2;
    ctx.save();
    ctx.translate(cx, y + h / 2); ctx.rotate(o.rot ?? -.012); ctx.translate(-cx, -(y + h / 2));
    KP.paper(ctx, KP.jitter(KP.tornRect(x0, y, w, h, o.seed || 9, 6), (o.seed || 9) + (o.boil || 0), 1), o.fill || P.rice, { shadow: KP.SH.near, shadowA: o.shadowA });
    lines.forEach((l, li) => {
      let xx = cx - lw[li] / 2; const by = y + padY + li * lh + size * .82;
      l.forEach(wd => {
        const ww = KP.measure(ctx, wd.s, fnt);
        if (wd.on) {
          const cp = KP.jitter(KP.rr(xx - 14, by - size * .9, ww + 28, size * 1.18, 10, 10), 21 + (o.boil || 0), 1);
          ctx.save(); ctx.translate(-3, -6); // raised: lifted off the strip with a bigger shadow
          KP.paper(ctx, cp, o.accent || P.pink, { shadow: [7, 10], mis: o.accentMis, misOff: [-3, 2] });
          KP.text(ctx, wd.s, xx, by, { ...fnt, color: o.onInk || P.ink });
          ctx.restore();
        } else KP.text(ctx, wd.s, xx, by, { ...fnt, color: wd.done ? P.ink : (o.later || '#A58BFF') });
        xx += ww + sp;
      });
    });
    ctx.restore();
  };
  // helper: split "a b *c* d" -> words, * marks current; words before current are 'done'
  KP.words = str => { let seen = false; return str.split('|').map(line => line.trim().split(/\s+/).map(s => { const on = s.startsWith('*'); if (on) seen = true; const r = { s: s.replace(/\*/g, ''), on, done: !seen }; return r; })); };

  // ---------------------------------------------------------------- distressed rubber stamp (cached)
  const stamps = {};
  KP.stamp = (key, lines, w, h, o = {}) => {
    if (stamps[key]) return stamps[key];
    const c = mkCanvas(w, h), g = c.getContext('2d'), r = KP.rng(o.seed || 55), ink = o.ink || P.tomato;
    g.fillStyle = ink; g.strokeStyle = ink;
    const outer = KP.jitter(KP.rr(10, 10, w - 20, h - 20, 18, 12), 3, 2.2), inner = KP.jitter(KP.rr(30, 30, w - 60, h - 60, 10, 12), 4, 1.6);
    const p = KP.path(outer); const pi = KP.path(KP.jitter(KP.rr(22, 22, w - 44, h - 44, 13, 12), 5, 1.6));
    g.fill(p); g.globalCompositeOperation = 'destination-out'; g.fill(pi); g.globalCompositeOperation = 'source-over';
    g.lineWidth = 5; g.stroke(KP.path(inner));
    g.textAlign = 'center'; g.textBaseline = 'alphabetic';
    const fs = o.size || 80, lh = fs * 1.12, top = h / 2 - (lines.length * lh) / 2 + fs * .82;
    g.font = `400 ${fs}px "Young Serif"`; g.letterSpacing = (o.ls || 2) + 'px';
    lines.forEach((s, i) => { g.fillText(s, w / 2, top + i * lh, w - 130); });
    // distress: uneven pressure (fades toward one corner), missing patches, speckle, dry streaks
    g.globalCompositeOperation = 'destination-out';
    for (let i = 0; i < 2600; i++) { const x = r() * w, y = r() * h, fall = Math.max(0, (x / w) * .6 + (1 - y / h) * .5 - .35); g.fillStyle = `rgba(0,0,0,${Math.min(1, .25 + fall * 1.4)})`; g.beginPath(); g.arc(x, y, .6 + r() * (1.2 + fall * 3), 0, TAU); g.fill(); }
    for (let i = 0; i < 26; i++) { const x = w * (.55 + r() * .45), y = h * r() * .55; g.fillStyle = `rgba(0,0,0,${.35 + r() * .5})`; g.beginPath(); g.ellipse(x, y, 6 + r() * 26, 3 + r() * 10, r() * 3, 0, TAU); g.fill(); }
    for (let i = 0; i < 9; i++) { const y = r() * h; g.fillStyle = `rgba(0,0,0,${.15 + r() * .25})`; g.fillRect(r() * w * .5, y, w * (.3 + r() * .5), 1.5 + r() * 2.5); }
    const sp = g.createPattern(KP.tile('grain'), 'repeat'); g.globalAlpha = .55; g.globalCompositeOperation = 'destination-in'; // grain eats the ink a little
    g.globalAlpha = 1; g.globalCompositeOperation = 'source-over';
    return (stamps[key] = c);
  };

  // jagged 'zap' bolt polygon from (x,y) along angle a
  KP.bolt = (x, y, len, a, w, seed) => {
    const r = KP.rng(seed), n = 4, L = [], R = [];
    for (let i = 0; i <= n; i++) {
      const u = i / n, d = len * u, side = (i % 2 ? 1 : -1) * w * (.9 + r() * .5), ww = w * (1 - u * .85) * .55;
      const px = x + Math.cos(a) * d - Math.sin(a) * side * (i && i < n ? 1 : 0), py = y + Math.sin(a) * d + Math.cos(a) * side * (i && i < n ? 1 : 0);
      L.push([px - Math.sin(a) * ww, py + Math.cos(a) * ww]); R.push([px + Math.sin(a) * ww, py - Math.cos(a) * ww]);
    }
    const tip = [x + Math.cos(a) * (len + w), y + Math.sin(a) * (len + w)];
    return L.concat([tip], R.reverse());
  };

  // ================================================================= FRAMES
  // Client rules (override the doc): max ONE 1-3 word label per frame (+ the DRAMATIZATION tag in A),
  // story told in pictures; every colour vivid and saturated.

  // ---------- Frame A: cold-open TV wall, fictional Daytime TV Doctor (s01 @ 7.5s)
  function frameA(ctx, t) {
    const T = 7.5 + (t - 1), b = KP.boil(T), cam = Math.sin(T * .3) * 6; // slow lateral truck
    ctx.fillStyle = P.plum; ctx.fillRect(0, 0, W, H);
    // back layer: wallpaper strips (parallax .3)
    ctx.save(); ctx.translate(cam * .3, 0);
    const rw = KP.rng(4);
    for (let i = 0; i < 9; i++) KP.paper(ctx, KP.jitter(KP.rr(-20 + i * 136, -40, 118, H + 80, 12, 30), 60 + i, 1.5), mix(P.plum, P.grape, .18 + rw() * .14), { shadow: false });
    ctx.restore();
    const cable = mix(P.plum, P.ink, .45);
    // ---- the other 8 sets (mid layer .6): grape cardboard, dark cobalt screens
    ctx.save(); ctx.translate(cam * .6, 0);
    const far = [
      [24, 40, 330, 280, { antenna: .5 }], [384, 76, 320, 250, {}], [736, 30, 370, 300, { antenna: .4 }],
      [928, 410, 240, 300, { panel: 'bottom' }], [934, 750, 230, 300, {}],
      [24, 1120, 330, 260, { feet: 1 }], [384, 1146, 330, 236, { panel: 'bottom', feet: 1 }], [740, 1110, 360, 280, { feet: 1 }],
    ];
    far.forEach(([x, y, w, h, o], i) => KP.tv(ctx, x, y, w, h, { seed: 100 + i * 7, boil: b, shadow: [8, 11], shadowA: .45, panel: o.panel || 'right', antenna: o.antenna, feet: o.feet, body: P.grape, screen: P.screenDim, antColor: P.ink }));
    // the sideboard the sets stand on (bottom zone: decor only)
    KP.paper(ctx, KP.jitter(KP.rr(-30, 1404, W + 60, 600, 16, 30), 120 + b, 1.5), P.coral, { shadow: [0, -10], shadowA: .35 });
    KP.paper(ctx, KP.jitter(KP.rr(-30, 1404, W + 60, 26, 10, 30), 121 + b, 1.2), P.lemon, { shadow: false });
    for (const dx of [60, 400, 740]) { KP.paper(ctx, KP.jitter(KP.rr(dx, 1470, 300, 170, 14, 20), 122 + dx + b, 1.2), mix(P.coral, P.magenta, .45), { shadow: KP.SH.far }); KP.fastener(ctx, dx + 150, 1555, 10); }
    ctx.restore();
    KP.brush(ctx, KP.curve([[930, 380], [905, 560], [915, 760]], false, 8), 9, cable, { taper: .1 });

    // ---- hero TV (action layer): turquoise set, lit lemon screen
    const hx = 34, hy = 372, hw = 872, hh = 716;
    const S = KP.tv(ctx, hx, hy, hw, hh, { seed: 7, boil: b, lit: true, body: P.turq, screen: P.lemon, panel: 'bottom', shadow: [14, 18], shadowA: .5 });
    ctx.save(); ctx.clip(KP.path(KP.rr(S.x, S.y, S.w, S.h, 20)));
    // the overexcited brain behind the puppet
    const bx = S.x + S.w * .72, by = S.y + S.h * .4, br = 112;
    const brain = KP.jitter(KP.curve([[bx - br, by + 10], [bx - br * .9, by - br * .5], [bx - br * .4, by - br * .82], [bx + br * .2, by - br * .85], [bx + br * .75, by - br * .6], [bx + br * 1.02, by - br * .05], [bx + br * .8, by + br * .5], [bx + br * .2, by + br * .62], [bx - br * .5, by + br * .58]], true, 8), 300 + b, 1.5);
    const zr = KP.rng(9);
    [[-2.5, 1], [-1.6, .9], [-.75, 1.1], [.1, 1], [.75, .85], [2.2, .7]].forEach(([a, k], i) => {
      const d = br * 1.05;
      KP.paper(ctx, KP.jitter(KP.bolt(bx + Math.cos(a) * d, by + Math.sin(a) * d * .8, 70 * k + zr() * 30, a, 18, 20 + i), 400 + i + b, 1.2), P.tomato, { shadow: KP.SH.far, mis: P.pink, misOff: [-4, 3], misA: .6 });
    });
    KP.paper(ctx, brain, P.bubble, { shadow: KP.SH.near, mis: P.tomato, misOff: [-3, 3], misA: .45 });
    [[[bx - 80, by - 40], [bx - 40, by - 70], [bx, by - 50], [bx + 30, by - 74]], [[bx + 50, by - 30], [bx + 80, by - 10], [bx + 66, by + 26]], [[bx - 90, by + 20], [bx - 50, by + 6], [bx - 30, by + 36]]].forEach(c => KP.brush(ctx, KP.curve(c, false, 8), 8, P.magenta, { taper: .6 }));
    KP.paper(ctx, KP.jitter(KP.rr(bx + 4, by + br * .5, 34, 60, 14, 10), 33 + b, 1), P.pink, { shadow: KP.SH.far });
    KP.face(ctx, bx + 8, by + 20, 110, { eyes: 'dot', mouth: 'wobble', cheek: P.magenta });
    KP.brush(ctx, [[bx - br - 30, by - 70], [bx - br - 46, by - 92]], 6, P.ink, { taper: .5 });
    KP.brush(ctx, [[bx - br - 34, by - 40], [bx - br - 58, by - 46]], 6, P.ink, { taper: .5 });
    // the puppet
    KP.puppetDoctor(ctx, S.x + S.w * .3, S.y + S.h * .52, 1.05, { boil: b, jaw: .75, stickTo: S.y + S.h + 20, shadowA: .2 });
    const mx = S.x + S.w * .3 + 135, my = S.y + S.h * .52 - 78;
    KP.brush(ctx, [[mx, my - 16], [mx + 26, my - 30]], 6, P.ink, { taper: .5 });
    KP.brush(ctx, [[mx + 6, my + 4], [mx + 36, my + 4]], 6, P.ink, { taper: .5 });
    KP.brush(ctx, [[mx, my + 24], [mx + 26, my + 38]], 6, P.ink, { taper: .5 });
    ctx.restore();
    // torn lower-third: the one label + the required DRAMATIZATION tag (cobalt slip, white print)
    KP.slip(ctx, [
      { s: 'TV DOCTOR', size: 44, ls: 2, color: P.rice },
      { s: 'DRAMATIZATION - fictional character', size: 28, weight: 700, color: P.lemon },
    ], hx + 30, S.y + S.h - 132, { rot: -.025, seed: 12, boil: b, padX: 34, padY: 18, tape: P.pink, fill: P.blue, shadowA: .3 });

    // ---- caption band
    KP.caption(ctx, KP.words('It overexcites your | *brain* cells.'), 470, 1320, { boil: b, accent: P.pink, accentMis: P.tomato, shadowA: .45 });
    // foreground (1.3): the back of the viewer's sofa
    ctx.save(); ctx.translate(cam * 1.3, 0);
    KP.paper(ctx, KP.jitter(KP.curve([[-60, 1760], [120, 1700], [330, 1710], [520, 1740], [700, 1708], [900, 1700], [1140, 1750], [1140, 1980], [-60, 1980]], true, 10), 130 + b, 2.2), P.magenta, { shadow: [0, -12], shadowA: .4, mis: P.grape, misOff: [0, -4], misA: .5 });
    KP.paper(ctx, KP.jitter(KP.curve([[-60, 1850], [300, 1820], [700, 1830], [1140, 1850], [1140, 1980], [-60, 1980]], true, 10), 131 + b, 2), mix(P.magenta, P.ink, .3), { shadow: [0, -8], shadowA: .3 });
    ctx.restore();
    KP.texture(ctx, 'grain', .1);
    KP.texture(ctx, 'speck', .28, 17, 41);
  }

  // ---------- Frame B: gut-lining cells eating glutamate (s05 @ 18.5s)
  function frameB(ctx, t) {
    const T = 18.5 + (t - 1), b = KP.boil(T), cam = Math.sin(T * .25) * 8;
    ctx.fillStyle = P.turq; ctx.fillRect(0, 0, W, H);
    // far layer (0.3): villi folds
    ctx.save(); ctx.translate(cam * .3, 0);
    const vr = KP.rng(12), vil = [[-60, 760]];
    for (let x = -40; x < W + 80; x += 120) { const hgt = 170 + vr() * 120, yb = 760 - x * .05; vil.push([x, yb - hgt * .2], [x + 30, yb - hgt], [x + 85, yb - hgt * .95], [x + 110, yb - hgt * .15]); }
    vil.push([W + 80, 1100], [-60, 1100]);
    KP.paper(ctx, KP.jitter(KP.curve(vil, true, 5), 5, 1.5), P.pink30, { shadow: KP.SH.far, mis: P.blue, misOff: [3, -2], misA: .3 });
    ctx.restore();
    // mid layer (0.6): back counter where the far cells sit
    ctx.save(); ctx.translate(cam * .6, 0);
    const cells = [[350, 852, 134, 5], [520, 846, 150, 4], [690, 862, 168, 3]];
    const backCounter = KP.curve([[-60, 870], [200, 868], [500, 852], [760, 880], [960, 940], [1140, 980], [1140, 1400], [-60, 1400]], true, 10);
    const drawCell = ([cx, cy, cw, id], o = {}) => {
      const chh = cw * 1.08;
      KP.arm(ctx, cx - cw * .38, cy - chh * .05, cw * .3, 2.3, cw * .17, P.bubble, { seed: id, boil: b });
      KP.arm(ctx, cx + cw * .38, cy - chh * .05, cw * .3, .85, cw * .17, P.bubble, { seed: id + 9, boil: b });
      return KP.cell(ctx, cx, cy - chh * .3, cw, chh, { seed: id, boil: b, fill: P.bubble, bib: P.rice, ...o });
    };
    const backOpts = [{ eyes: 'dot', mouth: 'o', look: -.06 }, { eyes: 'happy', mouth: 'smile' }, { eyes: 'dot', mouth: 'smile', look: -.08 }];
    cells.forEach((c, i) => drawCell(c, backOpts[i]));
    KP.paper(ctx, KP.jitter(backCounter, 6 + b, 1.5), P.pink, { shadow: KP.SH.near, mis: P.blue, misOff: [0, -3], misA: .3 });
    cells.forEach(([cx, cy, cw], i) => { for (const sx of [-1, 1]) KP.paper(ctx, KP.jitter(KP.egg(cx + sx * cw * .3, cy + 4, cw * .09, cw * .07, 0, 16), 90 + i * 2 + sx + b, .8), P.bubble, { shadow: KP.SH.far }); });
    ctx.restore();

    // action layer (1.0): the cell burning fuel (flame above)
    const c2 = [830, 1150, 236, 2];
    drawCell(c2, { eyes: 'happy', mouth: 'smile', look: .03 });
    const fx = 840, fy = 896;
    KP.paper(ctx, KP.jitter(KP.curve([[fx, fy - 110], [fx + 22, fy - 60], [fx + 52, fy - 30], [fx + 50, fy + 18], [fx + 22, fy + 44], [fx - 22, fy + 44], [fx - 50, fy + 14], [fx - 44, fy - 34], [fx - 26, fy - 22], [fx - 12, fy - 64]], true, 6), 81 + b, 1.4), P.gold, { shadow: KP.SH.near, mis: P.tomato, misOff: [-2.5, 2], misA: .4 });
    KP.paper(ctx, KP.jitter(KP.curve([[fx + 2, fy - 36], [fx + 22, fy + 2], [fx + 16, fy + 30], [fx - 14, fy + 30], [fx - 22, fy + 6]], true, 6), 82 + b, 1), P.lemon, { shadow: false });
    const star = (x, y, r) => KP.paper(ctx, KP.jitter([[x, y - r], [x + r * .25, y - r * .25], [x + r, y], [x + r * .25, y + r * .25], [x, y + r], [x - r * .25, y + r * .25], [x - r, y], [x - r * .25, y - r * .25]], 7, .6), P.gold, { shadow: KP.SH.far, mis: P.tomato, misOff: [-2, 2], misA: .4 });
    star(fx + 92, fy - 60, 22); star(fx - 88, fy - 16, 16); star(fx + 70, fy + 40, 12);
    // front counter strata (cut cake layers to the bottom of the frame)
    const front = (yTop, col, sd, mis) => KP.paper(ctx, KP.jitter(KP.curve([[-60, yTop + 30], [180, yTop], [460, yTop - 6], [760, yTop + 20], [1140, yTop + 90], [1140, H + 60], [-60, H + 60]], true, 10), sd + b, 1.8), col, { shadow: [0, -8], shadowA: .2, mis, misOff: [0, -3], misA: .3 });
    front(1140, P.pink, 1, P.blue);
    // the nearest cell, mid-chomp
    const cw = 330, chh = cw * 1.08, ccx = 320, ccy = 1230 - chh * .3;
    KP.arm(ctx, ccx - cw * .4, ccy + chh * .14, cw * .32, 3.6, cw * .16, P.bubble, { seed: 41, boil: b, thumb: 1 });
    KP.arm(ctx, ccx + cw * .4, ccy + chh * .14, cw * .3, -.55, cw * .16, P.bubble, { seed: 42, boil: b, thumb: 1 });
    const m = KP.cell(ctx, ccx, ccy, cw, chh, { seed: 1, boil: b, fill: P.bubble, bib: P.rice, eyes: 'happy', mouth: 'chomp', mouthDx: -.1, look: -.06 });
    front(1320, P.magenta, 2, P.grape);
    front(1520, P.bubble, 3, P.blue);
    front(1720, P.grape, 4, P.pink);
    // the Goldie stream
    const path = KP.curve([[-120, 470], [80, 520], [240, 640], [180, 820], [m.mx - 26, m.my]], false, 30);
    const at = u => { const i = Math.min(path.length - 2, Math.floor(u * (path.length - 1))), k = u * (path.length - 1) - i; const p = path[i], q = path[i + 1]; return [p[0] + (q[0] - p[0]) * k, p[1] + (q[1] - p[1]) * k, Math.atan2(q[1] - p[1], q[0] - p[0])]; };
    const flow = (T * .05) % (1 / 6);
    for (let i = 0; i < 6; i++) {
      const u = .1 + i / 6 + flow; if (u >= .97) continue;
      const [x, y, a] = at(u), hgt = 64 + u * 40;
      KP.brush(ctx, [[x - Math.cos(a) * hgt * .75, y - Math.sin(a) * hgt * .75 - 10], [x - Math.cos(a) * hgt * 1.1, y - Math.sin(a) * hgt * 1.1 - 10]], 5, P.ink, { taper: .6 });
      KP.goldie(ctx, x, y, hgt, { seed: 10 + i, boil: b, rot: Math.sin(T * 2 + i) * .15, eyes: i % 3 === 1 ? 'happy' : 'dot', mouth: i % 2 ? 'o' : 'smile', look: 3 });
    }
    // the Goldie going in: clipped to (outside the cell) + (the mouth)
    ctx.save();
    const clip = new Path2D(); clip.rect(-100, -100, W + 200, H + 200);
    clip.addPath(KP.path(KP.egg(ccx, ccy, cw / 2, chh / 2, .1, 60)));
    clip.addPath(KP.path(KP.ell(m.mx, m.my, m.mw, m.mh, 32)));
    ctx.clip(clip, 'evenodd');
    KP.goldie(ctx, m.mx - 20, m.my + 2, 112, { seed: 3, boil: b, rot: -1.0, eyes: 'happy', mouth: 'open', shadow: KP.SH.far });
    ctx.restore();
    KP.brush(ctx, [[m.mx - 60, m.my - 70], [m.mx - 80, m.my - 96]], 6, P.ink, { taper: .5 });
    KP.brush(ctx, [[m.mx - 88, m.my - 40], [m.mx - 118, m.my - 50]], 6, P.ink, { taper: .5 });
    // the one label
    KP.slip(ctx, [{ s: 'Fuel!', size: 72, ls: 1 }], 90, 250, { rot: -.05, seed: 30, boil: b, tape: P.blue, padX: 40, padY: 24 });
    KP.caption(ctx, KP.words('your intestines love | *glutamate.*'), 470, 1324, { boil: b, accent: P.gold, accentMis: P.tomato });
    // foreground torn scraps (1.3, darker)
    ctx.save(); ctx.translate(cam * 1.3, 0);
    const deep = mix(P.magenta, P.ink, .2);
    KP.paper(ctx, KP.torn(KP.curve([[1140, 1600], [960, 1700], [900, 1860], [930, 1980], [1140, 1980]], true, 8), 72 + b), deep, { shadow: [-10, -6] });
    KP.paper(ctx, KP.torn(KP.curve([[-40, 1740], [150, 1770], [230, 1880], [210, 1980], [-40, 1980]], true, 8), 73 + b), deep, { shadow: [10, -6] });
    ctx.restore();
    KP.texture(ctx, 'grain', .1);
  }

  // squiggle 'handwriting/typing' line: a wavy tapered brush stroke (no readable words)
  KP.squiggle = (ctx, x0, x1, y, seed, o = {}) => {
    const r = KP.rng(seed), pts = [], amp = o.amp ?? 6, wl = o.wl ?? 26, ph = r() * TAU;
    for (let x = x0; x <= x1; x += 4) pts.push([x, y + Math.sin((x - x0) / wl * TAU + ph) * amp * (.7 + .3 * Math.sin(x * .013 + ph))]);
    KP.brush(ctx, pts, o.w || 7, o.color || P.ink, { taper: .5 });
  };

  // ---------- Frame C: the 1968 letter + headline stamp (s06 @ 29.0s)
  function frameC(ctx, t) {
    const T = 29 + (t - 1), b = KP.boil(T);
    ctx.fillStyle = P.blue; ctx.fillRect(0, 0, W, H);
    // a hot-pink sheet underneath (far layer)
    KP.paper(ctx, KP.rot(KP.jitter(KP.rr(150, 300, 740, 990, 6, 30), 2, 1.5), 520, 800, .06), P.pink, { shadow: KP.SH.near, shadowA: .3 });
    // ---- the letter: lemon paper, squiggle lines only
    const lx = 110, ly = 240, lw = 720, lh = 1010, lc = [lx + lw / 2, ly + lh / 2], ang = -4 * Math.PI / 180;
    ctx.save(); ctx.translate(lc[0], lc[1]); ctx.rotate(ang); ctx.translate(-lc[0], -lc[1]);
    KP.paper(ctx, KP.jitter(KP.rr(lx, ly, lw, lh, 8, 30), 9 + b, 1.3), P.lemon, { shadow: KP.SH.near, shadowA: .35, mis: P.coral, misOff: [-3, 3], misA: .5 });
    KP.paper(ctx, [[lx, ly + lh * .34], [lx + lw, ly + lh * .34], [lx + lw, ly + lh * .34 + 3], [lx, ly + lh * .34 + 3]], P.brass, { shadow: false });
    const r = KP.rng(77), x0 = lx + 70, x1 = lx + lw - 70;
    KP.squiggle(ctx, x1 - 170, x1, ly + 90, 1, { amp: 5 });           // date line
    KP.squiggle(ctx, x0, x0 + 230, ly + 180, 2, { amp: 6, w: 9 });     // salutation
    let y = ly + 270;
    const lines = [1, 1, 1, 1, .55, 0, .9, 1, 1, 1, .4];
    lines.forEach((k, i) => {
      if (k === 0) { y += 36; return; }
      const ind = i === 0 || i === 6 ? 50 : 0;
      if (y > ly + 440 && y < ly + 790) { y += 54; return; } // keep the stamp's landing zone clean so the headline reads
      KP.squiggle(ctx, x0 + ind, x0 + ind + (x1 - x0 - ind) * k * (.92 + r() * .08), y, 10 + i, { amp: 5 + r() * 2, wl: 22 + r() * 8 });
      y += 54;
    });
    // anonymous signature scribble (author not named or drawn)
    KP.brush(ctx, KP.curve([[x0 + 330, y + 70], [x0 + 360, y + 40], [x0 + 380, y + 80], [x0 + 410, y + 36], [x0 + 440, y + 76], [x0 + 500, y + 52], [x0 + 560, y + 62]], false, 6), 6, P.ink, { taper: .6 });
    ctx.restore();

    // ---- pinned card: the suspects (wine, salt cellar, shaker), each with a ? cut-out
    ctx.save(); ctx.translate(250, 1180); ctx.rotate(.06);
    KP.paper(ctx, KP.jitter(KP.rr(-190, -120, 400, 250, 8, 20), 14 + b, 1.2), P.turq, { shadow: KP.SH.near, shadowA: .35 });
    KP.paper(ctx, KP.jitter(KP.curve([[-128, 98], [-128, 10], [-120, -14], [-108, -24], [-108, -60], [-98, -60], [-98, -24], [-86, -14], [-78, 10], [-78, 98]], true, 4), 15 + b, 1), P.grape, { shadow: KP.SH.far });
    KP.paper(ctx, KP.jitter(KP.rr(-124, 30, 42, 36, 4, 8), 16, .8), P.rice, { shadow: false });
    KP.paper(ctx, KP.jitter(KP.curve([[-40, 40], [40, 40], [32, 74], [10, 84], [14, 98], [-14, 98], [-10, 84], [-32, 74]], true, 4), 17 + b, 1), P.rice, { shadow: KP.SH.far });
    KP.paper(ctx, KP.jitter(KP.ell(0, 42, 40, 9, 20), 18, .6), P.pink, { shadow: false });
    KP.paper(ctx, KP.jitter(KP.rr(84, 14, 64, 84, 10, 8), 19 + b, 1), P.rice, { shadow: KP.SH.far });
    KP.paper(ctx, KP.jitter(KP.curve([[82, 20], [88, -8], [116, -20], [144, -8], [150, 20]], true, 5), 20 + b, 1), P.coral, { shadow: false });
    for (const [hx, hy] of [[104, -4], [116, -9], [128, -4]]) { ctx.fillStyle = P.ink; ctx.beginPath(); ctx.arc(hx, hy, 3.2, 0, TAU); ctx.fill(); }
    for (const [qx, qy, qr] of [[-104, -78, -.12], [0, -16, .08], [116, -48, -.05]]) {
      ctx.save(); ctx.translate(qx, qy); ctx.rotate(qr);
      ctx.font = KP.font(64, 'Young Serif', 400); ctx.textAlign = 'center';
      ctx.fillStyle = rgba(P.ink, .25); ctx.fillText('?', 4, 5); ctx.fillStyle = P.ink; ctx.fillText('?', 0, 0);
      ctx.restore();
    }
    KP.brush(ctx, KP.curve([[170, -80], [170, -150], [190, -164], [208, -150], [208, -60], [196, -50], [184, -60], [184, -130]], false, 6), 6, P.rice, { taper: .05 });
    ctx.restore();

    // ---- newspapers starting to slide in at the edges (next beat): hot pink stock, squiggle columns
    const sheet = (x, y, w, h, a, sd, col) => {
      ctx.save(); ctx.translate(x, y); ctx.rotate(a);
      KP.paper(ctx, KP.jitter(KP.rr(-w / 2, -h / 2, w, h, 4, 18), sd + b, 1.2), col, { shadow: KP.SH.near, shadowA: .35 });
      ctx.fillStyle = P.ink; ctx.fillRect(-w / 2 + 18, -h / 2 + 18, w - 36, 22);
      ctx.fillStyle = P.tomato; ctx.fillRect(-w / 2 + 18, -h / 2 + 48, w - 36, 5);
      const cwid = (w - 36 - 24) / 3;
      for (let c = 0; c < 3; c++) for (let i = 0; i < 13; i++) KP.squiggle(ctx, -w / 2 + 18 + c * (cwid + 12), -w / 2 + 18 + c * (cwid + 12) + cwid * .9, -h / 2 + 76 + i * 17, sd * 50 + c * 20 + i, { amp: 2, wl: 10, w: 4, color: P.rice });
      ctx.restore();
    };
    sheet(1040, 1120, 280, 320, .22, 61, P.coral);
    sheet(-40, 560, 260, 300, -.3, 62, P.grape);
    sheet(820, 1660, 300, 340, -.12, 63, P.coral);
    for (const [x, y, k] of [[880, 1060, 1], [890, 1140, 1], [130, 520, -1], [110, 610, -1]]) KP.brush(ctx, [[x, y], [x - k * 50, y + 6 * k]], 6, P.rice, { taper: .6 });

    // ---- the stamp impression: the one label (rotated -8deg)
    const stw = 800, sth = 310, scx = 470, scy = 860;
    const st = KP.stamp('crs', [String.fromCharCode(0x201C) + 'CHINESE-RESTAURANT', 'SYNDROME' + String.fromCharCode(0x201D)], stw, sth, { size: 66, ls: 1.5, seed: 55 });
    ctx.save(); ctx.translate(scx, scy); ctx.rotate(-8 * Math.PI / 180);
    ctx.drawImage(st, -stw / 2, -sth / 2);
    ctx.restore();
    // impact flecks
    const fr = KP.rng(88), sa = -8 * Math.PI / 180;
    [[-1.05, .2], [-1, -.6], [-.7, 1.15], [-.2, -1.2], [.3, 1.2], [.75, -1.15], [1.04, .5], [1.06, -.3], [.1, 1.25]].forEach(([u, v], i) => {
      const lx0 = u * stw / 2, ly0 = v * sth / 2, x = scx + lx0 * Math.cos(sa) - ly0 * Math.sin(sa), y = scy + lx0 * Math.sin(sa) + ly0 * Math.cos(sa);
      const dx = x - scx, dy = (y - scy) * 2.2, dl = Math.hypot(dx, dy), nx = dx / dl, ny = dy / dl, s = 7 + fr() * 9;
      ctx.save(); ctx.translate(x + nx * 34, y + ny * 26); ctx.rotate(fr() * 3);
      KP.paper(ctx, KP.jitter([[-s, -s * .6], [s, -s * .4], [s * .7, s * .7], [-s * .8, s * .5]], i, .8), [P.tomato, P.rice, P.lemon][i % 3], { shadow: KP.SH.far });
      ctx.restore();
      KP.brush(ctx, [[x + nx * 28, y + ny * 22], [x + nx * 4, y + ny * 2]], 5, P.ink, { taper: .6 });
    });
    // ---- the rubber stamp lifting out the top of frame
    const lift = pts => KP.rot(pts, 0, 0, -8 * Math.PI / 180);
    const stampBody = (dx, dy) => {
      const ox = 700 + dx, oy = 170 + dy;
      return [
        KP.jitter(KP.move(lift(KP.rr(-230, -50, 460, 100, 14, 18)), ox, oy), 91 + b, 1.5),
        KP.jitter(KP.move(lift(KP.rr(-64, -260, 128, 230, 36, 18)), ox, oy), 92 + b, 1.5),
        KP.jitter(KP.move(lift(KP.egg(0, -290, 130, 90, 0, 40)), ox, oy), 93 + b, 1.5),
      ];
    };
    stampBody(34, 64).forEach(p => { ctx.fillStyle = rgba(P.ink, .25); ctx.fill(KP.path(p)); });
    const [blk, neck, knob] = stampBody(0, 0);
    KP.paper(ctx, knob, P.ink, { shadow: false }); KP.paper(ctx, neck, P.ink, { shadow: false }); KP.paper(ctx, blk, P.ink, { shadow: false });
    KP.paper(ctx, KP.jitter(KP.move(lift(KP.rr(-222, 40, 444, 20, 6, 18)), 700, 170), 94 + b, 1), P.tomato, { shadow: false });
    KP.paper(ctx, KP.jitter(KP.move(lift(KP.rr(-196, -20, 392, 10, 4, 18)), 700, 170), 95 + b, 1), P.grape, { shadow: false });
    for (const [dx, dy] of [[-262, 60], [-284, 96], [262, 20], [284, 56]]) KP.brush(ctx, [[700 + dx, 170 + dy + 56], [700 + dx, 170 + dy]], 6, P.rice, { taper: .6 });

    KP.caption(ctx, KP.words('printed it under a | *catchy* headline'), 470, 1330, { boil: b, accent: P.tomato, onInk: P.rice });
    KP.texture(ctx, 'news', .07);
    KP.texture(ctx, 'grain', .12, 101, 57);
  }

  window.STYLEFRAMES = {
    size: [W, H],
    fonts: [
      { family: 'Young Serif', url: '/assets/YoungSerif.ttf', weight: '400' },
      { family: 'Nunito', url: '/assets/Nunito.ttf', weight: '200 1000' },
      { family: 'Courier Prime', url: '/assets/CourierPrime-Regular.ttf', weight: '400' },
      { family: 'Courier Prime', url: '/assets/CourierPrime-Bold.ttf', weight: '700' },
    ],
    frames: { A: frameA, B: frameB, C: frameC },
  };
})();
