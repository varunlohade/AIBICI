// KZ — tiny flat-vector animation kit in the spirit of Kurzgesagt.
// Everything is deterministic: a frame is a pure function of (scene, t).
// Canvas is always 1080x1920 (vertical 9:16 Reel). Never use Math.random() or Date in scenes; use KZ.rng(seed).
(function () {
  const W = 1080, H = 1920;
  const KZ = { W, H, scenes: {} };

  // ---------- palette (saturated flat colors on deep navy) ----------
  KZ.C = {
    night: '#0b1736', navy: '#14254f', deep: '#1c2f63', indigo: '#2b3f86', violet: '#5a3d9a',
    sky: '#3d7bd9', cyan: '#3fd0e8', teal: '#1fb5a6', mint: '#7be3b8', green: '#5cc96b',
    yellow: '#ffd23f', gold: '#ffb627', orange: '#ff8a3d', red: '#ff4d5a', pink: '#ff7eb6',
    magenta: '#d94fd1', cream: '#fff4dc', white: '#ffffff', ink: '#0a1128', grey: '#8a95b8',
    skin: '#ffc9a3', skin2: '#e8a77f', skin3: '#a86b4a', brown: '#8a5a3b', paper: '#f4ecd8',
  };

  // ---------- math ----------
  KZ.clamp = (x, a = 0, b = 1) => Math.max(a, Math.min(b, x));
  KZ.lerp = (a, b, t) => a + (b - a) * t;
  // progress of t through [a, b], clamped 0..1
  KZ.prog = (t, a, b) => KZ.clamp((t - a) / (b - a));
  KZ.ease = {
    linear: x => x,
    inOut: x => x < .5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2,
    out: x => 1 - Math.pow(1 - x, 3),
    in: x => x * x * x,
    outBack: x => { const c1 = 1.70158, c3 = c1 + 1; return 1 + c3 * Math.pow(x - 1, 3) + c1 * Math.pow(x - 1, 2); },
    outElastic: x => x === 0 ? 0 : x === 1 ? 1 : Math.pow(2, -10 * x) * Math.sin((x * 10 - .75) * (2 * Math.PI) / 3) + 1,
  };
  // eased progress: KZ.ep(t, a, b, 'outBack')
  KZ.ep = (t, a, b, e = 'inOut') => KZ.ease[e](KZ.prog(t, a, b));
  // fade in at [a, a+d], fade out at [b-d, b]
  KZ.window = (t, a, b, d = 0.4) => Math.min(KZ.prog(t, a, a + d), 1 - KZ.prog(t, b - d, b));
  KZ.rng = seed => { let s = seed >>> 0 || 1; return () => ((s = Math.imul(s ^ (s >>> 15), 1 | s), s = s + Math.imul(s ^ (s >>> 7), 61 | s) ^ s, ((s ^ (s >>> 14)) >>> 0) / 4294967296)); };
  KZ.wobble = (t, seed = 0, amp = 1, speed = 1) => amp * (Math.sin(t * 1.3 * speed + seed) * .6 + Math.sin(t * 2.1 * speed + seed * 1.7) * .4);
  KZ.mixColor = (a, b, t) => {
    const p = h => [1, 3, 5].map(i => parseInt(h.slice(i, i + 2), 16));
    const A = p(a), B = p(b);
    return '#' + A.map((v, i) => Math.round(KZ.lerp(v, B[i], t)).toString(16).padStart(2, '0')).join('');
  };
  KZ.alpha = (hex, a) => { const p = [1, 3, 5].map(i => parseInt(hex.slice(i, i + 2), 16)); return `rgba(${p[0]},${p[1]},${p[2]},${a})`; };

  // ---------- cues ----------
  // cues = [{voice,text,start,end}, ...] for the current scene (narration line timings in seconds).
  KZ.cue = (cues, i) => cues[Math.max(0, Math.min(cues.length - 1, i))];

  // ---------- drawing primitives ----------
  KZ.circle = (ctx, x, y, r, fill, stroke, lw = 4) => {
    ctx.beginPath(); ctx.arc(x, y, Math.max(0, r), 0, Math.PI * 2);
    if (fill) { ctx.fillStyle = fill; ctx.fill(); }
    if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = lw; ctx.stroke(); }
  };
  KZ.rrect = (ctx, x, y, w, h, r, fill, stroke, lw = 4) => {
    ctx.beginPath(); ctx.roundRect(x, y, w, h, r);
    if (fill) { ctx.fillStyle = fill; ctx.fill(); }
    if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = lw; ctx.stroke(); }
  };
  KZ.poly = (ctx, pts, fill, stroke, lw = 4) => {
    ctx.beginPath(); pts.forEach(([x, y], i) => i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)); ctx.closePath();
    if (fill) { ctx.fillStyle = fill; ctx.fill(); }
    if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = lw; ctx.stroke(); }
  };
  KZ.line = (ctx, x1, y1, x2, y2, color, lw = 4, cap = 'round') => {
    ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2);
    ctx.strokeStyle = color; ctx.lineWidth = lw; ctx.lineCap = cap; ctx.stroke();
  };
  // organic wobbly blob (cells, clouds, food). n lobes, deterministic per seed.
  KZ.blob = (ctx, x, y, r, t, seed, fill, opts = {}) => {
    const n = opts.points || 10, amp = opts.amp ?? 0.08, sp = opts.speed ?? 1;
    ctx.beginPath();
    const pts = [];
    for (let i = 0; i < n; i++) {
      const a = i / n * Math.PI * 2;
      const rr = r * (1 + amp * Math.sin(a * 3 + seed + t * sp) + amp * .6 * Math.sin(a * 5 - seed * 2 + t * 1.3 * sp));
      pts.push([x + Math.cos(a) * rr, y + Math.sin(a) * rr]);
    }
    for (let i = 0; i < n; i++) {
      const p0 = pts[i], p1 = pts[(i + 1) % n];
      const mx = (p0[0] + p1[0]) / 2, my = (p0[1] + p1[1]) / 2;
      if (i === 0) ctx.moveTo(mx, my); else ctx.quadraticCurveTo(p0[0], p0[1], mx, my);
    }
    const p0 = pts[0], p1 = pts[1];
    ctx.quadraticCurveTo(p0[0], p0[1], (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2);
    ctx.closePath();
    if (fill) { ctx.fillStyle = fill; ctx.fill(); }
    if (opts.stroke) { ctx.strokeStyle = opts.stroke; ctx.lineWidth = opts.lw || 4; ctx.stroke(); }
  };
  // soft radial glow
  KZ.glow = (ctx, x, y, r, color, a = 0.6) => {
    const g = ctx.createRadialGradient(x, y, 0, x, y, r);
    g.addColorStop(0, KZ.alpha(color, a)); g.addColorStop(1, KZ.alpha(color, 0));
    ctx.fillStyle = g; ctx.fillRect(x - r, y - r, r * 2, r * 2);
  };
  // vertical gradient background
  KZ.bg = (ctx, top = KZ.C.navy, bottom = KZ.C.night) => {
    const g = ctx.createLinearGradient(0, 0, 0, H);
    g.addColorStop(0, top); g.addColorStop(1, bottom);
    ctx.fillStyle = g; ctx.fillRect(0, 0, W, H);
  };
  // twinkling starfield with optional parallax drift (px/sec)
  KZ.stars = (ctx, t, seed = 7, count = 160, drift = 6, alpha = 1) => {
    const r = KZ.rng(seed);
    for (let i = 0; i < count; i++) {
      const x0 = r() * W, y = r() * H, s = r() * 2.4 + .6, ph = r() * 6.28, depth = r();
      const x = ((x0 - t * drift * (0.3 + depth)) % W + W) % W;
      const tw = .55 + .45 * Math.sin(t * (1 + depth * 2) + ph);
      ctx.globalAlpha = alpha * tw * (.4 + depth * .6);
      KZ.circle(ctx, x, y, s, '#ffffff');
    }
    ctx.globalAlpha = 1;
  };
  // floating dust particles (foreground depth)
  KZ.dust = (ctx, t, seed = 3, count = 40, color = '#ffffff', alpha = .25) => {
    const r = KZ.rng(seed);
    for (let i = 0; i < count; i++) {
      const x = ((r() * W + t * (10 + r() * 20)) % W + W) % W;
      const y = ((r() * H - t * (5 + r() * 12)) % H + H) % H;
      ctx.globalAlpha = alpha * (.5 + .5 * Math.sin(t + i));
      KZ.circle(ctx, x, y, 2 + r() * 4, color);
    }
    ctx.globalAlpha = 1;
  };

  // ---------- text ----------
  // KZ.text(ctx, 'Hello', x, y, {size, color, weight, align, font, alpha, shadow, maxW})
  KZ.text = (ctx, s, x, y, o = {}) => {
    ctx.save();
    ctx.globalAlpha *= o.alpha ?? 1;
    ctx.font = `${o.weight || 600} ${o.size || 64}px ${o.font || 'Fredoka'}`;
    ctx.textAlign = o.align || 'center'; ctx.textBaseline = o.baseline || 'middle';
    if (o.letterSpacing) ctx.letterSpacing = o.letterSpacing + 'px';
    if (o.shadow) { ctx.shadowColor = o.shadow; ctx.shadowBlur = o.shadowBlur ?? 24; }
    if (o.stroke) { ctx.strokeStyle = o.stroke; ctx.lineWidth = o.strokeW || 8; ctx.lineJoin = 'round'; ctx.strokeText(s, x, y, o.maxW); }
    ctx.fillStyle = o.color || KZ.C.white; ctx.fillText(s, x, y, o.maxW);
    ctx.restore();
  };
  // typewriter reveal: shows first n chars by progress p (0..1)
  KZ.typed = (s, p) => s.slice(0, Math.floor(s.length * KZ.clamp(p)));
  // pop-in scale for an element appearing at time a (overshoot)
  KZ.pop = (t, a, d = 0.45) => KZ.ease.outBack(KZ.prog(t, a, a + d));

  // ---------- transforms ----------
  KZ.withT = (ctx, x, y, s, rot, fn) => { ctx.save(); ctx.translate(x, y); if (rot) ctx.rotate(rot); if (s !== 1) ctx.scale(s, s); fn(); ctx.restore(); };
  // global camera: zoom s around (cx, cy)
  KZ.camera = (ctx, cx, cy, s, fn) => { ctx.save(); ctx.translate(W / 2, H / 2); ctx.scale(s, s); ctx.translate(-cx, -cy); fn(); ctx.restore(); };
  // vignette overlay — call last
  KZ.vignette = (ctx, a = .45) => {
    const g = ctx.createRadialGradient(W / 2, H / 2, H * .35, W / 2, H / 2, H * .95);
    g.addColorStop(0, 'rgba(0,0,0,0)'); g.addColorStop(1, `rgba(5,8,25,${a})`);
    ctx.fillStyle = g; ctx.fillRect(0, 0, W, H);
  };
  // fade to/from black: call with alpha
  KZ.fade = (ctx, a, color = '#05081a') => { if (a <= 0) return; ctx.globalAlpha = KZ.clamp(a); ctx.fillStyle = color; ctx.fillRect(0, 0, W, H); ctx.globalAlpha = 1; };

  // ---------- reusable characters & icons ----------
  // Kurzgesagt-ish person: rounded body, big round head, dot eyes. Facing front.
  // o: {skin, shirt, hair, s (scale), mood:'happy'|'neutral'|'sad'|'shock', blink(t), armUp(0..1), talk(0..1)}
  KZ.person = (ctx, x, y, t, o = {}) => {
    const s = o.s || 1, skin = o.skin || KZ.C.skin, shirt = o.shirt || KZ.C.sky;
    ctx.save(); ctx.translate(x, y); ctx.scale(s, s);
    const bob = Math.sin(t * 2 + (o.seed || 0)) * 3;
    // body
    KZ.rrect(ctx, -70, -40 + bob, 140, 190, 60, shirt);
    // arms
    const au = o.armUp || 0;
    ctx.lineCap = 'round';
    KZ.line(ctx, -62, 10 + bob, -95, 110 - au * 170 + bob, shirt, 34);
    KZ.line(ctx, 62, 10 + bob, 95, 110 + bob, shirt, 34);
    KZ.circle(ctx, -95, 110 - au * 170 + bob, 18, skin);
    KZ.circle(ctx, 95, 110 + bob, 18, skin);
    // head
    const hy = -120 + bob;
    KZ.circle(ctx, 0, hy, 82, skin);
    if (o.hair) { ctx.beginPath(); ctx.arc(0, hy - 8, 84, Math.PI * 1.05, Math.PI * 1.95); ctx.fillStyle = o.hair; ctx.fill(); }
    // eyes
    const bl = (Math.sin(t * 1.7 + (o.seed || 0) * 3) > .985) ? .15 : 1;
    const ey = hy - 4, er = o.mood === 'shock' ? 11 : 9;
    ctx.fillStyle = KZ.C.ink;
    ctx.beginPath(); ctx.ellipse(-28, ey, er, er * bl, 0, 0, 7); ctx.fill();
    ctx.beginPath(); ctx.ellipse(28, ey, er, er * bl, 0, 0, 7); ctx.fill();
    // mouth
    const talk = o.talk ? (0.5 + 0.5 * Math.sin(t * 22)) * o.talk : 0;
    ctx.strokeStyle = KZ.C.ink; ctx.lineWidth = 6; ctx.lineCap = 'round'; ctx.beginPath();
    if (o.mood === 'shock' || talk > .05) { ctx.ellipse(0, hy + 34, 12, 6 + talk * 12 + (o.mood === 'shock' ? 8 : 0), 0, 0, 7); ctx.fillStyle = KZ.C.ink; ctx.fill(); }
    else if (o.mood === 'sad') { ctx.arc(0, hy + 52, 20, Math.PI * 1.2, Math.PI * 1.8); ctx.stroke(); }
    else if (o.mood === 'neutral') { ctx.moveTo(-16, hy + 34); ctx.lineTo(16, hy + 34); ctx.stroke(); }
    else { ctx.arc(0, hy + 20, 22, Math.PI * .2, Math.PI * .8); ctx.stroke(); }
    ctx.restore();
  };

  // Glutamate molecule as a cute stylized ball-and-stick: carbon chain with COO- ends + NH3+.
  // o: {s, t, glow, face}
  KZ.glutamate = (ctx, x, y, t, o = {}) => {
    const s = o.s || 1;
    ctx.save(); ctx.translate(x, y); ctx.scale(s, s); ctx.rotate(o.rot || 0);
    const atoms = [
      [-150, 0, 'O'], [-110, -35, 'C'], [-150, -75, 'O'], [-55, -20, 'C'], [0, 10, 'C'], [55, -20, 'C'], [55, -85, 'N'], [110, 10, 'C'], [150, -25, 'O'], [140, 60, 'O'],
    ];
    const bonds = [[0, 1], [1, 2], [1, 3], [3, 4], [4, 5], [5, 6], [5, 7], [7, 8], [7, 9]];
    if (o.glow) KZ.glow(ctx, 0, -10, 260, KZ.C.yellow, .35 * o.glow);
    bonds.forEach(([a, b]) => KZ.line(ctx, atoms[a][0], atoms[a][1], atoms[b][0], atoms[b][1], KZ.C.cream, 12));
    const col = { C: KZ.C.grey, O: KZ.C.red, N: KZ.C.sky };
    const rad = { C: 26, O: 22, N: 24 };
    atoms.forEach(([ax, ay, k], i) => {
      const w = Math.sin(t * 2 + i) * 2;
      KZ.circle(ctx, ax, ay + w, rad[k], col[k]);
      KZ.circle(ctx, ax - rad[k] * .3, ay + w - rad[k] * .3, rad[k] * .3, 'rgba(255,255,255,.35)');
    });
    if (o.face) { // little face on the central carbon
      KZ.circle(ctx, -8, 6, 4, KZ.C.ink); KZ.circle(ctx, 8, 6, 4, KZ.C.ink);
      ctx.beginPath(); ctx.arc(0, 14, 7, .2 * Math.PI, .8 * Math.PI); ctx.strokeStyle = KZ.C.ink; ctx.lineWidth = 3; ctx.stroke();
    }
    ctx.restore();
  };

  // Shaker (salt/MSG). label text on body.
  KZ.shaker = (ctx, x, y, s, label, body = KZ.C.cream, cap = KZ.C.grey) => {
    ctx.save(); ctx.translate(x, y); ctx.scale(s, s);
    KZ.rrect(ctx, -60, -60, 120, 170, 26, body);
    KZ.rrect(ctx, -60, -60, 120, 170, 26, null, 'rgba(0,0,0,.12)', 6);
    KZ.rrect(ctx, -54, -110, 108, 60, [30, 30, 8, 8], cap);
    [-25, 0, 25].forEach(dx => KZ.circle(ctx, dx, -92, 5, KZ.C.ink));
    KZ.text(ctx, label, 0, 25, { size: 44, color: KZ.C.ink, weight: 700 });
    ctx.restore();
  };

  // chapter/lower-third label used between sections
  KZ.chapter = (ctx, t, a, b, label) => {
    const w = KZ.window(t, a, b, .5); if (w <= 0) return;
    const x = KZ.lerp(-400, 80, KZ.ease.out(KZ.prog(t, a, a + .6)));
    ctx.globalAlpha = w;
    KZ.rrect(ctx, x, 70, 18, 70, 9, KZ.C.yellow);
    KZ.text(ctx, label, x + 40, 106, { size: 46, align: 'left', weight: 600 });
    ctx.globalAlpha = 1;
  };

  // ---------- scene registry ----------
  // KZ.scene(id, { draw(ctx, t, cues, D), sfx(cues, D) => [{t, name, gain?}] })
  KZ.scene = (id, def) => { KZ.scenes[id] = def; };

  window.KZ = KZ;
})();
