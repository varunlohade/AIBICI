// "Abyss cross-section" style frames + the AB animation kit.
// Boesmansgat, 8 Jan 2005. Dark, tense, minimal. All helpers are deterministic (seeded RNG, no Math.random/Date).
//   node design/render_styleframes.js design/frames/abyss/frames.js A,B,C
(function () {
  const W = 1080, H = 1920;
  const AB = { W, H };

  // ---------------------------------------------------------------- palette
  AB.C = {
    sky0: '#05090d', sky1: '#0c1b22',            // night air above the pool
    rock: '#07080a', rock2: '#0d0f12',            // section-cut rock
    strata: 'rgba(150,175,185,0.06)',
    teal: '#2f8f8e', blue: '#123f5e', abyss: '#03070b',
    cut: '120,205,205',                           // rgb of the section cut line
    amber: '#ffb347', amberHot: '#ffe2a8',         // torch (the warm accent)
    red: '#ff3b2f',                                // danger (sparingly)
    pale: '#e9f1f0', mute: 'rgba(200,225,225,0.55)',
    cold: '200,230,255',                           // cold marker light
  };
  // water colour stops (depth in metres -> colour)
  AB.WATER = [[0, '#3a9a93'], [10, '#226f78'], [35, '#13465e'], [70, '#0b2840'], [120, '#071a2c'], [180, '#040e1a'], [240, '#03080f'], [300, '#010305']];
  AB.FONTS = [
    { family: 'Barlow Condensed', url: '/assets/BarlowCondensed-Light.ttf', weight: '300' },
    { family: 'Barlow Condensed', url: '/assets/BarlowCondensed-Medium.ttf', weight: '500' },
    { family: 'Barlow Condensed', url: '/assets/BarlowCondensed-SemiBold.ttf', weight: '600' },
    { family: 'JetBrains Mono', url: '/assets/JetBrainsMono.ttf', weight: '100 800' },
    { family: 'IBM Plex Sans', url: '/assets/IBMPlexSans.ttf', weight: '100 700' },
  ];

  // ---------------------------------------------------------------- math
  AB.rng = seed => { let s = seed >>> 0 || 1; return () => ((s = Math.imul(s ^ (s >>> 15), 1 | s), s = s + Math.imul(s ^ (s >>> 7), 61 | s) ^ s, ((s ^ (s >>> 14)) >>> 0) / 4294967296)); };
  AB.clamp = (x, a = 0, b = 1) => Math.max(a, Math.min(b, x));
  AB.lerp = (a, b, t) => a + (b - a) * t;
  AB.prog = (t, a, b) => AB.clamp((t - a) / (b - a));
  AB.smooth = x => x * x * (3 - 2 * x);
  AB.ease = { inOut: x => x < .5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2, out: x => 1 - Math.pow(1 - x, 3), in: x => x * x * x };
  AB.mod = (a, n) => ((a % n) + n) % n;
  // piecewise smooth interpolation through [[k, v], ...]
  AB.curve = (pts, k) => {
    if (k <= pts[0][0]) return pts[0][1];
    for (let i = 1; i < pts.length; i++) if (k <= pts[i][0]) {
      const [k0, v0] = pts[i - 1], [k1, v1] = pts[i], u = AB.smooth((k - k0) / (k1 - k0)); return v0 + (v1 - v0) * u;
    }
    return pts[pts.length - 1][1];
  };
  // 1-D value-ish noise from summed sines (cheap, smooth, deterministic)
  AB.n1 = (x, seed = 0) => Math.sin(x * 1.0 + seed) * .5 + Math.sin(x * 2.17 + seed * 1.9) * .28 + Math.sin(x * 4.63 + seed * 3.1) * .14 + Math.sin(x * 9.71 + seed * 5.3) * .08;
  const hexRgb = h => [1, 3, 5].map(i => parseInt(h.slice(i, i + 2), 16));
  AB.rgba = (hex, a) => { const p = hexRgb(hex); return `rgba(${p[0]},${p[1]},${p[2]},${a})`; };
  AB.waterAt = d => { // interpolated water colour at depth d -> [r,g,b]
    const S = AB.WATER; if (d <= S[0][0]) return hexRgb(S[0][1]);
    for (let i = 1; i < S.length; i++) if (d <= S[i][0]) { const u = (d - S[i - 1][0]) / (S[i][0] - S[i - 1][0]), a = hexRgb(S[i - 1][1]), b = hexRgb(S[i][1]); return a.map((v, j) => v + (b[j] - v) * u); }
    return hexRgb(S[S.length - 1][1]);
  };

  // ---------------------------------------------------------------- camera
  // A camera maps the cave (metres: x right of the shaft axis, d = depth below the pool surface) to pixels.
  // cam = { cx, cy, dRef, s }: the shaft axis sits at screen x=cx, depth dRef sits at screen y=cy, s = px per metre.
  AB.cam = (cx, cy, dRef, s) => ({ cx, cy, dRef, s, X: m => cx + m * s, Y: d => cy + (d - dRef) * s, D: y => dRef + (y - cy) / s, M: x => (x - cx) / s });

  // ---------------------------------------------------------------- cave geometry (stylised Boesmansgat)
  // half-widths of the water body (metres) to the left / right of the axis, by depth.
  AB.GROUND = -14;           // land level above the pool
  AB.PROFILE_L = [[-14, 50], [-6, 33], [0, 21], [8, 15], [20, 11], [30, 7.5], [42, 9], [52, 17], [66, 40], [85, 62], [112, 78], [128, 60], [146, 86], [175, 92], [210, 94], [240, 90], [262, 84], [290, 80]];
  AB.PROFILE_R = [[-14, 58], [-6, 41], [0, 25], [8, 18], [20, 13], [30, 8.5], [42, 11], [52, 24], [66, 50], [84, 80], [120, 104], [160, 118], [186, 106], [212, 126], [240, 120], [262, 112], [290, 108]];
  AB.DRIFT = [[40, 0], [120, 12], [270, 22]];   // the chamber leans right as it deepens
  AB.wallL = d => AB.curve(AB.PROFILE_L, d) - AB.curve(AB.DRIFT, d) + (d > 0 ? (2.2 + d * .02) * AB.n1(d * .11, 1.3) + .8 * AB.n1(d * .7, 4.1) + .25 * AB.n1(d * 3.1, 6) : 0);
  AB.wallR = d => AB.curve(AB.PROFILE_R, d) + AB.curve(AB.DRIFT, d) + (d > 0 ? (2.6 + d * .025) * AB.n1(d * .09, 7.7) + .9 * AB.n1(d * .8, 2.2) + .25 * AB.n1(d * 2.7, 1) : 0);
  // floor depth by x (metres): a rubble slope that falls away to the right, max ~283 m
  AB.floorD = x => 268 + x * .07 + 3.2 * AB.n1(x * .09, 3) + 1.3 * AB.n1(x * .45, 9) + 3 * Math.exp(-Math.pow((x + 30) / 9, 2)) * -1;
  AB.FLOOR_TOP = 262;        // depth at which the walls meet the floor region

  // polygon (in px) of the flooded volume: down the left wall, across the floor, up the right wall
  AB.waterPoly = (cam, dTop = 0) => {
    const pts = [], step = Math.max(.35, 3 / cam.s);
    let dl = dTop; while (dl < 300 && dl < AB.floorD(-AB.wallL(dl))) dl += step;   // where each wall meets the floor
    let dr = dTop; while (dr < 300 && dr < AB.floorD(AB.wallR(dr))) dr += step;
    const xl = -AB.wallL(dl), xr = AB.wallR(dr);
    for (let d = dTop; d < dl; d += step) pts.push([cam.X(-AB.wallL(d)), cam.Y(d)]);
    for (let x = xl; x <= xr; x += step) pts.push([cam.X(x), cam.Y(AB.floorD(x))]);
    for (let d = dr; d >= dTop; d -= step) pts.push([cam.X(AB.wallR(d)), cam.Y(d)]);
    return pts;
  };
  const pathPts = (ctx, pts, close = true) => { ctx.beginPath(); pts.forEach(([x, y], i) => i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)); if (close) ctx.closePath(); };
  AB.pathPts = pathPts;

  AB.waterGradient = (ctx, cam) => {
    const y0 = cam.Y(0), y1 = cam.Y(300), g = ctx.createLinearGradient(0, y0, 0, y1);
    for (const [d, c] of AB.WATER) g.addColorStop(AB.clamp(d / 300), c);
    return g;
  };

  // Full background: air + land above the pool, section-cut rock, the water body, receding wall layers, cut line.
  AB.cave = (ctx, cam, t = 0, o = {}) => {
    const s = cam.s;
    // rock everywhere
    ctx.fillStyle = AB.C.rock; ctx.fillRect(0, 0, W, H);
    // strata hatching in the rock (section-diagram texture)
    ctx.save(); ctx.strokeStyle = AB.C.strata; ctx.lineWidth = Math.max(1, s * .12);
    const dA = Math.floor(cam.D(0) / 5) * 5, dB = cam.D(H);
    for (let d = dA; d < dB; d += 5) {
      ctx.beginPath();
      for (let x = 0; x <= W; x += 12) { const m = cam.M(x); const y = cam.Y(d + 1.4 * AB.n1(m * .05 + d * .3, d)); x ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }
      ctx.stroke();
    }
    ctx.restore();
    // sky / air above ground level and the crater air down to the pool surface
    const yG = cam.Y(AB.GROUND);
    if (yG > 0) {
      const g = ctx.createLinearGradient(0, 0, 0, Math.max(1, cam.Y(0)));
      g.addColorStop(0, AB.C.sky0); g.addColorStop(1, AB.C.sky1);
      ctx.fillStyle = g;
      ctx.beginPath(); ctx.moveTo(0, 0); ctx.lineTo(W, 0); ctx.lineTo(W, yG);
      for (let d = AB.GROUND; d <= 0; d += .5) ctx.lineTo(cam.X(AB.wallR(d) + (d - AB.GROUND) * -.15), cam.Y(d));
      for (let d = 0; d >= AB.GROUND; d -= .5) ctx.lineTo(cam.X(-AB.wallL(d)), cam.Y(d));
      ctx.lineTo(0, yG); ctx.closePath(); ctx.fill();
      // land silhouette: a slightly lighter crust line + scrub tufts
      ctx.save(); ctx.strokeStyle = 'rgba(140,175,175,0.18)'; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(0, yG); ctx.lineTo(cam.X(-AB.wallL(AB.GROUND)), yG); ctx.moveTo(cam.X(AB.wallR(AB.GROUND)), yG); ctx.lineTo(W, yG); ctx.stroke();
      const r = AB.rng(77); ctx.fillStyle = AB.C.rock;
      for (let i = 0; i < 26; i++) {
        let x = r() * W; const m = cam.M(x); if (m > -AB.wallL(AB.GROUND) - 2 && m < AB.wallR(AB.GROUND) + 2) continue;
        const h = 4 + r() * 12, w = 6 + r() * 16; ctx.beginPath(); ctx.ellipse(x, yG, w, h, 0, Math.PI, 0); ctx.fill();
      }
      ctx.restore();
    }
    // the water body
    const poly = AB.waterPoly(cam, 0);
    ctx.save(); pathPts(ctx, poly); ctx.fillStyle = AB.waterGradient(ctx, cam); ctx.fill(); ctx.clip();
    // receding wall layers (far = closer to water colour, near = darker) give the chamber depth
    const layers = o.layers || [[.16, 22, .043, 1.1], [.30, 13, .071, 2.7], [.46, 6.5, .12, 5.3]];
    for (const [dark, amp, freq, ph] of layers) {
      const step = Math.max(.35, 3 / s), inset = d => {
        const k = AB.clamp((d - 45) / 40);              // layers only open up in the big chamber
        const b = .5 + .5 * Math.sin(d * freq + ph); return k * amp * Math.pow(b, 2.2) * (1 + .3 * AB.n1(d * .3, ph));
      };
      for (const side of [-1, 1]) {
        ctx.beginPath(); ctx.moveTo(side < 0 ? -50 : W + 50, cam.Y(0));
        for (let d = 0; d <= AB.FLOOR_TOP + 30; d += step) {
          const wall = side < 0 ? -AB.wallL(Math.min(d, AB.FLOOR_TOP)) + inset(d + 9 * ph) : AB.wallR(Math.min(d, AB.FLOOR_TOP)) - inset(d + 13 * ph);
          ctx.lineTo(cam.X(wall), cam.Y(d));
        }
        ctx.lineTo(side < 0 ? -50 : W + 50, cam.Y(AB.FLOOR_TOP + 30)); ctx.closePath();
        ctx.fillStyle = `rgba(1,3,6,${dark})`; ctx.fill();
      }
    }
    // rubble mounds on the floor (two layers)
    for (const [dark, lift, seed] of [[.35, 3.5, 5], [.7, 0, 11]]) {
      const step = Math.max(.35, 3 / s), x0 = -AB.wallL(AB.FLOOR_TOP) - 5, x1 = AB.wallR(AB.FLOOR_TOP) + 5;
      ctx.beginPath(); ctx.moveTo(cam.X(x0), cam.Y(300));
      for (let x = x0; x <= x1; x += step) ctx.lineTo(cam.X(x), cam.Y(AB.floorD(x) - lift - 2.2 * Math.abs(AB.n1(x * .21, seed))));
      ctx.lineTo(cam.X(x1), cam.Y(300)); ctx.closePath(); ctx.fillStyle = `rgba(1,2,4,${dark})`; ctx.fill();
    }
    ctx.restore();
    // section cut line along the rock edge, fading with depth
    ctx.save();
    const g = ctx.createLinearGradient(0, cam.Y(0), 0, cam.Y(290));
    g.addColorStop(0, `rgba(${AB.C.cut},0.55)`); g.addColorStop(.25, `rgba(${AB.C.cut},0.22)`); g.addColorStop(1, `rgba(${AB.C.cut},0.07)`);
    ctx.strokeStyle = g; ctx.lineWidth = o.cutWidth || 2; pathPts(ctx, poly, false); ctx.stroke();
    ctx.restore();
    return poly;
  };

  // Pool surface: bright meniscus line + a shimmer
  AB.surface = (ctx, cam, t = 0) => {
    const y = cam.Y(0); if (y < -20 || y > H + 20) return;
    const x0 = cam.X(-AB.wallL(0)), x1 = cam.X(AB.wallR(0));
    ctx.save(); ctx.globalCompositeOperation = 'lighter';
    ctx.strokeStyle = 'rgba(170,240,230,0.55)'; ctx.lineWidth = 2;
    ctx.beginPath(); for (let x = x0; x <= x1; x += 4) { const yy = y + 1.4 * Math.sin(x * .05 + t * 1.6); x === x0 ? ctx.moveTo(x, yy) : ctx.lineTo(x, yy); } ctx.stroke();
    const g = ctx.createLinearGradient(0, y - 30, 0, y); g.addColorStop(0, 'rgba(90,190,180,0)'); g.addColorStop(1, 'rgba(90,190,180,0.10)');
    ctx.fillStyle = g; ctx.fillRect(x0, y - 30, x1 - x0, 30);
    ctx.restore();
  };

  // Caustics: a net of wavering light lines in the top metres of water (clip to the water poly first)
  AB.caustics = (ctx, cam, t = 0, poly, o = {}) => {
    const depth = o.depth || 22, n = o.n || 34, r = AB.rng(o.seed || 9);
    ctx.save(); if (poly) { pathPts(ctx, poly); ctx.clip(); }
    ctx.globalCompositeOperation = 'lighter'; ctx.lineCap = 'round';
    const xa = cam.X(-AB.wallL(0) - 4), xb = cam.X(AB.wallR(0) + 4);
    for (let i = 0; i < n; i++) {
      const d0 = Math.pow(r(), 1.6) * depth, ph = r() * 6.28, fr = .02 + r() * .03, amp = (.6 + r() * 1.4) * cam.s, tilt = (r() - .5) * .6;
      const a = .16 * Math.pow(1 - d0 / depth, 1.5);
      ctx.strokeStyle = `rgba(170,245,230,${a})`; ctx.lineWidth = 1 + 1.6 * (1 - d0 / depth);
      ctx.beginPath();
      for (let x = xa; x <= xb; x += 5) { const y = cam.Y(d0) + amp * Math.sin(x * fr + ph + t * (.5 + r() * 0)) + (x - xa) * tilt * .2 + amp * .6 * Math.sin(x * fr * 2.7 - t * .8 + ph); x === xa ? ctx.moveTo(x, y) : ctx.lineTo(x, y); }
      ctx.stroke();
    }
    ctx.restore();
  };

  // God rays from the pool surface falling through the throat
  AB.rays = (ctx, cam, t = 0, poly, o = {}) => {
    const r = AB.rng(o.seed || 21), n = o.n || 7, len = o.len || 140;
    ctx.save(); if (poly) { pathPts(ctx, poly); ctx.clip(); }
    ctx.globalCompositeOperation = 'lighter';
    for (let i = 0; i < n; i++) {
      const m = -14 + r() * 30, w = (1.5 + r() * 4) * cam.s, ang = (r() - .5) * .18 + .03 * Math.sin(t * .3 + i), L = len * (.5 + r() * .5) * cam.s;
      const x = cam.X(m), y = cam.Y(0), a = (.05 + r() * .06) * (o.alpha || 1) * (.75 + .25 * Math.sin(t * .7 + i * 2));
      ctx.save(); ctx.translate(x, y); ctx.rotate(ang);
      const g = ctx.createLinearGradient(0, 0, 0, L); g.addColorStop(0, `rgba(150,235,220,${a})`); g.addColorStop(1, 'rgba(150,235,220,0)');
      ctx.fillStyle = g; ctx.beginPath(); ctx.moveTo(-w / 2, 0); ctx.lineTo(w / 2, 0); ctx.lineTo(w * 1.6, L); ctx.lineTo(-w * 1.6, L); ctx.closePath(); ctx.fill();
      ctx.restore();
    }
    ctx.restore();
  };

  // ---------------------------------------------------------------- torch + silt
  // light = { x, y, ang, len, spread } (px, radians). Soft amber cone, additive.
  AB.torch = (ctx, L, o = {}) => {
    const I = o.intensity == null ? 1 : o.intensity;
    ctx.save(); ctx.globalCompositeOperation = 'lighter';
    for (const [k, a] of [[1.0, .10], [.72, .12], [.45, .15]]) {
      const sp = L.spread * k, g = ctx.createRadialGradient(L.x, L.y, 0, L.x, L.y, L.len);
      g.addColorStop(0, `rgba(255,214,150,${a * 2.2 * I})`); g.addColorStop(.25, `rgba(255,180,90,${a * 1.2 * I})`);
      g.addColorStop(.7, `rgba(230,140,60,${a * .35 * I})`); g.addColorStop(1, 'rgba(200,110,40,0)');
      ctx.fillStyle = g; ctx.beginPath(); ctx.moveTo(L.x, L.y);
      ctx.arc(L.x, L.y, L.len, L.ang - sp, L.ang + sp); ctx.closePath(); ctx.fill();
    }
    // hot source
    const h = ctx.createRadialGradient(L.x, L.y, 0, L.x, L.y, (o.halo || 26));
    h.addColorStop(0, `rgba(255,240,205,${.95 * I})`); h.addColorStop(.3, `rgba(255,190,100,${.45 * I})`); h.addColorStop(1, 'rgba(255,150,60,0)');
    ctx.fillStyle = h; ctx.beginPath(); ctx.arc(L.x, L.y, o.halo || 26, 0, 6.29); ctx.fill();
    ctx.restore();
  };
  // In-water scatter: a broad faint warm haze around a light, so silhouettes read against it
  AB.scatter = (ctx, x, y, R = 260, a = .14, rgb = '255,170,90') => {
    ctx.save(); ctx.globalCompositeOperation = 'lighter';
    const g = ctx.createRadialGradient(x, y, 0, x, y, R); g.addColorStop(0, `rgba(${rgb},${a})`); g.addColorStop(.35, `rgba(${rgb},${a * .35})`); g.addColorStop(1, `rgba(${rgb},0)`);
    ctx.fillStyle = g; ctx.fillRect(x - R, y - R, R * 2, R * 2); ctx.restore();
  };
  // returns 0..1 brightness of a point inside the cone
  AB.inCone = (L, x, y) => {
    const dx = x - L.x, dy = y - L.y, dist = Math.hypot(dx, dy); if (dist > L.len || dist < 1) return 0;
    let da = Math.atan2(dy, dx) - L.ang; da = Math.atan2(Math.sin(da), Math.cos(da));
    const e = 1 - Math.abs(da) / L.spread; if (e <= 0) return 0;
    return AB.smooth(AB.clamp(e * 1.6)) * Math.pow(1 - dist / L.len, 1.3);
  };
  // Ambient silt across the frame: slow drift, cool and faint; amber where a light catches it.
  AB.silt = (ctx, t = 0, o = {}) => {
    const r = AB.rng(o.seed || 3), n = o.n || 260, lights = o.lights || [], dim = o.alpha == null ? 1 : o.alpha;
    ctx.save(); ctx.globalCompositeOperation = 'lighter';
    for (let i = 0; i < n; i++) {
      const bx = r() * W, by = r() * H, sz = .6 + Math.pow(r(), 3) * 2.6, sp = 4 + r() * 10, ph = r() * 6.28, base = .05 + r() * .16;
      const x = AB.mod(bx + Math.sin(t * .23 + ph) * 14 + t * sp * .4, W), y = AB.mod(by - t * sp * .25 + Math.cos(t * .17 + ph) * 10, H);
      if (o.clipY && (y < o.clipY[0] || y > o.clipY[1])) continue;
      let lit = 0; for (const L of lights) lit = Math.max(lit, AB.inCone(L, x, y));
      if (lit > .02) {
        ctx.fillStyle = `rgba(255,205,140,${AB.clamp(base * dim + lit * .85)})`;
        ctx.beginPath(); ctx.arc(x, y, sz * (1 + lit * .6), 0, 6.29); ctx.fill();
      } else {
        ctx.fillStyle = `rgba(180,220,230,${base * dim})`; ctx.fillRect(x - sz / 2, y - sz / 2, sz, sz);
      }
    }
    ctx.restore();
  };
  // Dense silt sampled inside a cone (the torch "catches" a column of particles)
  AB.coneSilt = (ctx, L, t = 0, o = {}) => {
    const r = AB.rng(o.seed || 41), n = o.n || 120;
    ctx.save(); ctx.globalCompositeOperation = 'lighter';
    for (let i = 0; i < n; i++) {
      const u = Math.sqrt(r()), v = (r() * 2 - 1), ph = r() * 6.28, sz = .7 + Math.pow(r(), 2.5) * 2.4;
      const dist = u * L.len, a = L.ang + v * L.spread;
      const x = L.x + Math.cos(a) * dist + Math.sin(t * .4 + ph) * 6, y = L.y + Math.sin(a) * dist + AB.mod(-t * 6 + ph * 3, 18) - 9;
      const lit = AB.inCone(L, x, y); if (lit <= 0) continue;
      ctx.fillStyle = `rgba(255,215,160,${AB.clamp(lit * (.5 + .5 * Math.sin(t * 2 + ph * 5)) + .1)})`;
      ctx.beginPath(); ctx.arc(x, y, sz, 0, 6.29); ctx.fill();
    }
    ctx.restore();
  };

  // ---------------------------------------------------------------- diver
  // Silhouette pointing along `ang` (head first). Scale k: 1 => ~130 px long. Returns torch light origin.
  AB.diver = (ctx, x, y, ang, k = 1, t = 0, o = {}) => {
    const kick = Math.sin(t * (o.kickSpeed || 1.6)) * (o.kick == null ? 1 : o.kick);
    ctx.save(); ctx.translate(x, y); ctx.rotate(ang); ctx.scale(k, k * (o.flip ? -1 : 1));
    const fill = o.fill || '#010203';
    const body = () => {
      // head + mask
      ctx.beginPath(); ctx.arc(49, -1, 9, 0, 6.29); ctx.fill();
      ctx.beginPath(); ctx.roundRect(52, -4, 9, 9, 3); ctx.fill();
      // torso (one smooth streamlined shape)
      ctx.beginPath(); ctx.moveTo(42, -9);
      ctx.bezierCurveTo(24, -14, 0, -12, -16, -8); ctx.lineTo(-19, 7);
      ctx.bezierCurveTo(0, 13, 24, 13, 42, 9); ctx.quadraticCurveTo(46, 0, 42, -9); ctx.fill();
      // rebreather / cylinder on the back + breathing hose
      ctx.beginPath(); ctx.roundRect(-12, -21, 44, 13, 6.5); ctx.fill();
      ctx.save(); ctx.strokeStyle = ctx.fillStyle; ctx.lineWidth = 2.6; ctx.lineCap = 'round';
      ctx.beginPath(); ctx.moveTo(30, -18); ctx.quadraticCurveTo(48, -20, 56, -4); ctx.stroke();
      // legs: thigh + shin, gently bent, scissoring
      for (const [sgn, off] of [[1, -3], [-1, 3]]) {
        const a = kick * .14 * sgn, kx = -40, ky = off + 2 * sgn + a * 40, ax = -62, ay = off + a * 70;
        ctx.lineWidth = 12; ctx.beginPath(); ctx.moveTo(-12, off * .8); ctx.lineTo(kx, ky); ctx.stroke();
        ctx.lineWidth = 9; ctx.beginPath(); ctx.moveTo(kx, ky); ctx.lineTo(ax, ay); ctx.stroke();
        // fin blade
        const fa = Math.atan2(ay - ky, ax - kx);
        ctx.save(); ctx.translate(ax, ay); ctx.rotate(fa);
        ctx.beginPath(); ctx.moveTo(-2, -5); ctx.quadraticCurveTo(16, -8, 34, -7 - 2 * sgn); ctx.lineTo(35, 5 - 2 * sgn); ctx.quadraticCurveTo(16, 5, -2, 5); ctx.closePath(); ctx.fill();
        ctx.restore();
      }
      // arm reaching forward with the torch
      ctx.lineWidth = 7; ctx.beginPath(); ctx.moveTo(32, 6); ctx.quadraticCurveTo(44, 18, 58, 13); ctx.stroke();
      ctx.restore();
      ctx.beginPath(); ctx.roundRect(56, 9, 15, 9, 3); ctx.fill();
    };
    // amber rim light (offset toward the torch), then the silhouette
    if (o.rim !== 0) {
      ctx.save(); ctx.translate(1.2, 1.4); ctx.fillStyle = `rgba(255,170,80,${o.rim == null ? .55 : o.rim})`; body(); ctx.restore();
    }
    ctx.fillStyle = fill; body();
    ctx.restore();
    // torch origin in world px
    const lx = 71, ly = 13.5 * (o.flip ? -1 : 1), c = Math.cos(ang), s = Math.sin(ang);
    return { x: x + (lx * c - ly * s) * k, y: y + (lx * s + ly * c) * k, headX: x + 46 * c * k, headY: y + 46 * s * k };
  };

  // Rising bubbles: a wobbling trail from (x, y) up to y - height
  AB.bubbles = (ctx, x, y, t = 0, o = {}) => {
    const r = AB.rng(o.seed || 13), n = o.n || 14, Hh = o.height || 700, k = o.scale || 1;
    ctx.save(); ctx.globalCompositeOperation = 'lighter';
    for (let i = 0; i < n; i++) {
      const sp = 40 + r() * 50, ph = r(), rad = (1.6 + r() * 3.4) * k, wob = 6 + r() * 14;
      const u = AB.mod(t * sp / Hh + ph, 1), yy = y - u * Hh, xx = x + Math.sin(u * 9 + i) * wob * u + (r() - .5) * 10 * k;
      const a = .55 * Math.pow(1 - u, .8) * Math.min(1, u * 12), R = rad * (1 + u * .7);
      ctx.strokeStyle = `rgba(190,235,240,${a})`; ctx.lineWidth = 1.3;
      ctx.beginPath(); ctx.ellipse(xx, yy, R, R * .86, 0, 0, 6.29); ctx.stroke();
      ctx.fillStyle = `rgba(230,250,255,${a * .8})`; ctx.beginPath(); ctx.arc(xx - R * .35, yy - R * .35, Math.max(.6, R * .25), 0, 6.29); ctx.fill();
    }
    ctx.restore();
  };

  // ---------------------------------------------------------------- cave line
  // The guideline / shotline: a thin bright filament. pts in px; lit by lights (amber) or faint otherwise.
  AB.line = (ctx, pts, o = {}) => {
    ctx.save(); ctx.lineCap = 'round'; ctx.lineJoin = 'round';
    pathPts(ctx, pts, false);
    if (o.glow) { ctx.globalCompositeOperation = 'lighter'; ctx.strokeStyle = `rgba(255,190,110,${o.glow})`; ctx.lineWidth = (o.width || 2) * 5; ctx.stroke(); }
    ctx.globalCompositeOperation = 'lighter'; ctx.strokeStyle = o.color || 'rgba(225,240,240,0.35)'; ctx.lineWidth = o.width || 1.6; ctx.stroke();
    ctx.restore();
  };
  // a messy tangle of loops around (x, y), radius R: returns points for AB.line.
  // Random control points alternate around the centre and are smoothed (Chaikin) into a scribble.
  AB.tangle = (x, y, R, t = 0, o = {}) => {
    const r = AB.rng(o.seed || 5), n = o.n || 14, sx = o.sx || 1.3, sy = o.sy || .8, rot = o.rot || 0;
    let pts = [], a = r() * 6.28;
    for (let i = 0; i < n; i++) {
      a += 1.9 + r() * 1.6;                         // jump most of the way round each time -> crossings
      const rr = R * (.35 + .65 * r()), w = Math.sin(t * .7 + i) * R * .03;
      const px = Math.cos(a) * rr * sx + w, py = Math.sin(a) * rr * sy;
      pts.push([x + px * Math.cos(rot) - py * Math.sin(rot), y + px * Math.sin(rot) + py * Math.cos(rot)]);
    }
    for (let k = 0; k < 3; k++) {                     // Chaikin smoothing (open curve)
      const q = [pts[0]];
      for (let i = 0; i < pts.length - 1; i++) { const [ax, ay] = pts[i], [bx, by] = pts[i + 1]; q.push([ax * .75 + bx * .25, ay * .75 + by * .25], [ax * .25 + bx * .75, ay * .25 + by * .75]); }
      q.push(pts[pts.length - 1]); pts = q;
    }
    return pts;
  };

  // ---------------------------------------------------------------- HUD
  const spaced = (ctx, str, x, y, sp) => { for (const ch of str) { ctx.fillText(ch, x, y); x += ctx.measureText(ch).width + sp; } return x; };
  AB.spaced = spaced;
  AB.spacedWidth = (ctx, str, sp) => [...str].reduce((w, ch) => w + ctx.measureText(ch).width + sp, -sp);
  // Depth gauge: small mono label + huge condensed number. warn (0..1) shifts it to red.
  AB.gauge = (ctx, x, y, depth, o = {}) => {
    const size = o.size || 230, warn = o.warn || 0, a = o.alpha == null ? 1 : o.alpha;
    ctx.save(); ctx.globalAlpha = a; ctx.textBaseline = 'alphabetic';
    const col = warn > 0 ? `rgba(255,${Math.round(235 - 175 * warn)},${Math.round(230 - 190 * warn)},1)` : AB.C.pale;
    // label row
    ctx.font = `500 ${Math.round(size * .115)}px "JetBrains Mono"`; ctx.fillStyle = warn > 0 ? `rgba(255,90,70,${.6 + .4 * warn})` : AB.C.mute;
    spaced(ctx, o.label || 'DEPTH', x + 6, y, size * .03);
    // number
    const num = (depth <= 0 ? '' : '-') + Math.round(Math.abs(depth));
    ctx.font = `300 ${size}px "Barlow Condensed"`; ctx.fillStyle = col;
    if (warn > 0) { ctx.shadowColor = `rgba(255,40,30,${.55 * warn})`; ctx.shadowBlur = 40; }
    const yN = y + size * .86; ctx.fillText(num, x, yN);
    const wN = ctx.measureText(num).width; ctx.shadowBlur = 0;
    ctx.font = `300 ${Math.round(size * .36)}px "Barlow Condensed"`; ctx.fillText('m', x + wN + size * .05, yN);
    // thin rule under the number with depth ticks
    const rw = o.ruleW || wN + size * .3, ry = yN + size * .12;
    ctx.fillStyle = warn > 0 ? `rgba(255,70,55,.7)` : 'rgba(200,230,230,.35)'; ctx.fillRect(x + 4, ry, rw, 2);
    for (let i = 0; i <= 10; i++) ctx.fillRect(x + 4 + rw * i / 10, ry, 2, i % 5 ? 7 : 14);
    if (o.max) { const p = AB.clamp(depth / o.max); ctx.fillStyle = warn > 0 ? AB.C.red : AB.C.amber; ctx.beginPath(); ctx.moveTo(x + 4 + rw * p, ry - 3); ctx.lineTo(x + 4 + rw * p - 8, ry - 15); ctx.lineTo(x + 4 + rw * p + 8, ry - 15); ctx.closePath(); ctx.fill(); }
    ctx.restore();
    return { bottom: ry + 14, width: Math.max(rw, wN) };
  };
  // Depth ruler down the left edge (metres), labelled every `every` m
  AB.ruler = (ctx, cam, x = 56, o = {}) => {
    const every = o.every || 50, minor = o.minor || 10, a = o.alpha || 1;
    ctx.save(); ctx.globalAlpha = a; ctx.font = `500 ${o.font || 22}px "JetBrains Mono"`; ctx.textBaseline = 'middle';
    const d0 = Math.max(0, Math.ceil(cam.D(o.y0 || 0) / minor) * minor), d1 = Math.min(o.dMax || 300, cam.D(o.y1 || H));
    ctx.fillStyle = 'rgba(200,230,230,.18)'; ctx.fillRect(x, Math.max(o.y0 || 0, cam.Y(d0)), 1.5, Math.min(o.y1 || H, cam.Y(d1)) - Math.max(o.y0 || 0, cam.Y(d0)));
    for (let d = d0; d <= d1; d += minor) {
      const y = cam.Y(d), major = d % every === 0;
      ctx.fillStyle = major ? 'rgba(210,235,235,.55)' : 'rgba(200,230,230,.22)';
      ctx.fillRect(x, y, major ? 18 : 9, 1.5);
      if (major && !(o.skip && o.skip(d))) { ctx.fillStyle = 'rgba(210,235,235,.5)'; ctx.fillText(d === 0 ? '0' : '-' + d, x + 26, y); }
    }
    ctx.restore();
  };
  // A thin dashed depth mark across the frame with a big label (e.g. the 270 m floor)
  AB.depthMark = (ctx, cam, d, o = {}) => {
    const y = cam.Y(d), x0 = o.x0 || 40, x1 = o.x1 || W - 140, col = o.color || 'rgba(230,240,240,0.5)';
    ctx.save(); ctx.setLineDash([10, 9]); ctx.strokeStyle = col; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(x0, y); ctx.lineTo(x1, y); ctx.stroke(); ctx.setLineDash([]);
    if (o.label) {
      ctx.font = `${o.weight || 500} ${o.size || 120}px "Barlow Condensed"`; ctx.textAlign = 'right'; ctx.textBaseline = 'alphabetic';
      ctx.fillStyle = o.textColor || AB.C.pale; ctx.fillText(o.label, x1, y - 18);
    }
    ctx.restore();
  };
  // Small mono label with letter spacing (dates, places). One at a time.
  AB.label = (ctx, str, x, y, o = {}) => {
    ctx.save(); ctx.font = `${o.weight || 500} ${o.size || 30}px "JetBrains Mono"`; ctx.textBaseline = 'alphabetic';
    ctx.fillStyle = o.color || AB.C.mute; ctx.globalAlpha = o.alpha == null ? 1 : o.alpha;
    let xx = x; if (o.align === 'center') xx = x - AB.spacedWidth(ctx, str, o.sp == null ? 6 : o.sp) / 2;
    if (o.bar) { ctx.fillRect(xx, y - (o.size || 30) * 1.25, 36, 2); }
    spaced(ctx, str, xx, y, o.sp == null ? 6 : o.sp); ctx.restore();
  };
  // Warning chip: triangle + "CO2" (subscript drawn manually), red, pulsing with p (0..1)
  AB.warnChip = (ctx, x, y, o = {}) => {
    const p = o.pulse == null ? 1 : o.pulse, s = o.scale || 1, text = o.text || 'CO', sub = o.sub == null ? '2' : o.sub;
    ctx.save(); ctx.translate(x, y); ctx.scale(s, s);
    ctx.font = '600 64px "Barlow Condensed"'; const tw = ctx.measureText(text).width; ctx.font = '600 38px "Barlow Condensed"'; const sw = sub ? ctx.measureText(sub).width + 4 : 0;
    const w = 104 + tw + sw + 26, h = 92;
    ctx.shadowColor = `rgba(255,40,30,${.7 * p})`; ctx.shadowBlur = 36 * p;
    ctx.fillStyle = `rgba(40,4,4,${.55})`; ctx.strokeStyle = `rgba(255,59,47,${.5 + .5 * p})`; ctx.lineWidth = 3;
    ctx.beginPath(); ctx.roundRect(0, 0, w, h, 10); ctx.fill(); ctx.stroke(); ctx.shadowBlur = 0;
    // triangle
    ctx.fillStyle = `rgba(255,59,47,${.6 + .4 * p})`; ctx.beginPath(); ctx.moveTo(52, 16); ctx.lineTo(84, 74); ctx.lineTo(20, 74); ctx.closePath(); ctx.fill();
    ctx.fillStyle = '#1a0202'; ctx.fillRect(49.5, 34, 5, 22); ctx.fillRect(49.5, 61, 5, 6);
    ctx.fillStyle = `rgba(255,${Math.round(120 + 80 * (1 - p))},${Math.round(110 + 60 * (1 - p))},1)`;
    ctx.font = '600 64px "Barlow Condensed"'; ctx.textBaseline = 'alphabetic'; ctx.fillText(text, 104, 70);
    if (sub) { ctx.font = '600 38px "Barlow Condensed"'; ctx.fillText(sub, 104 + tw + 3, 80); }
    ctx.restore();
    return { w: w * s, h: h * s };
  };
  // Red alarm wash on the frame edges (use sparingly). p 0..1
  AB.alarm = (ctx, p = 1) => {
    if (p <= 0) return;
    ctx.save();
    const g = ctx.createRadialGradient(W / 2, H * .5, H * .36, W / 2, H * .5, H * .7);
    g.addColorStop(0, 'rgba(255,30,20,0)'); g.addColorStop(.6, `rgba(190,18,12,${.16 * p})`); g.addColorStop(1, `rgba(220,24,16,${.5 * p})`);
    ctx.globalCompositeOperation = 'screen'; ctx.fillStyle = g; ctx.fillRect(0, 0, W, H);
    ctx.restore();
  };
  // Soft cold marker light (suggests a presence/place, never depicts a person)
  AB.marker = (ctx, x, y, o = {}) => {
    const R = o.r || 60, a = o.alpha == null ? 1 : o.alpha;
    ctx.save(); ctx.globalCompositeOperation = 'lighter';
    const g = ctx.createRadialGradient(x, y, 0, x, y, R); g.addColorStop(0, `rgba(${AB.C.cold},${.5 * a})`); g.addColorStop(.2, `rgba(${AB.C.cold},${.16 * a})`); g.addColorStop(1, `rgba(${AB.C.cold},0)`);
    ctx.fillStyle = g; ctx.beginPath(); ctx.arc(x, y, R, 0, 6.29); ctx.fill();
    ctx.fillStyle = `rgba(235,245,255,${.85 * a})`; ctx.beginPath(); ctx.arc(x, y, o.core || 3, 0, 6.29); ctx.fill();
    ctx.restore();
  };

  // ---------------------------------------------------------------- finishing
  let grainTile = null;
  AB.grain = (ctx, t = 0, amt = .07) => {
    if (!grainTile) {
      grainTile = document.createElement('canvas'); grainTile.width = grainTile.height = 256;
      const g = grainTile.getContext('2d'), img = g.createImageData(256, 256), r = AB.rng(1234);
      for (let i = 0; i < img.data.length; i += 4) { const v = Math.floor(r() * 255); img.data[i] = img.data[i + 1] = img.data[i + 2] = v; img.data[i + 3] = 255; }
      g.putImageData(img, 0, 0);
    }
    const f = Math.floor(t * 24), ox = (f * 97) % 256, oy = (f * 61) % 256;
    ctx.save(); ctx.globalAlpha = amt; ctx.globalCompositeOperation = 'overlay';
    ctx.translate(-ox, -oy); ctx.fillStyle = ctx.createPattern(grainTile, 'repeat'); ctx.fillRect(0, 0, W + 256, H + 256);
    ctx.restore();
  };
  AB.vignette = (ctx, amt = .75) => {
    ctx.save();
    const g = ctx.createRadialGradient(W / 2, H * .46, H * .22, W / 2, H * .5, H * .72);
    g.addColorStop(0, 'rgba(0,0,0,0)'); g.addColorStop(1, `rgba(0,0,0,${amt})`);
    ctx.fillStyle = g; ctx.fillRect(0, 0, W, H); ctx.restore();
  };

  // ---------------------------------------------------------------- captions
  // Burned-in narration caption, word by word. words: ['In', '2005,', ...]; k = words spoken so far (float:
  // the fractional part fades the current word in). Layout is fixed for the whole chunk so nothing jumps.
  AB.CAP = { y: 1660, size: 56, maxW: 820, cx: 540, lh: 1.24, font: 'IBM Plex Sans', weight: 600 };
  AB.caption = (ctx, words, k = Infinity, o = {}) => {
    const c = Object.assign({}, AB.CAP, o), sz = c.size;
    ctx.save(); ctx.font = `${c.weight} ${sz}px "${c.font}"`; ctx.textBaseline = 'alphabetic';
    const sp = ctx.measureText(' ').width;
    const wrap = mw => { const ls = [[]]; let lw = 0; for (const w of words) { const ww = ctx.measureText(w).width; if (ls[ls.length - 1].length && lw + sp + ww > mw) { ls.push([]); lw = 0; } ls[ls.length - 1].push(w); lw += (lw ? sp : 0) + ww; } return ls; };
    let lines = wrap(c.maxW);
    if (lines.length > 1) { let lo = 100, hi = c.maxW; while (hi - lo > 2) { const m = (lo + hi) / 2; wrap(m).length > lines.length ? lo = m : hi = m; } lines = wrap(hi); } // balance
    const widths = lines.map(l => ctx.measureText(l.join(' ')).width), bw = Math.max(...widths), lhp = sz * c.lh;
    const bh = lhp * lines.length, padX = 30, padY = 20, top = c.y - bh / 2;
    const vis = AB.clamp(k);  // plate fades in with the first word
    // plate
    ctx.globalAlpha = (c.alpha == null ? 1 : c.alpha) * vis;
    ctx.fillStyle = 'rgba(2,6,10,0.62)'; ctx.beginPath(); ctx.roundRect(c.cx - bw / 2 - padX, top - padY, bw + padX * 2, bh + padY * 2, 18); ctx.fill();
    ctx.strokeStyle = 'rgba(190,230,230,0.08)'; ctx.lineWidth = 1.5; ctx.stroke();
    // words
    let wi = 0;
    lines.forEach((l, li) => {
      let x = c.cx - widths[li] / 2; const y = top + lhp * li + sz * .98;
      for (const w of l) {
        const p = AB.clamp(k - wi), ww = ctx.measureText(w).width;
        if (p > 0) {
          const cur = k - wi < 1.6 && wi === Math.floor(Math.min(k, words.length - 1e-6));
          ctx.globalAlpha = (c.alpha == null ? 1 : c.alpha) * AB.ease.out(p);
          ctx.fillStyle = cur && c.hot !== false ? '#ffffff' : '#dfe9ea';
          ctx.fillText(w, x, y + (1 - AB.ease.out(p)) * 10);
          if (cur && c.hot !== false) { ctx.fillStyle = `rgba(255,179,71,${.9 * p})`; ctx.fillRect(x, y + sz * .2, ww, 3); }
        }
        x += ww + sp; wi++;
      }
    });
    ctx.restore();
  };

  window.AB = AB;

  // ================================================================ STYLE FRAMES
  // A — the hook: the whole sinkhole, surface to 270 m floor.
  function A(ctx, t) {
    const cam = AB.cam(470, 372, 0, 4.2);   // surface at y=372, 270 m at y≈1506
    const poly = AB.cave(ctx, cam, t);
    AB.rays(ctx, cam, t, poly, { n: 9, len: 120, alpha: 1.1 });
    AB.caustics(ctx, cam, t, poly, { depth: 26, n: 40 });
    AB.surface(ctx, cam, t);
    // faint shotline from the surface toward the deep; the diver just below the throat
    AB.line(ctx, [[cam.X(1), cam.Y(-1)], [cam.X(1.6), cam.Y(150)], [cam.X(6), cam.Y(AB.floorD(6) - .5)]], { color: 'rgba(220,240,240,0.13)', width: 1.2 });
    const dv = { x: cam.X(1.2), y: cam.Y(15) }, ang = Math.PI * .5 + .1;
    AB.scatter(ctx, dv.x, dv.y + 30, 150, .22);
    const L = AB.diver(ctx, dv.x, dv.y, ang, .36, t, { rim: .8 });
    const light = { x: L.x, y: L.y, ang, len: 260, spread: .3 };
    AB.torch(ctx, light, { intensity: .95, halo: 20 });
    AB.silt(ctx, t, { n: 220, lights: [light], alpha: .8, seed: 3 });
    AB.bubbles(ctx, dv.x - 3, dv.y - 12, t, { n: 7, height: 60, scale: .5 });
    // the floor mark
    AB.depthMark(ctx, cam, 270, { label: '270 m', size: 150, weight: 300, x0: 60, x1: 930, color: 'rgba(230,240,240,0.34)' });
    AB.ruler(ctx, cam, 52, { y0: cam.Y(0), every: 50, minor: 10, alpha: .9, dMax: 280, skip: d => d === 250 });
    AB.vignette(ctx, .7);
    AB.label(ctx, '8 JAN 2005', 80, 262, { size: 30, sp: 8, color: 'rgba(220,235,235,0.78)' });
    AB.grain(ctx, t, .08);
    AB.caption(ctx, 'In 2005, one diver went 270 metres down'.split(' '));
  }

  // B — mid-descent at -180 m: a speck of warm light in a vast black chamber.
  function B(ctx, t) {
    const cam = AB.cam(700, 980, 180, 10);
    AB.cave(ctx, cam, t, { cutWidth: 2.5 });
    const dv = { x: 600, y: 900 }, ang = Math.PI * .5 + .32;
    // shotline running down through the frame, slightly slack
    AB.line(ctx, [[640, 0], [620, 500], [dv.x + 14, dv.y - 30], [612, 1400], [626, H]], { color: 'rgba(220,240,240,0.14)', width: 1.4 });
    AB.bubbles(ctx, dv.x + 6, dv.y - 24, t, { n: 16, height: 860, seed: 13 });
    AB.scatter(ctx, dv.x - 20, dv.y + 80, 330, .13);
    const L = AB.diver(ctx, dv.x, dv.y, ang, .62, t, { rim: .8 });
    const light = { x: L.x, y: L.y, ang, len: 560, spread: .28 };
    AB.torch(ctx, light, { intensity: 1, halo: 26 });
    AB.coneSilt(ctx, light, t, { n: 180 });
    AB.silt(ctx, t, { n: 300, lights: [light], alpha: .75, seed: 8 });
    AB.vignette(ctx, .82);
    AB.gauge(ctx, 80, 300, 180, { size: 230, max: 270, ruleW: 380 });
    AB.grain(ctx, t, .085);
    AB.caption(ctx, 'He kept sinking, past 180 metres'.split(' '), 5.5);
  }

  // C — the bottom, -270 m: the line wraps the torch, CO2 climbing.
  function C(ctx, t) {
    const cam = AB.cam(470, 1180, 270, 13);
    AB.cave(ctx, cam, t, { cutWidth: 3 });
    // marker on the floor: a place, not a person
    const mx = cam.X(-22), my = cam.Y(AB.floorD(-22) - 1.4);
    AB.marker(ctx, mx, my, { r: 110, alpha: .7, core: 3.5 });
    const dv = { x: 600, y: 990 }, ang = Math.PI * .8;
    AB.scatter(ctx, dv.x - 60, dv.y + 40, 420, .17);
    const L = AB.diver(ctx, dv.x, dv.y, ang, 1.05, t, { rim: .9, kick: .3, flip: true });
    const light = { x: L.x, y: L.y, ang: ang + .1, len: 520, spread: .24 };
    AB.torch(ctx, light, { intensity: .85, halo: 34 });
    AB.coneSilt(ctx, light, t, { n: 200, seed: 77 });
    AB.silt(ctx, t, { n: 320, lights: [light], alpha: .7, seed: 19 });
    // the line: hangs from above, snarls around the torch arm, trails off
    const tg = AB.tangle(L.x + 34, L.y - 18, 74, t, { n: 15, seed: 12, sx: 1.45, sy: .7, rot: -.4 });
    const a0 = tg[0], a1 = tg[tg.length - 1];
    AB.line(ctx, [[cam.X(12), 0], [cam.X(11.5), 500], [a0[0] + 20, a0[1] - 140], a0], { color: 'rgba(235,245,245,0.42)', width: 1.6 });
    AB.line(ctx, tg, { color: 'rgba(255,238,210,0.92)', width: 2.2, glow: .14 });
    { const tail = []; for (let i = 0; i <= 30; i++) { const u = i / 30, x = a1[0] + u * 330, y = a1[1] + Math.sin(u * 1.9) * 150 + u * 40; tail.push([x, Math.min(y, cam.Y(AB.floorD(cam.M(x)) - .6))]); }
      AB.line(ctx, tail, { color: 'rgba(235,245,245,0.28)', width: 1.4 }); }
    AB.bubbles(ctx, L.headX, L.headY - 30, t, { n: 9, height: 520, seed: 31 });
    const pulse = .5 + .5 * Math.sin(t * 5.2);
    AB.vignette(ctx, .85);
    AB.alarm(ctx, .4 + .4 * pulse);
    const g = AB.gauge(ctx, 80, 300, 270, { size: 230, warn: 1, max: 270, ruleW: 380 });
    AB.warnChip(ctx, 84, g.bottom + 40, { pulse });
    AB.grain(ctx, t, .09);
    AB.caption(ctx, 'Then the line wrapped around his torch'.split(' '));
  }

  window.STYLEFRAMES = { fonts: AB.FONTS, frames: { A, B, C } };
})();
