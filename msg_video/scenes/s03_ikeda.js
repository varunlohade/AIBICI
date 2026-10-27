// s03_ikeda: 1908, the discovery of umami (Kombu Paper). Shots s03.01-.07 of design/SHOTLIST.md.
// Kikunae Ikeda is the only named real person drawn: a respectful paper doll (high collar, round glasses, neat
// moustache, period suit), never comedic. All timings come from cues.json words.
//   1 lab-kitchen, '1908': Ikeda carries a steaming bowl of seaweed soup to his desk
//   2 he sips, the kombu strip floats in the bowl, brows lift, '?' on 'odd'
//   3 overhead: rings spread from the bowl carrying sweet / sour / salty / bitter; the fifth ring blooms gold + Goldie
//   4 a mountain of kombu tumbles into a giant pot, Ikeda tiny on a stool
//   5 time-lapse: the pile boils down in steam puffs to a dish of crystals; Goldie + the glutamate diagram, 'Glutamic acid'
//   6 Ikeda holds up the dish, the gold ring returns, 'Umami'; Goldie and a sodium ion link mittens on 'sodium'
//   7 they hop into the MSG shaker and the cap snaps on, 'MSG'; paper slide out
(function () {
  const P = KP.P, TAU = Math.PI * 2;
  const times = (cues, D) => {
    const w = (re, o) => KP.word(cues, re, o);
    const T = { y1908: w(/1908/, { fb: .53 }), chemist: w(/chemist/i, { fb: 2.6 }), ikeda: w(/ikeda/i, { fb: 3.8 }),
      noticed: w(/noticed/i, { fb: 4.4 }), odd: w(/^odd/i, { fb: 5.64 }), soup: w(/soup/i, { fb: 6.84 }) };
    T.l2 = w(/sweet/i, { line: 1, fb: 7.46 }); T.sweet = w(/sweet/i, { fb: 7.87 }); T.sour = w(/sour/i, { fb: 8.29 });
    T.salty = w(/salty/i, { fb: 8.72 }); T.bitter = w(/bitter/i, { fb: 9.3 }); T.something = w(/something/i, { after: T.bitter, fb: 10.5 }); T.else = w(/else/i, { fb: 11.1 });
    T.l3 = w(/boiled/i, { line: 1, fb: 11.62 }); T.boiled = w(/boiled/i, { fb: 12.05 }); T.mountain = w(/mountain/i, { fb: 12.87 });
    T.found = w(/^found/i, { fb: 14.4 }); T.glutamic = w(/glutamic/i, { fb: 15.49 });
    T.l4 = w(/umami/i, { line: 1, fb: 16.88 }); T.umami = w(/umami/i, { fb: 17.89 }); T.paired = w(/paired/i, { fb: 18.88 }); T.sodium = w(/^sodium/i, { fb: 19.64 });
    T.l5 = w(/monosodium/i, { line: 1, fb: 21.47 }); T.mono = w(/monosodium/i, { fb: 21.52 }); T.glutamate = w(/glutamate/i, { after: T.mono, fb: 22.63 });
    T.s2 = T.noticed - .1; T.s5 = T.found - .2;
    return T;
  };

  // a soup bowl seen from the side: white bowl with a turquoise band, broth, kombu strip, steam
  function bowl(ctx, x, y, s, b, t, o = {}) {
    for (let i = 0; i < 3; i++) {
      const sx = x + (i - 1) * 34 * s, ph = KP.held(t) * 2.5 + i;
      KP.brush(ctx, KP.curve([[sx, y - 70 * s], [sx + 12 * s * Math.sin(ph), y - 110 * s], [sx - 10 * s * Math.sin(ph), y - 150 * s], [sx + 6 * s, y - 185 * s]], false, 6), 8 * s, P.rice, { taper: .7 });
    }
    KP.paper(ctx, KP.jitter(KP.curve([[x - 110 * s, y - 50 * s], [x + 110 * s, y - 50 * s], [x + 90 * s, y + 10 * s], [x + 40 * s, y + 40 * s], [x - 40 * s, y + 40 * s], [x - 90 * s, y + 10 * s]], true, 6), 2000 + b, 1.2), P.rice, { shadow: KP.SH.near, shadowA: .3, mis: P.turq, misOff: [-3, 2], misA: .4 });
    KP.paper(ctx, KP.jitter(KP.rr(x - 100 * s, y - 24 * s, 200 * s, 16 * s, 6, 10), 2001 + b, .8), P.turq, { shadow: false });
    KP.paper(ctx, KP.jitter(KP.ell(x, y - 50 * s, 110 * s, 20 * s, 32), 2002 + b, .8), P.lemon, { shadow: false });
    if (o.kombu !== false) KP.brush(ctx, KP.curve([[x - 60 * s, y - 52 * s], [x - 20 * s, y - 60 * s + Math.sin(KP.held(t) * 3) * 4 * s], [x + 30 * s, y - 48 * s], [x + 66 * s, y - 56 * s]], false, 8), 14 * s, P.kombu, { taper: .4 });
  }
  function lab(ctx, b, t) {
    ctx.fillStyle = P.lemon; ctx.fillRect(-600, -400, 2400, 2800);
    for (let i = -4; i < 12; i++) KP.paper(ctx, KP.jitter(KP.rr(-20 + i * 136, -400, 60, 2800, 10, 60), 2010 + i + b, 1.2), KP.mix(P.lemon, P.brass, .3), { shadow: false });
    // window (blue sky + paper sun is not gold: lemon/white) and a shelf of flasks
    KP.paper(ctx, KP.jitter(KP.rr(620, 280, 300, 280, 14, 20), 2020 + b, 1.2), P.coral, { shadow: KP.SH.near, shadowA: .3 });
    KP.paper(ctx, KP.jitter(KP.rr(642, 302, 256, 236, 8, 20), 2021 + b, 1), P.turq, { shadow: false });
    KP.paper(ctx, KP.jitter(KP.rr(762, 302, 16, 236, 4, 20), 2022 + b, .8), P.coral, { shadow: false });
    KP.paper(ctx, KP.jitter(KP.rr(-300, 560, 820, 26, 6, 30), 2023 + b, 1.2), P.coral, { shadow: KP.SH.near, shadowA: .3 });
    [[-120, P.pink], [20, P.turq], [150, P.kombu], [300, P.blue], [430, P.magenta]].forEach(([fx, c], i) => KP.flask(ctx, fx, 560, 120 + (i % 2) * 30, c, { seed: 2030 + i, boil: b, t: t + i * .3 }));
  }
  function desk(ctx, b, x0 = -400, x1 = 1500) {
    KP.paper(ctx, KP.jitter(KP.rr(x0, 1040, x1 - x0, 40, 8, 40), 2040 + b, 1.4), KP.mix(P.coral, P.lemon, .25), { shadow: KP.SH.near, shadowA: .35 });
    KP.paper(ctx, KP.jitter(KP.rr(x0 + 30, 1080, x1 - x0 - 60, 900, 8, 40), 2041 + b, 1.4), P.coral, { shadow: [0, -6], shadowA: .25, mis: P.pink, misOff: [0, -3], misA: .35 });
    for (let i = 0; i < 6; i++) KP.paper(ctx, KP.jitter(KP.rr(x0 + 80 + i * 300, 1120, 240, 120, 12, 20), 2042 + i + b, 1), KP.mix(P.coral, P.magenta, .4), { shadow: KP.SH.far });
  }
  // ---- shots 1-2
  function shotLab(ctx, t, b, T) {
    const walk = KP.ep(t, .3, T.ikeda - .1, 'inOut'), ix = KP.lerp(-80, 330, walk);
    const step = walk > 0 && walk < 1 ? Math.abs(Math.sin(KP.held(t) * 9)) * 12 : 0;
    const set = t >= T.ikeda;
    const push = KP.ep(t, T.s2, T.l2, 'inOut');
    const camX = KP.lerp(KP.lerp(330, 480, KP.ease.inOut(walk)), 330, push), camY = KP.lerp(860, 740, push), sc = 1 + .4 * push;
    KP.camera(ctx, camX + 10, camY, sc, () => {
      lab(ctx, b, t);
      // sip: hands to the mouth on 'noticed', down again before 'odd'
      const sip = KP.ep(t, T.noticed - .1, T.noticed + .25, 'out') * (1 - KP.ep(t, T.odd - .55, T.odd - .2, 'inOut'));
      const carry = !set, arms = carry ? .6 : sip > 0 ? KP.lerp(1.1, 2.35, sip) : 1.1;
      const bowlUp = carry || sip > 0 || t >= T.noticed;
      const lift = t >= T.odd ? KP.backOut(KP.prog(t, T.odd, T.odd + .3)) : 0;
      const h = KP.ikeda(ctx, ix, 1250 - step, 1, { boil: b, armL: -arms, armR: arms, brows: lift, mouth: sip > .6 ? 'o' : 'smile' });
      desk(ctx, b);
      if (set && !(bowlUp && t >= T.noticed - .1)) bowl(ctx, ix + 60, 1036, 1, b, t);
      if (bowlUp && (carry || t >= T.noticed - .1)) { const bx = (h.handL[0] + h.handR[0]) / 2, by = (h.handL[1] + h.handR[1]) / 2 + 20; bowl(ctx, bx, by, carry ? .9 : 1, b, t); }
      // '?' pops above his head on 'odd'
      if (t >= T.odd) { const q = KP.arrive(t, T.odd, .3); ctx.save(); KP.pose(ctx, q, h.head[0] + 130, h.head[1] - 60, 'pop'); KP.qmark(ctx, h.head[0] + 140, h.head[1] - 140, 150, { rot: .12 + KP.wobA(t, 2, .04) }); ctx.restore(); }
    });
    if (t >= T.y1908 && t < T.s2) {
      const ar = KP.arrive(t, T.y1908, .3), out = KP.leave(t, T.s2, .2);
      ctx.save(); ctx.globalAlpha = 1 - out; KP.pose(ctx, ar, 230, 330, 'pop');
      KP.slip(ctx, [{ s: '1908', size: 110, font: 'Young Serif', weight: 400, ls: 4 }], 70, 222, { rot: -.04, seed: 50, boil: b, tape: P.pink, padX: 36, padY: 20 });
      ctx.restore();
    }
  }
  // ---- shot 3: overhead bowl + the Fifth Ripple of tastes
  const TASTES = [['sweet', -2.35], ['sour', -.8], ['salty', 2.35], ['bitter', .8]];
  function tasteIcon(ctx, kind, x, y, s, b) {
    ctx.save(); ctx.translate(x, y); ctx.scale(s, s);
    if (kind === 'sweet') {      // wrapped candy
      for (const sx of [-1, 1]) KP.paper(ctx, KP.jitter([[sx * 40, 0], [sx * 86, -34], [sx * 86, 34]], 2100 + sx + b, 1), P.pink30, { shadow: KP.SH.far });
      KP.paper(ctx, KP.jitter(KP.egg(0, 0, 54, 40, 0, 32), 2102 + b, 1), P.pink, { shadow: KP.SH.near, mis: P.magenta, misOff: [-3, 2], misA: .4 });
      KP.brush(ctx, [[-30, -20], [20, 24]], 9, P.rice, { taper: .3 });
    } else if (kind === 'sour') { // lemon slice
      KP.paper(ctx, KP.jitter(KP.egg(0, 0, 66, 66, 0, 40), 2110 + b, 1), P.lemon, { shadow: KP.SH.near, mis: P.kombu, misOff: [-3, 2], misA: .4 });
      KP.paper(ctx, KP.jitter(KP.egg(0, 0, 52, 52, 0, 40), 2111 + b, .8), P.rice, { shadow: false });
      for (let i = 0; i < 8; i++) { const a = i / 8 * TAU; KP.paper(ctx, KP.jitter([[Math.cos(a) * 6, Math.sin(a) * 6], [Math.cos(a - .32) * 46, Math.sin(a - .32) * 46], [Math.cos(a + .32) * 46, Math.sin(a + .32) * 46]], 2112 + i + b, .6), P.lemon, { shadow: false }); }
    } else if (kind === 'salty') { // salt crystal
      KP.paper(ctx, KP.jitter([[0, -60], [56, -30], [56, 34], [0, 64], [-56, 34], [-56, -30]], 2120 + b, 1), P.rice, { shadow: KP.SH.near, shadowA: .3, mis: P.turq, misOff: [-3, 2], misA: .5 });
      KP.brush(ctx, [[0, 2], [0, 62]], 5, P.turq, { taper: .1 }); KP.brush(ctx, [[0, 2], [-54, -28]], 5, P.turq, { taper: .1 }); KP.brush(ctx, [[0, 2], [54, -28]], 5, P.turq, { taper: .1 });
    } else {                       // coffee bean
      KP.paper(ctx, KP.jitter(KP.rot(KP.egg(0, 0, 44, 62, .08, 32), 0, 0, .4), 2130 + b, 1), P.grape, { shadow: KP.SH.near, mis: P.blue, misOff: [-3, 2], misA: .4 });
      KP.brush(ctx, KP.rot(KP.curve([[0, -50], [10, -10], [-10, 20], [0, 52]], false, 6), 0, 0, .4), 7, P.ink, { taper: .4 });
    }
    ctx.restore();
  }
  function shotTastes(ctx, t, b, T) {
    ctx.fillStyle = P.turq; ctx.fillRect(0, 0, KP.W, KP.H);
    const cx = 490, cy = 800, push = 1 + .12 * KP.ep(t, T.something, T.l3, 'inOut');
    KP.camera(ctx, cx, cy, push, () => {
      for (let i = 0; i < 9; i++) KP.paper(ctx, KP.jitter(KP.rr(-400 + i * 200, -400, 100, 2800, 10, 60), 2140 + i + b, 1.2), KP.mix(P.turq, P.blue, .12), { shadow: false });
      // overhead bowl
      KP.paper(ctx, KP.jitter(KP.ell(cx, cy, 230, 230, 64), 2150 + b, 1.6), P.rice, { shadow: [10, 14], shadowA: .35, mis: P.blue, misOff: [-4, 3], misA: .4 });
      KP.paper(ctx, KP.jitter(KP.ell(cx, cy, 196, 196, 60), 2151 + b, 1.2), P.lemon, { shadow: false });
      KP.brush(ctx, KP.curve([[cx - 120, cy + 30], [cx - 50, cy - 10 + Math.sin(KP.held(t) * 3) * 8], [cx + 30, cy + 40], [cx + 110, cy]], false, 10), 34, P.kombu, { taper: .3 });
      // a ring per taste, carrying its icon, popping on its word
      const ts = [T.sweet, T.sour, T.salty, T.bitter];
      TASTES.forEach(([k, a], i) => {
        if (t < ts[i]) return;
        const p = KP.ease.out(KP.prog(t, ts[i], ts[i] + .5)), r = KP.lerp(200, 300 + i * 34, p);
        const ring = KP.jitter(KP.ell(cx, cy, r, r, 64), 2160 + i + b, 1.2); ring.push(ring[0]);
        KP.brush(ctx, ring, 7, P.rice, { taper: .1 });
        const ar = KP.arrive(t, ts[i] + .05, .3), bob = Math.sin(KP.held(t) * 3 + i) * 6;
        const ix = cx + Math.cos(a) * r, iy = cy + Math.sin(a) * r * 1.05 + bob;
        ctx.save(); KP.pose(ctx, ar, ix, iy, 'pop'); tasteIcon(ctx, k, ix, iy, 1.1, b); ctx.restore();
      });
      // the fifth ring: wide and gold, then Goldie rises from the centre
      if (t >= T.something) {
        const p = KP.ease.out(KP.prog(t, T.something, T.something + .7)), r = KP.lerp(200, 470, p);
        const ring = KP.jitter(KP.ell(cx, cy, r, r, 72), 2170 + b, 1.6); ring.push(ring[0]);
        KP.brush(ctx, ring, 30, P.gold, { taper: .05 });
      }
      if (t >= T.else) {
        const g = KP.arrive(t, T.else, .4), bob = Math.sin(KP.held(t) * 3) * 6;
        ctx.save(); KP.pose(ctx, g, cx, cy + 60, 'up', 120); KP.goldie(ctx, cx, cy - 10 + bob, 150, { boil: b, eyes: 'happy' }); ctx.restore();
        if (t < T.else + .5) KP.marks(ctx, cx, cy - 30, 110, 6, 2180 + b, { a0: -Math.PI, span: Math.PI, len: 26, color: P.rice });
      }
    });
  }
  // ---- shots 4-5: the seaweed mountain and the time-lapse
  function kombuStrip(ctx, x, y, a, len, b, sd) {
    const pts = []; for (let i = 0; i <= 10; i++) { const u = i / 10; pts.push([x + Math.cos(a) * len * (u - .5) - Math.sin(a) * Math.sin(u * 6 + sd) * 16, y + Math.sin(a) * len * (u - .5) + Math.cos(a) * Math.sin(u * 6 + sd) * 16]); }
    KP.brush(ctx, pts.map(([px, py]) => [px + 4, py + 6]), 34, KP.rgba(P.ink, .2), { taper: .3 });
    KP.brush(ctx, pts, 34, sd % 3 ? P.kombu : KP.mix(P.kombu, P.turq, .3), { taper: .3 });
  }
  function shotMountain(ctx, t, b, T, D) {
    const tilt = KP.ep(t, T.l3, T.l3 + 1.6, 'inOut'), boil5 = KP.prog(t, T.s5, T.s5 + 1.2);
    const push = 1 + .35 * KP.ep(t, T.s5, T.l4, 'inOut');
    ctx.fillStyle = P.pink; ctx.fillRect(0, 0, KP.W, KP.H);
    KP.camera(ctx, 490, KP.lerp(560, 860, tilt) + (push - 1) * 300, push, () => {
      for (let i = -3; i < 11; i++) KP.paper(ctx, KP.jitter(KP.rr(-20 + i * 136, -900, 60, 3600, 10, 60), 2200 + i + b, 1.2), KP.mix(P.pink, P.bubble, .4), { shadow: false });
      // the pot (behind the pile)
      const px = 560, py = 1180;
      KP.paper(ctx, KP.jitter(KP.rr(px - 300, py - 380, 600, 400, 30, 30), 2210 + b, 1.6), P.blue, { shadow: [12, 16], shadowA: .35, mis: P.grape, misOff: [-4, 3], misA: .45 });
      for (const sx of [-1, 1]) KP.paper(ctx, KP.jitter(KP.rr(px + sx * 320 - 40, py - 320, 80, 36, 14, 12), 2211 + sx + b, 1), P.grape, { shadow: KP.SH.far });
      // falling strips build the mountain, then it shrinks in steps (stop-motion time-lapse)
      const shrink = 1 - Math.floor(boil5 * 5) / 5;
      const r = KP.rng(2220);
      for (let i = 0; i < 26; i++) {
        const t0 = T.l3 + .1 + i * .07, land = t0 + .55, fx = px - 220 + r() * 440, fa = r() * 3, len = 140 + r() * 80;
        if (t < t0) continue;
        const k = KP.ease.in(KP.prog(t, t0, land)), pileY = py - 400 - (i / 26) * 340 * shrink + Math.abs(fx - px) * .5 * shrink;
        if (shrink <= 0) continue;
        const y = KP.lerp(-700, pileY, k);
        if (t > land && pileY > py - 380 + 10) continue;
        kombuStrip(ctx, KP.lerp(fx, px + (fx - px) * shrink, t > T.s5 ? 1 : 0), t > T.s5 ? KP.lerp(py - 390, pileY, shrink) : y, fa + (1 - k) * 2, len * (t > T.s5 ? Math.max(.3, shrink) : 1), b, i);
      }
      KP.paper(ctx, KP.jitter(KP.rr(px - 320, py - 400, 640, 50, 20, 30), 2212 + b, 1.4), P.grape, { shadow: KP.SH.near, shadowA: .3 });
      // bubbles in the pot rim from 'mountain'
      if (t >= T.mountain) for (let i = 0; i < 6; i++) { const ph = (KP.held(t) * 1.3 + i * .17) % 1; ctx.fillStyle = P.rice; ctx.beginPath(); ctx.arc(px - 240 + i * 96, py - 410 - ph * 60, 14 * (1 - ph), 0, TAU); ctx.fill(); }
      // steam puffs during the time-lapse
      if (t >= T.s5) for (let i = 0; i < 7; i++) {
        const t0 = T.s5 + i * .16, ph = KP.prog(t, t0, t0 + .9); if (ph <= 0 || ph >= 1) continue;
        const x = px - 200 + (i * 73) % 400, y = py - 460 - ph * 520, rr = 50 + ph * 70;
        KP.paper(ctx, KP.jitter(KP.egg(x, y, rr, rr * .8, 0, 28), 2230 + i + b, 2), P.rice, { shadow: KP.SH.far, shadowA: .2 });
      }
      // Ikeda, tiny on a stool
      KP.paper(ctx, KP.jitter(KP.rr(70, 1110, 150, 30, 8, 12), 2240 + b, 1), P.coral, { shadow: KP.SH.far });
      for (const lx of [90, 190]) KP.paper(ctx, KP.jitter(KP.rr(lx - 8, 1140, 16, 140, 6, 12), 2241 + lx + b, 1), P.coral, { shadow: KP.SH.far });
      KP.ikeda(ctx, 145, 1180, .4, { boil: b, armL: .2, armR: t > T.mountain ? -2.2 : -.2, brows: t > T.mountain ? 1 : 0 });
      // the dish of crystals + Goldie + the glutamate diagram (shot 5)
      if (t >= T.s5 + .7) {
        const dA = KP.arrive(t, T.s5 + .7, .35), dx = 560, dy = 1030;
        ctx.save(); KP.pose(ctx, dA, dx, dy + 40, 'up', 260);
        crystals(ctx, dx, dy, 1, b, t, t >= T.s5 + 1.1);
        ctx.restore();
      }
      if (t >= T.glutamic - .1) { const g = KP.arrive(t, T.glutamic - .1, .35); ctx.save(); KP.pose(ctx, g, 700, 640, 'right', 700); KP.glu(ctx, 700, 640, .9, { boil: b }); ctx.restore(); }
    });
    if (t >= T.glutamic) {
      const ar = KP.arrive(t, T.glutamic, .3);
      ctx.save(); KP.pose(ctx, ar, 300, 330, 'pop'); KP.slip(ctx, [{ s: 'Glutamic acid', size: 72, ls: 1 }], 70, 236, { rot: -.03, seed: 52, boil: b, tape: P.turq, padX: 36, padY: 20 }); ctx.restore();
    }
  }
  // a white dish heaped with gold crystals; Goldie peeks out of the heap
  function crystals(ctx, x, y, s, b, t, peek) {
    KP.paper(ctx, KP.jitter(KP.ell(x, y, 170 * s, 40 * s, 40), 2250 + b, 1.2), P.rice, { shadow: KP.SH.near, shadowA: .35, mis: P.turq, misOff: [-3, 2], misA: .4 });
    if (peek) { const pk = KP.ep(t, 0, 1, 'out'); KP.goldie(ctx, x + 30 * s, y - 80 * s + Math.sin(KP.held(t) * 3) * 5, 110 * s, { boil: b, eyes: 'happy', mitts: [.5, .5] }); void pk; }
    const r = KP.rng(2251);
    for (let i = 0; i < 40; i++) { const u = r(), cx = x + (u - .5) * 240 * s * (1 - Math.abs(u - .5)), cy = y - 10 * s - r() * 70 * s * (1 - Math.abs(u - .5) * 1.8), cs = (8 + r() * 6) * s; KP.paper(ctx, KP.rot([[cx - cs, cy - cs], [cx + cs, cy - cs], [cx + cs, cy + cs], [cx - cs, cy + cs]], cx, cy, r() * 3), i % 3 ? P.gold : P.lemon, { shadow: false }); }
  }
  // ---- shots 6-7: umami, sodium, the shaker
  function shotUmami(ctx, t, b, T, D) {
    ctx.fillStyle = P.blue; ctx.fillRect(0, 0, KP.W, KP.H);
    const push = 1 + .08 * KP.ep(t, T.l4, T.l5, 'inOut');
    KP.camera(ctx, 490, 860, push, () => {
      for (let i = -2; i < 10; i++) KP.paper(ctx, KP.jitter(KP.rr(-20 + i * 136, -400, 60, 2800, 10, 60), 2300 + i + b, 1.2), KP.mix(P.blue, P.grape, .3), { shadow: false });
      // the gold ring returns behind the dish on 'umami'
      if (t >= T.umami - .1) { const p = KP.ease.out(KP.prog(t, T.umami - .1, T.umami + .6)); KP.ripple(ctx, 490, 960, 420 * p + 60, { p: 1, boil: b, squash: .4 }); }
      const shot7 = t >= T.l5;
      if (!shot7) {
        const h = KP.ikeda(ctx, 490, 1560, 1.15, { boil: b, armL: -2.0, armR: 2.0, brows: t >= T.umami ? .7 : 0 });
        const hx = (h.handL[0] + h.handR[0]) / 2, hy = (h.handL[1] + h.handR[1]) / 2;
        crystals(ctx, hx, hy - 10, .8, b, t, false);
      }
      // Goldie hops off the dish to meet a sodium ion; they link mittens on 'sodium'
      const hop = KP.ep(t, T.paired - .1, T.paired + .5, 'inOut'), meet = KP.ep(t, T.sodium - .5, T.sodium, 'inOut');
      const into = KP.ep(t, T.mono, T.mono + .55, 'inOut'), into2 = KP.ep(t, T.mono + .2, T.mono + .75, 'inOut');
      const gx0 = KP.lerp(KP.lerp(490, 330, hop), 430, meet), gy0 = KP.lerp(960, 600, hop) - Math.sin(Math.PI * hop) * 160;
      const nx0 = KP.lerp(1180, 560, meet), ny0 = 610;
      const sx = 490, sy = 1150;   // shaker
      if (shot7) {
        const cap = KP.ep(t, T.l5 - .1, T.l5 + .2, 'out') * (1 - KP.ep(t, T.glutamate - .1, T.glutamate + .05, 'in'));
        KP.shaker(ctx, sx, sy, 420, { boil: b, seed: 7, capDy: -220 * cap, capRot: -.3 * cap });
      }
      if (t >= T.paired - .1 && into < 1) {
        const gx = KP.lerp(gx0, sx - 20, into), gy = KP.lerp(gy0, sy - 300, into) - Math.sin(Math.PI * into) * 260;
        KP.goldie(ctx, gx, gy + Math.sin(KP.held(t) * 4) * 5, 130 * (1 - into * .5), { boil: b, eyes: t >= T.sodium ? 'happy' : 'dot', mitts: [t >= T.sodium ? 1.8 : .4, .4] });
      }
      if (t >= T.sodium - .6 && into2 < 1) {
        const nx = KP.lerp(nx0, sx + 20, into2), ny = KP.lerp(ny0, sy - 300, into2) - Math.sin(Math.PI * into2) * 240;
        KP.na(ctx, nx, ny + Math.sin(KP.held(t) * 4 + 1) * 5, 54 * (1 - into2 * .5), { boil: b, arm: t >= T.sodium ? 1 : 0, side: -1, eyes: t >= T.sodium ? 'happy' : 'dot' });
      }
      if (t >= T.sodium && t < T.sodium + .4) KP.marks(ctx, 495, 600, 70, 5, 2310 + b, { a0: -Math.PI, span: Math.PI, len: 22 });
      // shaker front of the cap opening: the two drop in behind its rim
    });
    // labels: 'Umami' (Young Serif), then 'MSG'
    if (t >= T.umami && t < T.paired) {
      const ar = KP.arrive(t, T.umami, .3), out = KP.leave(t, T.paired, .15);
      ctx.save(); ctx.globalAlpha = 1 - out; KP.pose(ctx, ar, 490, 360, 'pop');
      KP.slip(ctx, [{ s: 'Umami', size: 130, font: 'Young Serif', weight: 400, ls: 4 }], 270, 220, { rot: -.03, seed: 53, boil: b, tape: P.pink, padX: 44, padY: 24 });
      ctx.restore();
    }
    if (t >= T.mono) {
      const ar = KP.arrive(t, T.mono, .3);
      ctx.save(); KP.pose(ctx, ar, 490, 330, 'pop');
      KP.slip(ctx, [{ s: 'MSG', size: 130, font: 'Young Serif', weight: 400, ls: 6 }], 330, 220, { rot: .03, seed: 54, boil: b, tape: P.turq, padX: 44, padY: 24 });
      ctx.restore();
    }
  }

  KP.scene('s03_ikeda', {
    sfx(cues, D) {
      const T = times(cues, D), a = [], e = (t, name, gain) => a.push({ t: Math.max(0, +t.toFixed(3)), name, gain });
      e(T.y1908, 'pop', .4); e(T.chemist, 'paper', .3); e(T.ikeda, 'pop_small', .3);
      e(T.noticed, 'slurp', .35); e(T.odd, 'pop_small', .35);
      e(T.sweet, 'pop', .35); e(T.sour, 'pop', .35); e(T.salty, 'pop', .35); e(T.bitter, 'pop', .35); e(T.something, 'sparkle', .4); e(T.else, 'chime', .45);
      e(T.l3 - .3, 'paper', .3); e(T.boiled, 'thud', .45); e(T.mountain, 'bubbling_pot', .35);
      e(T.s5, 'swoosh_down', .35); e(T.s5 + .7, 'sparkle', .4); e(T.glutamic, 'pop', .4);
      e(T.umami, 'impact_soft', .6); e(T.umami + .15, 'chime', .4); e(T.sodium, 'click', .45);
      e(T.mono, 'pop', .4); e(T.glutamate, 'click', .4); e(D - .45, 'whoosh', .35);
      return a.sort((x, y) => x.t - y.t);
    },
    draw(ctx, t, cues, D) {
      const T = times(cues, D), b = KP.boil(t);
      if (t < T.l2) shotLab(ctx, t, b, T);
      else if (t < T.l3) {
        shotTastes(ctx, t, b, T);
        KP.slide(ctx, KP.ep(t, T.l3 - .35, T.l3, 'out'), P.pink, c => shotMountain(c, T.l3, b, T, D));   // paper slide in
      } else if (t < T.l4) shotMountain(ctx, t, b, T, D);
      else shotUmami(ctx, t, b, T, D);
      KP.texture(ctx, 'grain', .1);
      KP.goldIn(ctx, KP.prog(t, 0, .5), 490, 860, b);
      KP.slide(ctx, KP.ep(t, D - .5, D, 'inOut'), KP.GROUND.s04_everywhere);
    },
  });
})();
