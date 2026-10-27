// s09_outro: So... is MSG bad for you? (Kombu Paper). Shots s09.01-.05 of design/SHOTLIST.md.
// All timings come from cues.json words (KP.word). The LAST FRAME is s01's frame 0 (nested draw of s01 at t=0), so the
// Reel loops: the ramen end card pulls back into the hero TV of the s01 wall, a channel flicker, then s01 frame 0.
//   1 TV-wall callback: every screen shows vivid riso static; the shaker in front with a '?'; on 'Not really' the '?'
//     flips away and Goldie pops out of the shaker smiling ('Not really')
//   2 three weights drop onto the shaker on their phrases: ink '?' (guess), mini stamp (catchy name), red tag (prejudice)
//   3 pull back: the TVs switch off one by one, a tall stack of vivid research papers rises, the weights slide off
//   4 the NO MSG sign: 'NO' flickers and burns out on 'sign...', leaving 'MSG' lit gold
//   5 lamp-lit kitchen: ramen bowl, a mitten hand shakes the shaker, one gold drop -> the Fifth Ripple + 'MSG' slip;
//     pull back into the hero TV of the s01 wall (exact s01 frame 0 layout); channel flicker; loop
(function () {
  const P = KP.P, TAU = Math.PI * 2;
  const times = (cues, D) => {
    const w = (re, o) => KP.word(cues, re, o);
    const T = { msg1: w(/msg/i, { fb: .9 }), not: w(/^not$/i, { fb: 2.66 }), really: w(/really/i, { fb: 2.85 }) };
    T.s2 = w(/guess/i, { line: 1, fb: T.really + .9 }); T.guess = w(/guess/i, { fb: T.s2 + .25 }); T.catchy = w(/catchy/i, { fb: T.guess + .65 });
    T.prej = w(/prejudice/i, { fb: T.catchy + 1.6 }); T.outlived = w(/outlived/i, { fb: T.prej + .6 }); T.s3 = T.outlived - .15;
    T.decades = w(/decades/i, { fb: T.outlived + .6 }); T.of = w(/^of$/i, { after: T.decades, fb: T.decades + .6 }); T.science = w(/science/i, { fb: T.of + .2 });
    T.s4 = w(/^so$/i, { after: T.science, line: 1, fb: T.science + .6 }); T.no = w(/^no$/i, { after: T.s4, fb: T.s4 + 1.2 }); T.sign = w(/sign/i, { after: T.no, fb: T.no + .8 });
    T.reach = w(/reach/i, { after: T.sign, fb: T.sign + .6 }); T.s5 = T.reach - .1; T.shaker = w(/shaker/i, { after: T.reach, fb: T.reach + .6 });
    T.drop = T.shaker + .4; T.pull = Math.max(T.drop + .7, D - 1.05); T.lastF = (Math.round(D * 30) - 1) / 30; T.flick = T.lastF - 8 / 30; T.clean = T.lastF - 4 / 30;
    return T;
  };
  const s01 = () => KZ.scenes.s01_coldopen;
  const SHK = [490, 1330];
  // a TV screen switching off: static -> white band collapses -> dot -> dark (p 0..1)
  function offScreen(ctx, S, p, f) {
    // calm colour bars (no harsh static) -> a white band collapses to a dot -> dark glass with a glint + power dot
    if (p <= 0) { KP.colorBars(ctx, S, f); return; }
    if (p >= .85) { KP.screenOff(ctx, S); return; }
    ctx.save(); ctx.clip(KP.path(KP.rr(S.x, S.y, S.w, S.h, 20)));
    ctx.fillStyle = P.screenDim; ctx.fillRect(S.x, S.y, S.w, S.h);
    const cx = S.x + S.w / 2, cy = S.y + S.h / 2;
    if (p < .5) { const h = S.h * (1 - p * 2) + 6; ctx.fillStyle = P.rice; ctx.fillRect(S.x, cy - h / 2, S.w, h); }
    else { const k = 1 - (p - .5) / .35; ctx.fillStyle = P.rice; ctx.fillRect(cx - S.w / 2 * k, cy - 3, S.w * k, 6); ctx.beginPath(); ctx.arc(cx, cy, 8, 0, TAU); ctx.fill(); }
    ctx.restore();
  }
  function table(ctx, b) {
    KP.paper(ctx, KP.jitter(KP.rr(SHK[0] - 34, SHK[1] + 20, 68, 380, 10, 20), 5500 + b, 1.2), KP.mix(P.turq, P.ink, .3), { shadow: KP.SH.near, shadowA: .4 });
    KP.paper(ctx, KP.jitter(KP.ell(SHK[0], SHK[1] + 6, 230, 46, 44), 5501 + b, 1.5), P.turq, { shadow: [10, 14], shadowA: .45, mis: P.blue, misOff: [-4, 3], misA: .45 });
  }
  // shots 1-3: the wall + shaker
  function shotWall(ctx, t, b, T) {
    const api = s01().api, f = Math.floor(t * 30);
    const closeK = KP.ep(t, T.s2 - .3, T.s2 + .2, 'inOut') * (1 - KP.ep(t, T.s3, T.s3 + 1, 'inOut'));
    const sc = KP.lerp(.8 + .05 * KP.prog(t, 0, T.s2), 1.45, closeK), cy = KP.lerp(880, SHK[1] - 300, closeK), cx = KP.lerp(500, SHK[0], closeK);
    // TV switch-off order on 'outlived' / 'decades' / 'of' + the rest staggered after
    const offT = i => { const base = [T.outlived, T.decades, T.of][i % 3]; return base + Math.floor(i / 3) * .08; };
    const offP = i => KP.prog(t, offT(i), offT(i) + .35);
    KP.camera(ctx, cx, cy, sc, () => {
      api.wall(ctx, t, b, {
        extra: true, cues: [], lit: () => 1, heroScreen: P.screenDim,
        litDraw: (c, S, i) => offScreen(c, S, offP(i + 1), f + i * 7),
        hero: (c, S) => offScreen(c, S, offP(0), f),
      });
      // the research papers stack rises at the right (shot 3)
      const rise = KP.ep(t, T.s3, T.science + .3, 'out');
      if (rise > 0) { const cols = [P.lemon, P.pink, P.turq, P.coral, P.rice, P.grape, P.lemon, P.kombu, P.bubble, P.blue];
        cols.forEach((c, i) => { const y = 1400 - i * 70 * rise; if (i / cols.length > rise) return;
          KP.paper(ctx, KP.jitter(KP.rr(820 + (i % 3 - 1) * 14, y - 60, 280, 60, 6, 16), 5510 + i + b, 1.2), c, { shadow: KP.SH.near, shadowA: .3 });
          KP.squiggle(ctx, 850 + (i % 3 - 1) * 14, 1040, y - 30, 5520 + i, { amp: 3, wl: 14, w: 5, color: c === P.rice ? P.blue : P.rice }); }); }
      table(ctx, b);
      // weights on the shaker (shot 2), sliding off in shot 3
      const press = [T.guess, T.catchy, T.prej].reduce((a, ts) => a + (t >= ts && t < ts + .18 ? 1 : 0), 0);
      const slide = KP.ep(t, T.s3 + .2, T.s3 + .8, 'in');
      ctx.save(); if (press) { ctx.translate(SHK[0], SHK[1]); ctx.scale(1.06, .9); ctx.translate(-SHK[0], -SHK[1]); }
      KP.shaker(ctx, SHK[0], SHK[1], 330, { boil: b, seed: 7, shadowA: .4 });
      ctx.restore();
      const top = SHK[1] - 330 * .95;
      const wts = [[T.guess, c => KP.qmark(c, -150, -40, 150, { rot: -.15 })],
        [T.catchy, c => { KP.paper(c, KP.jitter(KP.rr(-210, -160, 420, 170, 10, 16), 5535 + b, 1.2), P.rice, { shadow: KP.SH.near, shadowA: .35 }); const st = KP.stamp('crs-mini', [String.fromCharCode(0x201C) + 'CHINESE-RESTAURANT', 'SYNDROME' + String.fromCharCode(0x201D)], 400, 150, { size: 32, ls: 1, seed: 56 }); c.drawImage(st, -200, -150); }],
        [T.prej, c => { KP.brush(c, [[-60, -150], [-10, -60]], 4, P.ink, { taper: .1 }); KP.paper(c, KP.jitter([[-120, -150], [40, -150], [70, -110], [40, -70], [-120, -70]], 5530 + b, 1), P.tomato, { shadow: KP.SH.near, shadowA: .35, mis: P.pink, misOff: [-3, 2], misA: .45 }); c.fillStyle = P.rice; c.beginPath(); c.arc(30, -110, 9, 0, TAU); c.fill(); }]];
      wts.forEach(([ts, fn], i) => {
        if (t < ts) return;
        const fall = KP.ease.in(KP.prog(t, ts - .25, ts)), y = KP.lerp(top - 900, top - i * 95, fall);
        ctx.save(); ctx.translate(SHK[0] + (i - 1) * 40 + slide * (i - 1 || 1) * 700, y + slide * 300); ctx.rotate(slide * (i - 1 || 1) * 1.5); fn(ctx); ctx.restore();
      });
      // shot 1: the '?' above the shaker; it flips away on 'Not' and Goldie pops out smiling
      if (t < T.s2) {
        const qa = KP.arrive(t, T.msg1, .3), flip = KP.prog(t, T.not - .05, T.not + .2);
        if (qa.on && flip < 1) { ctx.save(); ctx.translate(SHK[0], SHK[1] - 520); ctx.scale(Math.cos(flip * Math.PI / 2) * qa.p, qa.p); KP.qmark(ctx, 0, 0, 220, { rot: .08 + Math.sin(KP.held(t) * 2) * .06 }); ctx.restore(); }
        if (t >= T.not) { const g = KP.arrive(t, T.not + .1, .35), bob = Math.sin(KP.held(t) * 4) * 8; ctx.save(); KP.pose(ctx, g, SHK[0], top, 'up', 160); KP.goldie(ctx, SHK[0], top - 110 + bob, 170, { boil: b, eyes: 'happy', mitts: [2.2, 2.2] }); ctx.restore();
          if (t < T.not + .6) KP.marks(ctx, SHK[0], top - 110, 140, 7, 5540 + b, { a0: -Math.PI, span: Math.PI, len: 26 }); }
      }
    });
    if (t >= T.not && t < T.s2 + .1) {
      const ar = KP.arrive(t, T.not, .3), out = KP.leave(t, T.s2 + .1, .15);
      ctx.save(); ctx.globalAlpha = out > 0 ? 0 : 1; KP.pose(ctx, ar, 490, 330, 'pop');
      KP.slip(ctx, [{ s: 'Not really', size: 96, font: 'Young Serif', weight: 400, ls: 2 }], 200, 230, { rot: -.03, seed: 90, boil: b, tape: P.kombu, padX: 40, padY: 22 });
      ctx.restore();
    }
  }
  // shot 4: the NO MSG sign; 'NO' flickers then burns out on 'sign...'
  function shotSign(ctx, t, b, T) {
    const push = 1 + .12 * KP.ep(t, T.s4, T.s5, 'inOut');
    ctx.fillStyle = P.blue; ctx.fillRect(0, 0, KP.W, KP.H);
    KP.camera(ctx, 490, 760, push, () => {
      for (let i = -2; i < 10; i++) KP.paper(ctx, KP.jitter(KP.rr(-20 + i * 136, -400, 60, 2800, 10, 60), 5600 + i + b, 1.2), KP.mix(P.blue, P.grape, .22), { shadow: false });
      const wx = 70, wy = 380, ww = 840, wh = 760;
      KP.paper(ctx, KP.jitter(KP.rr(wx, wy, ww, wh, 18, 24), 5610 + b, 1.5), P.lemon, { shadow: [10, 14], shadowA: .4 });
      KP.paper(ctx, KP.jitter(KP.rr(wx + 26, wy + 26, ww - 52, wh - 52, 10, 24), 5611 + b, 1.2), P.plum, { shadow: false });
      ctx.save(); ctx.clip(KP.path(KP.rr(wx + 26, wy + 26, ww - 52, wh - 52, 10, 24)));
      for (let i = 0; i < 5; i++) KP.paper(ctx, KP.jitter(KP.rr(wx + 50 + i * 160, wy + 470 - (i % 2) * 80, 130, 400, 6, 20), 5620 + i, 1), P.screenDim, { shadow: false });
      const flickering = t >= T.no && t < T.sign, noOn = t < T.sign && !(flickering && KP.hash(Math.floor(t * 15), 3) < .45);
      KP.noMsgSign(ctx, wx + ww / 2, wy + wh / 2 - 20, 640, { boil: b, bulbs: 1, no: noOn ? 1 : 0, msg: 1, flick: flickering ? KP.boil(t) + 1 : 0, letter: t >= T.sign ? P.gold : P.lemon, dim: t >= T.sign ? KP.mix(P.magenta, P.ink, .35) : undefined });
      if (t >= T.sign && t < T.sign + .5) { KP.marks(ctx, wx + ww / 2, wy + wh / 2 - 130, 60, 5, 5630 + b, { a0: -Math.PI, span: Math.PI, len: 22, color: P.lemon }); }
      if (t >= T.sign) { const k = KP.prog(t, T.sign, T.sign + .8); for (let i = 0; i < 3; i++) { const ph = (k + i * .3) % 1; KP.paper(ctx, KP.jitter(KP.egg(wx + ww / 2 - 40 + i * 40, wy + wh / 2 - 140 - ph * 160, 16 + ph * 20, 12 + ph * 14, 0, 16), 5640 + i + b, 1.2), P.rice, { shadow: false }); } }  // little smoke puffs
      ctx.restore();
      KP.paper(ctx, KP.jitter(KP.rr(wx - 20, wy + wh - 10, ww + 40, 36, 8, 20), 5612 + b, 1.2), P.coral, { shadow: KP.SH.near, shadowA: .35 });
    });
  }
  // shot 5 contents (drawn into its own canvas so it can sit inside the hero TV during the pull-back)
  const ramenCanvas = (() => { let c; return () => c || (c = Object.assign(document.createElement('canvas'), { width: KP.W, height: KP.H })); })();
  function ramen(ctx, t, b, T) {
    const C = [540, 960];
    ctx.fillStyle = P.coral; ctx.fillRect(0, 0, KP.W, KP.H);
    for (let i = -1; i < 10; i++) KP.paper(ctx, KP.jitter(KP.rr(-20 + i * 136, -40, 60, 2000, 10, 60), 5700 + i + b, 1.2), KP.mix(P.coral, P.pink, .35), { shadow: false });
    // lamp wash: flat lemon paper pool on the table (no glow)
    KP.paper(ctx, KP.jitter(KP.ell(C[0], C[1] + 60, 470, 260, 60), 5701 + b, 2), P.lemon, { shadow: false });
    // bowl (side-on) + broth + noodles + egg + kombu
    KP.paper(ctx, KP.jitter(KP.curve([[C[0] - 230, C[1] - 20], [C[0] + 230, C[1] - 20], [C[0] + 190, C[1] + 110], [C[0] + 90, C[1] + 170], [C[0] - 90, C[1] + 170], [C[0] - 190, C[1] + 110]], true, 8), 5710 + b, 1.6), P.blue, { shadow: [12, 16], shadowA: .35, mis: P.grape, misOff: [-4, 3], misA: .4 });
    for (let i = 0; i < 5; i++) KP.paper(ctx, KP.jitter(KP.egg(C[0] - 150 + i * 75, C[1] + 80 - Math.abs(i - 2) * 12, 18, 18, 0, 16), 5711 + i + b, .8), P.lemon, { shadow: false });
    KP.paper(ctx, KP.jitter(KP.ell(C[0], C[1] - 20, 230, 64, 48), 5712 + b, 1.4), P.rice, { shadow: false });
    KP.paper(ctx, KP.jitter(KP.ell(C[0], C[1] - 20, 206, 52, 48), 5713 + b, 1.2), P.turq, { shadow: false });
    for (let i = 0; i < 5; i++) KP.brush(ctx, KP.curve([[C[0] - 160 + i * 18, C[1] - 6], [C[0] - 120 + i * 18, C[1] - 40], [C[0] - 70 + i * 18, C[1] - 4], [C[0] - 30 + i * 18, C[1] - 36]], false, 6), 9, P.lemon, { taper: .2 });
    KP.paper(ctx, KP.jitter(KP.egg(C[0] + 100, C[1] - 30, 46, 30, 0, 24), 5714 + b, 1), P.rice, { shadow: KP.SH.far, shadowA: .25 });
    KP.paper(ctx, KP.jitter(KP.egg(C[0] + 100, C[1] - 30, 22, 15, 0, 20), 5715 + b, .8), P.gold, { shadow: false });
    KP.brush(ctx, KP.curve([[C[0] + 20, C[1] - 50], [C[0] + 60, C[1] - 64], [C[0] + 40, C[1] - 30]], false, 6), 16, P.kombu, { taper: .3 });
    // the Fifth Ripple when the drop lands
    if (t >= T.drop) KP.ripple(ctx, C[0] - 40, C[1] - 22, 190, { p: KP.ease.out(KP.prog(t, T.drop, T.drop + .8)), boil: b, squash: .26 });
    for (let i = 0; i < 3; i++) { const sx = C[0] - 100 + i * 100, ph = KP.held(t) * 2.5 + i; KP.brush(ctx, KP.curve([[sx, C[1] - 90], [sx + 14 * Math.sin(ph), C[1] - 150], [sx - 10 * Math.sin(ph), C[1] - 210]], false, 6), 9, P.rice, { taper: .7 }); }
    // the mitten hand reaches in from the right, takes the shaker, shakes; one gold drop falls
    const reach = KP.ep(t, T.s5, T.reach + .45, 'out'), lift = KP.ep(t, T.reach + .45, T.shaker, 'inOut');
    const shake = t >= T.shaker && t < T.drop + .2 ? Math.sin(KP.held(t) * 40) * .3 : 0;
    const sx = KP.lerp(840, C[0] - 40, lift), sy = KP.lerp(C[1] + 40, C[1] - 260, lift);
    ctx.save(); ctx.translate(sx, sy); ctx.rotate(KP.lerp(0, Math.PI * .85, lift) + shake);
    KP.shaker(ctx, 0, 0, 220, { boil: b, seed: 7 });
    ctx.restore();
    const hx = KP.lerp(1300, sx + 70, reach), hy = KP.lerp(C[1] + 200, sy - 40, reach);
    KP.brush(ctx, [[1200, hy + 220], [hx + 40, hy + 20]], 90, P.pink, { taper: .05 });
    KP.paper(ctx, KP.jitter(KP.egg(hx, hy, 70, 60, 0, 28), 5720 + b, 1.2), P.bubble, { shadow: KP.SH.near, shadowA: .3, mis: P.blue, misOff: [2, -2], misA: .3 });
    KP.paper(ctx, KP.jitter(KP.egg(hx - 56, hy - 24, 20, 30, 0, 18), 5721 + b, .8), P.bubble, { shadow: false });
    if (t >= T.shaker && t < T.drop) { const k = KP.ease.in(KP.prog(t, T.shaker + .1, T.drop)); KP.paper(ctx, KP.jitter(KP.drop(C[0] - 40, KP.lerp(sy + 120, C[1] - 40, k), 50), 5730 + b, 1), P.gold, { shadow: KP.SH.far, mis: P.tomato, misOff: [-2, 2], misA: .4 }); }
    // the end card label: 'MSG' slip on the ripple
    if (t >= T.shaker) { const ar = KP.arrive(t, T.shaker, .3); ctx.save(); KP.pose(ctx, ar, C[0] + 180, C[1] - 250, 'pop');
      KP.slip(ctx, [{ s: 'MSG', size: 100, font: 'Young Serif', weight: 400, ls: 6 }], C[0] + 40, C[1] - 330, { rot: -.03, seed: 91, boil: b, tape: P.turq, padX: 40, padY: 20 }); ctx.restore(); }
    KP.texture(ctx, 'grain', .1);
  }
  const HS = () => { const [x, y, w, h] = s01().api.HERO; return KP.tvScreen(x, y, w, h, { panel: 'bottom' }); };
  function shotEnd(ctx, t, b, T, D) {
    const rc = ramenCanvas(), g = rc.getContext('2d');
    g.setTransform(1, 0, 0, 1, 0, 0); g.clearRect(0, 0, KP.W, KP.H);
    ramen(g, t, b, T);
    const S = s01().api.screen0(), coverK = Math.max(S.w / KP.W, S.h / KP.H);   // hero screen rect in s01 frame 0 (frame coords)
    const zEnd = 1 / coverK;                                           // ramen push-in ends where the pull-back starts
    if (t < T.pull) {
      const z = KP.lerp(1, zEnd, KP.ep(t, T.s5, T.pull, 'inOut'));
      ctx.save(); ctx.translate(KP.W / 2, KP.H / 2); ctx.scale(z, z); ctx.translate(-KP.W / 2, -KP.H / 2); ctx.drawImage(rc, 0, 0); ctx.restore();
      return;
    }
    // pull-back: s01 frame 0 (nested) under a zoom Z that maps the hero screen centre to the frame centre
    const p = KP.ep(t, T.pull, T.flick, 'inOut'), Z = KP.lerp(zEnd * coverK, 1, p);
    const scx = S.x + S.w / 2, scy = S.y + S.h / 2;
    const ox = KP.lerp(KP.W / 2, scx, p), oy = KP.lerp(KP.H / 2, scy, p);
    const cues01 = window.CUES && window.CUES.s01_coldopen;
    ctx.save(); ctx.translate(ox, oy); ctx.scale(Z, Z); ctx.translate(-scx, -scy);
    const sc = s01(), clean = t >= T.clean - 1e-6;   // the last 4 frames are exactly s01 frame 0
    sc.hideLow = t < T.clean - 1e-6;   // tag only in the final frames, when the screen shows his clip frame (= s01 frame 0)                 // credit tag + shelf only once the bowl is gone and the TV is in frame
    sc.heroOverride = clean ? null : (c, SS) => {
      if (t >= T.flick) { KP.colorBars(c, SS, Math.floor(t * 30)); return; }
      const k = Math.max(SS.w / KP.W, SS.h / KP.H), dw = KP.W * k, dh = KP.H * k;
      c.drawImage(rc, SS.x + (SS.w - dw) / 2, SS.y + (SS.h - dh) / 2, dw, dh);
    };
    try { sc.draw(ctx, 0, cues01 ? cues01.lines : [], cues01 ? cues01.duration : 21); } finally { sc.heroOverride = null; sc.hideLow = false; }
    ctx.restore();
  }

  KP.scene('s09_outro', {
    sfx(cues, D) {
      const T = times(cues, D), a = [], e = (t, name, gain) => a.push({ t: Math.max(0, +t.toFixed(3)), name, gain });
      e(.2, 'static', .25); e(T.msg1, 'pop', .35); e(T.not, 'boing', .35); e(T.really, 'sparkle', .35);
      e(T.guess, 'thud', .35); e(T.catchy, 'stamp', .35); e(T.prej, 'thud', .4);
      e(T.outlived, 'tv_off', .45); e(T.decades, 'tv_off', .45); e(T.of, 'tv_off', .45); e(T.science, 'paper', .4);
      e(T.s4 + .1, 'neon_buzz', .3); e(T.no, 'neon_flicker', .45); e(T.sign, 'click', .45); e(T.sign + .3, 'sparkle', .35);
      e(T.reach, 'pop_small', .35); e(T.shaker, 'pop_small', .35); e(T.drop, 'bubble', .4); e(T.drop + .2, 'chime', .4); e(T.flick, 'channel_switch', .45);
      return a.sort((x, y) => x.t - y.t);
    },
    draw(ctx, t, cues, D) {
      const T = times(cues, D), b = KP.boil(t);
      if (t < T.s4) shotWall(ctx, t, b, T);
      else if (t < T.s5) shotSign(ctx, t, b, T);
      else { shotEnd(ctx, t, b, T, D); return; }   // the end shot carries its own grain (and s01's on the last frame)
      KP.texture(ctx, 'grain', .1);
      KP.texture(ctx, 'speck', .2, 17, 41);
      if (t < T.s4) KP.slide(ctx, KP.ep(t, T.s4 - .35, T.s4, 'out'), P.blue, c => shotSign(c, T.s4, b, T));   // paper slide to the sign
    },
  });
})();
