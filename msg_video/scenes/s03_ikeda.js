// s03_ikeda: 1908, the discovery of umami (Kombu Paper). Shots s03.01-.05 of design/SHOTLIST.md (138 s cut).
// Kikunae Ikeda is the only named real person drawn: a respectful paper doll (high collar, round glasses, neat
// moustache, period suit), never comedic. All timings come from cues.json words (KP.word).
//   1 lab-kitchen, '1908': Ikeda carries a steaming bowl of seaweed soup to his desk
//   2 he sips; a long ruffled kombu blade hangs over the rim, 'Seaweed'; brows lift, '?' on 'odd'
//   3 overhead: rings spread from the bowl carrying sweet / sour / salty / bitter
//   4 the fifth ring blooms gold, Goldie rises, 'Umami'; tilt down: gold crystals pour into the MSG shaker
//   5 the cap snaps on, a Goldie sticker presses on, the shaker hops, 'MSG'; paper slide out
(function () {
  const P = KP.P, TAU = Math.PI * 2;
  const times = (cues, D) => {
    const w = (re, o) => KP.word(cues, re, o);
    const T = { y1908: w(/1908/, { fb: .55 }), chemist: w(/chemist/i, { fb: 2.7 }), ikeda: w(/ikeda/i, { fb: 4 }) };
    T.noticed = w(/tasted|noticed/i, { fb: T.ikeda + .6 }); T.odd = w(/^odd/i, { fb: T.noticed + 1 }); T.seaweed = w(/seaweed/i, { after: T.odd, fb: T.odd + .6 });
    T.l2 = w(/sweet/i, { line: 1, fb: T.seaweed + 1 }); T.sweet = w(/sweet/i, { fb: T.l2 + .2 }); T.sour = w(/sour/i, { fb: T.sweet + .4 });
    T.salty = w(/salty/i, { fb: T.sour + .4 }); T.bitter = w(/bitter/i, { fb: T.salty + .5 });
    T.he = w(/^(he|it)$/i, { after: T.bitter, fb: T.bitter + .7 }); T.umami = w(/umami/i, { fb: T.he + .7 });
    T.turned = w(/turned|paired/i, { after: T.umami, fb: T.umami + .9 }); T.seasoning = w(/seasoning/i, { after: T.umami, fb: T.turned + 1 });
    T.l5 = w(/monosodium/i, { line: 1, fb: T.seasoning + .8 }); T.mono = w(/monosodium/i, { fb: T.l5 }); T.glutamate = w(/glutamate/i, { after: T.mono, fb: T.mono + 1.1 });
    T.s2 = T.noticed - .1; T.s4 = T.he - .1; T.s5 = T.l5;
    T.else = T.he; T.something = T.he - .05; T.l3 = D + 1;   // aliases used by the ripple shot
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
    if (o.kombu !== false) {
      // a long ruffled kombu blade draped over the rim: wavy-edged ribbon + midrib, holdfast in the broth
      const sw = Math.sin(KP.held(t) * 2.4) * 6 * s, L = [], R = [];
      const path = KP.curve([[x - 30 * s, y - 56 * s], [x + 60 * s, y - 70 * s], [x + 118 * s, y - 40 * s], [x + 130 * s, y + 30 * s + sw], [x + 122 * s, y + 110 * s + sw]], false, 8);
      path.forEach(([px, py], i) => { const u = i / (path.length - 1), wd = (10 + 16 * Math.sin(Math.PI * Math.min(1, u * 1.2))) * s, wv = Math.sin(u * 26 + KP.held(t) * 3) * 4 * s; L.push([px - wd - wv, py]); R.push([px + wd + wv, py]); });
      KP.paper(ctx, KP.jitter(L.concat(R.reverse()), 2003 + b, 1), P.kombu, { shadow: KP.SH.far, shadowA: .3, mis: P.blue, misOff: [-2, 2], misA: .3 });
      KP.brush(ctx, path, 4 * s, KP.mix(P.kombu, P.lemon, .5), { taper: .5 });
      KP.paper(ctx, KP.jitter(KP.egg(x - 34 * s, y - 54 * s, 16 * s, 8 * s, 0, 16), 2004 + b, .6), KP.mix(P.kombu, P.ink, .35), { shadow: false });
    }
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
  function deskProps(ctx, b) {
    // a stack of books and an open notebook on the desk (period lab bench)
    [[P.blue, 70], [P.magenta, 56], [P.kombu, 62]].forEach(([c, h], i) => KP.paper(ctx, KP.jitter(KP.rr(640 + i * 8, 1040 - 58 - i * 58, 240 - i * 20, 54, 8, 14), 2050 + i + b, 1), c, { shadow: KP.SH.far, shadowA: .3 }));
    KP.paper(ctx, KP.jitter([[900, 1040], [1030, 1000], [1160, 1040], [1030, 1060]], 2054 + b, 1), P.rice, { shadow: KP.SH.far, shadowA: .3 });
    for (let i = 0; i < 3; i++) KP.squiggle(ctx, 960 + i * 6, 1020 + i * 6, 1022 + i * 10, 2055 + i, { amp: 2, wl: 10, w: 3 });
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
    const camX = KP.lerp(KP.lerp(ix + 170, 500, KP.ease.inOut(walk) * .3), ix + 60, push), camY = KP.lerp(780, 760, push), sc = KP.lerp(1.2, 1.75, KP.ease.inOut(push));
    KP.camera(ctx, camX + 10, camY, sc, () => {
      lab(ctx, b, t);
      // sip: hands to the mouth on 'noticed', down again before 'odd'
      const sip = KP.ep(t, T.noticed - .1, T.noticed + .25, 'out') * (1 - KP.ep(t, T.odd - .55, T.odd - .2, 'inOut'));
      const carry = !set, arms = carry ? .6 : sip > 0 ? KP.lerp(1.1, 2.35, sip) : 1.1;
      const bowlUp = carry || sip > 0 || t >= T.noticed;
      const lift = t >= T.odd ? KP.backOut(KP.prog(t, T.odd, T.odd + .3)) : 0;
      const h = KP.ikeda(ctx, ix, 1250 - step, 1, { boil: b, armL: -arms, armR: arms, brows: lift, mouth: sip > .6 ? 'o' : 'smile' });
      desk(ctx, b);
      deskProps(ctx, b);
      if (set && !(bowlUp && t >= T.noticed - .1)) bowl(ctx, ix + 60, 1036, 1, b, t);
      if (bowlUp && (carry || t >= T.noticed - .1)) { const bx = (h.handL[0] + h.handR[0]) / 2, by = (h.handL[1] + h.handR[1]) / 2 + 20; bowl(ctx, bx, by, carry ? .9 : 1, b, t); }
      // '?' pops above his head on 'odd'
      if (t >= T.odd) { const q = KP.arrive(t, T.odd, .3); ctx.save(); KP.pose(ctx, q, h.head[0] + 130, h.head[1] - 60, 'pop'); KP.qmark(ctx, h.head[0] + 140, h.head[1] - 140, 150, { rot: .12 + KP.wobA(t, 2, .04) }); ctx.restore(); }
    });
    if (t >= T.seaweed) {
      const ar = KP.arrive(t, T.seaweed, .3);
      ctx.save(); KP.pose(ctx, ar, 760, 1200, 'pop');
      KP.slip(ctx, [{ s: 'Seaweed', size: 76, ls: 1 }], 600, 1120, { rot: .04, seed: 49, boil: b, tape: P.kombu, padX: 34, padY: 18 });
      ctx.restore();
    }
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
  function shotTastes(ctx, t, b, T, D) {
    ctx.fillStyle = P.turq; ctx.fillRect(0, 0, KP.W, KP.H);
    const cx = 490, cy = 800;
    const tilt = KP.ep(t, T.turned, T.seasoning, 'inOut'), push = 1 + .1 * KP.ep(t, T.he, T.turned, 'inOut') * (1 - tilt);
    KP.camera(ctx, cx, KP.lerp(cy, 1330, tilt), push, () => {
      for (let i = 0; i < 9; i++) KP.paper(ctx, KP.jitter(KP.rr(-400 + i * 200, -400, 100, 3200, 10, 60), 2140 + i + b, 1.2), KP.mix(P.turq, P.blue, .12), { shadow: false });
      KP.paper(ctx, KP.jitter(KP.ell(cx, cy, 230, 230, 64), 2150 + b, 1.6), P.rice, { shadow: [10, 14], shadowA: .35, mis: P.blue, misOff: [-4, 3], misA: .4 });
      KP.paper(ctx, KP.jitter(KP.ell(cx, cy, 196, 196, 60), 2151 + b, 1.2), P.lemon, { shadow: false });
      KP.brush(ctx, KP.curve([[cx - 120, cy + 30], [cx - 50, cy - 10 + Math.sin(KP.held(t) * 3) * 8], [cx + 30, cy + 40], [cx + 110, cy]], false, 10), 34, P.kombu, { taper: .3 });
      const ts = [T.sweet, T.sour, T.salty, T.bitter];
      TASTES.forEach(([k, a], i) => {
        if (t < ts[i]) return;
        const p = KP.ease.out(KP.prog(t, ts[i], ts[i] + .45)), r = KP.lerp(200, 300 + i * 34, p);
        const ring = KP.jitter(KP.ell(cx, cy, r, r, 64), 2160 + i + b, 1.2); ring.push(ring[0]);
        KP.brush(ctx, ring, 7, P.rice, { taper: .1 });
        const ar = KP.arrive(t, ts[i] + .05, .28), bob = Math.sin(KP.held(t) * 3 + i) * 6;
        const ix = cx + Math.cos(a) * r, iy = cy + Math.sin(a) * r * 1.05 + bob;
        ctx.save(); KP.pose(ctx, ar, ix, iy, 'pop'); tasteIcon(ctx, k, ix, iy, 1.1, b); ctx.restore();
      });
      // the fifth ring blooms wide and gold; Goldie rises from the centre
      if (t >= T.he) {
        const p = KP.ease.out(KP.prog(t, T.he, T.he + .7)), r = KP.lerp(200, 470, p);
        const ring = KP.jitter(KP.ell(cx, cy, r, r, 72), 2170 + b, 1.6); ring.push(ring[0]);
        KP.brush(ctx, ring, 30, P.gold, { taper: .05 });
        const g = KP.arrive(t, T.he + .2, .4), bob = Math.sin(KP.held(t) * 3) * 6;
        ctx.save(); KP.pose(ctx, g, cx, cy + 60, 'up', 120); KP.goldie(ctx, cx, cy - 10 + bob, 150, { boil: b, eyes: 'happy' }); ctx.restore();
      }
      // the MSG shaker below the bowl; on 'seasoning' gold crystals pour from the ring into it
      const sx = 490, sy = 1720, cap = t < T.mono ? 1 : 1 - KP.ep(t, T.mono - .05, T.mono + .12, 'in');
      const hop = t >= T.glutamate + .25 ? Math.pow(Math.sin(Math.PI * KP.prog(t, T.glutamate + .25, T.glutamate + .65)), 1) * 60 : 0;
      if (t >= T.turned - .2) {
        ctx.save(); ctx.translate(0, -hop);
        KP.shaker(ctx, sx, sy, 400, { boil: b, seed: 7, capDx: 190 * cap, capDy: -90 * cap, capRot: .5 * cap });
        // Goldie sticker presses onto the shaker on 'glutamate'
        if (t >= T.glutamate) { const st = KP.arrive(t, T.glutamate, .22); ctx.save(); KP.pose(ctx, st, sx, sy - 120, 'pop');
          KP.paper(ctx, KP.jitter(KP.egg(sx, sy - 150, 62, 62, 0, 32), 2190 + b, 1), P.lemon, { shadow: KP.SH.far, shadowA: .3, mis: P.tomato, misOff: [-2, 2], misA: .4 });
          KP.goldie(ctx, sx, sy - 146, 80, { boil: b, eyes: 'happy', steam: false, shadow: false }); ctx.restore(); }
        ctx.restore();
      }
      if (t >= T.seasoning - .1 && t < T.mono + .1) {
        const r = KP.rng(2195);
        for (let i = 0; i < 30; i++) { const ph = ((t - T.seasoning) * 1.6 + r()) % 1, x = sx + (r() - .5) * 120 * (1 - ph), y = KP.lerp(1240, sy - 270, ph), cs = 8 + r() * 6; KP.paper(ctx, KP.rot([[x - cs, y - cs], [x + cs, y - cs], [x + cs, y + cs], [x - cs, y + cs]], x, y, r() * 3 + ph * 4), i % 3 ? P.gold : P.lemon, { shadow: false }); }
      }
    });
    // labels: 'Umami' (Young Serif) until the shaker shot, then 'MSG'
    if (t >= T.umami && t < T.s5) {
      const ar = KP.arrive(t, T.umami, .3), out = KP.leave(t, T.s5, .12);
      ctx.save(); ctx.globalAlpha = 1 - out; KP.pose(ctx, ar, 490, 340, 'pop');
      KP.slip(ctx, [{ s: 'Umami', size: 130, font: 'Young Serif', weight: 400, ls: 4 }], 270, 220, { rot: -.03, seed: 53, boil: b, tape: P.pink, padX: 44, padY: 24 });
      ctx.restore();
    }
    if (t >= T.mono) {
      const ar = KP.arrive(t, T.mono, .3);
      ctx.save(); KP.pose(ctx, ar, 490, 340, 'pop');
      KP.slip(ctx, [{ s: 'MSG', size: 130, font: 'Young Serif', weight: 400, ls: 6 }], 330, 220, { rot: .03, seed: 54, boil: b, tape: P.turq, padX: 44, padY: 24 });
      ctx.restore();
    }
  }

  KP.scene('s03_ikeda', {
    sfx(cues, D) {
      const T = times(cues, D), a = [], e = (t, name, gain) => a.push({ t: Math.max(0, +t.toFixed(3)), name, gain });
      e(T.y1908, 'pop', .4); e(T.chemist, 'paper', .3); e(T.ikeda, 'pop_small', .3);
      e(T.noticed, 'slurp', .35); e(T.odd, 'pop_small', .35); e(T.seaweed, 'bubble', .3);
      e(T.sweet, 'pop', .35); e(T.sour, 'pop', .35); e(T.salty, 'pop', .35); e(T.bitter, 'pop', .35);
      e(T.umami, 'impact_soft', .6); e(T.umami + .15, 'chime', .4); e(T.turned, 'swoosh_down', .35); e(T.seasoning, 'pop', .35);
      e(T.mono, 'click', .4); e(T.glutamate, 'pop', .4); e(D - .45, 'whoosh', .35);
      return a.sort((x, y) => x.t - y.t);
    },
    draw(ctx, t, cues, D) {
      const T = times(cues, D), b = KP.boil(t);
      if (t < T.l2) shotLab(ctx, t, b, T);
      else shotTastes(ctx, t, b, T, D);
      KP.texture(ctx, 'grain', .1);
      // in: revealed by s02's Gold Ripple wipe (frame 0 is the iris's final frame)
      KP.slideTo(ctx, KP.ep(t, D - .5, D, 'inOut'), 's04_everywhere');   // slide carrying s04's first frame
    },
  });
})();
