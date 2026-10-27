// s04_everywhere: Glutamate is everywhere (Kombu Paper). Shots s04.01-.05 of design/SHOTLIST.md.
// All timings come from cues.json words (KP.word).
//   1 'alarmingly chemical': lab flasks sprout from the shaker, Goldie looks nervous, 'Chemical?' warning card flaps
//   2 the flasks fall away; zoom out: the shaker stands on a paper Earth where Goldies pop up all over
//   3 three foods stacked: tomato, parmesan, milk drop + bottle, each with a mini Goldie peeking out
//   4 split: a tomato releases a Goldie / the shaker shakes one out; each beside the same skeletal diagram
//   5 the divider tears away; the two Goldies and diagrams slide together and overlap; '=' + 'Same molecule'
(function () {
  const P = KP.P, TAU = Math.PI * 2;
  const times = (cues, D) => {
    const w = (re, o) => KP.word(cues, re, o);
    const T = { sounds: w(/sounds/i, { fb: .55 }), alarm: w(/alarming/i, { fb: .75 }), chemical: w(/chemical/i, { fb: 1.45 }) };
    T.s2 = T.chemical + .9;
    T.glu1 = w(/glutamate/i, { after: T.chemical, fb: T.s2 + .7 }); T.every = w(/everywhere/i, { fb: T.glu1 + .7 });
    T.tomatoes = w(/tomatoes/i, { fb: T.every + .9 }); T.s3 = T.tomatoes - .1;
    T.parm = w(/parmesan/i, { fb: T.tomatoes + .8 }); T.milk = w(/milk/i, { fb: T.parm + 1.3 });
    T.s4 = w(/body/i, { line: 1, fb: T.milk + .9 });
    T.glu2 = w(/glutamate/i, { after: T.s4, fb: T.s4 + 1 }); T.tomato = w(/^tomato/i, { after: T.s4, fb: T.glu2 + .9 });
    T.msg = w(/msg/i, { after: T.tomato, fb: T.tomato + .9 }); T.s5 = w(/^is$/i, { after: T.msg, fb: T.msg + .5 }) - .1;
    T.exact = w(/exact/i, { after: T.msg, fb: T.s5 + .45 }); T.same = w(/same/i, { after: T.msg, fb: T.exact + .4 });
    T.molecule = w(/molecule/i, { after: T.same - .01, fb: T.same + .2 }); T.wink = T.molecule + .6;
    return T;
  };
  function stars(ctx, b) {
    const r = KP.rng(4000);
    for (let i = 0; i < 40; i++) { const x = r() * 2200 - 600, y = r() * 3200 - 1000, s = 6 + r() * 10; KP.paper(ctx, KP.jitter([[x, y - s], [x + s * .3, y - s * .3], [x + s, y], [x + s * .3, y + s * .3], [x, y + s], [x - s * .3, y + s * .3], [x - s, y], [x - s * .3, y - s * .3]], 4001 + i + (b % 2), .4), i % 3 ? P.lemon : P.rice, { shadow: false }); }
  }
  // ---- shots 1-2: one world. The shaker stands on top of a paper Earth; shot 1 is a 3x close-up of it.
  const EC = [490, 900], ER = 380, SB = [490, EC[1] - ER + 6];   // Earth centre/radius, shaker base
  function earth(ctx, t, b, T) {
    KP.paper(ctx, KP.jitter(KP.egg(EC[0], EC[1], ER, ER, 0, 80), 4010 + b, 2), P.turq, { shadow: [16, 20], shadowA: .3, mis: P.blue, misOff: [-6, 4], misA: .45 });
    // continents (generic blobs, no real geography) + a paper cloud band
    [[-150, -120, 120, 90], [120, -40, 150, 110], [-60, 170, 130, 80], [220, 210, 70, 60]].forEach(([dx, dy, rx, ry], i) =>
      KP.paper(ctx, KP.jitter(KP.egg(EC[0] + dx, EC[1] + dy, rx, ry, .1, 36), 4020 + i + b, 3), P.kombu, { shadow: false, mis: P.lemon, misOff: [-3, 2], misA: .25 }));
    for (const [dx, dy, r] of [[-200, 40, 34], [-150, 30, 26], [60, 120, 30], [110, 112, 22]]) KP.paper(ctx, KP.jitter(KP.egg(EC[0] + dx, EC[1] + dy, r * 1.6, r, 0, 20), 4030 + dx + b, 1), P.rice, { shadow: false });  // paper clouds
    // Goldies pop up all over the surface (from 'glutamate', a burst on 'everywhere')
    for (let i = 0; i < 12; i++) {
      const a = -Math.PI / 2 + (i - 5.5) * .45, t0 = T.glu1 + i * .08 + (i % 3 === 0 ? .7 : 0);
      if (t < t0 || Math.abs(a + Math.PI / 2) < .3) continue;
      const ar = KP.arrive(t, t0, .3), x = EC[0] + Math.cos(a) * (ER + 10), y = EC[1] + Math.sin(a) * (ER + 10);
      ctx.save(); ctx.translate(x, y); ctx.rotate(a + Math.PI / 2); ctx.scale(ar.p, ar.p * ar.sy);
      KP.goldie(ctx, 0, -40 + Math.sin(KP.held(t) * 3 + i) * 4, 90, { seed: 4040 + i, boil: b, steam: false, eyes: i % 2 ? 'happy' : 'dot', shadow: KP.SH.far });
      ctx.restore();
    }
  }
  function shot12(ctx, t, b, T, D) {
    ctx.fillStyle = P.coral; ctx.fillRect(0, 0, KP.W, KP.H);
    const zoom = KP.ep(t, T.s2, T.s2 + .9, 'inOut'), sc = KP.lerp(3, 1, zoom);
    const shake = t >= T.chemical && t < T.chemical + .2 ? KP.wob(t * 3, 8, 10) : [0, 0];
    // shot 1 framing: shaker base at screen (490, 1150); shot 2: the whole Earth
    const cy = KP.lerp(SB[1] - (1150 - KP.CAM[1]) / 3, 900, zoom);
    ctx.save(); ctx.translate(shake[0], shake[1]);
    KP.camera(ctx, 490, cy, sc, () => {
      for (let i = -6; i < 14; i++) KP.paper(ctx, KP.jitter(KP.rr(-20 + i * 136, -1200, 60, 3600, 10, 60), 4050 + i + b, 1.2), KP.mix(P.coral, P.pink, .35), { shadow: false });
      stars(ctx, b);
      earth(ctx, t, b, T);
      // lab flasks sprout from the shaker (tubes from its cap), fall away at the zoom
      const s = 1 / 3, fall = KP.ep(t, T.s2 - .1, T.s2 + .45, 'in');
      [[-150, -300, P.pink, T.sounds], [0, -380, P.blue, T.alarm], [150, -300, P.kombu, T.alarm + .25]].forEach(([dx, dy, c, t0], i) => {
        const ar = KP.arrive(t, t0, .3); if (!ar.on || fall >= 1) return;
        const fx = SB[0] + dx * s, fy = SB[1] + (dy + fall * 900) * s;
        KP.brush(ctx, [[SB[0], SB[1] - 120 * s], [fx, fy + 6 * s]], 8 * s, P.rice, { taper: .1 });
        ctx.save(); KP.pose(ctx, ar, fx, fy, 'pop'); ctx.translate(fx, fy); ctx.rotate(fall * (i - 1) * 1.2); ctx.translate(-fx, -fy);
        KP.flask(ctx, fx, fy, 150 * s, c, { seed: 4060 + i, boil: b, t: t * 1.5 });
        ctx.restore();
      });
      KP.shaker(ctx, SB[0], SB[1], 420 * s, { boil: b, seed: 7, rot: t >= T.chemical && t < T.s2 ? KP.wobA(t, 3, .05) : 0 });
      // nervous Goldie beside it (shot 1)
      if (zoom < 1) {
        const gx = SB[0] - 210 * s, gy = SB[1] - 60 * s, trem = KP.wob(t, 9, 1.5);
        KP.goldie(ctx, gx + trem[0] * s, gy, 150 * s, { boil: b, eyes: t >= T.alarm ? 'worried' : 'dot', mouth: t >= T.alarm ? 'wobble' : 'smile', look: 2 });
        if (t >= T.alarm) KP.marks(ctx, gx, gy - 30 * s, 70 * s, 3, 4070 + b, { a0: -2.6, span: 1.2, len: 20 * s, w: 5 * s });
      }
    });
    ctx.restore();
    // the label: a tomato-red warning card that flaps, 'Chemical?'
    if (t >= T.chemical && t < T.s2 + .2) {
      const ar = KP.arrive(t, T.chemical, .3), out = KP.leave(t, T.s2 + .2, .2), flap = Math.sin(KP.held(t) * 9) * .06;
      ctx.save(); ctx.globalAlpha = 1 - out; KP.pose(ctx, ar, 700, 330, 'pop'); ctx.translate(700, 300); ctx.rotate(flap); ctx.translate(-700, -300);
      KP.slip(ctx, [{ s: 'Chemical?', size: 76, ls: 1, color: P.rice }], 500, 240, { rot: .05, seed: 60, boil: b, fill: P.tomato, tape: P.lemon, padX: 34, padY: 20, shadowA: .35 });
      ctx.restore();
    }
  }
  // ---- shot 3: three foods stacked, each with a mini Goldie peeking out
  function parmesan(ctx, x, y, s, b) {
    KP.paper(ctx, KP.jitter([[x - 150 * s, y + 50 * s], [x + 150 * s, y - 70 * s], [x + 160 * s, y + 40 * s], [x - 130 * s, y + 100 * s]], 4100 + b, 1.2), P.brass, { shadow: KP.SH.near, shadowA: .35, mis: P.coral, misOff: [-3, 2], misA: .4 });
    KP.paper(ctx, KP.jitter([[x - 150 * s, y + 50 * s], [x + 150 * s, y - 70 * s], [x + 140 * s, y - 92 * s], [x - 160 * s, y + 30 * s]], 4101 + b, 1), P.lemon, { shadow: false });
    ctx.fillStyle = KP.mix(P.brass, P.wood, .5);
    for (const [hx, hy, r] of [[-60, 40, 14], [30, 10, 10], [90, 10, 16], [-10, 60, 9]]) { ctx.beginPath(); ctx.arc(x + hx * s, y + hy * s, r * s, 0, TAU); ctx.fill(); }
  }
  function milk(ctx, x, y, s, b) {
    // a simple milk drop + a baby bottle (icon level)
    KP.paper(ctx, KP.jitter(KP.drop(x - 100 * s, y, 150 * s), 4110 + b, 1.2), P.rice, { shadow: KP.SH.near, shadowA: .35, mis: P.turq, misOff: [-3, 2], misA: .4 });
    KP.paper(ctx, KP.jitter(KP.rr(x + 40 * s, y - 70 * s, 110 * s, 160 * s, 24 * s, 12), 4111 + b, 1.2), P.rice, { shadow: KP.SH.near, shadowA: .35, mis: P.blue, misOff: [-3, 2], misA: .4 });
    KP.paper(ctx, KP.jitter(KP.rr(x + 40 * s, y - 92 * s, 110 * s, 30 * s, 10 * s, 10), 4112 + b, 1), P.blue, { shadow: false });
    KP.paper(ctx, KP.jitter(KP.egg(x + 95 * s, y - 112 * s, 24 * s, 28 * s, 0, 18), 4113 + b, .8), P.pink, { shadow: KP.SH.far });
    for (let i = 0; i < 3; i++) KP.brush(ctx, [[x + 50 * s, y - 30 * s + i * 34 * s], [x + 76 * s, y - 30 * s + i * 34 * s]], 5 * s, P.blue, { taper: .2 });
  }
  function shot3(ctx, t, b, T) {
    ctx.fillStyle = P.coral; ctx.fillRect(0, 0, KP.W, KP.H);
    const tilt = KP.ep(t, T.s3, T.s4, 'inOut');
    KP.camera(ctx, 490, KP.lerp(760, 860, tilt), 1, () => {
      for (let i = -2; i < 10; i++) KP.paper(ctx, KP.jitter(KP.rr(-20 + i * 136, -400, 60, 3000, 10, 60), 4050 + i + b, 1.2), KP.mix(P.coral, P.pink, .35), { shadow: false });
      const rows = [[T.tomatoes, 420, P.turq, c => KP.tomato(c, 470, 430, 120, { boil: b })], [T.parm, 780, P.blue, c => parmesan(c, 470, 780, 1, b)], [T.milk, 1140, P.grape, c => milk(c, 470, 1150, 1, b)]];
      rows.forEach(([t0, y, card, draw], i) => {
        const ar = KP.arrive(t, t0, .32); if (!ar.on) return;
        ctx.save(); KP.pose(ctx, ar, 490, y + 150, 'pop');
        KP.paper(ctx, KP.jitter(KP.rr(130, y - 160, 700, 320, 20, 24), 4120 + i + b, 1.4), card, { shadow: KP.SH.near, shadowA: .35 });
        // the mini Goldie peeks out from behind the food
        const pk = KP.ease.out(KP.prog(t, t0 + .25, t0 + .6)), bob = Math.sin(KP.held(t) * 3 + i) * 5;
        KP.goldie(ctx, 640, y - 30 - pk * 60 + bob, 90, { seed: 4130 + i, boil: b, eyes: 'happy', steam: false });
        draw(ctx);
        ctx.restore();
      });
    });
  }
  // ---- shots 4-5: tomato vs shaker, then the merge
  function halves(ctx, t, t0, top, bottom, under) {
    const p = KP.ep(t, t0, t0 + .35, 'out');
    if (p < 1 && under) { ctx.save(); under(); ctx.restore(); }
    ctx.save(); ctx.beginPath(); ctx.rect(0, 0, KP.W, 760); ctx.clip(); ctx.translate((1 - p) * -KP.W, 0); top(ctx); ctx.restore();
    ctx.save(); ctx.beginPath(); ctx.rect(0, 760, KP.W, KP.H - 760); ctx.clip(); ctx.translate((1 - p) * KP.W, 0); bottom(ctx); ctx.restore();
  }
  function shot45(ctx, t, b, T, D) {
    const merge = KP.ep(t, T.s5, T.exact, 'inOut'), apart = KP.ep(t, T.s5, T.s5 + .4, 'in');
    const push = 1 + .3 * KP.ep(t, T.s5, T.same, 'inOut') + .05 * KP.ep(t, T.same, D, 'inOut');
    ctx.fillStyle = P.lemon; ctx.fillRect(0, 0, KP.W, KP.H);
    const r = KP.rng(4200); for (let i = 0; i < 9; i++) KP.paper(ctx, KP.jitter(KP.rr(-20 + i * 136, -40, 60, 2000, 10, 60), 4201 + i + b, 1.2), KP.mix(P.lemon, P.brass, .3), { shadow: false }); void r;
    const topBg = c => { c.fillStyle = P.turq; c.fillRect(0, -200, KP.W, 960); };
    const botBg = c => { c.fillStyle = P.pink; c.fillRect(0, 760, KP.W, 1400); };
    const drawGroup = (c, y, srcFn, srcT, sd) => {
      const out = KP.ep(t, T.s5, T.s5 + .5, 'in');
      c.save(); c.translate(-out * 700, 0); srcFn(c, 230, y); c.restore();
      if (t >= srcT - .3) {
        const k = KP.ease.out(KP.prog(t, srcT - .3, srcT + .1)), gx = KP.lerp(230, 430, k), gy = y - 30 - Math.sin(Math.PI * k) * 120;
        const wink = t >= T.wink && t < T.wink + .35;
        KP.goldie(c, gx, gy + Math.sin(KP.held(t) * 3 + sd) * 4, 130, { seed: 4210, boil: b, eyes: wink ? 'closed' : 'happy', mitts: [.4, .4] });
      }
      if (t >= T.glu2) { const g = KP.arrive(t, T.glu2, .35); c.save(); KP.pose(c, g, 740, y + 40, 'right', 500); KP.glu(c, 740, y - 20, .8, { boil: b, seed: 1600 }); c.restore(); }
    };
    const topY = KP.lerp(470, 800, merge), botY = KP.lerp(1050, 800, merge);
    KP.camera(ctx, KP.lerp(490, 560, push - 1), KP.lerp(860, 800, Math.min(1, (push - 1) * 3)), push, () => {
      const under = t < T.s4 + .35 ? () => shot3(ctx, T.s4 - 1 / 60, b, T) : null;
      if (t < T.s5) halves(ctx, t, T.s4, c => { topBg(c); drawGroup(c, 470, (cc, x, y) => KP.tomato(cc, x, y + 10, 110, { boil: b }), T.tomato, 0); },
        c => { botBg(c); drawGroup(c, 1050, (cc, x, y) => KP.shaker(cc, x, y + 120, 300, { boil: b, seed: 7, rot: t > T.msg - .3 && t < T.msg + .2 ? Math.sin(KP.held(t) * 30) * .3 : 0 }), T.msg, 1); }, under);
      else {
        // backgrounds part (top half up, bottom half down) revealing the lemon ground; groups slide together
        ctx.save(); ctx.translate(0, -apart * 900); topBg(ctx); ctx.restore();
        ctx.save(); ctx.translate(0, apart * 1300); botBg(ctx); ctx.restore();
        if (t >= T.exact) KP.ripple(ctx, 470, 800, 360 * KP.ease.out(KP.prog(t, T.exact, T.exact + .8)) + 40, { p: 1, boil: b, squash: .35 });
        drawGroup(ctx, botY, (cc, x, y) => KP.shaker(cc, x, y + 120, 300, { boil: b, seed: 7 }), T.msg, 1);
        ctx.save(); ctx.globalAlpha = merge < 1 ? 1 : .0; drawGroup(ctx, topY, (cc, x, y) => KP.tomato(cc, x, y + 10, 110, { boil: b }), T.tomato, 0); ctx.restore();
      }
      // divider: an ink torn strip; tears away (two halves fly up / down) at the merge
      if (apart < 1) {
        ctx.save(); ctx.translate(-apart * 600, -apart * 300); ctx.rotate(-apart * .3);
        KP.paper(ctx, KP.jitter(KP.torn(KP.rr(-40, 746, 600, 28, 6, 40), 990, 3, 12), 991, 1), P.ink, { shadow: [0, 6], shadowA: .3 }); ctx.restore();
        ctx.save(); ctx.translate(apart * 600, apart * 300); ctx.rotate(apart * .3);
        KP.paper(ctx, KP.jitter(KP.torn(KP.rr(540, 746, 580, 28, 6, 40), 992, 3, 12), 993, 1), P.ink, { shadow: [0, 6], shadowA: .3 }); ctx.restore();
      }
    });
    // '=' cut-out (TOP) on 'exact', label 'Same molecule' (LOW) on 'same'
    if (t >= T.exact) {
      const ar = KP.arrive(t, T.exact, .3);
      ctx.save(); KP.pose(ctx, ar, 490, 360, 'pop');
      KP.paper(ctx, KP.jitter(KP.egg(490, 330, 100, 100, 0, 40), 4220 + b, 1.2), P.rice, { shadow: KP.SH.near, shadowA: .35 });
      for (const dy of [-24, 24]) KP.paper(ctx, KP.jitter(KP.rr(430, 330 + dy - 13, 120, 26, 8, 12), 4221 + dy + b, 1), P.ink, { shadow: false });
      ctx.restore();
    }
    if (t >= T.same) {
      const ar = KP.arrive(t, T.same, .3);
      ctx.save(); KP.pose(ctx, ar, 490, 1180, 'pop');
      KP.slip(ctx, [{ s: 'Same molecule', size: 76, ls: 1 }], 220, 1110, { rot: -.03, seed: 61, boil: b, tape: P.turq, padX: 36, padY: 20 });
      ctx.restore();
    }
  }

  KP.scene('s04_everywhere', {
    sfx(cues, D) {
      const T = times(cues, D), a = [], e = (t, name, gain) => a.push({ t: Math.max(0, +t.toFixed(3)), name, gain });
      e(.2, 'bubble', .3); e(T.alarm, 'bubble', .35); e(T.chemical, 'error', .3);
      e(T.glu1 - .15, 'swoosh_up', .4); e(T.glu1, 'pop_small', .3); e(T.every, 'sparkle', .4);
      e(T.tomatoes, 'pop', .4); e(T.parm, 'pop', .4); e(T.milk, 'pop', .4);
      e(T.s4 + .05, 'whoosh', .35); e(T.tomato, 'pop', .4); e(T.msg, 'pop', .4);
      e(T.exact, 'impact_soft', .55); e(T.same, 'success', .4); e(D - .45, 'whoosh', .35);
      return a.sort((x, y) => x.t - y.t);
    },
    draw(ctx, t, cues, D) {
      const T = times(cues, D), b = KP.boil(t);
      if (t < T.s3) shot12(ctx, t, b, T, D);
      else if (t < T.s4) shot3(ctx, t, b, T);
      else shot45(ctx, t, b, T, D);
      KP.texture(ctx, 'grain', .1);
      KP.irisTo(ctx, KP.prog(t, D - .45, D), 's05_body', 490, 860, b);   // out: Gold Ripple wipe reveals s05
    },
  });
})();
