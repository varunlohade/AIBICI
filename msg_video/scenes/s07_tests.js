// s07_tests: Putting MSG to the test (Kombu Paper). Shots s07.01-.03 of design/SHOTLIST.md.
// All timings come from cues.json words (KP.word).
//   1 cobalt graph-paper lab: two identical capsules drop under two cups and shuffle; scientist + volunteer wear paper
//     blindfolds ('Double-blind'); tilt down to a row of volunteers (three skin tones) with worried thought bubbles
//   2 a flattened Fifth Ripple becomes the chart baseline; gold (MSG) and cobalt (placebo) bars rise nearly equal
//   3 a generic official document slides in, the stamp slams GRAS; the MSG shaker hops onto a shelf beside salt
(function () {
  const P = KP.P, TAU = Math.PI * 2;
  const times = (cues, D) => {
    const w = (re, o) => KP.word(cues, re, o);
    const T = { dbl: w(/double/i, { fb: .56 }), studies: w(/studies/i, { fb: 1.2 }), people: w(/people/i, { fb: 1.9 }) };
    T.made = w(/blamed|^made/i, { fb: T.people + .6 }); T.sick = w(/msg|sick/i, { after: T.made, fb: T.made + .4 });
    T.s2 = w(/couldn|reacted/i, { after: T.sick, fb: T.sick + .5 }) - .1; T.often = w(/reliably|often/i, { after: T.s2, fb: T.s2 + .3 }); T.placebo = w(/placebo/i, { fb: T.often + .8 });
    T.s3 = w(/fda/i, { line: 1, fb: T.placebo + 1.4 }); T.fda = w(/fda/i, { fb: T.s3 + .2 });
    T.gen = w(/generally/i, { fb: T.fda + .9 }); T.safe = w(/safe/i, { after: T.gen, fb: T.gen + 1.6 }); T.salt = w(/salt/i, { after: T.safe, fb: T.safe + .9 });
    T.just = w(/^just$/i, { after: T.safe, fb: T.salt - .35 });
    return T;
  };
  function lab(ctx, b) {
    ctx.fillStyle = P.blue; ctx.fillRect(-400, -1200, 2000, 4000);
    ctx.fillStyle = KP.mix(P.blue, P.turq, .3);
    for (let x = -400; x < 1600; x += 60) ctx.fillRect(x, -1200, 3, 4000);
    for (let y = -1200; y < 2800; y += 60) ctx.fillRect(-400, y, 2000, 3);
  }
  function capsule(ctx, x, y, a, b, sd) {
    ctx.save(); ctx.translate(x, y); ctx.rotate(a);
    KP.paper(ctx, KP.jitter(KP.rr(-40, -16, 80, 32, 16, 8), sd + b, .8), P.rice, { shadow: KP.SH.far, shadowA: .3 });
    KP.paper(ctx, KP.jitter(KP.rr(0, -16, 40, 32, 16, 8), sd + 1 + b, .6), P.pink, { shadow: false });
    ctx.restore();
  }
  function cup(ctx, x, y, b, sd) {
    KP.paper(ctx, KP.jitter([[x - 80, y], [x + 80, y], [x + 56, y - 170], [x - 56, y - 170]], sd + b, 1.2), P.coral, { shadow: KP.SH.near, shadowA: .35, mis: P.pink, misOff: [-3, 2], misA: .45 });
    KP.paper(ctx, KP.jitter(KP.rr(x - 84, y - 14, 168, 20, 6, 10), sd + 1 + b, .8), P.tomato, { shadow: false });
  }
  // two cups lifted over two identical capsules resting on the bench; on 'double-blind' the cups come down over them,
  // then they shuffle (swap along arcs, 3 swaps on the words)
  function shuffle(ctx, t, b, T, y0) {
    const xs = [400, 580];
    const down = KP.ep(t, T.dbl - .1, T.dbl + .2, 'in'), lift = 1 - down;
    xs.forEach((x, i) => capsule(ctx, x, y0 - 18, 0, b, 4700 + i * 3));
    const swaps = [T.studies, (T.studies + T.people) / 2, T.people], n = swaps.filter(s => t >= s + .3).length;
    let pos = [0, 1]; for (let k = 0; k < n; k++) pos = [pos[1], pos[0]];
    const cur = swaps.find(s => t >= s && t < s + .3);
    xs.forEach((_, i) => {
      let x = xs[pos[i]], y = y0 - lift * 120;
      if (cur !== undefined) { const k = KP.ease.inOut(KP.prog(t, cur, cur + .3)), from = xs[pos[i]], to = xs[pos[1 - i]]; x = KP.lerp(from, to, k); y -= Math.sin(Math.PI * k) * 80 * (i ? 1 : -1); }
      ctx.save(); ctx.translate(x, y); ctx.scale(.8, .8); ctx.translate(-x, -y); cup(ctx, x, y, b, 4710 + i * 5); ctx.restore();
    });
    if (cur !== undefined) KP.marks(ctx, 490, y0 - 80, 170, 6, 4720 + b, { a0: -Math.PI * .9, span: Math.PI * .8, len: 24 });
  }
  function shot1(ctx, t, b, T) {
    // one composition inside the safe area: the blindfolded bench row (TOP/MID) and the volunteers (LOW), all in frame
    lab(ctx, b);
    const BY = 690, VY = 1250;
    KP.paper(ctx, KP.jitter(KP.rr(-60, BY, 1200, 44, 10, 30), 4730 + b, 1.4), P.lemon, { shadow: KP.SH.near, shadowA: .35 });
    KP.doll(ctx, 150, BY, .95, { seed: 4740, boil: b, coat: true, skin: P.skin2, hair: P.ink, blind: P.lemon, arm: KP.ep(t, T.studies, T.studies + .2, 'out') * .4 });
    KP.doll(ctx, 830, BY, .95, { seed: 4750, boil: b, shirt: P.pink, skin: P.skin3, hair: P.grape, blind: P.lemon });
    shuffle(ctx, t, b, T, BY);
    KP.paper(ctx, KP.jitter(KP.rr(-60, VY - 40, 1200, 700, 10, 30), 4760 + b, 1.4), P.turq, { shadow: [0, -8], shadowA: .3 });
    [[260, P.skin, P.grape, P.coral], [490, P.skin3, P.ink, P.lemon], [720, P.skin2, P.magenta, P.kombu]].forEach(([x, sk, hair, sh], i) => {
      const worried = t >= T.made, bob = Math.sin(KP.held(t) * 3 + i * 2) * 4;
      KP.doll(ctx, x, VY + bob, .95, { seed: 4770 + i * 10, boil: b, skin: sk, hair, shirt: sh, eyes: worried ? 'worried' : 'dot', mouth: worried ? 'wobble' : 'smile' });
      const ta = KP.arrive(t, T.made + i * .1, .3);
      if (ta.on) { ctx.save(); KP.pose(ctx, ta, x + 30, 900, 'pop');
        KP.thought(ctx, x + 36, 840, 74, (c, cx, cy, r) => {
          KP.shaker(c, cx - 22, cy + 32, 64, { boil: b, seed: 7, shadow: false });
          if (t >= T.sick && i === 1) { KP.paper(c, KP.jitter(KP.egg(cx + 28, cy + 6, 24, 28, 0, 20), 4790 + b, .6), P.pink, { shadow: false }); KP.brush(c, [[cx + 14, cy + 6], [cx + 22, cy], [cx + 30, cy + 8], [cx + 40, cy + 2]], 4, P.ink, { taper: .3 }); }
          else KP.text(c, '?', cx + 30, cy + 20, { size: 52, font: 'Young Serif', weight: 400, align: 'center' });
        }, { boil: b }); ctx.restore(); }
    });
    if (t >= T.dbl) {
      const ar = KP.arrive(t, T.dbl, .3), out = KP.leave(t, T.s2, .001);
      ctx.save(); ctx.globalAlpha = out > 0 ? 0 : 1; KP.pose(ctx, ar, 490, 300, 'pop');
      KP.slip(ctx, [{ s: 'Double-blind', size: 72, ls: 1 }], 280, 226, { rot: -.03, seed: 70, boil: b, tape: P.lemon, padX: 34, padY: 18 });
      ctx.restore();
    }
  }
  function shot2(ctx, t, b, T) {
    lab(ctx, b);
    // a flattened Fifth Ripple becomes the baseline
    const flat = KP.ep(t, T.s2, T.s2 + .5, 'out');
    KP.ripple(ctx, 490, 1180, 440, { p: 1, boil: b, squash: KP.lerp(.35, .06, flat) });
    // test rounds repeat every ~0.45 s: the reaction bars for MSG and for placebo jump up and down unpredictably
    // (sometimes one is higher, sometimes the other): people couldn't reliably tell them apart. No numbers.
    const RD = .45, r0 = Math.floor(Math.max(0, t - T.s2) / RD), rt = Math.max(0, t - T.s2) - r0 * RD;
    const hAt = (r, i) => 200 + KP.hash(r * 7 + 3, i * 11 + 5) * 440;
    const bars = [[330, P.gold, 'msg'], [650, P.blue, 'pl']];
    bars.forEach(([x, c, kind], i) => {
      const grow = KP.ep(t, T.s2 + i * .1, T.s2 + .35 + i * .1, 'out');
      const hgt = grow * KP.lerp(r0 ? hAt(r0 - 1, i) : hAt(0, i), hAt(r0, i), KP.backOut(KP.prog(rt, 0, .25)));
      if (hgt <= 0) return;
      KP.paper(ctx, KP.jitter(KP.rr(x - 100, 1180 - hgt, 200, hgt, 12, 20), 4800 + i + b, 1.4), c, { shadow: KP.SH.near, shadowA: .35, mis: kind === 'msg' ? P.tomato : P.turq, misOff: [-4, 3], misA: .4 });
      const iy = 1180 - hgt - 70;   // the icon rides on top of its bar from the start
      if (kind === 'msg') { KP.paper(ctx, KP.jitter(KP.egg(x, iy, 62, 62, 0, 32), 4810 + b, 1), P.lemon, { shadow: KP.SH.far, shadowA: .3 }); KP.goldie(ctx, x, iy + 4, 80, { boil: b, eyes: 'happy', steam: false, shadow: false }); }
      else { KP.paper(ctx, KP.jitter(KP.egg(x, iy, 62, 62, 0, 32), 4811 + b, 1), P.rice, { shadow: KP.SH.far, shadowA: .3 }); capsule(ctx, x, iy, -.3, b, 4820); }
      if (rt < .12 && r0 > 0) KP.marks(ctx, x, iy, 70, 4, 4830 + i + r0, { a0: -Math.PI * .9, span: Math.PI * .8, len: 18, w: 5 });
    });
    // a round counter of little ticks along the baseline (pictures only, no digits) + a shrugging '?'
    for (let r = 0; r <= Math.min(r0, 12); r++) KP.paper(ctx, KP.jitter(KP.rr(160 + r * 56, 1250, 34, 34, 8, 8), 4840 + r + b, .6), r % 2 ? P.pink : P.lemon, { shadow: KP.SH.far, shadowA: .3 });
    if (t >= T.placebo) {
      const ar = KP.arrive(t, T.placebo, .3);
      ctx.save(); KP.pose(ctx, ar, 650, 430, 'pop');
      KP.slip(ctx, [{ s: 'Placebo', size: 76, ls: 1 }], 500, 360, { rot: .04, seed: 71, boil: b, tape: P.pink, padX: 36, padY: 20 });
      ctx.restore();
    }
  }
  function shot3(ctx, t, b, T, D) {
    const hit = T.gen, back = KP.ep(t, T.just - .2, T.salt, 'inOut');
    const sh = t >= hit && t < hit + 4 / 30 ? KP.wob(t * 2, 6, 12) : [0, 0];
    const sc = KP.lerp(1.12 + .05 * KP.prog(t, T.s3, hit), .82, back);
    ctx.fillStyle = P.lemon; ctx.fillRect(0, 0, KP.W, KP.H);
    ctx.save(); ctx.translate(sh[0], sh[1]);
    KP.camera(ctx, 490, KP.lerp(720, 880, back), sc, () => {
      for (let i = -4; i < 12; i++) KP.paper(ctx, KP.jitter(KP.rr(-20 + i * 136, -800, 60, 3600, 10, 60), 4830 + i + b, 1.2), KP.mix(P.lemon, P.brass, .3), { shadow: false });
      // generic official document (no seal, no logo): white sheet, squiggle lines, a coral rule
      const din = KP.ep(t, T.s3, T.fda + .2, 'out');
      ctx.save(); ctx.translate((1 - din) * -1200, 0); ctx.rotate(-.04);
      KP.paper(ctx, KP.jitter(KP.rr(170, 300, 640, 820, 8, 30), 4840 + b, 1.3), P.rice, { shadow: KP.SH.near, shadowA: .35, mis: P.turq, misOff: [-3, 3], misA: .4 });
      KP.paper(ctx, [[220, 400], [760, 400], [760, 408], [220, 408]], P.coral, { shadow: false });
      for (let i = 0; i < 10; i++) { if (i > 2 && i < 7) continue; KP.squiggle(ctx, 230, 230 + (i % 3 ? 500 : 340), 460 + i * 60, 4850 + i, { amp: 5, wl: 22, w: 6 }); }
      ctx.restore();
      if (t >= hit) { const st = KP.stamp('safe', ['SAFE'], 520, 240, { size: 140, ls: 10, seed: 57, ink: P.kombu }); ctx.save(); ctx.translate(490, 700); ctx.rotate(-.12); const sa = KP.arrive(t, hit, .12); ctx.scale(1 + (1 - sa.p) * .3, 1 + (1 - sa.p) * .3); ctx.drawImage(st, -260, -120); ctx.restore(); }
      // the stamp handle: descends, hits on 'generally', lifts away
      const down = KP.ease.in(KP.prog(t, hit - .45, hit)), up = KP.ep(t, hit + .1, hit + .5, 'out');
      if (t > hit - .45 && up < 1) {
        const y = t < hit ? KP.lerp(-500, 560, down) : KP.lerp(560, -500, up);
        KP.paper(ctx, KP.jitter(KP.rr(330, y + 60, 320, 70, 12, 16), 4860 + b, 1.2), P.ink, { shadow: KP.SH.near, shadowA: .3 });
        KP.paper(ctx, KP.jitter(KP.rr(450, y - 140, 80, 210, 26, 16), 4861 + b, 1.2), P.ink, { shadow: false });
        KP.paper(ctx, KP.jitter(KP.egg(490, y - 160, 90, 60, 0, 30), 4862 + b, 1.2), P.ink, { shadow: false });
        KP.paper(ctx, KP.jitter(KP.rr(336, y + 124, 308, 14, 4, 16), 4863 + b, 1), P.kombu, { shadow: false });
      }
      // shelf with the salt shaker slides up from below on 'Just'; the MSG shaker hops on beside it on 'salt'
      const shA = KP.arrive(t, T.just - .15, .3);   // the shelf pops in (scale), never peeking up from the frame edge
      if (!shA.on) { ctx.save(); ctx.globalAlpha = 0; } else { ctx.save(); KP.pose(ctx, shA, 490, 1440, 'pop'); }
      KP.paper(ctx, KP.jitter(KP.rr(140, 1420, 700, 34, 8, 20), 4870 + b, 1.2), P.coral, { shadow: KP.SH.near, shadowA: .35 });
      for (const bx of [200, 780]) KP.paper(ctx, KP.jitter([[bx - 20, 1454], [bx + 20, 1454], [bx, 1520]], 4871 + bx + b, .8), P.coral, { shadow: KP.SH.far });
      KP.shaker(ctx, 380, 1422, 320, { boil: b, seed: 8, salt: true });
      if (t >= T.salt - .45) {
        const k = KP.ease.out(KP.prog(t, T.salt - .45, T.salt)), x = KP.lerp(1200, 620, k), y = 1422 - Math.sin(Math.PI * k) * 300, land = KP.arrive(t, T.salt, .2);
        KP.shaker(ctx, x, y, 320, { boil: b, seed: 7, sy: t >= T.salt ? land.sy : 1, rot: (1 - k) * .5 });
        if (t >= T.salt && t < T.salt + .5) KP.marks(ctx, 620, 1300, 150, 5, 4880 + b, { a0: -Math.PI * .9, span: Math.PI * .8, len: 24 });
      }
      ctx.restore();
      if (t >= T.safe) for (let i = 0; i < 3; i++) { const sa = KP.arrive(t, T.safe + i * .1, .25); if (!sa.on) continue; const x = [250, 760, 700][i], y = [560, 470, 900][i], r = 26 * sa.p; KP.paper(ctx, KP.jitter([[x, y - r], [x + r * .3, y - r * .3], [x + r, y], [x + r * .3, y + r * .3], [x, y + r], [x - r * .3, y + r * .3], [x - r, y], [x - r * .3, y - r * .3]], 4890 + i + b, .6), P.gold, { shadow: KP.SH.far, mis: P.tomato, misOff: [-2, 2], misA: .4 }); }
    });
    ctx.restore();
  }

  KP.scene('s07_tests', {
    sfx(cues, D) {
      const T = times(cues, D), a = [], e = (t, name, gain) => a.push({ t: Math.max(0, +t.toFixed(3)), name, gain });
      e(.35, 'whoosh', .3); e(T.dbl, 'pop', .35); e(T.studies, 'whoosh', .3); e(T.people, 'whoosh', .3); e(T.made, 'bubble', .35); e(T.sick, 'bubble', .3);
      e(T.s2 + .1, 'swoosh_up', .4); e(T.often, 'blip', .3); e(T.often + .45, 'blip', .3); e(T.placebo, 'ding', .35);
      e(T.fda, 'paper', .35); e(T.gen, 'stamp', .8); e(T.gen + .05, 'impact_soft', .4); e(T.safe, 'sparkle', .3); e(T.salt, 'pop', .4); e(D - .45, 'whoosh', .35);
      return a.sort((x, y) => x.t - y.t);
    },
    draw(ctx, t, cues, D) {
      const T = times(cues, D), b = KP.boil(t);
      if (t < T.s2) shot1(ctx, t, b, T); else if (t < T.s3) shot2(ctx, t, b, T); else shot3(ctx, t, b, T, D);
      KP.texture(ctx, 'grain', .1);
      // out: cut; s08's page curl peels this scene's last frame away
    },
  });
})();
