// s08_twist: Plot twist, MSG vs salt (Kombu Paper). Shots s08.01-.02 of design/SHOTLIST.md.
// All timings come from cues.json words (KP.word). No % reduction on screen.
//   1 the page curls ('Plot twist') to reveal a balance scale: salt shaker vs MSG shaker; cobalt sodium ions pop
//     out, exactly three over salt and one over MSG ('1/3 sodium'); the scale wobbles level on 'table'
//   2 a cooking pot: Goldie bobs happily in the broth (same flavour); a sodium meter beside the pot drops ('Less sodium')
(function () {
  const P = KP.P, TAU = Math.PI * 2;
  const times = (cues, D) => {
    const w = (re, o) => KP.word(cues, re, o);
    const T = { plot: w(/plot/i, { fb: .39 }), has: w(/^has$/i, { fb: 1.2 }), third: w(/third/i, { fb: 1.6 }), sodium: w(/sodium/i, { fb: 2.2 }) };
    T.table = w(/table/i, { after: T.sodium, fb: T.sodium + .6 });
    T.s2 = w(/same|big/i, { after: T.table, fb: T.table + 1 }) - .1; T.flavor = w(/flavou?r/i, { fb: T.s2 + .3 });
    T.less = w(/^less/i, { fb: T.flavor + .6 }); T.sodium2 = w(/sodium/i, { after: T.less, fb: T.less + .2 });
    return T;
  };
  const PY = 760, ARM = 300;
  const tiltAt = (t, T) => { const settle = KP.prog(t, T.table, T.table + .8); return t < T.table ? .12 : .12 * Math.cos(settle * 9) * (1 - settle); };
  function scale(ctx, t, b, T) {
    // pivot at (490, 640); beam tilts toward salt until 'table', then wobbles level
    const tilt = tiltAt(t, T);
    KP.paper(ctx, KP.jitter([[430, 1260], [550, 1260], [510, PY + 20], [470, PY + 20]], 5000 + b, 1.2), P.grape, { shadow: KP.SH.near, shadowA: .35 });
    KP.paper(ctx, KP.jitter(KP.rr(330, 1250, 320, 40, 12, 16), 5001 + b, 1), P.grape, { shadow: KP.SH.near, shadowA: .3 });
    ctx.save(); ctx.translate(490, PY); ctx.rotate(-tilt);
    KP.paper(ctx, KP.jitter(KP.rr(-ARM - 50, -14, 2 * ARM + 100, 28, 12, 20), 5002 + b, 1.2), P.coral, { shadow: KP.SH.near, shadowA: .3, mis: P.pink, misOff: [-3, 2], misA: .4 });
    KP.fastener(ctx, 0, 0, 14);
    const ends = [];
    for (const sx of [-1, 1]) {
      const ex = sx * ARM;
      ctx.save(); ctx.translate(ex, 0); ctx.rotate(tilt);   // pans hang straight down
      KP.brush(ctx, [[-90, 0], [0, 260]], 5, P.ink, { taper: .1 }); KP.brush(ctx, [[90, 0], [0, 260]], 5, P.ink, { taper: .1 });
      KP.brush(ctx, [[0, 0], [0, 250]], 0.1, P.ink);
      KP.paper(ctx, KP.jitter(KP.curve([[-130, 250], [130, 250], [100, 300], [-100, 300]], true, 6), 5003 + sx + b, 1), P.lemon, { shadow: KP.SH.near, shadowA: .3 });
      KP.shaker(ctx, 0, 254, 230, { boil: b, seed: sx < 0 ? 8 : 7, salt: sx < 0 });
      ctx.restore();
    }
    ctx.restore();
    return [-1, 1].map(sx => [490 + sx * ARM * Math.cos(tilt), PY - sx * ARM * Math.sin(-tilt) * -1]);
  }
  function ions(ctx, t, b, T, ends, k) {
    // exactly three ions over salt, one over MSG; they pop on 'has a third of the sodium'
    const sets = [[ends[0], 3, [T.has, T.has + .1, T.has + .2]], [ends[1], 1, [T.third]]];
    sets.forEach(([[ex, ey], n, ts], j) => {
      for (let i = 0; i < n; i++) {
        const ar = KP.arrive(t, ts[i], .3); if (!ar.on) continue;
        const x = ex + (n === 3 ? (i - 1) * 100 : 0), y = ey - 130 + (n === 3 && i === 1 ? -80 : 0) + Math.sin(KP.held(t) * 3 + i + j) * 6;
        ctx.save(); KP.pose(ctx, ar, x, y + 50, 'pop'); KP.na(ctx, x, y, 46, { seed: 5010 + j * 5 + i, boil: b, eyes: 'happy' }); ctx.restore();
      }
    });
  }
  function shot1(ctx, t, b, T) {
    ctx.fillStyle = P.pink; ctx.fillRect(0, 0, KP.W, KP.H);
    for (let i = 0; i < 9; i++) KP.paper(ctx, KP.jitter(KP.rr(-20 + i * 136, -40, 60, 2000, 10, 60), 5020 + i + b, 1.2), KP.mix(P.pink, P.bubble, .4), { shadow: false });
    const push = 1 + .1 * KP.ep(t, T.third, T.s2, 'inOut');
    KP.camera(ctx, 490, KP.lerp(860, 800, KP.prog(push, 1, 1.1)), push, () => {
      const ends = scale(ctx, t, b, T);
      ions(ctx, t, b, T, ends, 1);
    });
    // page curl (plot twist): the previous (lemon) page peels away from the bottom-right corner
    const c = KP.ep(t, 0, T.plot + .5, 'inOut');
    if (c < 1) {
      const d = c * 2600;   // fold line distance from the corner
      const fx = KP.W, fy = KP.H, ux = -.6, uy = -.8;
      const pt = (s, n) => [fx + ux * s - uy * n, fy + uy * s + ux * n];
      const rem = [[0, 0], [KP.W, 0], [KP.W, KP.H], [0, KP.H]].filter(([x, y]) => ((x - fx) * ux + (y - fy) * uy) > d);
      const a = pt(d, -3000), bb = pt(d, 3000);
      ctx.save(); ctx.beginPath(); ctx.moveTo(...a); ctx.lineTo(...bb); ctx.lineTo(...pt(4000, 3000)); ctx.lineTo(...pt(4000, -3000)); ctx.closePath(); ctx.clip();
      // the page being peeled is s07's last frame (so the curl hands content across)
      const c7 = window.CUES && window.CUES.s07_tests;
      if (!c7 || !KP.drawScene(ctx, 's07_tests', (Math.round(c7.duration * 30) - 1) / 30)) { ctx.fillStyle = P.lemon; ctx.fillRect(0, 0, KP.W, KP.H); }
      ctx.restore();
      // the folded-back flap (mirror of the corner across the fold), coral underside with a hard shadow
      const flap = [pt(d, -d * .9), pt(d, d * .9), pt(d * 1.0 - d, 0)].map(([x, y]) => [x, y]);
      const mid = pt(d, 0), cx = fx, cy = fy, mx = 2 * mid[0] - cx, my = 2 * mid[1] - cy;
      const tri = [pt(d, -Math.min(d, 1800)), pt(d, Math.min(d, 1800)), [mx, my]];
      void flap; void rem;
      KP.paper(ctx, KP.jitter(tri, 5030, 1.5), P.coral, { shadow: [-14, -18], shadowA: .35 });
    }
    // labels: 'Plot twist' then '1/3 sodium'
    if (t >= T.plot && t < T.third) {
      const ar = KP.arrive(t, T.plot, .3), out = KP.leave(t, T.third, .12);
      ctx.save(); ctx.globalAlpha = out > 0 ? 0 : 1; KP.pose(ctx, ar, 490, 330, 'pop');
      KP.slip(ctx, [{ s: 'Plot twist', size: 90, font: 'Young Serif', weight: 400, ls: 2 }], 230, 230, { rot: -.04, seed: 80, boil: b, tape: P.lemon, padX: 36, padY: 20 });
      ctx.restore();
    }
    if (t >= T.third) {
      const ar = KP.arrive(t, T.third, .3);
      ctx.save(); KP.pose(ctx, ar, 490, 330, 'pop');
      KP.slip(ctx, [{ s: '1/3 sodium', size: 90, font: 'Young Serif', weight: 400, ls: 2 }], 220, 230, { rot: .03, seed: 81, boil: b, tape: P.blue, padX: 36, padY: 20 });
      ctx.restore();
    }
  }
  function shot2(ctx, t, b, T, D) {
    ctx.fillStyle = P.turq; ctx.fillRect(0, 0, KP.W, KP.H);
    for (let i = 0; i < 9; i++) KP.paper(ctx, KP.jitter(KP.rr(-20 + i * 136, -40, 60, 2000, 10, 60), 5040 + i + b, 1.2), KP.mix(P.turq, P.blue, .15), { shadow: false });
    KP.paper(ctx, KP.jitter(KP.rr(-60, 1150, 1200, 900, 10, 30), 5041 + b, 1.4), P.coral, { shadow: [0, -8], shadowA: .3 });
    // stove flame (paper) + pot
    for (let i = 0; i < 4; i++) { const fx = 270 + i * 70, fl = 1 + .15 * Math.sin(KP.held(t) * 20 + i); KP.paper(ctx, KP.jitter([[fx - 22, 1150], [fx + 22, 1150], [fx, 1150 - 70 * fl]], 5050 + i + b, 1), P.tomato, { shadow: false, mis: P.lemon, misOff: [-3, 3], misA: .5 }); }
    const px = 380, py = 1120;
    KP.paper(ctx, KP.jitter(KP.rr(px - 230, py - 300, 460, 300, 30, 24), 5060 + b, 1.4), P.blue, { shadow: KP.SH.near, shadowA: .35, mis: P.grape, misOff: [-4, 3], misA: .45 });
    for (const sx of [-1, 1]) KP.paper(ctx, KP.jitter(KP.rr(px + sx * 250 - 30, py - 250, 60, 28, 12, 10), 5061 + sx + b, 1), P.grape, { shadow: KP.SH.far });
    KP.paper(ctx, KP.jitter(KP.ell(px, py - 300, 230, 40, 40), 5062 + b, 1.2), P.lemon, { shadow: false });
    for (let i = 0; i < 5; i++) { const ph = (KP.held(t) * 1.4 + i * .2) % 1; ctx.fillStyle = P.rice; ctx.beginPath(); ctx.arc(px - 160 + i * 80, py - 300 - ph * 40, 12 * (1 - ph), 0, TAU); ctx.fill(); }
    // Goldie bobs happily in the broth
    const ga = KP.arrive(t, T.s2 + .1, .35), bob = Math.sin(KP.held(t) * 4) * 10;
    ctx.save(); KP.pose(ctx, ga, px, py - 300, 'up', 160);
    ctx.save(); ctx.beginPath(); ctx.rect(0, 0, KP.W, py - 300 + 10); ctx.clip();
    KP.goldie(ctx, px, py - 360 + bob, 170, { boil: b, eyes: t >= T.flavor ? 'happy' : 'dot', mitts: [1.6 + Math.sin(KP.held(t) * 5) * .3, 1.6] });
    ctx.restore(); ctx.restore();
    for (let i = 0; i < 3; i++) { const sx = px - 100 + i * 100, ph = KP.held(t) * 2.5 + i; KP.brush(ctx, KP.curve([[sx, py - 480], [sx + 14 * Math.sin(ph), py - 540], [sx - 10 * Math.sin(ph), py - 600]], false, 6), 9, P.rice, { taper: .7 }); }
    // sodium meter: a tall card gauge with a cobalt fill that drops on 'less'
    const mx = 800, my = 1100, mh = 560, lvl = KP.lerp(.85, .3, KP.ep(t, T.less - .1, T.sodium2 + .2, 'out'));
    KP.paper(ctx, KP.jitter(KP.rr(mx - 70, my - mh, 140, mh, 30, 20), 5070 + b, 1.2), P.rice, { shadow: KP.SH.near, shadowA: .35 });
    ctx.save(); ctx.clip(KP.path(KP.rr(mx - 50, my - mh + 20, 100, mh - 40, 22, 20)));
    KP.paper(ctx, KP.jitter(KP.rr(mx - 60, my - 20 - (mh - 40) * lvl, 120, (mh - 40) * lvl + 10, 6, 14), 5071 + b, 1), P.blue, { shadow: false });
    ctx.restore();
    KP.na(ctx, mx, my - mh - 40, 40, { boil: b, eyes: t >= T.less ? 'happy' : 'dot' });
    if (t >= T.less - .1 && t < T.sodium2 + .3) KP.brush(ctx, [[mx + 100, my - mh * .7], [mx + 100, my - mh * .3]], 10, P.ink, { taper: .6 });
    if (t >= T.less) {
      const ar = KP.arrive(t, T.less, .3);
      ctx.save(); KP.pose(ctx, ar, 490, 330, 'pop');
      KP.slip(ctx, [{ s: 'Less sodium', size: 84, ls: 1 }], 220, 240, { rot: -.03, seed: 82, boil: b, tape: P.pink, padX: 36, padY: 20 });
      ctx.restore();
    }
  }

  KP.scene('s08_twist', {
    sfx(cues, D) {
      const T = times(cues, D), a = [], e = (t, name, gain) => a.push({ t: Math.max(0, +t.toFixed(3)), name, gain });
      e(T.plot, 'swoosh_up', .45); e(T.has, 'pop_small', .3); e(T.has + .1, 'pop_small', .3); e(T.has + .2, 'pop_small', .3); e(T.third, 'pop', .35); e(T.sodium, 'ding', .45); e(T.table, 'thud', .3);
      e(T.s2 + .1, 'bubbling_pot', .3); e(T.flavor, 'pop', .35); e(T.less, 'swoosh_down', .35); e(T.sodium2, 'success', .4); e(D - .45, 'whoosh', .35);
      return a.sort((x, y) => x.t - y.t);
    },
    draw(ctx, t, cues, D) {
      const T = times(cues, D), b = KP.boil(t);
      if (t < T.s2) shot1(ctx, t, b, T); else shot2(ctx, t, b, T, D);
      KP.texture(ctx, 'grain', .1);
      KP.slideTo(ctx, KP.ep(t, D - .45, D, 'inOut'), 's09_outro');   // slide carrying s09's TV wall
    },
  });
})();
