// s02_title: Title card (Kombu Paper). Picks up s01's out-slide: a turquoise surface with a falling gold drop.
// On 'MSG.' the drop lands, the Fifth Ripple spreads (4 white rings + a wide gold 5th), Goldie pops up out of
// the centre and the 'MSG' title slip is slapped down. Out: Gold Ripple wipe (act change).
// All times come from cues.json word timings.
(function () {
  const P = KP.P, CX = 490, CY = 760;
  const times = (cues, D) => {
    const ws = (cues[0] && cues[0].words) || [], find = re => (ws.find(w => re.test(w.w)) || {}).start;
    const land = find(/msg/i) ?? .35, mis = find(/misunder/i) ?? land + 1.4;
    return { land, mis, pop: land + .3, wipe: D - .45 };
  };
  KP.scene('s02_title', {
    sfx(cues, D) {
      const T = times(cues, D);
      const r = x => Math.max(0, +x.toFixed(3));
      return [
        { t: r(T.land - .3), name: 'swoosh_down', gain: .4 }, { t: r(T.land), name: 'bubble', gain: .5 },
        { t: r(T.land + .08), name: 'impact_soft', gain: .55 }, { t: r(T.pop), name: 'pop', gain: .4 },
        { t: r(T.mis), name: 'sparkle', gain: .45 }, { t: r(T.wipe), name: 'whoosh', gain: .4 },
      ];
    },
    draw(ctx, t, cues, D) {
      const T = times(cues, D), b = KP.boil(t);
      ctx.fillStyle = P.turq; ctx.fillRect(0, 0, KP.W, KP.H);
      const push = 1 + .04 * KP.ep(t, T.land, D, 'inOut');
      KP.camera(ctx, CX, CY + 100, push, () => {
        // overhead-ish bowl: coral table, a big white rim, the turquoise broth with kombu, scallion rings and steam
        ctx.fillStyle = P.coral; ctx.fillRect(-400, -400, 1900, 2800);
        for (let i = -3; i < 12; i++) KP.paper(ctx, KP.jitter(KP.rr(-20 + i * 136, -400, 60, 2800, 10, 60), 60 + i + b, 1.2), KP.mix(P.coral, P.pink, .35), { shadow: false });
        KP.paper(ctx, KP.jitter(KP.ell(CX, CY + 40, 660, 300, 72), 80 + b, 2), P.rice, { shadow: [14, 20], shadowA: .35, mis: P.blue, misOff: [-5, 4], misA: .4 });
        KP.paper(ctx, KP.jitter(KP.ell(CX, CY + 40, 590, 250, 72), 81 + b, 1.6), P.turq, { shadow: false });
        for (let i = 0; i < 4; i++) KP.paper(ctx, KP.jitter(KP.ell(CX, CY + 40, 200 + i * 110, (200 + i * 110) * .4, 60), 70 + i + b, 2), i % 2 ? KP.mix(P.turq, P.blue, .12) : P.turq, { shadow: false });
        KP.brush(ctx, KP.curve([[CX - 470, CY + 120], [CX - 380, CY + 80 + Math.sin(KP.held(t) * 2) * 8], [CX - 290, CY + 140], [CX - 210, CY + 110]], false, 8), 30, P.kombu, { taper: .3 });
        for (const [sx, sy] of [[CX + 330, CY - 20], [CX + 400, CY + 90], [CX - 300, CY - 60]]) { const rp = KP.jitter(KP.ell(sx, sy, 22, 12, 20), 90 + sx + b, .6); rp.push(rp[0]); KP.brush(ctx, rp, 8, P.kombu, { taper: 0 }); }
        for (let i = 0; i < 3; i++) { const sx = CX - 160 + i * 160, ph = KP.held(t) * 2.5 + i; KP.brush(ctx, KP.curve([[sx, CY - 160], [sx + 18 * Math.sin(ph), CY - 230], [sx - 14 * Math.sin(ph), CY - 300], [sx + 8, CY - 360]], false, 6), 10, P.rice, { taper: .7 }); }
        // the drop falling in, then the Fifth Ripple (a new set of rings every 1.6 s)
        if (t < T.land) {
          const y = KP.lerp(CY - 420, CY, KP.ease.in(KP.prog(t, T.land - .35, T.land)));
          KP.paper(ctx, KP.jitter(KP.drop(CX, y, 130), 960 + b, 1.2), P.gold, { shadow: KP.SH.near, mis: P.tomato, misOff: [-3, 2], misA: .4 });
          KP.brush(ctx, [[CX, y - 150], [CX, y - 230]], 6, P.ink, { taper: .6 });
        }
        for (let k = 0; k < 3; k++) {
          const t0 = T.land + k * 1.6;
          if (t < t0) continue;
          const p = KP.ease.out(KP.prog(t, t0, t0 + 2.2));
          KP.ripple(ctx, CX, CY + 40, 520 + p * 120, { p, boil: b + k * 5, squash: .38 });
        }
        // Goldie pops up out of the centre and bobs in the gold ring
        const ar = KP.arrive(t, T.pop, .35);
        if (ar.on) {
          const bob = Math.sin(KP.held(t) * 3.2) * 8;
          ctx.save(); KP.pose(ctx, ar, CX, CY + 60, 'up', 220);
          KP.goldie(ctx, CX, CY - 40 + bob, 260, { boil: b, eyes: t > T.mis ? 'happy' : 'dot', mouth: 'smile', look: 0 });
          ctx.restore();
          // splash flecks
          if (t < T.pop + .6) KP.marks(ctx, CX, CY - 40, 170 + 60 * KP.prog(t, T.pop, T.pop + .6), 7, 30 + b, { a0: -Math.PI, span: Math.PI, len: 30, color: P.rice });
        }
      });
      // the one label: the title slip (Young Serif), slapped down on 'MSG.'
      const s = KP.arrive(t, T.land + .05, .3);
      if (s.on) {
        ctx.save(); KP.pose(ctx, s, 490, 1250, 'pop');
        KP.slip(ctx, [{ s: 'MSG', size: 150, font: 'Young Serif', weight: 400, ls: 6 }], 270, 1030, { rot: -.03, seed: 41, boil: b, tape: P.pink, padX: 50, padY: 26, align: 'center' });
        ctx.restore();
      }
      // out: Gold Ripple wipe, a wide gold ring sweeping out past the frame
      KP.texture(ctx, 'grain', .1);
      KP.goldOut(ctx, KP.prog(t, T.wipe, D), CX, 860, b);
    },
  });
})();
