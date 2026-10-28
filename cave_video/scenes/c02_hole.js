// c02 Boesmansgat: the pool on the dry farm, the push down the throat into the vast chamber.
// 1994: an older light sinks away below its teammates and settles on the floor as a cold point (never a body).
(function () {
  const ID = 'c02_hole';
  const THROAT = { d: 58, x: 1, s: 12 }, CHAMBER = { d: 158, x: 8, s: 5.4 }, FLOORV = { d: 250, x: -13, s: 11 }, FLOOR2 = { d: 255, x: -14, s: 13 };

  const times = (cues, D) => {
    const c1 = window.CUES.c01_hook, D1 = c1.duration, tz1 = AB.at(c1.lines, 'home', 0, D1 - 1.5);
    return {
      D1, tz1,
      tOpen: AB.at(cues, 'opens', 0, 3.9), tCham: AB.at(cues, 'chamber', 0, 6), tDeep: AB.atEnd(cues, 'deep', 0, 8.5),
      t94: AB.at(cues, '1994', 0, 9), tSank: AB.at(cues, 'sank', 0, 12), tUp: AB.atEnd(cues, 'up', 0, 15.6),
      tStay: AB.at(cues, 'stayed', 0, 16.8), tYears: AB.atEnd(cues, 'years', 0, 19),
    };
  };
  // the 1994 light: from just below the throat, sinking and drifting to Deon's place on the floor
  const LIGHT0 = { m: -4, d: 50 };
  const lightAt = (t, k) => {
    const u = AB.ease.inOut(AB.prog(t, k.t94 + .9, k.tUp));
    const m = AB.lerp(LIGHT0.m, AB.MK.x, AB.smooth(AB.prog(t, k.t94 + .9, k.tUp))) + 2.5 * Math.sin(u * 3.2) * (1 - u);
    return { m, d: AB.lerp(LIGHT0.d, AB.MK.d, u), u };
  };
  const viewAt = (t, k, D) => {
    if (t < 1.6) {   // finish c01's glide into the pool
      const u = AB.ease.inOut(AB.clamp((t + k.D1 - k.tz1) / (k.D1 - k.tz1 + 1.4)));
      if (u < 1) return AB.lerpView(AB.V.FULL, AB.V.POOL, u);
    }
    const P = AB.V.POOL, drift = { d: P.d + .8 * AB.smooth(AB.prog(t, 1.4, k.tOpen)), x: P.x, s: P.s };
    return AB.path([
      [k.tOpen - .1, drift],
      [k.tCham + .2, THROAT, { ease: 'in' }],
      [k.tDeep + .1, CHAMBER, { ease: 'out' }],
      [k.t94 + .9, CHAMBER],
      [k.tUp, FLOORV],
      [D, FLOOR2],
    ], t);
  };

  KZ.scene(ID, {
    draw(ctx, t, cues, D) {
      const T = AB.amb(ID, t), k = times(cues, D), v = viewAt(t, k, D), L = lightAt(t, k);
      const appear = AB.ep(t, k.t94, k.t94 + .7, 'out');
      const settle = AB.ep(t, k.tUp - .8, k.tUp + 1.2, 'inOut');     // warm torch -> cold point
      AB.frame(ctx, v, T, {
        mid(cam) {
          // teammates' lights, holding at ~45 m while his light sinks away below them
          const tm = appear * (1 - AB.ep(t, k.tSank, k.tUp, 'inOut'));
          [[3, 44.5, 0], [7.5, 46.5, 2]].forEach(([m, d, ph]) => AB.glow(ctx, cam.X(m + .5 * Math.sin(T * .4 + ph)), cam.Y(d + .4 * Math.sin(T * .5 + ph)), { r: Math.max(18, 4 * cam.s), alpha: .7 * tm, core: 1.8 }));
          // the sinking light: warm and dimming as it goes, then cold and still
          if (appear > 0 && settle < 1) {
            const x = cam.X(L.m), y = cam.Y(L.d), dim = 1 - .45 * AB.smooth(L.u);
            AB.scatter(ctx, x, y, Math.max(60, 9 * cam.s), .12 * appear * dim * (1 - settle));
            AB.glow(ctx, x, y, { r: Math.max(22, 5 * cam.s), alpha: appear * dim * (1 - settle), rgb: '255,200,130', core: 2.2 });
          }
          AB.markerAt(ctx, cam, T, settle);
          AB.snow(ctx, T, .9 * AB.ep(t, k.tUp, k.tStay + .5, 'inOut'));   // ten years of silt settling
        },
        ruler: 1 - AB.ep(t, k.t94 - .5, k.t94 + .2, 'inOut'),
        hud() {
          const ga = AB.ep(t, k.t94 + .3, k.t94 + 1.1, 'out');
          if (ga > 0) AB.gauge(ctx, 80, 300, AB.lerp(LIGHT0.d, AB.MK.d, AB.ease.inOut(AB.prog(t, k.t94 + .9, k.tUp))), { size: 230, max: 270, ruleW: 380, alpha: ga });
          // one label: 1994, which counts on to 2004 over "ten years"
          const yr = 1994 + Math.floor(10 * AB.ease.inOut(AB.prog(t, k.tStay, k.tYears)) + 1e-6);
          AB.slot(ctx, String(yr), AB.ep(t, k.t94, k.t94 + .6, 'out'));
        },
      });
    },
    sfx(cues, D) {
      const k = times(cues, D), ev = [
        { t: k.tOpen - .1, name: 'whoosh_long', gain: .22 },
        { t: k.tOpen + .5, name: 'water_muffle_hit', gain: .28 },
        { t: k.tCham, name: 'deep_rumble', gain: .5 },
        { t: k.t94, name: 'drone_swell', gain: .3 },
        { t: k.t94 + 1.2, name: 'bubbles_rise', gain: .22 },
        { t: k.tUp + .3, name: 'sonar_ping', gain: .4 },
      ];
      // a soft tick for each year that passes
      for (let y = 1; y <= 10; y++) {
        const u = y / 10; let lo = 0, hi = 1;   // invert the eased year counter
        for (let i = 0; i < 30; i++) { const m = (lo + hi) / 2; AB.ease.inOut(m) < u ? lo = m : hi = m; }
        ev.push({ t: +(AB.lerp(k.tStay, k.tYears, hi)).toFixed(3), name: 'tick', gain: .12 });
      }
      return ev;
    },
  });
})();
