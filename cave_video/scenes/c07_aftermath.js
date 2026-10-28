// c07 Aftermath: 12 JAN 2005. The line is lifted; two soft lights rise together toward the caustics and the pool.
// "Dave Shaw did bring Deon home." with the plaque on the rim lit softly. In the tail the camera settles under the
// pool surface and the opening diver of c01 returns: the last frame is c01's first frame (the loop).
(function () {
  const ID = 'c07_aftermath';
  const DEEP7 = { d: 236, x: -8, s: 7 }, SHALLOW7 = { d: 24, x: -2, s: 9 }, POOL7 = { d: -5, x: -26, s: 15 };

  const times = (cues, D) => ({
    tFour: AB.at(cues, 'Four', 0, .3), tLifted: AB.at(cues, 'lifted', 0, 2), tRose: AB.at(cues, 'rose', 0, 4), tTog: AB.atEnd(cues, 'together', 0, 5.9),
    tDave: AB.ln(cues, 1).start, tHome: AB.atEnd(cues, 'home', 0, 8.6), tLast: AB.lastT(D),
  });
  const viewAt = (t, k, D) => AB.path([[0, AB.V.FULL], [2.1, DEEP7], [k.tLifted + .3, DEEP7], [k.tTog + .2, SHALLOW7],
    [k.tHome - .5, POOL7], [k.tHome + .35, POOL7], [k.tLast, window.C01.HOOK]], t);
  const rise = (t, k) => AB.ease.inOut(AB.prog(t, k.tLifted + .2, k.tTog + .4));
  const lightsAt = (t, k) => {
    const u = rise(t, k), top = 9 - 5 * AB.ep(t, k.tTog + .4, k.tDave + 1.4, 'inOut'), S = window.C05.SHAW, M = AB.MK;
    const sway = 1 - Math.abs(2 * u - 1);
    const dave = { m: AB.lerp(S.m, .9, AB.smooth(u)) + 1.5 * sway * Math.sin(u * 5), d: AB.lerp(S.d, top, u) };
    const deon = { m: AB.lerp(M.x - .2, -1.1, AB.smooth(u)) + 1.5 * sway * Math.sin(u * 5 + .6), d: AB.lerp(M.d - 1.3, top + 1.2, u) };
    return { dave, deon, u };
  };

  KZ.scene(ID, {
    draw(ctx, t, cues, D) {
      const T = AB.amb(ID, t), k = times(cues, D), v = viewAt(t, k, D), Lt = lightsAt(t, k);
      const vis = 1 - AB.ep(t, k.tDave + .6, k.tHome + .1, 'inOut');                                   // the lights pass into the surface light
      const plaque = AB.ep(t, k.tHome - 1.2, k.tHome - .2, 'inOut') * (1 - AB.ep(t, k.tHome + .5, k.tLast - .5, 'inOut'));
      const lineA = AB.ep(t, k.tLifted - .2, k.tLifted + .6) * (1 - AB.ep(t, k.tTog, k.tDave + 1, 'inOut'));
      const back = AB.ep(t, k.tLast - 1.5, k.tLast, 'inOut');                                          // the opening diver returns
      let lights = [];
      AB.frame(ctx, v, T, {
        plaque,
        mid(cam) {
          if (vis > 0) {
            const a = { x: cam.X(Lt.dave.m), y: cam.Y(Lt.dave.d) }, b = { x: cam.X(Lt.deon.m), y: cam.Y(Lt.deon.d) };
            if (lineA > 0) AB.line(ctx, [[cam.X(1), cam.Y(-1)], [cam.X(1.3), cam.Y(Math.min(150, Lt.dave.d * .6))], [(a.x + b.x) / 2, Math.min(a.y, b.y) - 6]], { color: `rgba(225,240,240,${.32 * lineA})`, width: 1.4 });
            const warmUp = AB.ep(t, k.tTog - .5, k.tHome, 'inOut'), R = Math.max(40, 7.5 * cam.s);
            AB.scatter(ctx, (a.x + b.x) / 2, (a.y + b.y) / 2, R * 3.2, .16 * vis, AB.mixRgb('255,190,120', '190,240,230', warmUp));
            AB.glow(ctx, a.x, a.y, { r: R, alpha: (.9 + .1 * warmUp) * vis, rgb: AB.mixRgb('255,190,120', '230,245,235', warmUp), core: 3 });
            AB.glow(ctx, b.x, b.y, { r: R, alpha: (.9 + .1 * warmUp) * vis, rgb: AB.mixRgb(AB.C.cold, '230,245,235', warmUp), core: 3 });
            if (Lt.u > 0 && Lt.u < 1) { ctx.save(); ctx.globalAlpha = vis * Math.min(1, Lt.u * 6); AB.bubbles(ctx, (a.x + b.x) / 2, Math.min(a.y, b.y) - 10, T, { n: 10, height: 260, scale: .7, seed: 23 }); ctx.restore(); }
            lights.push({ x: a.x, y: a.y, ang: -Math.PI / 2, len: R * 2.2, spread: 1.4 });
          }
          if (back > 0) { const L = window.C01.hook(ctx, cam, T, 0, back); if (L) lights.push(L); }
        },
        get lights() { return lights; },
        hud() {
          AB.slot(ctx, '12 JAN 2005', AB.win(t, k.tFour, k.tLifted + .4, .5, .5));
          const ga = AB.ep(t, k.tLifted + .5, k.tLifted + 1.2, 'out') * (1 - AB.ep(t, k.tTog - .2, k.tTog + .4));
          if (ga > 0) AB.gauge(ctx, 80, 300, Math.max(0, (Lt.dave.d + Lt.deon.d) / 2), { size: 230, max: 270, ruleW: 380, alpha: ga });
          window.C01.gauge(ctx, 0, back);
        },
      });
    },
    sfx(cues, D) {
      const k = times(cues, D);
      return [
        { t: .1, name: 'drone_swell', gain: .28 },
        { t: k.tLifted, name: 'line_creak', gain: .35 },
        { t: k.tRose - .3, name: 'bubbles_rise', gain: .32 },
        { t: k.tRose + 1.6, name: 'bubbles_rise', gain: .22 },
        { t: k.tTog + .2, name: 'water_muffle_hit', gain: .22 },
        { t: k.tHome - .4, name: 'sonar_ping', gain: .25 },
        { t: k.tLast - 1.5, name: 'torch_click', gain: .3 },
      ];
    },
  });
})();
