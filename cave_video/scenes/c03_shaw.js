// c03 Dave Shaw: 2004. A rebreather silhouette (pilot's wings) sinks to the floor; his torch sweeps and finds the
// cold point; he ties his line to it. Then up the line to the water's edge: two silhouettes, an offer.
(function () {
  const ID = 'c03_shaw';
  const FLOOR2 = { d: 255, x: -14, s: 13 }, WIDE3 = { d: 196, x: -7, s: 6.5 }, BOTTOM3 = { d: 256, x: -7, s: 12 };
  const SHAW = { m: -1.5, d: 265.2 };   // where he hovers by the marker

  const times = (cues, D) => ({
    tShaw: AB.at(cues, 'Dave', 0, .8), tPilot: AB.at(cues, 'pilot', 0, 3.4), tBottom: AB.atEnd(cues, 'bottom', 0, 6.5),
    tTorch: AB.at(cues, 'torch', 0, 7.3), tDeon: AB.at(cues, 'Deon', 0, 8.3), tNobody: AB.at(cues, 'Nobody', 0, 9.3),
    tOffer: AB.at(cues, 'offered', 0, 11.5),
  });
  const D_START = 118;
  const shawAt = (t, k) => {
    const u = AB.ease.out(AB.prog(t, .4, k.tBottom));
    return { m: AB.lerp(-2, SHAW.m, AB.smooth(AB.prog(t, k.tBottom - 3, k.tBottom))) + .6 * Math.sin(t * .5), d: AB.lerp(D_START, SHAW.d, u) + .5 * Math.sin(t * .7) };
  };
  const rise = (t, k) => AB.prog(t, k.tNobody - .15, k.tOffer - .2);
  const viewAt = (t, k, D) => {
    const r = rise(t, k);
    if (r > 0) return AB.lerpView(BOTTOM3, AB.V.EDGE, AB.ease.inOut(r), { arc: 1.55 });
    return AB.path([[0, FLOOR2], [2.4, WIDE3], [k.tBottom + .3, BOTTOM3]], t);
  };
  KZ.scene(ID, {
    draw(ctx, t, cues, D) {
      const T = AB.amb(ID, t), k = times(cues, D), v = viewAt(t, k, D), S = shawAt(t, k), r = rise(t, k);
      const level = AB.ep(t, k.tBottom - 2.2, k.tBottom + .4, 'inOut');
      const ang = AB.lerp(Math.PI * .5 + .22, Math.PI * .93, level) + .04 * Math.sin(T * .5);
      const found = AB.ep(t, k.tDeon - .15, k.tDeon + .5, 'out');
      const tie = AB.ep(t, k.tDeon + .35, k.tDeon + 1.3, 'inOut');
      let light = null, mk = null;
      AB.frame(ctx, v, T, {
        mid(cam) {
          AB.snow(ctx, T, .9 * (1 - AB.ep(t, 0, 1.6, 'inOut')));   // c02's settling silt clears
          mk = AB.markerAt(ctx, cam, T, 1, { r: 1 + .25 * found });
          if (found > 0) AB.marker(ctx, mk.x, mk.y, { r: Math.max(30, 7 * cam.s) * 1.6, alpha: .25 * found * (1 - .4 * r), core: 0 });
          // the torch: down along the body on the way down, then sweeping the floor, coming to rest on the marker
          const ox = cam.X(S.m), oy = cam.Y(S.d);
          const toMk = Math.atan2(mk.y - oy, mk.x - ox) + .05;
          const sweep = AB.ep(t, k.tTorch - .4, k.tDeon, 'inOut');
          const torchAbs = AB.lerp(Math.PI * .42, toMk, sweep) + .03 * Math.sin(T * .8) * (1 - found);
          const cone = AB.lerp(0, torchAbs - ang, level);
          // his line: a faint thread back up toward the shot line, and the tie to the marker
          const hand = { x: ox + Math.cos(ang) * 60 * AB.diverK(cam.s), y: oy + Math.sin(ang) * 60 * AB.diverK(cam.s) };
          const lineA = AB.ep(t, k.tBottom - .5, k.tBottom + .5);
          if (lineA > 0) AB.line(ctx, [[cam.X(1.6), cam.Y(150)], [cam.X(-2), cam.Y(220)], [hand.x, hand.y]], { color: `rgba(225,240,240,${.2 * lineA})`, width: 1.3 });
          if (tie > 0) {
            const pts = []; for (let i = 0; i <= 24; i++) { const u = i / 24 * tie; pts.push([AB.lerp(hand.x, mk.x, u), AB.lerp(hand.y, mk.y, u) + Math.sin(u * Math.PI) * 18 * (cam.s / 12)]); }
            AB.line(ctx, pts, { color: `rgba(255,240,215,${.95 - .35 * r})`, width: 2.4, glow: .22 * (1 - r) });
          }
          const res = AB.diverAt(ctx, cam, S, ang, { t: T, cone, flip: ang > Math.PI / 2 + .6, bubbles: 10, edge: .22, rim: .85, lenK: 1.05, kick: .8 });
          light = res && res.light;
          // the water's edge: two figures on the crater bank above the pool (only once we surface)
          const fa = AB.ep(t, k.tOffer - .9, k.tOffer - .2, 'out');
          AB.edgeFigures(ctx, cam, fa, AB.ep(t, k.tOffer, k.tOffer + .7, 'inOut'));
        },
        get lights() { return light ? [light] : []; },
        hud() {
          // one continuous gauge: it holds Deon's depth from c02, re-targets to Shaw's light as he comes into view, and
          // hides for the wide climb to the surface (no reading without a visible diver)
          const ga = 1 - AB.ep(t, k.tNobody - .2, k.tNobody + .3, 'inOut');
          AB.gauge(ctx, 80, 300, AB.lerp(AB.MK.d, S.d, AB.ep(t, k.tShaw, k.tShaw + 1.6, 'inOut')), { size: 230, max: 270, ruleW: 380, alpha: ga });
          AB.year(ctx, 2004, 1 - AB.ep(t, k.tPilot - .8, k.tPilot - .1));   // carried from c02's counter
        },
      });
    },
    sfx(cues, D) {
      const k = times(cues, D);
      return [
        { t: .2, name: 'drone_swell', gain: .3 },
        { t: k.tShaw + .2, name: 'torch_click', gain: .45 },
        { t: k.tShaw + 1.2, name: 'regulator_breath', gain: .25 },
        { t: k.tPilot, name: 'bubbles_rise', gain: .25 },
        { t: k.tBottom - .4, name: 'deep_rumble', gain: .35 },
        { t: k.tDeon - .1, name: 'sonar_ping', gain: .45 },
        { t: k.tDeon + .4, name: 'line_creak', gain: .4 },
        { t: k.tNobody - .1, name: 'swoosh_up', gain: .3 },
        { t: k.tOffer - .6, name: 'water_muffle_hit', gain: .3 },
      ];
    },
  });
})();
