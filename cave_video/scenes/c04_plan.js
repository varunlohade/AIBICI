// c04 The Plan: pull back from the water's edge to the whole section and draw the dive plan on it:
// a fast line down, a short box at the bottom, then a long staircase of decompression stops: 12 HRS.
(function () {
  const ID = 'c04_plan';
  const times = (cues, D) => ({
    tPlan: AB.at(cues, 'plan', 0, .5), tFew: AB.at(cues, 'few', 0, 1), tBottom: AB.atEnd(cues, 'bottom', 0, 2.3),
    t12: AB.at(cues, 'twelve', 0, 2.6), tLight: AB.atEnd(cues, 'light', 0, 5.2),
  });
  // shared with c05/c06: how much of the plan is drawn at time t
  const planState = (t, k) => ({
    desc: AB.ep(t, k.tPlan - .2, k.tFew + .2, 'inOut'),
    box: AB.ep(t, k.tFew, k.tFew + .5, 'out'),
    stairs: AB.ep(t, k.tBottom - .1, k.tLight, 'inOut'),
    stations: AB.ep(t, k.tBottom + .2, k.tLight + .2, 'inOut'),
  });

  KZ.scene(ID, {
    draw(ctx, t, cues, D) {
      const T = AB.amb(ID, t), k = times(cues, D);
      const v = AB.lerpView(AB.V.EDGE, AB.V.FULL, AB.ease.inOut(AB.prog(t, 0, 1.7)));
      const P = planState(t, k);
      AB.frame(ctx, v, T, {
        mid(cam) {
          AB.edgeFigures(ctx, cam, 1 - AB.ep(t, .15, .9), 1);
          AB.plan(ctx, cam, Object.assign({ t: T }, P));
          // the pen tip: a small warm point riding the end of whatever is being drawn
          let tip = null;
          if (P.stairs > 0 && P.stairs < 1) tip = AB.planAt(P.stairs);
          else if (P.desc > 0 && P.desc < 1) tip = { x: AB.lerp(AB.PLAN.desc[0][0], AB.PLAN.desc[1][0], P.desc), d: AB.lerp(0, 262, P.desc) };
          if (tip) AB.glow(ctx, cam.X(tip.x), cam.Y(tip.d), { r: 26, alpha: .9, core: 2.6 });
        },
        hud(cam) {
          // one label at a time: "6 MIN" by the box, then "12 HRS" in the chamber
          const b = AB.PLAN.box;
          AB.bigLabel(ctx, '6 MIN', 930, cam.Y(150), AB.win(t, k.tFew + .3, k.t12, .4, .3), { size: 132, color: '#ffd9a6' });   // same size and place as 12 HRS
          AB.bigLabel(ctx, '12 HRS', 930, cam.Y(150), AB.ep(t, k.t12 + .05, k.t12 + .7, 'out'), { size: 132, color: '#ffd9a6' });
        },
      });
    },
    sfx(cues, D) {
      const k = times(cues, D), ev = [
        { t: 0, name: 'whoosh_long', gain: .18 },
        { t: k.tFew + .1, name: 'click', gain: .2 },
        { t: k.t12 + .05, name: 'sonar_ping', gain: .28 },
        { t: k.tBottom, name: 'drone_swell', gain: .25 },
      ];
      // a soft tick as each stop is drawn
      const P = AB.PLAN;
      for (let i = 0; i < P.stops.length; i += 2) {
        const u = P.len[1 + 2 * i] / P.total; let lo = 0, hi = 1;
        for (let j = 0; j < 30; j++) { const m = (lo + hi) / 2; AB.ease.inOut(m) < u ? lo = m : hi = m; }
        ev.push({ t: +AB.lerp(k.tBottom - .1, k.tLight, hi).toFixed(3), name: 'tick', gain: .1 });
      }
      return ev;
    },
  });
  window.C04 = { times, planState };   // c05 fades this exact plan out
})();
