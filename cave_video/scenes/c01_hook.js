// c01 Hook: frame 0 reads on mute. Close on a diver just under the pool surface, head down, torch on, with the
// depth counter already ticking. Then the pull back to the whole sinkhole in section. 8 JAN 2005.
// Frame 0 is also where the Reel ends (c07's last frame draws C01.hook at c01-time 0), so the Reel loops.
(function () {
  const ID = 'c01_hook';
  const BRIDGE = 1.4;   // the glide into the pool continues this long into c02
  const HOOK = { d: 7.5, x: 1.2, s: 22 };
  const PULL0 = .45, PULL1 = 3.3;
  // depth of the diver (m): already sinking at frame 0
  const depthAt = t => 4.5 + 4.2 * t + .28 * t * t;
  const viewAt = (t, cues, D) => {
    const tz = AB.at(cues, 'home', 0, D - 1.5);
    if (t >= tz) return AB.lerpView(AB.V.FULL, AB.V.POOL, AB.ease.inOut(AB.clamp((t - tz) / (D - tz + BRIDGE))));
    return AB.lerpView(HOOK, AB.V.FULL, AB.ease.inOut(AB.prog(t, PULL0, PULL1)));
  };
  // the diver (+ torch): t01 = c01 time (pose/depth), T = ambient time, a = alpha/intensity. Returns the light.
  const hook = (ctx, cam, T, t01, a = 1) => {
    if (a <= 0) return null;
    const d = depthAt(t01), sway = .04 * Math.sin(T * .6);
    const r = AB.diverAt(ctx, cam, { m: 1.2 + .4 * Math.sin(T * .3), d }, Math.PI * .5 + .1 + sway, { I: a, alpha: a, t: T, bubbles: 7, rim: .9, edge: .3, lenK: 1 });
    return r && r.light;
  };
  const gauge = (ctx, t01, a, rulerFade) => { if (a > 0) AB.gauge(ctx, 80, 300, depthAt(t01), { size: 230, max: 270, ruleW: 380, alpha: a }); };
  window.C01 = { HOOK, hook, gauge, depthAt };

  KZ.scene(ID, {
    draw(ctx, t, cues, D) {
      const T = AB.amb(ID, t), v = viewAt(t, cues, D);
      const tz = AB.at(cues, 'home', 0, D - 1.5), tJan = AB.at(cues, 'January'), t70 = AB.at(cues, 'seventy'), tStr = AB.at(cues, 'stranger');
      const I = 1 - AB.ep(t, tz + .2, AB.lastT(D), 'inOut');
      const ga = 1 - AB.ep(t, tz - .2, tz + .6, 'inOut');
      let light = null;
      AB.frame(ctx, v, T, {
        mid(cam) {
          AB.markerAt(ctx, cam, T, AB.ep(t, tStr, tStr + .9, 'out'));   // the cold point, first glimpsed on "stranger"
          light = hook(ctx, cam, T, t, I);
        },
        get lights() { return light ? [light] : []; },
        label: AB.ep(t, t70, t70 + .8, 'out') * (1 - AB.ep(t, tz - .1, tz + .5)),
        ruler: 1 - .75 * ga,
        hud() {
          gauge(ctx, t, ga);
          AB.slot(ctx, '8 JAN 2005', AB.win(t, tJan, t70 - .15, .5, .5));
        },
      });
    },
    sfx(cues, D) {
      const tz = AB.at(cues, 'home', 0, D - 1.5);
      return [
        { t: 0, name: 'drone_swell', gain: .45 },
        { t: .05, name: 'bubbles_rise', gain: .3 },
        { t: .9, name: 'regulator_breath', gain: .22 },
        { t: PULL0 + .1, name: 'whoosh_long', gain: .18 },
        { t: AB.at(cues, 'seventy'), name: 'deep_rumble', gain: .45 },
        { t: AB.at(cues, 'stranger'), name: 'sonar_ping', gain: .35 },
        { t: tz + .1, name: 'swoosh_up', gain: .22 },
      ];
    },
  });
})();
