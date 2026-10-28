// c01 Hook: the whole sinkhole in section; a single torch light starts sinking. 8 JAN 2005.
// Frame 0 is the "quiet section" the Reel also ends on (c07's last frame), so the Reel loops.
(function () {
  const ID = 'c01_hook';
  const BRIDGE = 1.4;   // the zoom to the pool continues this long into c02
  // camera: hold the full section, then glide into the pool on "home" (finishes in c02)
  const viewAt = (t, cues, D) => {
    const tz = AB.at(cues, 'home', 0, D - 1.5);
    return AB.lerpView(AB.V.FULL, AB.V.POOL, AB.ease.inOut(AB.clamp((t - tz) / (D - tz + BRIDGE))));
  };
  // depth of the sinking light (m)
  const T_ON = .45;
  const depthAt = (t, D) => 62 * Math.pow(AB.prog(t, T_ON, D), 1.25);

  KZ.scene(ID, {
    draw(ctx, t, cues, D) {
      const T = AB.amb(ID, t), v = viewAt(t, cues, D);
      const tz = AB.at(cues, 'home', 0, D - 1.5), tJan = AB.at(cues, 'January'), t70 = AB.at(cues, 'seventy'), tStr = AB.at(cues, 'stranger');
      const I = AB.ep(t, T_ON, T_ON + .35, 'out') * (1 - AB.ep(t, tz + .2, AB.lastT(D), 'inOut'));
      let light = null;
      AB.frame(ctx, v, T, {
        mid(cam) {
          // the cold point on the floor, first glimpsed on "stranger"
          const mA = AB.ep(t, tStr, tStr + .9, 'out');
          AB.markerAt(ctx, cam, T, mA);
          if (I > 0) {
            const d = depthAt(t, D), sway = .04 * Math.sin(T * .6);
            const r = AB.diverAt(ctx, cam, { m: 1.2 + .4 * Math.sin(T * .3), d }, Math.PI * .5 + .1 + sway, { I, alpha: I, t: T, bubbles: 7, rim: .8, edge: 0 });
            light = r && r.light;
          }
        },
        get lights() { return light ? [light] : []; },
        label: AB.ep(t, t70, t70 + .8, 'out') * (1 - AB.ep(t, tz - .1, tz + .5)),
        hud() {
          AB.slot(ctx, '8 JAN 2005', AB.win(t, tJan, t70 - .15, .5, .5), { x: 80, y: 262, size: 30, bar: false, color: 'rgba(220,235,235,0.78)' });
        },
      });
    },
    sfx(cues, D) {
      const tz = AB.at(cues, 'home', 0, D - 1.5);
      return [
        { t: 0, name: 'drone_swell', gain: .45 },
        { t: T_ON, name: 'torch_click', gain: .55 },
        { t: T_ON + .25, name: 'water_muffle_hit', gain: .3 },
        { t: 1.2, name: 'bubbles_rise', gain: .3 },
        { t: 2.0, name: 'regulator_breath', gain: .22 },
        { t: AB.at(cues, 'seventy'), name: 'deep_rumble', gain: .45 },
        { t: AB.at(cues, 'stranger'), name: 'sonar_ping', gain: .35 },
        { t: tz + .1, name: 'swoosh_up', gain: .22 },
      ];
    },
  });
})();
