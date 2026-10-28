// c06 Don Shirley: from above, a second light sees the still light below and goes down after it. At -250 m his
// rebreather's controller is crushed (red, only here and in c05); he turns back. Then the long climb up the
// decompression staircase, the world tilting slightly (vertigo). 12½ HRS.
(function () {
  const ID = 'c06_shirley';
  const PULL = { d: 249, x: -7, s: 8.5 }, SEE = { d: 202, x: -3, s: 6.6 }, NEAR = { d: 249, x: -3, s: 11 };
  const START = { m: 4, d: 148 }, TURN = { m: -.5, d: 250 };

  const times = (cues, D) => ({
    tSaw: AB.at(cues, 'saw', 0, 1.5), tMoving: AB.at(cues, 'moving', 0, 3), tWent: AB.at(cues, 'went', 0, 4), t250: AB.atEnd(cues, 'metres', 0, 7),
    tCrush: AB.at(cues, 'crushed', 0, 8.5), tTurn: AB.at(cues, 'turn', 0, 10.9), tL2: AB.ln(cues, 2).start, t12: AB.at(cues, 'twelve', 0, 12.5),
    tSurf: AB.atEnd(cues, 'surface', 0, 15.4),
  });
  const climbU = (t, k, D) => AB.ease.inOut(AB.prog(t, k.tL2 - .3, D - .55));
  const shirleyAt = (t, k, D) => {
    const dn = AB.ease.inOut(AB.prog(t, k.tWent - .1, k.t250 + .2));
    let p = { m: AB.lerp(START.m, TURN.m, dn) + .5 * Math.sin(t * .6), d: AB.lerp(START.d, TURN.d, dn) + .4 * Math.sin(t * .8) };
    const u0 = AB.planU(TURN.d), u = AB.lerp(u0, 1, climbU(t, k, D)), q = AB.planAt(u), j = AB.ep(t, k.tL2 - .7, k.tL2 + .6, 'inOut');
    return { m: AB.lerp(p.m, q.x, j), d: AB.lerp(p.d, q.d, j) };
  };
  const viewAt = (t, k, D) => {
    const env = AB.ep(t, k.tL2 + .2, k.tL2 + 1.4, 'inOut') * (1 - AB.ep(t, D - 1.5, D - .25, 'inOut'));
    const v = AB.path([[0, PULL], [2.2, SEE], [k.tWent, SEE], [k.t250 + .4, NEAR], [k.tL2 - .3, NEAR], [k.tL2 + 1.5, AB.V.FULL]], t);
    const rot = env * (.045 * Math.sin((t - k.tL2) * 1.25) + .015 * Math.sin((t - k.tL2) * 2.9));   // vertigo
    return rot ? Object.assign({}, v, { rot }) : v;
  };

  KZ.scene(ID, {
    draw(ctx, t, cues, D) {
      const T = AB.amb(ID, t), k = times(cues, D), v = viewAt(t, k, D), S = shirleyAt(t, k, D);
      const danger = AB.ep(t, k.tCrush - .25, k.tCrush + .35, 'out') * (1 - AB.ep(t, k.tTurn + .4, k.tL2 + .9, 'inOut'));
      const pulse = .5 + .5 * Math.cos((t - k.tCrush) * 2.6);   // a gentle pulse
      const on = AB.ep(t, .5, .9, 'out') * (1 - AB.ep(t, k.tSurf - .1, AB.lastT(D) - .05, 'inOut'));
      const turn = AB.ep(t, k.tTurn - .5, k.tTurn + .9, 'inOut');
      const planA = AB.ep(t, k.tL2 - .4, k.tL2 + .8) * (1 - AB.ep(t, D - .9, AB.lastT(D))) * .7;
      const gaugeA = AB.ep(t, .7, 1.5, 'out') * (1 - AB.ep(t, k.tSurf - .2, AB.lastT(D) - .1));
      const lights = [];
      AB.frame(ctx, v, T, {
        mid(cam) {
          AB.plan(ctx, cam, { desc: 0, box: 0, stairs: 1, stations: 1, t: T, alpha: planA });
          // Dave's light below: still and dim, exactly as c05 left it
          const r0 = window.C05.tableau(ctx, cam, T, window.C05.END);
          if (r0 && r0.light) lights.push(r0.light);
          if (on <= 0) return;
          // Shirley: head down toward the still light, then the turn (pitching up through facing right), then up
          let a0 = Math.atan2(cam.Y(window.C05.SHAW.d) - cam.Y(S.d), cam.X(window.C05.SHAW.m) - cam.X(S.m));
          const look = AB.ep(t, k.tSaw, k.tMoving, 'inOut');
          const ang = AB.lerp(AB.lerp(Math.PI * .5 - .35, a0, look), -Math.PI * .5, turn) + .05 * Math.sin(T * .6);
          const r = AB.diverAt(ctx, cam, S, ang, { t: T, I: on, alpha: on, edge: .2, rim: .8, bubbles: 8, seed: 17, kick: .9, lenK: AB.lerp(1.2, .8, AB.ep(t, k.tL2, k.tL2 + 1)) });
          if (r && r.light) lights.push(r.light);
        },
        lights,
        ruler: 1 - .8 * gaugeA * AB.ep(t, k.tL2, k.tL2 + 1),
        hud(cam) {
          const warn = danger;
          AB.alarm(ctx, warn * (.14 + .26 * pulse));
          const g0 = 1 - AB.ep(t, 0, .7);
          if (g0 > 0) AB.gauge(ctx, 80, 300, window.C05.SHAW.d + .4, { size: 230, max: 270, ruleW: 380, alpha: g0 });
          const g = AB.gauge(ctx, 80, 300, Math.max(0, S.d), { size: 230, max: 270, ruleW: 380, warn, alpha: gaugeA });
          if (danger > 0) { ctx.save(); ctx.globalAlpha = danger; AB.warnChip(ctx, 84, g.bottom + 40, { pulse, text: 'RIG FAILURE', sub: '' }); ctx.restore(); }
          // one label: 12½ HRS, by the staircase (as in c04)
          AB.bigLabel(ctx, '12½ HRS', 930, cam.Y(150), AB.ep(t, k.t12, k.t12 + .7, 'out') * (1 - AB.ep(t, D - .9, AB.lastT(D))), { size: 132, color: '#ffd9a6' });
        },
      });
    },
    sfx(cues, D) {
      const k = times(cues, D);
      return [
        { t: .1, name: 'drone_swell', gain: .28 },
        { t: .55, name: 'torch_click', gain: .35 },
        { t: k.tSaw + .3, name: 'sonar_ping', gain: .35 },
        { t: k.tWent, name: 'swoosh_down', gain: .22 },
        { t: k.tWent + .5, name: 'regulator_breath', gain: .2 },
        { t: k.tCrush - .3, name: 'deep_rumble', gain: .5 },
        { t: k.tCrush, name: 'water_muffle_hit', gain: .4 },
        { t: k.tCrush + .25, name: 'alarm_beep', gain: .3 },
        { t: k.tCrush + 1.05, name: 'alarm_beep', gain: .22 },
        { t: k.tCrush + .6, name: 'heartbeat_fast', gain: .2 },
        { t: k.tTurn - .2, name: 'swoosh_up', gain: .3 },
        { t: k.tL2, name: 'drone_swell', gain: .3 },
        { t: k.tL2 + .4, name: 'bubbles_rise', gain: .3 },
        { t: k.tL2 + 1.6, name: 'regulator_breath', gain: .16 },
        { t: D - .7, name: 'water_muffle_hit', gain: .2 },
      ];
    },
  });
})();
