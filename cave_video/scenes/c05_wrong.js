// c05 What Went Wrong: down to the floor at -270 m. The gas in the loop packs tight (density). The body comes loose;
// the line tangles around his torch hand; CO2 builds (the only red in the film is here and in c06).
// On "He never came back up." the torch goes still and dims. Quiet and dignified: no breathing sounds, no drama on
// the figure; the camera draws back and lets the light speak.
(function () {
  const ID = 'c05_wrong';
  const BOTTOM5 = { d: 257, x: -9, s: 15 }, B5b = { d: 258, x: -9.2, s: 16.5 }, PULL = { d: 249, x: -7, s: 8.5 };
  const SHAW = { m: -2.2, d: 267.6 };

  const times = (cues, D) => ({
    tGas: AB.at(cues, 'gas', 0, 1.5), tDense: AB.at(cues, 'dense', 0, 3.5), tL1: AB.ln(cues, 1).start, tL1e: AB.ln(cues, 1).end,
    tLoose: AB.at(cues, 'loose', 0, 8.5), tLine: AB.at(cues, 'line', 0, 9.6), tStrain: AB.at(cues, 'Straining', 0, 11.5),
    tCarbon: AB.at(cues, 'carbon', 0, 13.8), tBuilt: AB.at(cues, 'built', 0, 16.4), tPassed: AB.at(cues, 'passed', 0, 17.6),
    tOut: AB.atEnd(cues, 'out', 0, 18.4), tNever: AB.ln(cues, 4).start, tUp: AB.ln(cues, 4).end,
  });
  const viewAt = (t, k, D) => AB.path([[0, AB.V.FULL], [2.5, BOTTOM5], [k.tPassed - .3, B5b], [D, PULL]], t);

  // the loupe: molecules of breathing gas, packing tight as the pressure rises
  const MOL = (() => { const r = AB.rng(7), out = []; for (let i = 0; i < 170; i++) { const a = r() * 6.283, rr = Math.sqrt(r()); out.push([Math.cos(a) * rr, Math.sin(a) * rr, r() * 6.28, .8 + r() * .4]); } return out; })();
  const loupe = (ctx, x, y, R, T, c, a, from) => {
    if (a <= 0) return;
    ctx.save(); ctx.globalAlpha = a;
    // leader from the rebreather to the loupe
    const ang = Math.atan2(from.y - y, from.x - x), ex = x + Math.cos(ang) * R, ey = y + Math.sin(ang) * R;
    ctx.strokeStyle = 'rgba(190,230,235,.45)'; ctx.lineWidth = 1.5; ctx.setLineDash([6, 6]);
    ctx.beginPath(); ctx.moveTo(from.x, from.y); ctx.lineTo(ex, ey); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = 'rgba(190,230,235,.8)'; ctx.beginPath(); ctx.arc(from.x, from.y, 3.5, 0, 6.29); ctx.fill();
    ctx.fillStyle = 'rgba(3,10,16,.82)'; ctx.beginPath(); ctx.arc(x, y, R, 0, 6.29); ctx.fill();
    ctx.save(); ctx.clip();
    const z = AB.lerp(3.8, 1.02, c) * (1 + .03 * Math.sin(T * 1.05) * c), amp = AB.lerp(9, 2, c), mr = AB.lerp(7, 7.5, c);
    for (const [mx, my, ph, sz] of MOL) {
      const px = x + mx * z * (R - 10) + amp * Math.sin(T * (.9 - .5 * c) + ph), py = y + my * z * (R - 10) + amp * Math.cos(T * (.8 - .45 * c) + ph * 1.7);
      if (Math.hypot(px - x, py - y) > R + 10) continue;
      ctx.fillStyle = 'rgba(160,220,228,.55)'; ctx.beginPath(); ctx.arc(px, py, mr * sz, 0, 6.29); ctx.fill();
      ctx.fillStyle = 'rgba(230,250,252,.6)'; ctx.beginPath(); ctx.arc(px - mr * .3, py - mr * .3, mr * .28, 0, 6.29); ctx.fill();
    }
    ctx.restore();
    ctx.strokeStyle = 'rgba(200,235,240,.6)'; ctx.lineWidth = 2.5; ctx.beginPath(); ctx.arc(x, y, R, 0, 6.29); ctx.stroke();
    ctx.restore();
  };

  // the bottom tableau: marker (Deon's place), Dave's silhouette + torch, and the line. Shared with c06, which
  // holds its final state (still, dim). st = {I, live, strain, loose, tg, appear, bubA, cue}
  const tableau = (ctx, cam, T, st) => {
    const { I, live, strain, loose, tg, appear } = st;
    const mk = AB.markerAt(ctx, cam, T, 1, { m: loose * (.35 + .3 * Math.sin(T * .37)), d: -loose * (1.3 + .3 * Math.sin(T * .55)) });
    const wob = (.18 + .3 * strain) * live, S = { m: SHAW.m + wob * Math.sin(T * .9), d: SHAW.d + wob * .6 * Math.sin(T * .7 + 1) };
    const ang = Math.PI * .9 + (.05 + .07 * strain) * live * Math.sin(T * .8);
    const ox = cam.X(S.m) + Math.cos(ang) * 71 * AB.diverK(cam.s), oy = cam.Y(S.d);
    let a0 = Math.atan2(mk.y - oy, mk.x - ox); if (a0 < 0) a0 += Math.PI * 2;
    const aim = AB.clamp(a0, Math.PI * .78, Math.PI * 1.04) - ang;
    const cone = aim + (.04 + .1 * strain) * live * Math.sin(T * 1.1 + .5);
    const res = AB.diverAt(ctx, cam, S, ang, { t: T, I: I * appear, alpha: appear, cone, flip: true, edge: .26 * (.4 + .6 * I), rim: .9, kick: (.7 + .5 * strain) * live, bubbles: 9, bubA: st.bubA, lenK: .95, spread: .24 });
    if (!res) return null;
    const kk = cam.s / 16, top = [cam.X(-4.5), cam.Y(150)], via = [cam.X(-6), cam.Y(244)];
    const tang = AB.tangle(res.L.x + 12 * kk, res.L.y - 2 * kk, 66 * kk, T * live, { n: 15, seed: 12, sx: 1.45, sy: .7, rot: -.4 });
    const nT = Math.max(1, Math.round(tang.length * tg)), part = tang.slice(0, nT);
    const lineA = .34 + .2 * tg;
    AB.line(ctx, [top, via, [part[0][0] + 16 * kk, part[0][1] - 120 * kk], part[0]], { color: `rgba(235,245,245,${lineA * appear})`, width: 1.6 });
    if (tg > 0) AB.line(ctx, part, { color: `rgba(255,238,210,${.92 * appear * (.55 + .45 * I)})`, width: 2.2, glow: .14 * I });
    const end = part[part.length - 1], tail = [];
    for (let i = 0; i <= 16; i++) { const u = i / 16; tail.push([AB.lerp(end[0], mk.x, u), AB.lerp(end[1], mk.y, u) + Math.sin(u * Math.PI) * 26 * kk]); }
    AB.line(ctx, tail, { color: `rgba(235,245,245,${.32 * appear})`, width: 1.4 });
    return res;
  };
  window.C05 = { tableau, SHAW, END: { I: .32, live: 0, strain: 1, loose: 1, tg: 1, appear: 1, bubA: 0 } };

  KZ.scene(ID, {
    draw(ctx, t, cues, D) {
      const T = AB.amb(ID, t), k = times(cues, D), v = viewAt(t, k, D);
      const strain = AB.ep(t, k.tStrain - .3, k.tStrain + 1, 'inOut');
      const live = 1 - AB.ep(t, k.tPassed, k.tOut + .7, 'inOut');          // motion fades out after "passed out"
      const I = 1 - .68 * AB.ep(t, k.tNever + .1, k.tNever + 1.7, 'inOut'); // the torch dims and holds
      const danger = 1 - AB.ep(t, k.tNever, k.tNever + 1.4, 'inOut');
      const warn = AB.ep(t, k.tStrain + .4, k.tCarbon + .6, 'inOut') * danger;
      const loose = AB.ep(t, k.tLoose - .2, k.tLoose + 1.8, 'inOut');
      const tg = AB.ep(t, k.tLine - .3, k.tL1e + .7, 'inOut');
      const pulse = Math.pow(.5 + .5 * Math.cos((t - k.tCarbon) * 5.2), 2);
      let res = null;
      AB.frame(ctx, v, T, {
        mid(cam) {
          AB.plan(ctx, cam, { desc: 1, box: 1, stairs: 1, stations: 1, t: T, alpha: 1 - AB.ep(t, 0, 1.4) });
          res = tableau(ctx, cam, T, { I, live, strain, loose, tg, appear: AB.ep(t, .5, 1.6), bubA: 1 - AB.ep(t, k.tStrain, k.tCarbon + .6) });
        },
        get lights() { return res && res.light ? [res.light] : []; },
        hud(cam) {
          AB.bigLabel(ctx, '12 HRS', 930, cam.Y(150), 1 - AB.ep(t, 0, .7), { size: 132, color: '#ffd9a6' });
          AB.alarm(ctx, warn * (.16 + .3 * pulse));
          const ga = AB.ep(t, 1.2, 2.2, 'out');
          const g = AB.gauge(ctx, 80, 300, SHAW.d + .4, { size: 230, max: 270, ruleW: 380, warn, alpha: ga });
          // the gas loupe (line 1)
          if (res) loupe(ctx, 730, 560, 150, T, AB.ep(t, k.tDense - 1.5, k.tDense + .4, 'inOut'), AB.win(t, k.tGas - .5, k.tL1 + .7, .6, .7), { x: res.x + 8, y: res.y - 16 * res.k });
          // CO2: the chip, and a meter that fills as it builds
          const ca = AB.ep(t, k.tCarbon - .1, k.tCarbon + .4, 'out') * danger;
          if (ca > 0) {
            ctx.save(); ctx.globalAlpha = ca; const c = AB.warnChip(ctx, 84, g.bottom + 40, { pulse });
            const fill = AB.ep(t, k.tCarbon, k.tOut, 'inOut'), by = g.bottom + 40 + c.h + 22;
            ctx.strokeStyle = 'rgba(255,90,70,.7)'; ctx.lineWidth = 2; ctx.strokeRect(84, by, 300, 12);
            ctx.fillStyle = `rgba(255,59,47,${.7 + .3 * pulse})`; ctx.fillRect(87, by + 3, 294 * fill, 6);
            ctx.restore();
          }
        },
      });
    },
    sfx(cues, D) {
      const k = times(cues, D);
      // no breathing sounds anywhere in this scene, out of respect
      return [
        { t: 0, name: 'deep_rumble', gain: .5 },
        { t: 2.3, name: 'water_muffle_hit', gain: .28 },
        { t: k.tGas, name: 'heartbeat_slow', gain: .16 },
        { t: k.tDense - .6, name: 'drone_swell', gain: .22 },
        { t: k.tLoose, name: 'line_creak', gain: .35 },
        { t: k.tLine + .3, name: 'line_creak', gain: .45 },
        { t: k.tStrain, name: 'heartbeat_fast', gain: .18 },
        { t: k.tCarbon, name: 'alarm_beep', gain: .3 },
        { t: k.tCarbon + .8, name: 'alarm_beep', gain: .24 },
        { t: k.tCarbon + 1.6, name: 'alarm_beep', gain: .18 },
        { t: k.tBuilt, name: 'alarm_beep', gain: .14 },
        { t: k.tNever - .25, name: 'music_mute', dur: +(D - k.tNever + .25 + 1.2).toFixed(2), fade_out: .5, fade_in: 2.0 },
        { t: k.tNever - .1, name: 'silence_drop', gain: .45 },
      ];
    },
  });
})();
