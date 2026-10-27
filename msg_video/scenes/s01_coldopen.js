// s01_coldopen: Cold Open, The Accusations (Kombu Paper). Shots s01.01-.07 of design/SHOTLIST.md (138 s cut).
// Shots 1-3 play REAL interview clips (lib/clips.js) inside the paper hero TV; until a clip is delivered a vivid
// paper placeholder card with a play icon shows instead. Paper cut-outs sit BESIDE the screen, never on the person.
// The lower third shows the real name + credit from CLIPS.info (fallback "Clip 1/2/3"); no DRAMATIZATION line.
// Every time is looked up from cues.json words (KP.word), never hard-coded, so a re-recorded narration re-times it.
//   1 clip 1 on the hero TV; a pink paper brain pinned above the TV, zap bolts burst on 'overexcites'
//   2 channel flick -> clip 2; a paper shaker on the TV's top edge, tomato X stamp on 'poison'
//   3 channel flick -> clip 3; headache zig-zags beside the TV, the shaker hops onto the TV on 'MSG'
//   4 fast pull-back: the whole wall lights up (paused clip frames / lemon play cards); '50 YEARS'
//   5 paper slide -> restaurant window, NO MSG sign, three tables; an envelope slides onto the top table on 'started'
//   6 FLASH-FORWARD (~1.7 s): smash cut to the s06 stamp shot (nested draw of s06), landing on '1968'
//   7 smash cut back + record scratch: the shaker alone under a paper spotlight, '?' on 'bad', pull back to the wall;
//     paper slide out carrying s02's first frame
(function () {
  const P = KP.P, TAU = Math.PI * 2;
  const speakers = cues => cues.filter(l => l.voice !== 'NARR').slice(0, 3);
  const times = (cues, D) => {
    const w = (re, o) => KP.word(cues, re, o), sp = speakers(cues);
    const ls = i => sp[i] || { start: i * 3.9, end: i * 3.9 + 3.6, words: [] };
    const mid = (i, k) => ls(i).start + (ls(i).end - ls(i).start) * k;
    const narr = cues.find(l => l.voice === 'NARR') || { start: D * .55 };
    const T = { c2: ls(1).start, c3: ls(2).start, wall: narr.start };
    // beats on the clips: matched by word if the quote contains it, else a fraction of the clip line
    T.over = w(/overexc|brain/i, { fb: mid(0, .5) }); if (T.over >= T.c2) T.over = mid(0, .5);
    T.poison = w(/poison/i, { fb: mid(1, .6) }); if (T.poison < T.c2 || T.poison >= T.c3) T.poison = mid(1, .6);
    T.headache = w(/headache/i, { fb: mid(2, .4) }); if (T.headache < T.c3 || T.headache >= T.wall) T.headache = mid(2, .4);
    T.msgG = w(/msg/i, { after: T.headache, fb: mid(2, .75) }); if (T.msgG >= T.wall) T.msgG = mid(2, .75);
    T.fifty = w(/fifty|years/i, { after: T.wall, fb: T.wall + .3 });
    T.lit = w(/everyone/i, { after: T.wall, fb: T.wall + 1.2 });
    T.scared = w(/scared|afraid/i, { after: T.wall, fb: T.wall + 2.2 });
    T.tables0 = w(/^of$/i, { after: T.scared, fb: T.scared + .35 }) - .1;   // cut just before 'of'; the sign switches on there
    T.msgN = w(/msg/i, { after: T.tables0, fb: T.tables0 + .7 });
    T.and = w(/^and$/i, { after: T.msgN, fb: T.msgN + 1 }); T.started = w(/started/i, { after: T.msgN, fb: T.and + .6 });
    T.ff0 = w(/^one$/i, { after: T.started, fb: T.started + .6 }) - .1;
    T.y1968 = w(/1968/, { after: T.started, fb: T.ff0 + 1.2 }); T.ff1 = Math.max(T.y1968 + .4, w(/1968/, { after: T.started, end: 1, fb: T.y1968 + .45 }));
    T.is = w(/^is$/i, { after: T.ff1, fb: T.ff1 + .9 }); T.actually = w(/actually/i, { after: T.ff1, fb: T.is + .4 });
    T.bad = w(/^bad/i, { after: T.ff1, fb: T.actually + .6 }); T.you = w(/^you/i, { after: T.bad, fb: T.bad + .4 });
    T.out = D - .45;
    T.terrified = T.tables0 + .1; T.pops = [T.msgN, T.msgN + .15, T.msgN + .3]; T.msgA = T.ff1 - .3;
    return T;
  };

  // ---------- the TV wall (world coordinates; style frame A layout at camera scale 1)
  const HERO = [34, 372, 872, 716];
  const FAR = [
    [24, 40, 330, 280, { antenna: .5 }], [384, 76, 320, 250, {}], [736, 30, 370, 300, { antenna: .4 }],
    [928, 410, 240, 300, { panel: 'bottom' }], [934, 750, 230, 300, {}],
    [24, 1120, 330, 260, { feet: 1 }], [384, 1146, 330, 236, { panel: 'bottom', feet: 1 }], [740, 1110, 360, 280, { feet: 1 }],
  ];
  // extra sets outside frame A, revealed by the pull-back
  const EXTRA = [];
  [-380, 0, 360, 720, 1090, 1440].forEach((x, i) => EXTRA.push([x, -480 + (i % 2) * 26, 320 + (i % 3) * 14, 290, { panel: i % 2 ? 'bottom' : 'right' }]));
  [-220, 150, 530, 900, 1270].forEach((x, i) => EXTRA.push([x, -840 + (i % 2) * 20, 330, 300, { panel: i % 2 ? 'right' : 'bottom', antenna: i % 2 ? .5 : 0 }]));
  [40, 410, 760, 1110].forEach((y, i) => EXTRA.push([-390, y + (i % 2) * 18, 350, 300, { panel: i % 2 ? 'bottom' : 'right' }]));
  [40, 410, 760, 1110].forEach((y, i) => EXTRA.push([1205, y + (i % 2) * 22, 340, 300, { panel: i % 2 ? 'right' : 'bottom' }]));
  const ALL = FAR.concat(EXTRA);
  const SCREENS = [P.lemon, P.bubble, P.lemon, P.turq];
  // lighting order for the pull-back: nearest the hero first
  const ORDER = ALL.map((tv, i) => [i, Math.hypot(tv[0] + tv[2] / 2 - 470, tv[1] + tv[3] / 2 - 730)]).sort((a, b) => a[1] - b[1]).map(a => a[0]);
  const RANK = []; ORDER.forEach((i, k) => RANK[i] = k);

  function wallBack(ctx, b) {
    const rw = KP.rng(4);
    for (let i = -5; i < 13; i++) KP.paper(ctx, KP.jitter(KP.rr(-20 + i * 136, -1000, 118, 4200, 12, 60), 60 + i + b, 1.5), KP.mix(P.plum, P.grape, .18 + rw() * .14), { shadow: false });
  }
  function sideboard(ctx, b, extra) {
    KP.paper(ctx, KP.jitter(KP.rr(-560, 1404, 2300, 1800, 16, 40), 120 + b, 1.5), P.coral, { shadow: [0, -10], shadowA: .35 });
    KP.paper(ctx, KP.jitter(KP.rr(-560, 1404, 2300, 26, 10, 40), 121 + b, 1.2), P.lemon, { shadow: false });
    const drawers = extra ? [-620, -280, 60, 400, 740, 1080, 1420] : [60, 400, 740];
    for (const dx of drawers) { KP.paper(ctx, KP.jitter(KP.rr(dx, 1470, 300, 170, 14, 20), 122 + dx + b, 1.2), KP.mix(P.coral, P.magenta, .45), { shadow: KP.SH.far }); KP.fastener(ctx, dx + 150, 1555, 10); }
    if (extra) for (const dx of [-450, -110, 230, 570, 910, 1250]) KP.paper(ctx, KP.jitter(KP.rr(dx, 1700, 300, 170, 14, 20), 140 + dx + b, 1.2), KP.mix(P.coral, P.magenta, .45), { shadow: KP.SH.far });
  }
  // a small lit set: either a paused frame of one of the clips, or a lemon/pink card with a play icon
  function playIcon(ctx, cx, cy, r, col) {
    KP.paper(ctx, KP.jitter(KP.egg(cx, cy, r, r, 0, 32), 3100 + (cx | 0), 1), P.rice, { shadow: KP.SH.far, shadowA: .3 });
    KP.paper(ctx, KP.jitter([[cx - r * .32, cy - r * .48], [cx + r * .55, cy], [cx - r * .32, cy + r * .48]], 3101 + (cx | 0), .8), col, { shadow: false });
  }
  function smallLit(ctx, tv, i, t, b, cues, o = {}) {
    const [x, y, w, h, opt] = tv;
    const S = KP.tvScreen(x, y, w, h, { panel: opt.panel || 'right' });
    ctx.save(); ctx.clip(KP.path(KP.rr(S.x, S.y, S.w, S.h, 20)));
    const sp = speakers(cues)[i % 3], id = sp && sp.clip;
    const drawn = id && CLIPS.draw(ctx, id, 1 + (i % 5) * .7, S.x, S.y, S.w, S.h);
    if (!drawn) { ctx.fillStyle = SCREENS[i % 4]; ctx.fillRect(S.x, S.y, S.w, S.h); playIcon(ctx, S.x + S.w / 2, S.y + S.h / 2 + KP.wob(t, i, 2)[1], Math.min(S.w, S.h) * .22, P.tomato); }
    ctx.restore();
  }
  // whole wall. o: {lit(i) -> 0|1, hero(ctx, S), heroScreen, extra, cam}
  function wall(ctx, t, b, o) {
    wallBack(ctx, b);
    const sets = o.extra ? ALL : FAR;
    sets.forEach((tv, i) => {
      const [x, y, w, h, opt] = tv;
      const lit = o.lit ? o.lit(i) : 0;
      KP.tv(ctx, x, y, w, h, { seed: 100 + i * 7, boil: b, shadow: [8, 11], shadowA: .45, panel: opt.panel || 'right', antenna: opt.antenna, feet: opt.feet, body: P.grape, screen: P.screenDim, antColor: P.ink, lit: lit > 0 });
      if (lit > 0) { if (o.litDraw) o.litDraw(ctx, KP.tvScreen(x, y, w, h, { panel: opt.panel || 'right' }), i); else smallLit(ctx, tv, i, t, b, o.cues || [], o.litOpts ? o.litOpts(i) : {}); }
    });
    sideboard(ctx, b, o.extra);
    KP.brush(ctx, KP.curve([[930, 380], [905, 560], [915, 760]], false, 8), 9, KP.mix(P.plum, P.ink, .45), { taper: .1 });
    const [hx, hy, hw, hh] = HERO;
    const S = KP.tv(ctx, hx, hy, hw, hh, { seed: 7, boil: b, lit: true, body: P.turq, screen: o.heroScreen || P.lemon, panel: 'bottom', shadow: [14, 18], shadowA: .5, knob: o.knob || 0 });
    if (o.hero) { ctx.save(); ctx.clip(KP.path(KP.rr(S.x, S.y, S.w, S.h, 20))); o.hero(ctx, S); ctx.restore(); }
    return S;
  }
  function sofa(ctx, b, dy = 0) {
    ctx.save(); ctx.translate(0, dy);
    KP.paper(ctx, KP.jitter(KP.curve([[-60, 1760], [120, 1700], [330, 1710], [520, 1740], [700, 1708], [900, 1700], [1140, 1750], [1140, 1980], [-60, 1980]], true, 10), 130 + b, 2.2), P.magenta, { shadow: [0, -12], shadowA: .4, mis: P.grape, misOff: [0, -4], misA: .5 });
    KP.paper(ctx, KP.jitter(KP.curve([[-60, 1850], [300, 1820], [700, 1830], [1140, 1850], [1140, 1980], [-60, 1980]], true, 10), 131 + b, 2), KP.mix(P.magenta, P.ink, .3), { shadow: [0, -8], shadowA: .3 });
    ctx.restore();
  }

  // ---------- the hero screen: the real clip, or a vivid paper placeholder card until it is delivered
  const PH = [P.pink, P.coral, P.grape];
  function heroClip(ctx, S, i, t, b, cues) {
    const sp = speakers(cues)[i], id = sp && sp.clip;
    if (id && CLIPS.draw(ctx, id, t - sp.start, S.x, S.y, S.w, S.h)) return;
    ctx.fillStyle = PH[i]; ctx.fillRect(S.x, S.y, S.w, S.h);
    // film-strip sprockets top and bottom + a pulsing play icon (placeholder only: no grey, never static)
    for (let k = 0; k < 14; k++) for (const yy of [S.y + 16, S.y + S.h - 40]) KP.paper(ctx, KP.jitter(KP.rr(S.x + 20 + k * 56, yy, 30, 24, 6, 8), 3200 + k + yy + b, .6), KP.mix(PH[i], P.ink, .3), { shadow: false });
    const pulse = 1 + .05 * Math.sin(KP.held(t) * 5);
    playIcon(ctx, S.x + S.w / 2, S.y + S.h / 2, 120 * pulse, PH[i]);
  }
  function lowerThirdFor(ctx, i, t0, t, b, cues) {
    const sp = speakers(cues)[i], info = sp && sp.clip ? CLIPS.info(sp.clip) : null;
    const name = (info && info.speaker) || 'Clip ' + (i + 1), credit = info ? (info.credit || null) : null;
    const ar = t0 <= 0 ? { p: 1, sx: 1, sy: 1 } : KP.arrive(t, t0, .32);
    if (t0 > 0 && !ar.on) return;
    const [hx] = HERO, y = 372 + 50 + 520 - 112;
    ctx.save(); KP.pose(ctx, ar, hx + 30, y + 50, 'left', 760);
    KP.lowerThird(ctx, name.toUpperCase(), hx + 30, y, { boil: b, seed: 12 + i, sub: credit });
    ctx.restore();
  }
  // paper cut-outs beside the TV (never on the clip)
  function brainCut(ctx, t, b, T) {
    const bx = 720, by = 318, br = 86, wb = t >= T.over ? KP.wob(t, 5, 4) : [0, 0];
    ctx.save(); ctx.translate(wb[0], wb[1]);
    [[-2.6, 1], [-1.9, .9], [-.9, 1.1], [-.2, 1], [.5, .85], [2.6, .7], [-1.3, .6], [3.0, .65]].forEach(([a, k], i) => {
      let sc = i < 5 ? .75 : 0;
      if (t >= T.over) sc = KP.lerp(sc, 1.2, KP.arrive(t, T.over + i * .04, .22).p);
      if (sc <= 0 || (t > .05 && t < T.over && KP.hash(i, b) < .25)) return;
      KP.paper(ctx, KP.jitter(KP.bolt(bx + Math.cos(a) * br * 1.05, by + Math.sin(a) * br * .8, 54 * k * sc, a, 15 * Math.min(1.2, sc), 20 + i), 400 + i + b, 1.2), P.tomato, { shadow: KP.SH.far, mis: P.pink, misOff: [-4, 3], misA: .6 });
    });
    KP.brain(ctx, bx, by, br, { boil: b, eyes: t >= T.over ? 'worried' : 'dot', mouth: 'wobble' });
    ctx.restore();
    KP.paper(ctx, KP.jitter(KP.rot(KP.rr(bx - 60, by - br - 14, 120, 30, 4, 10), bx, by - br, -.2), 410 + b, .8), KP.rgba(P.pink, .85), { shadow: false });  // tape
  }
  function shakerOnTV(ctx, t, b, T, phase) {
    if (phase === 2) {       // on the TV's top edge, left; X stamp on 'poison'
      const sx = 220, sy = 378, hop = t >= T.poison && t < T.poison + .3 ? Math.sin(Math.PI * KP.prog(t, T.poison, T.poison + .3)) * 30 : 0;
      KP.shaker(ctx, sx, sy - hop, 150, { boil: b, seed: 7, rot: hop * .004 });
      if (t >= T.poison) {
        const ar = KP.arrive(t, T.poison + .03, .18);
        ctx.save(); ctx.translate(sx, sy - 70); ctx.rotate(-.12); ctx.scale(1 + (1 - ar.p) * 1.2, 1 + (1 - ar.p) * 1.2);
        for (const a of [.78, -.78]) KP.paper(ctx, KP.jitter(KP.rot(KP.rr(-100, -18, 200, 36, 10, 12), 0, 0, a), 620 + a * 10 + b, 1.4), P.tomato, { shadow: KP.SH.near, shadowA: .35, mis: P.pink, misOff: [-4, 3], misA: .5 });
        ctx.restore();
      }
    } else {                 // clip 3: headache zig-zags beside the TV, then the shaker hops onto it on 'MSG'
      if (t >= T.headache) [[-2.5, 0], [-1.9, .06], [-1.2, .12]].forEach(([a, dl], i) => {
        const ar = KP.arrive(t, T.headache + dl, .2), cx = 170, cy = 300;
        if (!ar.on || (KP.hash(i, b) < .2 && t > T.headache + .5)) return;
        ctx.save(); ctx.translate(cx, cy); ctx.scale(ar.p, ar.p); ctx.translate(-cx, -cy);
        KP.paper(ctx, KP.jitter(KP.bolt(cx + Math.cos(a) * 50, cy + Math.sin(a) * 50, 100, a, 24, 30 + i), 720 + i + b, 1), P.tomato, { shadow: KP.SH.far, mis: P.pink, misOff: [-3, 2], misA: .5 });
        ctx.restore();
      });
      if (t >= T.msgG - .35) {
        const k = KP.ease.out(KP.prog(t, T.msgG - .35, T.msgG)), x = KP.lerp(1080, 720, k), y = 378 - Math.sin(Math.PI * k) * 220;
        const land = KP.arrive(t, T.msgG, .2);
        KP.shaker(ctx, x, y, 150, { boil: b, seed: 7, sy: t >= T.msgG ? land.sy : 1, rot: (1 - k) * .6 });
        if (t >= T.msgG && t < T.msgG + .4) KP.marks(ctx, x, y - 70, 90, 5, 9 + b, { a0: -2.8, span: 2.4, len: 22, w: 5 });
      }
    }
  }

  // ---------- shot 5: restaurant window + dinner tables (separate sheet)
  function tablesSheet(ctx, t, b, T) {
    ctx.fillStyle = P.blue; ctx.fillRect(-100, -200, 1300, 2400);
    // wallpaper stripes on the restaurant wall
    for (let i = 0; i < 9; i++) KP.paper(ctx, KP.jitter(KP.rr(-20 + i * 136, -200, 60, 2400, 10, 60), 800 + i + b, 1.2), KP.mix(P.blue, P.grape, .22), { shadow: false });
    // window + the NO MSG sign
    const wx = 110, wy = 200, ww = 760, wh = 430;
    KP.paper(ctx, KP.jitter(KP.rr(wx, wy, ww, wh, 18, 24), 820 + b, 1.5), P.lemon, { shadow: [10, 14], shadowA: .4 });
    KP.paper(ctx, KP.jitter(KP.rr(wx + 26, wy + 26, ww - 52, wh - 52, 10, 24), 821 + b, 1.2), P.plum, { shadow: false });
    ctx.save(); ctx.clip(KP.path(KP.rr(wx + 26, wy + 26, ww - 52, wh - 52, 10, 24)));
    for (let i = 0; i < 5; i++) KP.paper(ctx, KP.jitter(KP.rr(wx + 40 + i * 150, wy + 260 - (i % 2) * 60, 120, 300, 6, 20), 830 + i, 1), P.screenDim, { shadow: false }); // night skyline
    const s0 = T.terrified, bulbs = KP.prog(t, s0, s0 + .7), fl = t < s0 + .9 ? KP.boil(t) + 1 : 0;
    const flick = k => t >= s0 + k && !(t < s0 + .9 && KP.hash(k * 10 | 0, KP.boil(t)) < .35);
    KP.noMsgSign(ctx, wx + ww / 2, wy + wh / 2 + 20, 540, { boil: b, bulbs, no: flick(.1) ? 1 : 0, msg: flick(.3) ? 1 : 0, flick: fl });
    ctx.restore();
    // three tables stacked down the frame, each with a folded no-MSG card that pops in on the narration
    const cloths = [[P.coral, P.lemon], [P.turq, P.pink], [P.grape, P.lemon]];
    const pops = T.pops;
    [[300, 790], [640, 975], [330, 1160]].forEach(([cx, cy], i) => {
      const rattle = t > T.terrified && t < T.terrified + .9 ? 4 : 0;
      const [rx, ry] = KP.wob(t, 40 + i, rattle);
      const tw = 470;
      // legs, cloth skirt + top
      for (const lx of [-tw * .36, tw * .36]) KP.paper(ctx, KP.jitter(KP.rr(cx + lx - 12, cy + 120, 24, 110, 8, 12), 845 + i + lx + b, 1), KP.mix(cloths[i][0], P.ink, .35), { shadow: KP.SH.far });
      KP.paper(ctx, KP.jitter(KP.curve([[cx - tw / 2, cy], [cx + tw / 2, cy], [cx + tw / 2 + 10, cy + 130], [cx + tw / 4, cy + 142], [cx, cy + 130], [cx - tw / 4, cy + 142], [cx - tw / 2 - 10, cy + 130]], true, 6), 850 + i + b, 1.6), cloths[i][0], { shadow: [10, 12], shadowA: .4, mis: cloths[i][1], misOff: [-4, 3], misA: .4 });
      KP.paper(ctx, KP.jitter(KP.ell(cx, cy, tw / 2 + 6, 40, 40), 860 + i + b, 1.4), KP.mix(cloths[i][0], P.rice, .25), { shadow: false });
      // a bowl of food + a cup (rattle on 'terrified')
      ctx.save(); ctx.translate(rx, ry);
      const px = cx - 90;
      KP.paper(ctx, KP.jitter(KP.ell(px, cy - 2, 92, 22, 28), 870 + i + b, 1), P.rice, { shadow: KP.SH.far, shadowA: .3 });
      KP.paper(ctx, KP.jitter(KP.curve([[px - 66, cy - 40], [px + 66, cy - 40], [px + 50, cy - 6], [px - 50, cy - 6]], true, 6), 872 + i + b, 1), cloths[(i + 1) % 3][0], { shadow: KP.SH.far, shadowA: .3 });
      KP.paper(ctx, KP.jitter(KP.ell(px, cy - 42, 66, 14, 24), 873 + i + b, .8), P.lemon, { shadow: false });
      for (let k = 0; k < 3; k++) KP.brush(ctx, KP.curve([[px - 30 + k * 30, cy - 60], [px - 22 + k * 30, cy - 84], [px - 34 + k * 30, cy - 104]], false, 5), 6, P.rice, { taper: .7 });
      KP.paper(ctx, KP.jitter(KP.rr(cx + 150, cy - 70, 46, 62, 10, 10), 874 + i + b, 1), P.rice, { shadow: KP.SH.far, shadowA: .3 });
      ctx.restore();
      if (rattle) { KP.marks(ctx, px, cy - 40, 100, 3, 50 + i + b, { a0: -2.8, span: 1.0, len: 20, w: 5 }); KP.marks(ctx, cx + 173, cy - 40, 50, 3, 60 + i + b, { a0: -1.2, span: 1.0, len: 20, w: 5 }); }
      // tent card: no-symbol over a mini shaker (pictures, no lettering)
      const ar = KP.arrive(t, pops[i], .3);
      if (ar.on) {
        const kx = cx + 70, ky = cy - 8;
        ctx.save(); KP.pose(ctx, ar, kx, ky, 'pop');
        KP.paper(ctx, KP.jitter([[kx - 70, ky], [kx - 56, ky - 118], [kx + 56, ky - 118], [kx + 70, ky]], 880 + i + b, 1), P.rice, { shadow: KP.SH.near, shadowA: .35 });
        KP.shaker(ctx, kx, ky - 24, 70, { boil: b, seed: 7, shadow: false });
        ctx.lineWidth = 9; ctx.strokeStyle = P.tomato;
        ctx.beginPath(); ctx.arc(kx, ky - 60, 40, 0, TAU); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(kx - 28, ky - 88); ctx.lineTo(kx + 28, ky - 32); ctx.stroke();
        ctx.restore();
      }
    });
    // 'it all started': a lemon envelope slides in onto the top table
    if (t >= T.and) {
      const k = KP.ep(t, T.and, T.started, 'out'), land = KP.arrive(t, T.started, .2);
      const ex = KP.lerp(1300, 420, k), ey = 735 - Math.sin(Math.PI * k) * 120;
      ctx.save(); ctx.translate(ex, ey); ctx.rotate(KP.lerp(.6, -.08, k)); if (t >= T.started) ctx.scale(land.sx, land.sy);
      KP.paper(ctx, KP.jitter(KP.rr(-110, -66, 220, 132, 8, 16), 895 + b, 1.2), P.lemon, { shadow: KP.SH.near, shadowA: .4, mis: P.coral, misOff: [-3, 2], misA: .45 });
      KP.brush(ctx, [[-106, -62], [0, 8], [106, -62]], 6, P.coral, { taper: .1 });
      for (let i = 0; i < 2; i++) KP.squiggle(ctx, -40, 40 - i * 20, 26 + i * 22, 896 + i, { amp: 3, wl: 14, w: 5 });
      ctx.restore();
    }
    // floor below the caption band (decor)
    KP.paper(ctx, KP.jitter(KP.rr(-60, 1460, 1200, 800, 10, 40), 890 + b, 1.5), P.magenta, { shadow: [0, -8], shadowA: .3 });
  }

  // ---------- shots 6-7: the shaker on a turquoise table in front of the TV wall
  const SHK = [470, 1330];                   // world: shaker bottom centre, standing on the little table
  function shakerTable(ctx, t, b, T, tb, back = 0) {
    KP.paper(ctx, KP.jitter(KP.rr(SHK[0] - 34, SHK[1] + 20, 68, 380, 10, 20), 900 + b, 1.2), KP.mix(P.turq, P.ink, .3), { shadow: KP.SH.near, shadowA: .4 });
    KP.paper(ctx, KP.jitter(KP.ell(SHK[0], SHK[1] + 6, 230, 46, 44), 901 + b, 1.5), P.turq, { shadow: [10, 14], shadowA: .45, mis: P.blue, misOff: [-4, 3], misA: .45 });
    const ar = KP.arrive(t, T.msgA - .15, .3);
    const sweat = t > T.bad + .3 ? 1 : 0;
    // nervous secondary motion while it waits under the light: little hops + a tremble (stops in the freeze)
    const since = tb - T.msgA - .3, ph = since > 0 ? (since / .75) % 1 : 0, hop = since > 0 ? Math.pow(Math.sin(Math.PI * ph), 3) * 26 : 0;
    const [tx] = since > 0 ? KP.wob(tb, 12, 3) : [0];
    ctx.save(); KP.pose(ctx, ar, SHK[0], SHK[1], 'drop', 500); ctx.translate(tx, -hop);
    if (hop < 2 && since > 0) { ctx.translate(SHK[0], SHK[1]); ctx.scale(1.04, .96); ctx.translate(-SHK[0], -SHK[1]); }
    KP.shaker(ctx, SHK[0], SHK[1], 330, { boil: tb, seed: 7, shadowA: .4 });
    ctx.restore();
    if (sweat) KP.marks(ctx, SHK[0], SHK[1] - 170, 190, 4, 70 + tb, { a0: -2.6, span: 2.1, len: 26 });
    // the '?' drops on 'bad' (boing)
    const q = KP.arrive(t, T.bad, .36);
    if (q.on) {
      const qx = SHK[0] + 10 + back * 210, qy = SHK[1] - 470 + back * 250, dropY = (1 - q.p) * -600;
      ctx.save(); ctx.translate(qx, qy + 90); ctx.scale(q.sx, q.sy); ctx.translate(-qx, -qy - 90);
      KP.qmark(ctx, qx, qy + dropY + Math.sin(KP.held(tb) * 3) * 12 * q.k, 230 - back * 60, { rot: .08 + Math.sin(KP.held(tb) * 2) * .08 });
      ctx.restore();
    }
  }
  // the pool of light on the table top (flat lemon paper, no glow)
  function spotlight(ctx) {
    KP.paper(ctx, KP.jitter(KP.ell(SHK[0], SHK[1] + 10, 210, 48, 44), 942, 1.5), P.lemon, { shadow: false });
  }

  // ---------- paper slide transitions
  function sheetSlide(ctx, p, fill, inner) {
    if (p <= 0) return;
    const y = (1 - p) * (KP.H + 60);
    ctx.save(); ctx.translate(0, y);
    KP.paper(ctx, KP.jitter(KP.rr(-40, 0, KP.W + 80, KP.H + 120, 18, 40), 950, 2), fill, { shadow: [0, -22], shadowA: .4 });
    if (inner) { ctx.save(); ctx.clip(KP.path(KP.rr(-40, 0, KP.W + 80, KP.H + 120, 18, 40))); inner(ctx); ctx.restore(); }
    ctx.restore();
  }

  KP.scene('s01_coldopen', {
    // shared with s09 (the loop): the TV wall drawer + a hero-screen override hook (set only around a nested draw)
    api: { wall: (...a) => wall(...a), HERO, ALL, FAR },
    heroOverride: null,
    sfx(cues, D) {
      const T = times(cues, D), a = [], e = (t, name, gain) => a.push({ t: Math.max(0, +t.toFixed(3)), name, gain });
      e(0, 'tv_on', .5); e(T.over, 'glitch', .35);
      e(T.c2 - .05, 'channel_switch', .55); e(T.poison, 'stamp', .4);
      e(T.c3 - .05, 'channel_switch', .55); e(T.headache, 'pop_small', .35); e(T.msgG, 'pop', .35);
      e(T.wall, 'whoosh', .45); e(T.fifty, 'pop', .4); e(T.lit, 'tv_on', .3);
      e(T.tables0 - .05, 'paper', .35); e(T.tables0 + .1, 'neon_flicker', .45); e(T.msgN, 'pop_small', .3); e(T.started, 'paper', .35);
      e(T.ff0 + .05, 'whoosh', .45); e(T.y1968, 'stamp', .8); e(T.y1968 + .05, 'impact_big', .4);
      e(T.ff1, 'record_scratch', .6); e(T.is, 'boing', .3); e(T.bad, 'pop', .4); e(T.out, 'whoosh', .4);
      return a.sort((x, y) => x.t - y.t);
    },
    draw(ctx, t, cues, D) {
      const T = times(cues, D), b = KP.boil(t);
      ctx.fillStyle = P.plum; ctx.fillRect(0, 0, KP.W, KP.H);
      if (t < T.tables0 + .32) {
        // ===== shots 1-4: the TV wall
        const pull = KP.ep(t, T.wall, T.wall + 1.2, 'out');
        const push = 1 + .03 * KP.ep(t, 0, T.wall, 'inOut');
        const sc = t < T.wall ? push : KP.lerp(1.03, .6, pull) * (1 - .03 * KP.prog(t, T.wall + 1.2, T.tables0));
        const cx = KP.lerp(KP.CAM[0], 560, pull), cy = KP.lerp(KP.CAM[1], 760, pull);
        const truck = Math.sin(t * .3) * 6;
        const shot = t < T.c2 ? 0 : t < T.c3 ? 1 : t < T.wall ? 2 : 3;
        const flicker = [T.c2, T.c3].some(s => t >= s - .05 && t < s + .12);
        const knob = (t >= T.c2 ? 1.2 : 0) + (t >= T.c3 ? 1.2 : 0);
        KP.camera(ctx, cx + truck * .4, cy, sc, () => {
          wall(ctx, t, b, {
            extra: t >= T.wall - .2, knob, cues,
            lit: i => shot < 3 ? 0 : (t >= T.wall + .15 + RANK[i] * Math.max(.03, (T.lit - T.wall) / ALL.length) ? 1 : 0),
            hero: (c, S) => { const ov = KZ.scenes.s01_coldopen.heroOverride; if (ov) ov(c, S); else if (flicker) KP.static(c, S, Math.floor(t * 30)); else heroClip(c, S, Math.min(shot, 2), t, b, cues); },
          });
          if (shot === 0 || shot === 3) brainCut(ctx, t, b, T);
          if (shot === 1) shakerOnTV(ctx, t, b, T, 2);
          if (shot === 2) shakerOnTV(ctx, t, b, T, 3);
        });
        sofa(ctx, b, pull * 700);
        if (shot < 3) lowerThirdFor(ctx, shot, shot === 0 ? 0 : [0, T.c2, T.c3][shot], t, b, cues);
        if (shot === 3 && t >= T.fifty) {
          const r2 = KP.arrive(t, T.fifty, .3);
          ctx.save(); KP.pose(ctx, r2, 300, 330, 'pop');
          KP.slip(ctx, [{ s: '50 YEARS', size: 96, font: 'Young Serif', weight: 400, ls: 3 }], 90, 220, { rot: -.04, seed: 51, boil: b, tape: P.pink, padX: 40, padY: 22 });
          ctx.restore();
        }
        KP.texture(ctx, 'grain', .1);
        KP.texture(ctx, 'speck', .28, 17, 41);
      }
      if (t >= T.tables0 && t < T.ff0) {
        // ===== shot 5: paper slide in, restaurant + tables; the envelope arrives on 'started'
        const p = KP.ep(t, T.tables0, T.tables0 + .32, 'out'), tilt = KP.ep(t, T.tables0, T.ff0, 'inOut') * 110;
        sheetSlide(ctx, p, P.blue, c => { c.translate(0, 60 - tilt); tablesSheet(c, t, b, T); });
        if (p >= 1) KP.texture(ctx, 'grain', .1);
      }
      if (t >= T.ff0 && t < T.ff1) {
        // ===== shot 6: FLASH-FORWARD, the s06 stamp shot nested (same art, slightly faster), landing on '1968'
        const s6 = window.CUES && window.CUES.s06_origin, sc6 = KZ.scenes.s06_origin;
        if (s6 && sc6 && sc6.stampTimes) {
          const st = sc6.stampTimes(s6.lines, s6.duration);
          const ts = Math.max(st.shot, st.catchy - .05, st.hit + (t - T.y1968) * 1.25);
          const snap = 1 + .08 * KP.ep(t, T.ff0, T.ff0 + .2, 'out'), sh = t >= T.y1968 && t < T.y1968 + 4 / 30 ? KP.wob(t * 2, 4, 12) : [0, 0];
          ctx.save(); ctx.translate(490 + sh[0], 860 + sh[1]); ctx.scale(snap, snap); ctx.translate(-490, -860);
          sc6.draw(ctx, ts, s6.lines, s6.duration, true);
          ctx.restore();
          KP.texture(ctx, 'news', .05); KP.texture(ctx, 'grain', .1, 101, 57);
        }
      }
      if (t >= T.ff1) {
        // ===== shot 7: smash cut back + record scratch; shaker under the spotlight; '?' on 'bad'; pull back to the wall
        const back = KP.ep(t, T.you, T.out, 'out');
        const wobble = t < T.ff1 + .35 ? Math.sin((t - T.ff1) * 60) * 8 * (1 - KP.prog(t, T.ff1, T.ff1 + .35)) : 0;
        const sc = KP.lerp(1.55, 1.0, back), cx = KP.lerp(SHK[0], 490, back), cy = KP.lerp(SHK[1] - 230, 900, back);
        const swing = Math.sin(Math.PI * 2 * KP.prog(t, T.actually, T.actually + .9)) * .12 * (1 - KP.prog(t, T.actually, T.actually + .9));
        ctx.save(); ctx.translate(wobble, 0);
        KP.camera(ctx, cx, cy, sc, () => {
          wall(ctx, t, b, { extra: true, cues, lit: i => back > .2 && i < 8 && t >= T.you + .1 + i * .04 ? 1 : 0, heroScreen: back > 0 ? P.lemon : P.screenDim, hero: back > 0 ? (c, S) => heroClip(c, S, 0, 1.5, b, cues) : null });
        });
        if (back < 1) {
          ctx.save(); ctx.translate(0, -back * 2100);
          KP.paper(ctx, KP.jitter(KP.rr(-40, -200, KP.W + 80, KP.H + 260, 18, 40), 940 + b, 2), P.plum, { shadow: [0, 24], shadowA: .4 });
          ctx.save(); ctx.translate(490, -200); ctx.rotate(swing); ctx.translate(-490, 200);
          KP.paper(ctx, KP.jitter([[380, -200], [600, -200], [1000, 1700], [-20, 1700]], 941 + b, 2), P.grape, { shadow: false });
          ctx.restore();
          for (let i = 0; i < 18; i++) { const r = KP.hash(i, 7), y = 1500 - ((KP.held(t) * (40 + r * 50) + i * 97) % 1500), x = 490 + (KP.hash(i, 9) - .5) * 360; ctx.fillStyle = i % 3 ? P.lemon : P.pink30; ctx.fillRect(x, y, 6 + r * 6, 6 + r * 6); }
          ctx.restore();
        }
        KP.camera(ctx, cx, cy, sc, () => {
          if (back < .05) spotlight(ctx);
          shakerTable(ctx, t, b, T, KP.boil(t), back);
        });
        ctx.restore();
        KP.texture(ctx, 'grain', .1);
        KP.texture(ctx, 'speck', .22, 17, 41);
      }
      // ===== out: paper slide carrying s02's first frame
      const outP = KP.ep(t, T.out, D, 'inOut');
      if (outP > 0) sheetSlide(ctx, outP, P.turq, c => { const next = KZ.scenes.s02_title; if (next) { c.save(); next.draw(c, 0, [{ words: [] }], 4.6); c.restore(); } });
    },
  });
})();
