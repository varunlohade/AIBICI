// s01_coldopen: Cold Open, The Accusations (Kombu Paper). Shots s01.01-.07 of design/SHOTLIST.md (138 s cut).
// Shots 1-3 play REAL interview clips (lib/clips.js) inside the paper hero TV; until a clip is delivered a vivid
// paper placeholder card with a play icon shows instead. Paper cut-outs sit BESIDE the screen, never on the person.
// The lower third shows the real name + credit from CLIPS.info (fallback "Clip 1/2/3"); no DRAMATIZATION line.
// Every time is looked up from cues.json words (KP.word), never hard-coded, so a re-recorded narration re-times it.
//   1 clip 1 on the hero TV; a pink paper brain taped above the TV, zap bolts burst on 'destroys', crackle on 'brain'
//   2 jump-cut flicker -> clip 2; a shaker on the TV top pours onto a plate that grows on 'hungry' / 'eat more'
//   (an optional third clip line still works: headache zig-zags + the shaker hops onto the TV)
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
    const T = { c2: ls(1).start, wall: narr.start };
    T.c3 = sp[2] ? sp[2].start : T.wall;                 // a third clip is optional (current cut: two Blaylock clips)
    // beats on the clips: matched by word if the quote contains it, else a fraction of the clip line.
    // clip 1 (hungry): the shaker pours on 'MSG'; the plate grows on 'hungry' and again on 'eat more'
    const inside = (v, a, z, fb) => (v >= a && v < z ? v : fb);
    T.pour = inside(w(/msg/i, { fb: -1 }), 0, T.c2, mid(0, .2));
    T.hungry = inside(w(/hungry/i, { fb: -1 }), 0, T.c2, mid(0, .6));
    T.eatMore = inside(w(/^eat$/i, { after: T.hungry, fb: -1 }), 0, T.c2, mid(0, .8));
    // clip 2 (brain): zap bolts burst on 'destroys' and crackle on 'brain'
    T.over = inside(w(/destroy|overexc/i, { after: T.c2, fb: -1 }), T.c2, T.wall, mid(1, .55));
    T.brainW = inside(w(/^brain/i, { after: T.over, fb: -1 }), T.c2, T.wall, T.over + .4);
    T.fifty = w(/fifty|years/i, { after: T.wall, fb: T.wall + .3 });
    T.lit = w(/everyone/i, { after: T.wall, fb: T.wall + 1.2 });
    T.scared = w(/scared|afraid/i, { after: T.wall, fb: T.wall + 2.2 });
    T.tables0 = w(/^of$/i, { after: T.scared, fb: T.scared + .35 }) - .1;   // cut just before 'of'; the sign switches on there
    T.msgN = w(/msg/i, { after: T.tables0, fb: T.tables0 + .7 });
    T.and = w(/^and$/i, { after: T.msgN, fb: T.msgN + 1 }); T.started = w(/started/i, { after: T.msgN, fb: T.and + .6 });
    T.ff0 = w(/^one$/i, { after: T.started, fb: T.started + .6 }) - .1;           // shot 5: the envelope handoff
    T.letter = w(/letter/i, { after: T.ff0, fb: T.ff0 + .3 });
    T.y1968 = w(/1968/, { after: T.started, fb: T.ff0 + 1.1 });
    T.land = Math.min(T.y1968 - .15, T.letter + .7);                             // the envelope lands on the s06 desk
    T.terrified = T.tables0 + .1; T.pops = [T.msgN, T.msgN + .15, T.msgN + .3];
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
    const x = 48, y = 1012;                                     // LOW zone, over the TV's bottom panel, clear of the screen + captions
    ctx.save(); KP.pose(ctx, ar, x, y + 50, 'left', 760);
    KP.lowerThird(ctx, name.toUpperCase(), x, y, { boil: b, seed: 12, sub: credit });
    ctx.restore();
  }
  // ---- the big-screen framing (s01.01-.02): the hero screen is ~860 px wide across TOP+MID (client: clip big)
  const BIG = { cx: 470, cy: 826, s: 1.114 };
  // screen-space shelf in LOW (right of the lower third) for the cut-outs; top edge at y 1282, clear of captions
  function shelf(ctx, b) {
    KP.paper(ctx, KP.jitter(KP.rr(560, 1270, 380, 22, 8, 16), 3290 + b, 1.2), P.coral, { shadow: KP.SH.near, shadowA: .4 });
    for (const bx of [600, 900]) KP.paper(ctx, KP.jitter([[bx - 16, 1292], [bx + 16, 1292], [bx, 1340]], 3291 + bx + b, .8), P.coral, { shadow: KP.SH.far });
  }
  // clip 1: the shaker tips and pours gold crystals onto a plate; the plate grows on 'hungry' and again on 'eat more'
  function pourPlate(ctx, t, b, T) {
    shelf(ctx, b);
    const px = 700, py = 1270, sx = 880, sy = 1270;
    const grow = 1 + .45 * KP.backOut(KP.prog(t, T.hungry, T.hungry + .35)) + .5 * KP.backOut(KP.prog(t, T.eatMore, T.eatMore + .35));
    ctx.save(); ctx.translate(px, py); ctx.scale(grow, grow);
    KP.paper(ctx, KP.jitter(KP.ell(0, -8, 92, 20, 32), 3300 + b, 1), P.rice, { shadow: KP.SH.far, shadowA: .35, mis: P.turq, misOff: [-2, 2], misA: .4 });
    const heap = KP.prog(t, T.pour, T.pour + 1) * .3 + (grow - 1) * .35;
    if (heap > 0) {
      KP.paper(ctx, KP.jitter(KP.egg(0, -18, 74 * Math.min(1, .55 + heap), 8 + 40 * heap, -.2, 28), 3301 + b, 1), P.lemon, { shadow: false, mis: P.coral, misOff: [-2, 2], misA: .5 });
      for (let i = 0; i < 3; i++) KP.brush(ctx, KP.curve([[-40 + i * 30, -20], [-30 + i * 30, -22 - 30 * heap], [-44 + i * 30, -24 - 44 * heap]], false, 5), 7, P.coral, { taper: .3 });
    }
    const r = KP.rng(3302);
    for (let i = 0; i < Math.floor(KP.prog(t, T.pour, T.pour + 1.6) * 24); i++) { const x = (r() - .5) * 110, y = -14 - r() * 30 * (1 + heap); KP.paper(ctx, KP.rot([[x - 5, y - 5], [x + 5, y - 5], [x + 5, y + 5], [x - 5, y + 5]], x, y, r() * 3), P.gold, { shadow: false }); }
    ctx.restore();
    if (t >= T.hungry && t < T.hungry + .35) KP.marks(ctx, px, py - 40, 110 * grow, 6, 3303 + b, { a0: -Math.PI, span: Math.PI, len: 24, w: 6 });
    if (t >= T.eatMore && t < T.eatMore + .35) KP.marks(ctx, px, py - 50, 120 * grow, 7, 3304 + b, { a0: -Math.PI, span: Math.PI, len: 28, w: 6 });
    // the shaker on the right of the shelf tips left toward the plate; crystals stream while it pours
    const tip = KP.ep(t, T.pour - .25, T.pour, 'out'), shk = t >= T.pour ? Math.sin(KP.held(t) * 30) * .06 : 0;
    ctx.save(); ctx.translate(sx, sy - 70); ctx.rotate(-(tip * 1.25 + shk)); ctx.translate(0, 70); KP.shaker(ctx, 0, 0, 150, { boil: b, seed: 7 }); ctx.restore();
    if (t >= T.pour) {
      const rr = KP.rng(3305), ox = sx - Math.sin(1.25) * 75 + 30, oy = sy - 70 - Math.cos(1.25) * 75;
      for (let i = 0; i < 14; i++) { const ph = ((t - T.pour) * 1.8 + rr()) % 1, x = KP.lerp(ox, px + 30 + (rr() - .5) * 50, ph), y = KP.lerp(oy, py - 24, ph) - Math.sin(Math.PI * ph) * 18, cs = 6 + rr() * 4;
        KP.paper(ctx, KP.rot([[x - cs, y - cs], [x + cs, y - cs], [x + cs, y + cs], [x - cs, y + cs]], x, y, ph * 5), i % 3 ? P.gold : P.lemon, { shadow: false }); }
    }
  }
  // clip 2: a pink paper brain on the shelf; zap bolts burst on 'destroys' and crackle on 'brain'; it trembles
  function brainCut(ctx, t, b, T) {
    shelf(ctx, b);
    const ba = KP.arrive(t, T.c2 + .05, .3), bx = 740, by = 1170, br = 90, wb = t >= T.c2 + .8 ? KP.wob(t, 5, t >= T.over ? 5 : 2) : [0, 0];
    if (!ba.on) return;
    ctx.save(); KP.pose(ctx, ba, bx, 1270, 'pop'); ctx.translate(wb[0], wb[1]);
    [[-2.6, 1], [-1.9, .9], [-.9, 1.1], [-.2, 1], [.5, .85], [2.6, .7], [-1.3, .6], [3.0, .65]].forEach(([a, k], i) => {
      if (t < T.over) return;
      const sc = 1.2 * KP.arrive(t, T.over + i * .04, .22).p;
      if (sc <= 0 || (t > T.over + .5 && KP.hash(i, b) < .25)) return;
      KP.paper(ctx, KP.jitter(KP.bolt(bx + Math.cos(a) * br * 1.05, by + Math.sin(a) * br * .8, 54 * k * sc, a, 15 * Math.min(1.2, sc), 20 + i), 400 + i + b, 1.2), P.tomato, { shadow: KP.SH.far, mis: P.pink, misOff: [-4, 3], misA: .6 });
    });
    if (t >= T.brainW && t < T.brainW + .6) KP.marks(ctx, bx, by, br + 30, 8, 420 + b, { len: 26, w: 6, color: P.tomato });
    KP.brain(ctx, bx, by, br, { boil: b, eyes: t >= T.over ? 'worried' : 'dot', mouth: 'wobble' });
    ctx.restore();
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
    if (t >= T.and && t < T.ff0 + .32) {   // (the same envelope art as s06's, so the handoff is one object)
      const k = KP.ep(t, T.and, T.started, 'out'), land = KP.arrive(t, T.started, .2);
      const ex = KP.lerp(1300, ENV0[0], k), ey = ENV0[1] - Math.sin(Math.PI * k) * 120;
      ctx.save(); ctx.translate(ex, ey); ctx.rotate(KP.lerp(.6, 0, k)); ctx.scale(ENV0[2], ENV0[2]); if (t >= T.started) ctx.scale(land.sx, land.sy);
      const s6 = KZ.scenes.s06_origin; if (s6 && s6.api) s6.api.envelope(ctx, 0, 0, b, 0);
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

  // ===== shot 5: HANDOFF TO s06. Push into the envelope on the top table ('one letter'); it lifts toward camera,
  // the background becomes s06's cobalt desk (its frame 0, nested), and it lands at s06's envelope spot; the '1968'
  // slip pins on '1968'. From then on the frame is s06 frame 0 itself (match cut, no blank frame).
  const ENV0 = [420, 735, .62];                 // envelope on the top table (tables-sheet coords) + its scale
  function handoff(ctx, t, b, T, D) {
    const sheetY = 60 - 110, ex = ENV0[0], ey = ENV0[1] + sheetY;   // envelope in screen space at the end of shot 4
    const swap = T.ff0 + .4, push = KP.ep(t, T.ff0, swap, 'in');
    const s6c = window.CUES && window.CUES.s06_origin, sc6 = KZ.scenes.s06_origin;
    if (t < swap || !sc6) {
      const z = KP.lerp(1, 3.2, push);
      ctx.fillStyle = P.blue; ctx.fillRect(0, 0, KP.W, KP.H);
      ctx.save(); ctx.translate(KP.lerp(ex, 490, push), KP.lerp(ey, 860, push)); ctx.scale(z, z); ctx.translate(-ex, -ey);
      ctx.translate(0, sheetY); tablesSheet(ctx, Math.min(t, T.ff0), b, T);
      ctx.restore();
      KP.texture(ctx, 'grain', .1);
      return;
    }
    const api = sc6.api, E1 = api.E1, fly = KP.ep(t, swap, T.land, 'out');
    const settled = t >= D - .2;               // the last frames are exactly s06 frame 0
    const flapT = (T.y1968 + D) / 2, flap = Math.sin(Math.PI * KP.prog(t, flapT - .25, flapT + .35)) * .06;   // the flap lifts a little
    const landHop = t >= T.land ? (1 - KP.arrive(t, T.land, .2).sy) * 2 : 0;
    sc6.envOverride = settled ? null : c => {
      const x = KP.lerp(490, E1[0], fly), y = KP.lerp(860, E1[1], fly), k = KP.lerp(3.2 * ENV0[2], 1, fly);
      c.save(); c.translate(x, y); c.scale(k * (1 + landHop * .1), k * (1 - landHop * .1)); c.rotate(KP.lerp(-.1, 0, fly) - flap); c.translate(-x, -y);
      api.envelope(c, x, y, 0, 0); c.restore();
      if (t >= T.y1968) api.slip1968(c, 0, 0, KP.arrive(t, T.y1968, .3));
    };
    try { sc6.draw(ctx, 0, s6c.lines, s6c.duration); } finally { sc6.envOverride = null; }
  }
  KP.scene('s01_coldopen', {
    // shared with s09 (the loop): the TV wall drawer + a hero-screen override hook (set only around a nested draw)
    api: { wall: (...a) => wall(...a), HERO, ALL, FAR,
      // the hero screen's rect in FRAME coordinates at t = 0 (the big framing), for s09's loop pull-back
      screen0: () => { const [x, y, w, h] = HERO, S = KP.tvScreen(x, y, w, h, { panel: 'bottom' }), f = (px, py) => [(px - BIG.cx) * BIG.s + KP.CAM[0], (py - BIG.cy) * BIG.s + KP.CAM[1]];
        const [x0, y0] = f(S.x, S.y), [x1, y1] = f(S.x + S.w, S.y + S.h); return { x: x0, y: y0, w: x1 - x0, h: y1 - y0, world: S }; } },
    heroOverride: null,
    sfx(cues, D) {
      const T = times(cues, D), a = [], e = (t, name, gain) => a.push({ t: Math.max(0, +t.toFixed(3)), name, gain });
      e(0, 'tv_on', .45); e(T.pour, 'pop', .35); e(T.hungry, 'bubble', .35); e(T.eatMore, 'chomp', .35);
      e(T.c2 - .05, 'channel_switch', .45); e(T.c2, 'static', .25); e(T.over, 'glitch', .4); e(T.brainW, 'blip', .3);
      if (T.c3 < T.wall) { e(T.c3 - .05, 'channel_switch', .55); e(T.headache, 'pop_small', .35); e(T.msgG, 'pop', .35); }
      e(T.wall, 'whoosh', .45); e(T.fifty, 'pop', .4); e(T.lit, 'tv_on', .3);
      e(T.tables0 - .05, 'paper', .35); e(T.tables0 + .1, 'neon_flicker', .45); e(T.msgN, 'pop_small', .3); e(T.started, 'paper', .35);
      e(T.ff0 + .1, 'whoosh', .4); e(T.letter, 'paper', .4); e(T.y1968, 'pop', .45);
      return a.sort((x, y) => x.t - y.t);
    },
    draw(ctx, t, cues, D) {
      const T = times(cues, D), b = KP.boil(t);
      ctx.fillStyle = P.plum; ctx.fillRect(0, 0, KP.W, KP.H);
      if (t < T.tables0 + .32) {
        // ===== shots 1-3: big screen (two clips, jump-cut flicker), then a fast pull-back to the whole TV wall
        const pull = KP.ep(t, T.wall, T.wall + 1.0, 'out');
        const drift = Math.sin(KP.held(t, 30) * .6) * 4 * (1 - pull);
        const sc = KP.lerp(BIG.s, .6, pull) * (1 - .03 * KP.prog(t, T.wall + 1.0, T.tables0));
        const cx = KP.lerp(BIG.cx + drift, 560, pull), cy = KP.lerp(BIG.cy, 760, pull);
        const shot = t < T.c2 ? 0 : t < T.c3 ? 1 : t < T.wall ? 2 : 3;
        const flicker = [T.c2, T.c3].some(s => s < T.wall && t >= s - .05 && t < s + .12);
        const knob = (t >= T.c2 ? 1.2 : 0) + (t >= T.c3 && T.c3 < T.wall ? 1.2 : 0);
        KP.camera(ctx, cx, cy, sc, () => {
          wall(ctx, t, b, {
            extra: t >= T.wall - .2, knob, cues,
            lit: i => shot < 3 ? 0 : (t >= T.wall + .15 + RANK[i] * Math.max(.03, (T.lit - T.wall) / ALL.length) ? 1 : 0),
            hero: (c, S) => { const ov = KZ.scenes.s01_coldopen.heroOverride; if (ov) ov(c, S); else if (flicker) KP.static(c, S, Math.floor(t * 30)); else heroClip(c, S, shot >= 3 ? 0 : Math.min(shot, 2), shot >= 3 ? 1.5 : t, b, cues); },
          });
          if (shot === 2) shakerOnTV(ctx, t, b, T, 3);
        });
        sofa(ctx, b, pull * 700);
        if (shot < 3) {
          if (shot === 0) pourPlate(ctx, t, b, T); else if (shot === 1) brainCut(ctx, t, b, T);
          lowerThirdFor(ctx, shot === 0 ? 0 : 1, 0, t, b, cues);
        } else if (t < T.wall + .25) {   // the LOW cut-outs and the tag fly off as the camera pulls back
          const k = KP.ep(t, T.wall, T.wall + .25, 'in');
          ctx.save(); ctx.translate(0, k * 900); brainCut(ctx, t, b, T); lowerThirdFor(ctx, 1, 0, t, b, cues); ctx.restore();
        }
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
        // ===== shot 4: paper slide carrying the restaurant window + tables; the envelope arrives on 'started'
        const p = KP.ep(t, T.tables0, T.tables0 + .32, 'out'), tilt = KP.ep(t, T.tables0, T.ff0, 'inOut') * 110;
        sheetSlide(ctx, p, P.blue, c => { c.translate(0, 60 - tilt); tablesSheet(c, t, b, T); });
        if (p >= 1) KP.texture(ctx, 'grain', .1);
      }
      if (t >= T.ff0) handoff(ctx, t, b, T, D);
    },
  });
})();
