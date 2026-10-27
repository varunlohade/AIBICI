// s01_coldopen: Cold Open, The Accusations (Kombu Paper). Pilot scene for the motion style.
// Every time below is read from cues.json word timings (no hard-coded seconds), so a re-recorded narration re-times it.
// Shots (design/SHOTLIST.md s01.01-.07):
//   1 TV Doctor + zapping brain (frame 0 = style frame A, also the loop landing point)
//   2 channel flick -> Celebrity Chef stirring a pot, flicks the shaker off on 'poison', X stamp
//   3 channel flick -> Wellness Influencer inside a phone frame, headache zig-zags, shaker pops in on 'MSG'
//   4 fast pull-back: the whole wall lights up with puppets; '50 YEARS' on the lower third
//   5 paper slide -> restaurant window with the NO MSG sign switching on + three dinner tables with no-MSG cards
//   6 cut -> the shaker alone on a turquoise table under a paper spotlight; '?' drops on 'bad'
//   7 record scratch after "isn't...": freeze (boil stops), then pull back to the TV wall, all puppets point at it;
//     paper slide out carrying a falling gold drop into s02
(function () {
  const P = KP.P, TAU = Math.PI * 2;
  // ---------- cue helpers: words are found by voice + text across all lines (never by fixed index or second),
  // so a re-recorded or re-split narration re-times the scene automatically.
  const lineOf = (cues, voice) => cues.find(l => l.voice === voice) || cues[0];
  const ws = (cues, voice) => lineOf(cues, voice).words || [];
  const allWords = cues => cues.flatMap(l => (l.words || []).map(w => ({ ...w, voice: l.voice, line: l })));
  // start of the first word matching re (optionally of a voice, at/after a time); fallback if missing
  const W = (cues, re, o = {}) => {
    const w = allWords(cues).find(w => re.test(w.w) && (!o.voice || w.voice === o.voice) && w.start >= (o.after ?? -1));
    return w ? (o.end ? w.end : o.line ? w.line.start : w.start) : (o.fb ?? 0);
  };
  const times = cues => {
    const n = { voice: 'NARR' };
    const T = {
      chef: lineOf(cues, 'CHEF').start, guru: lineOf(cues, 'GURU').start, wall: lineOf(cues, 'NARR').start,
      excito: W(cues, /excito/i), over: W(cues, /overexc/i), brain: W(cues, /^brain/i),
      never: W(cues, /^never/i), poison: W(cues, /poison/i),
      headache: W(cues, /headache/i), msgG: W(cues, /msg/i, { voice: 'GURU' }),
    };
    T.fifty = W(cues, /fifty|years/i, { ...n, fb: T.wall + .3 });
    T.letters = W(cues, /letters/i, { ...n, fb: T.wall + 1.8 });
    T.terrified = W(cues, /terrif/i, { ...n, fb: T.wall + 2.4 });
    T.dinner = W(cues, /dinner/i, { ...n, fb: T.terrified + .6 });
    T.tables = W(cues, /tables/i, { ...n, fb: T.dinner + .4 });
    T.ask = W(cues, /^bad/i, { ...n, line: 1, fb: T.tables + 1 });          // the "is MSG bad for you?" line
    T.msgA = W(cues, /msg/i, { ...n, after: T.ask, fb: T.ask + .4 });
    T.bad = W(cues, /^bad/i, { ...n, fb: T.ask + 1.5 });
    T.and = W(cues, /^and$/i, { ...n, after: T.bad, fb: T.bad + 1 });
    T.isnt = W(cues, /isn/i, { ...n, after: T.bad, fb: T.and + .8 });
    T.why = W(cues, /^why/i, { ...n, after: T.bad, fb: T.isnt + .6 });
    T.everyone = W(cues, /everyone/i, { ...n, after: T.bad, fb: T.why + .4 });
    T.tables0 = W(cues, /^have$/i, { ...n, after: T.letters, fb: T.terrified - .3 }) - .1;   // shot 5 starts on 'have'
    T.pops = [T.dinner, (T.dinner + T.tables) / 2, T.tables];
    const isntEnd = (allWords(cues).find(w => /isn/i.test(w.w) && w.start > T.bad) || {}).line;
    T.scratch = Math.min(T.why - .1, isntEnd ? isntEnd.end : T.isnt + .4);   // record scratch at the end of "isn't...", freeze until 'why'
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
  const KINDS = ['doctor', 'chef', 'guru'];
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
  // a small lit set: lemon screen + a tiny puppet. The screen art is rendered once per
  // (kind, flip, colour, jaw step, gesture step, boil phase) into a canonical 300x260 canvas and reused:
  // a pure function of its key, so frames stay deterministic in any order.
  const screenCache = new Map();
  function screenArt(kind, flip, color, jq, gq, bq) {
    const key = [kind, flip, color, jq, gq, bq].join('|');
    let c = screenCache.get(key);
    if (c) return c;
    c = document.createElement('canvas'); c.width = 300; c.height = 260;
    const g = c.getContext('2d');
    g.fillStyle = color; g.fillRect(0, 0, 300, 260);
    const s = 260 / 560, px = flip ? 198 : 102, py = 260 * .62;
    if (flip) { g.translate(300, 0); g.scale(-1, 1); }
    const fx = flip ? 300 - px : px;
    KP.puppet(g, kind, fx, py, s, { boil: bq * 3 + 1, jaw: jq / 2, gesture: gq, stickTo: 270, strap: false, shadowA: .2 });
    screenCache.set(key, c);
    return c;
  }
  function smallLit(ctx, tv, i, t, b, o = {}) {
    const [x, y, w, h, opt] = tv, kind = o.kind || KINDS[i % 3];
    const S = KP.tvScreen(x, y, w, h, { panel: opt.panel || 'right' });
    const flip = o.flip ?? (i % 2 === 1);
    const jaw = o.jaw ?? (.5 + .5 * Math.sin(KP.held(t, 15) * (9 + i % 4) + i));
    const gest = o.gesture ?? Math.sin(KP.held(t) * 3 + i) * .25;
    const art = screenArt(kind, flip, o.screen || SCREENS[i % 4], Math.round(jaw * 2), Math.round(gest * 10) / 10, (b + i) % 2);
    ctx.save(); ctx.clip(KP.path(KP.rr(S.x, S.y, S.w, S.h, 20)));
    ctx.fillStyle = o.screen || SCREENS[i % 4]; ctx.fillRect(S.x, S.y, S.w, S.h);
    const k = S.h / 260, dw = 300 * k, dy = KP.wob(t, i, 2)[1];
    ctx.drawImage(art, S.x + (S.w - dw) / 2, S.y + dy, dw, S.h);
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
      if (lit > 0) smallLit(ctx, tv, i, t, b, o.litOpts ? o.litOpts(i) : {});
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

  // ---------- hero screen contents
  const bump = (t, ts, k = 7) => t < ts ? 0 : Math.exp(-(t - ts) * k);
  function heroDoctor(ctx, S, t, b, T, cues, o = {}) {
    // the overexcited brain
    const bx = S.x + S.w * .72, by = S.y + S.h * .4, br = 112;
    const shake = t >= T.over ? KP.wob(t, 5, 4) : [0, 0];
    ctx.save(); ctx.translate(shake[0], shake[1]);
    const zr = KP.rng(9), big = KP.arrive(t, T.over, .25);
    [[-2.5, 1], [-1.6, .9], [-.75, 1.1], [.1, 1], [.75, .85], [2.2, .7], [-2.05, .6], [1.45, .65]].forEach(([a, k], i) => {
      const d = br * 1.05;
      // frame 0 = style frame A (six bolts); they flicker on the boil, then burst bigger on 'overexcites'
      let sc = i < 6 ? .85 : 0;
      if (t >= T.over) { const ar = KP.arrive(t, T.over + i * .04, .22); sc = KP.lerp(sc, 1.25, ar.p); }
      if (o.calm) sc *= .6;
      if (sc <= 0) return;
      if (t > .05 && t < T.over && KP.hash(i, b) < .25) return;   // stop-motion flicker
      const len = (70 * k + zr() * 30) * sc;
      KP.paper(ctx, KP.jitter(KP.bolt(bx + Math.cos(a) * d, by + Math.sin(a) * d * .8, len, a, 18 * Math.min(1.2, sc), 20 + i), 400 + i + b, 1.2), P.tomato, { shadow: KP.SH.far, mis: P.pink, misOff: [-4, 3], misA: .6 });
    });
    const sq = 1 + .05 * big.p * Math.sin(KP.held(t, 15) * 40);
    ctx.translate(bx, by); ctx.scale(sq, 2 - sq); ctx.translate(-bx, -by);
    const brain = KP.jitter(KP.curve([[bx - br, by + 10], [bx - br * .9, by - br * .5], [bx - br * .4, by - br * .82], [bx + br * .2, by - br * .85], [bx + br * .75, by - br * .6], [bx + br * 1.02, by - br * .05], [bx + br * .8, by + br * .5], [bx + br * .2, by + br * .62], [bx - br * .5, by + br * .58]], true, 8), 300 + b, 1.5);
    KP.paper(ctx, brain, P.bubble, { shadow: KP.SH.near, mis: P.tomato, misOff: [-3, 3], misA: .45 });
    [[[bx - 80, by - 40], [bx - 40, by - 70], [bx, by - 50], [bx + 30, by - 74]], [[bx + 50, by - 30], [bx + 80, by - 10], [bx + 66, by + 26]], [[bx - 90, by + 20], [bx - 50, by + 6], [bx - 30, by + 36]]].forEach(c => KP.brush(ctx, KP.curve(c, false, 8), 8, P.magenta, { taper: .6 }));
    KP.paper(ctx, KP.jitter(KP.rr(bx + 4, by + br * .5, 34, 60, 14, 10), 33 + b, 1), P.pink, { shadow: KP.SH.far });
    KP.face(ctx, bx + 8, by + 20, 110, { eyes: t >= T.over ? 'worried' : 'dot', eyeScale: t >= T.brain ? 1.4 : 1, mouth: 'wobble', cheek: P.magenta });
    KP.brush(ctx, [[bx - br - 30, by - 70], [bx - br - 46, by - 92]], 6, P.ink, { taper: .5 });
    KP.brush(ctx, [[bx - br - 34, by - 40], [bx - br - 58, by - 46]], 6, P.ink, { taper: .5 });
    ctx.restore();
    // the puppet: jaw from the TV Doctor's word timings, finger jabs on the stressed words
    const th = KP.held(t), jab = bump(th, T.excito) + bump(th, T.over) + bump(th, T.brain);
    const jaw = o.jaw ?? (t < .05 ? .75 : KP.speak(t, ws(cues, 'TVDOC')));
    const [wx, wy] = KP.wob(t, 2, 1.5), bob = Math.sin(th * 5.5) * 4;
    const px = S.x + S.w * .3 + wx, py = S.y + S.h * .52 + bob + wy;
    KP.puppet(ctx, 'doctor', px, py, 1.05, { boil: b, jaw, gesture: o.gesture ?? -.16 * jab, stickTo: S.y + S.h + 20, shadowA: .2 });
    if (jaw > .45) {
      const mx = px + 135, my = py - 78;
      KP.brush(ctx, [[mx, my - 16], [mx + 26, my - 30]], 6, P.ink, { taper: .5 });
      KP.brush(ctx, [[mx + 6, my + 4], [mx + 36, my + 4]], 6, P.ink, { taper: .5 });
      KP.brush(ctx, [[mx, my + 24], [mx + 26, my + 38]], 6, P.ink, { taper: .5 });
    }
  }
  function heroChef(ctx, S, t, b, T, cues) {
    const th = KP.held(t), cy = S.y + S.h - 64;
    // counter
    KP.paper(ctx, KP.jitter(KP.rr(S.x - 20, cy, S.w + 40, 120, 8, 30), 610 + b, 1.4), P.magenta, { shadow: [0, -6], shadowA: .25 });
    KP.paper(ctx, KP.jitter(KP.rr(S.x - 20, cy, S.w + 40, 14, 6, 30), 611 + b, 1), P.lemon, { shadow: false });
    // pot with steam
    const potX = S.x + S.w * .6, potW = 190;
    for (let i = 0; i < 3; i++) {
      const sx = potX - 50 + i * 50, ph = th * 2 + i;
      KP.brush(ctx, KP.curve([[sx, cy - 150], [sx + 14 * Math.sin(ph), cy - 200], [sx - 10 * Math.sin(ph), cy - 250], [sx + 6, cy - 290]], false, 6), 9, P.rice, { taper: .7 });
    }
    // the shaker: sits on the counter, flicked off on 'poison' (tumbles right), X stamp lands on it
    const fl = KP.prog(t, T.poison, T.poison + .45), shX = S.x + S.w * .8 + fl * 40, shY = cy + 4 - Math.sin(Math.PI * Math.min(1, fl * 1.2)) * 120 + fl * 10;
    const shRot = fl * 1.9;
    KP.shaker(ctx, shX, shY, 150, { boil: b, rot: shRot, seed: 7 });
    if (t >= T.poison + .05) {
      const ar = KP.arrive(t, T.poison + .05, .18);
      ctx.save(); ctx.translate(shX - 10, shY - 80); ctx.rotate(-.12); ctx.scale(1 + (1 - ar.p) * 1.2, 1 + (1 - ar.p) * 1.2); ctx.globalAlpha = KP.clamp(ar.k * 3);
      for (const a of [.78, -.78]) KP.paper(ctx, KP.jitter(KP.rot(KP.rr(-110, -20, 220, 40, 10, 12), 0, 0, a), 620 + a * 10 + b, 1.4), P.tomato, { shadow: KP.SH.near, shadowA: .35, mis: P.pink, misOff: [-4, 3], misA: .5 });
      ctx.restore();
    }
    // the chef: ladle stirs the pot, then flicks up and right on 'poison'
    const jaw = KP.speak(t, ws(cues, 'CHEF')), [wx, wy] = KP.wob(t, 3, 1.5);
    const stir = 2.45 + .16 * Math.sin(th * 7), flick = KP.ep(t, T.poison - .12, T.poison + .06, 'out') * (1 - KP.ep(t, T.poison + .5, T.poison + .9, 'inOut'));
    const nev = bump(th, T.never, 5) * .25;
    const px = S.x + S.w * .26 + wx, py = S.y + S.h * .6 + wy + Math.sin(th * 5) * 3;
    KP.puppet(ctx, 'chef', px, py, .9, { boil: b, jaw, gesture: KP.lerp(stir, .9, flick) - nev, stickTo: S.y + S.h + 20, shadowA: .2 });
    // pot front (over the ladle so it dips in)
    KP.paper(ctx, KP.jitter(KP.curve([[potX - potW / 2, cy - 130], [potX + potW / 2, cy - 130], [potX + potW / 2 - 10, cy - 10], [potX + potW / 2 - 40, cy + 4], [potX - potW / 2 + 40, cy + 4], [potX - potW / 2 + 10, cy - 10]], true, 6), 612 + b, 1.3), P.grape, { shadow: KP.SH.near, shadowA: .3, mis: P.blue, misOff: [-3, 2], misA: .4 });
    KP.paper(ctx, KP.jitter(KP.rr(potX - potW / 2 - 14, cy - 142, potW + 28, 22, 8, 10), 613 + b, 1), KP.mix(P.grape, P.ink, .3), { shadow: false });
    if (jaw > .45) KP.marks(ctx, px + 122, py - 78, 30, 3, 5 + b, { a0: -.6, span: 1.2, len: 24 });
  }
  function heroGuru(ctx, S, t, b, T, cues) {
    const th = KP.held(t);
    ctx.fillStyle = P.lemon; ctx.fillRect(S.x, S.y, S.w, S.h);
    // the phone frame (vertical video inside the TV)
    const pw = 440, ph = S.h - 40, pX = S.x + S.w * .42 - pw / 2, pY = S.y + 20;
    KP.paper(ctx, KP.jitter(KP.rr(pX, pY, pw, ph + 60, 44, 20), 700 + b, 1.4), P.grape, { shadow: [10, 12], shadowA: .35 });
    const inner = KP.rr(pX + 18, pY + 34, pw - 36, ph + 20, 28, 20);
    KP.paper(ctx, KP.jitter(inner, 701 + b, 1), P.pink30, { shadow: false });
    KP.paper(ctx, KP.jitter(KP.rr(pX + pw / 2 - 40, pY + 12, 80, 12, 6, 8), 702, .6), P.ink, { shadow: false });
    ctx.save(); ctx.clip(KP.path(inner));
    const px = pX + 150, py = pY + 300 + Math.sin(th * 4) * 3;
    // ring light halo behind the head
    const hx = px + 4, hy = py - 102 * .8;
    const ring = KP.jitter(KP.ell(hx, hy, 116, 116, 48), 710 + b, 1.4), ringIn = KP.ell(hx, hy, 90, 90, 40).reverse();
    const rp = KP.path(ring); rp.addPath(KP.path(ringIn));
    ctx.save(); ctx.translate(5, 6); ctx.fillStyle = KP.rgba(P.ink, .25); ctx.fill(rp, 'evenodd'); ctx.restore();
    ctx.fillStyle = P.rice; ctx.fill(rp, 'evenodd');
    // gestures: wave while talking, clutch the head on 'headaches', point at the shaker on 'MSG'
    const clutch = KP.ep(t, T.headache - .1, T.headache + .1, 'out') * (1 - KP.ep(t, T.msgG - .25, T.msgG, 'inOut'));
    const point = KP.ep(t, T.msgG - .2, T.msgG + .05, 'out');
    const g = Math.sin(th * 6) * .22 * (1 - clutch) * (1 - point) - 1.25 * clutch + .55 * point;
    const tilt = -.12 * clutch;
    ctx.save(); ctx.translate(px, py + 120); ctx.rotate(tilt); ctx.translate(-px, -py - 120);
    KP.puppet(ctx, 'guru', px, py, .8, { boil: b, jaw: KP.speak(t, ws(cues, 'GURU')), gesture: g, stickTo: pY + ph + 80, shadowA: .2 });
    ctx.restore();
    // headache zig-zags
    if (t >= T.headache) {
      [[-2.4, 0], [-1.6, .06], [-.9, .12]].forEach(([a, dl], i) => {
        const ar = KP.arrive(t, T.headache + dl, .2);
        if (!ar.on || (KP.hash(i, b) < .2 && t > T.headache + .5)) return;
        const r0 = 104;
        ctx.save(); ctx.translate(hx, hy); ctx.scale(ar.p, ar.p); ctx.translate(-hx, -hy);
        KP.paper(ctx, KP.jitter(KP.bolt(hx + Math.cos(a) * r0, hy + Math.sin(a) * r0, 46, a, 14, 30 + i), 720 + i + b, 1), P.tomato, { shadow: KP.SH.far, mis: P.pink, misOff: [-3, 2], misA: .5 });
        ctx.restore();
      });
    }
    // the shaker pops into the phone frame on 'MSG'
    if (t >= T.msgG) {
      const ar = KP.arrive(t, T.msgG, .28), sx = pX + pw - 78, sy = pY + 360;
      ctx.save(); KP.pose(ctx, ar, sx, sy, 'pop');
      KP.shaker(ctx, sx, sy, 130, { boil: b, rot: .12, seed: 7 });
      ctx.restore();
      if (ar.k >= 1) KP.marks(ctx, sx, sy - 70, 80, 5, 9 + b, { a0: -2.8, span: 2.4, len: 22, w: 5 });
    }
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
    ctx.save(); KP.pose(ctx, ar, SHK[0], SHK[1], 'drop', 500);
    KP.shaker(ctx, SHK[0], SHK[1], 330, { boil: tb, seed: 7, shadowA: .4 });
    ctx.restore();
    if (sweat) KP.marks(ctx, SHK[0], SHK[1] - 170, 190, 4, 70 + tb, { a0: -2.6, span: 2.1, len: 26 });
    // the '?' drops on 'bad' (boing)
    const q = KP.arrive(t, T.bad, .36);
    if (q.on) {
      const qx = SHK[0] + 10 + back * 210, qy = SHK[1] - 470 + back * 250, dropY = (1 - q.p) * -600;
      ctx.save(); ctx.translate(qx, qy + 90); ctx.scale(q.sx, q.sy); ctx.translate(-qx, -qy - 90);
      KP.qmark(ctx, qx, qy + dropY, 230 - back * 60, { rot: .08 + KP.wobA(tb / KP.BOIL_FPS, 3, .03) });
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
    sfx(cues, D) {
      const T = times(cues), a = [];
      const e = (t, name, gain) => a.push({ t: Math.max(0, +t.toFixed(3)), name, gain });
      e(0, 'tv_on', .5); e(T.excito, 'static', .25); e(T.over, 'glitch', .35); e(T.brain, 'blip', .3);
      e(T.chef - .05, 'channel_switch', .55); e(T.never, 'pop_small', .3); e(T.poison, 'stamp', .4);
      e(T.guru - .05, 'channel_switch', .55); e(T.headache, 'pop_small', .35); e(T.msgG, 'pop', .35);
      e(T.wall, 'whoosh', .45); e(T.fifty, 'pop', .4); e(T.letters, 'tv_on', .3);
      e(T.tables0 - .05, 'paper', .4); e(T.terrified, 'neon_flicker', .45); T.pops.forEach((p, i) => e(p, 'pop_small', i === 1 ? .2 : .3));
      e(T.msgA, 'pop', .4); e(T.bad, 'boing', .3);
      e(T.scratch, 'record_scratch', .65); e(T.why, 'pop', .4);
      e(D - .6, 'whoosh', .4);
      return a.sort((x, y) => x.t - y.t);
    },
    draw(ctx, t, cues, D) {
      const T = times(cues);
      // record-scratch freeze: the boil and idle motion stop between the scratch and 'why'
      const frozen = t >= T.scratch && t < T.why;
      const tb = frozen ? T.scratch : t, b = KP.boil(tb);
      ctx.fillStyle = P.plum; ctx.fillRect(0, 0, KP.W, KP.H);

      if (t < T.tables0 + .32) {
        // ===== shots 1-4: the TV wall
        const pull = KP.ep(t, T.wall, T.wall + 1.3, 'out'), s = KP.lerp(1.03 * (1 - 0) , .6, pull);
        const push = 1 + .03 * KP.ep(t, 0, T.wall, 'inOut');       // slow multiplane push-in on shots 1-3
        const sc = t < T.wall ? push : KP.lerp(1.03, .6, pull) * (1 - .03 * KP.prog(t, T.wall + 1.3, T.tables0));
        const cx = KP.lerp(KP.CAM[0], 560, pull), cy = KP.lerp(KP.CAM[1], 760, pull);
        const truck = Math.sin(t * .3) * 6;
        const shot = t < T.chef ? 'doc' : t < T.guru ? 'chef' : t < T.wall ? 'guru' : 'wall';
        const sw = sh => t >= sh - .05 && t < sh + .12;         // channel flicker window
        const flicker = sw(T.chef) || sw(T.guru);
        const knob = (t >= T.chef ? 1.2 : 0) + (t >= T.guru ? 1.2 : 0);
        KP.camera(ctx, cx + truck * .4, cy, sc, () => {
          const S = wall(ctx, t, b, {
            extra: t >= T.wall - .2, knob,
            heroScreen: shot === 'chef' ? P.bubble : P.lemon,
            lit: i => shot !== 'wall' ? 0 : (t >= T.wall + .15 + RANK[i] * Math.max(.04, (T.letters - T.wall - .3) / ALL.length) ? 1 : 0),
            hero: (c, S) => {
              if (flicker) { KP.static(c, S, Math.floor(t * 30)); return; }
              if (shot === 'doc' || shot === 'wall') heroDoctor(c, S, t, b, T, cues, shot === 'wall' ? { jaw: .5 + .5 * Math.sin(KP.held(t, 15) * 11), gesture: Math.sin(KP.held(t) * 4) * .2 } : {});
              else if (shot === 'chef') heroChef(c, S, t, b, T, cues);
              else heroGuru(c, S, t, b, T, cues);
            },
          });
        });
        sofa(ctx, b, pull * 700);
        // lower third: role + DRAMATIZATION (one label at a time); slides in from the left on each channel
        const [hx] = HERO, ltY = 372 + 50 + 520 - 132;
        let role = 'TV DOCTOR', t0 = -1, y = ltY;
        if (shot === 'chef') { role = 'CELEBRITY CHEF'; t0 = T.chef; }
        else if (shot === 'guru') { role = 'WELLNESS INFLUENCER'; t0 = T.guru; }
        else if (shot === 'wall') { role = null; t0 = T.wall + .2; y = 1196; }
        const ar = t0 < 0 ? { p: 1, sx: 1, sy: 1 } : KP.arrive(t, t0, .32);
        if (t0 < 0 || ar.on) {
          ctx.save(); KP.pose(ctx, ar, hx + 30, y + 60, 'left', 760);
          KP.lowerThird(ctx, role, shot === 'wall' ? 64 : hx + 30, y, { boil: b, seed: shot === 'wall' ? 15 : 12 });
          ctx.restore();
        }
        // '50 YEARS': its own white torn slip (the one label), separate from the puppets' DRAMATIZATION line
        if (shot === 'wall' && t >= T.fifty) {
          const r2 = KP.arrive(t, T.fifty, .3);
          ctx.save(); KP.pose(ctx, r2, 300, 330, 'pop');
          KP.slip(ctx, [{ s: '50 YEARS', size: 96, font: 'Young Serif', weight: 400, ls: 3 }], 90, 220, { rot: -.04, seed: 51, boil: b, tape: P.pink, padX: 40, padY: 22 });
          ctx.restore();
        }
        KP.texture(ctx, 'grain', .1);
        KP.texture(ctx, 'speck', .28, 17, 41);
      }
      if (t >= T.tables0 && t < T.ask) {
        // ===== shot 5: paper slide in, restaurant + tables, slight tilt down the stack
        const p = KP.ep(t, T.tables0, T.tables0 + .32, 'out'), tilt = KP.ep(t, T.tables0, T.ask, 'inOut') * 110;
        sheetSlide(ctx, p, P.blue, c => { c.translate(0, 60 - tilt); tablesSheet(c, t, b, T); });
        if (p >= 1) KP.texture(ctx, 'grain', .1);
      }
      if (t >= T.ask) {
        // ===== shots 6-7: cut to the shaker under a spotlight, then pull back to the wall (puppets point at it)
        const back = KP.ep(t, T.why, T.why + 1.1, 'out');
        const sc = KP.lerp(1.55 + .08 * KP.prog(tb, T.ask, T.scratch), .85, back);
        const cx = KP.lerp(SHK[0], 490, back), cy = KP.lerp(SHK[1] - 230, 945, back);
        const shakeS = t >= T.scratch && t < T.scratch + .12 ? KP.wob(t * 4, 9, 10) : [0, 0];
        ctx.save(); ctx.translate(shakeS[0], shakeS[1]);
        KP.camera(ctx, cx, cy, sc, () => {
          wall(ctx, tb, b, {
            extra: true,
            lit: i => (back > 0 && (i === 0 || i === 2 || i === 5 || i === 7) && t >= T.why + .1 + (i % 3) * .08) ? 1 : 0,
            litOpts: i => ({ kind: { 0: 'chef', 2: 'guru', 5: 'guru', 7: 'chef' }[i], flip: i === 2 || i === 7, gesture: 2.05 + Math.sin(KP.held(tb) * 5 + i) * .08, jaw: .6 }),
            heroScreen: back > 0 ? P.lemon : P.screenDim,
            hero: back > 0 ? (c, S) => {
              c.fillStyle = P.lemon; c.fillRect(S.x, S.y, S.w, S.h);
              const px = S.x + S.w * .4, py = S.y + S.h * .5 + KP.wob(tb, 2, 1.5)[1];
              KP.puppet(c, 'doctor', px, py, 1.0, { boil: b, jaw: .7, gesture: 2.0 + Math.sin(KP.held(tb) * 6) * .08, stickTo: S.y + S.h + 20, shadowA: .2 });
            } : null,
          });
        });
        // close-up backdrop: a violet paper sheet with a paper spotlight; it lifts away on 'why' to reveal the wall
        if (back < 1) {
          ctx.save(); ctx.translate(0, -back * 2100);
          KP.paper(ctx, KP.jitter(KP.rr(-40, -200, KP.W + 80, KP.H + 260, 18, 40), 940 + b, 2), P.plum, { shadow: [0, 24], shadowA: .4 });
          KP.paper(ctx, KP.jitter([[380, -200], [600, -200], [1000, 1700], [-20, 1700]], 941 + b, 2), P.grape, { shadow: false });
          ctx.restore();
        }
        KP.camera(ctx, cx, cy, sc, () => {
          if (back < .05) spotlight(ctx);
          shakerTable(ctx, t, b, T, KP.boil(tb), back);
        });
        ctx.restore();
        if (back > 0) {
          const ar = KP.arrive(t, T.why + .15, .3);
          ctx.save(); KP.pose(ctx, ar, 64, 1240, 'left', 760);
          KP.lowerThird(ctx, null, 64, 1196, { boil: b, seed: 15 });
          ctx.restore();
        }
        KP.texture(ctx, 'grain', .1);
        KP.texture(ctx, 'speck', .22, 17, 41);
      }
      // ===== out: paper slide carrying a falling gold drop into s02
      const outP = KP.ep(t, D - .6, D, 'inOut');
      if (outP > 0) {
        sheetSlide(ctx, outP, P.turq, c => {
          const dy = 300 + KP.ep(t, D - .5, D, 'in') * 380;
          KP.paper(c, KP.jitter(KP.drop(490, dy, 130), 960 + b, 1.2), P.gold, { shadow: KP.SH.near, mis: P.tomato, misOff: [-3, 2], misA: .4 });
          KP.brush(c, [[490, dy - 150], [490, dy - 230]], 6, P.ink, { taper: .6 });
        });
      }
    },
  });
})();
