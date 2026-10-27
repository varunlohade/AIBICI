// s05_body: What happens when you eat it (Kombu Paper). Shots s05.01-.07 of design/SHOTLIST.md.
//   1 a fork lifts a bite sprinkled with tiny Goldies to a paper-doll mouth (chomp on 'eat')
//   2 tongue: Goldies dock on receptor cups (small gold ripples), a signal zips up a nerve, the brain says 'Protein!'
//   3 dive down through pink paper layers into the gut; the front cell chomps on 'eaten'
//   4 style frame B: gut cells eat the Goldie stream, the next cell burns one as fuel, 'Fuel!'
//   5 pull-back to the gut wall: '95%' rolls up, only a trickle of Goldies reaches the blood vessel
//   6 track up the vessel to a cobalt castle wall: 'Blood-brain barrier'
//   7 friendly guards at the gate: a guard raises a mitten, Goldie bounces back and shrugs; paper slide out
// All timings are looked up from cues.json words (KP.word), never hard-coded.
(function () {
  const P = KP.P, TAU = Math.PI * 2;
  const times = (cues, D) => {
    const w = (re, o) => KP.word(cues, re, o);
    const T = { happens: w(/happens/i, { fb: .8 }), eat: w(/^eat/i, { fb: 1.7 }) };
    T.dive = w(/^it/i, { after: T.eat, fb: T.eat + .2 });                  // the dive starts on the bite
    T.l3 = T.dive;
    T.l4 = w(/^cells/i, { line: 1, fb: T.dive + .5 }); T.cells = w(/^cells/i, { fb: T.l4 + .25 });
    T.gut = w(/^gut/i, { after: T.cells, fb: T.cells + .9 }); T.burn = w(/^burn/i, { fb: T.gut + .25 }); T.fuel = w(/fuel/i, { fb: T.burn + 1.2 });
    T.eaten = T.cells;
    T.around = w(/^around/i, { fb: T.fuel + .7 }); T.ninety = w(/ninety|95/i, { fb: T.around + .45 });
    T.never = w(/^never/i, { after: T.ninety, fb: T.ninety + 1.4 }); T.blood = w(/^blood\.?$/i, { fb: T.never + 1.6 });
    T.l5 = w(/barrier/i, { line: 1, fb: T.blood + .6 }); T.bbb = w(/blood-brain|barrier/i, { after: T.l5 - .01, fb: T.l5 + 1.1 });
    T.one = w(/^one$/i, { after: T.bbb, fb: T.bbb + 1.1 });
    T.best = w(/guarded|best/i, { after: T.bbb, fb: T.one + .6 }); T.body = w(/^body/i, { after: T.bbb, fb: T.best + 1.5 });
    T.s5 = T.around - .1; T.s7 = w(/^of$/i, { after: T.one, fb: T.one + .2 }) - .1;   // cut just before 'of'
    T.chomps = [T.cells, T.gut, (T.burn + T.fuel) / 2];
    return T;
  };

  // ======================================================== shot 1: fork to mouth
  function shotFork(ctx, t, b, T) {
    ctx.fillStyle = P.turq; ctx.fillRect(0, 0, KP.W, KP.H);
    const camY = KP.lerp(110, 0, KP.ep(t, T.happens - .3, T.eat, 'inOut'));
    ctx.save(); ctx.translate(0, camY);
    // the diner (behind the table)
    const open = KP.ep(t, T.eat - .55, T.eat - .2, 'out') * (1 - KP.ep(t, T.eat, T.eat + .12, 'out'));
    const chew = t > T.eat ? .5 + .5 * Math.sin(KP.held(t) * 16) : 0;
    head(ctx, 520, 520, b, open, t > T.eat ? 'happy' : 'dot', chew * (t < T.eat + 1.2 ? 1 : 0), t);
    // table + plate (LOW)
    KP.paper(ctx, KP.jitter(KP.rr(-60, 990, 1200, 1100, 12, 40), 10 + b, 1.6), P.coral, { shadow: [0, -10], shadowA: .3, mis: P.pink, misOff: [0, -4], misA: .4 });
    for (let i = 0; i < 6; i++) KP.paper(ctx, KP.jitter(KP.rr(-40 + i * 200, 990, 90, 1100, 6, 40), 11 + i + b, 1.2), KP.mix(P.coral, P.pink, .35), { shadow: false });
    KP.paper(ctx, KP.jitter(KP.ell(420, 1070, 250, 62, 48), 12 + b, 1.4), P.rice, { shadow: KP.SH.near, shadowA: .35, mis: P.turq, misOff: [-3, 3], misA: .4 });
    KP.paper(ctx, KP.jitter(KP.ell(420, 1066, 170, 38, 40), 13 + b, 1), P.pink30, { shadow: false });
    for (const [x, y, s] of [[360, 1050, 1], [470, 1060, .9], [420, 1038, .8]]) bite(ctx, x, y, s * .8, b, 20 + x);
    // the fork: rises from the plate to the mouth, delivers the bite, retracts
    const up = KP.ep(t, T.happens, T.eat - .1, 'inOut'), back = KP.ep(t, T.eat + .1, T.eat + .7, 'inOut');
    const tip = [KP.lerp(400, 520, up) - back * 80, KP.lerp(1010, 850, up) + back * 420];
    fork(ctx, tip[0], tip[1], KP.lerp(-.35, 0, up) - back * .3, b, t < T.eat);
    ctx.restore();
    if (t > T.eat && t < T.eat + .35) KP.marks(ctx, 520, 690 + camY, 120, 6, 30 + b, { a0: Math.PI * .9, span: Math.PI * 1.2, len: 26 });
  }
  function bite(ctx, x, y, s, b, sd) {
    KP.paper(ctx, KP.jitter(KP.egg(x, y, 44 * s, 32 * s, .1, 28), sd + b, 1), P.tomato, { shadow: KP.SH.far, mis: P.pink, misOff: [-2, 2], misA: .45 });
    KP.paper(ctx, KP.jitter(KP.egg(x - 10 * s, y - 8 * s, 14 * s, 9 * s, 0, 16), sd + 1, .6), P.coral, { shadow: false });
    for (let i = 0; i < 3; i++) KP.goldie(ctx, x - 22 * s + i * 22 * s, y - 34 * s - (i % 2) * 8 * s, 30 * s, { seed: sd + i, boil: b, steam: false, shadow: KP.SH.far, eyes: i === 1 ? 'happy' : 'dot' });
  }
  function fork(ctx, x, y, a, b, loaded) {
    ctx.save(); ctx.translate(x, y); ctx.rotate(a);
    // handle down-right, tines up
    KP.paper(ctx, KP.jitter(KP.rr(-14, 40, 28, 560, 12, 20), 40 + b, 1.2), P.blue, { shadow: KP.SH.near, shadowA: .35, mis: P.grape, misOff: [-3, 2], misA: .4 });
    KP.paper(ctx, KP.jitter(KP.curve([[-50, -40], [50, -40], [44, 30], [16, 54], [-16, 54], [-44, 30]], true, 6), 41 + b, 1.2), P.blue, { shadow: KP.SH.near, shadowA: .35 });
    for (const tx of [-40, -14, 12, 38]) KP.paper(ctx, KP.jitter(KP.rr(tx - 6, -130, 13, 100, 6, 12), 42 + tx + b, .8), P.blue, { shadow: KP.SH.far });
    // a mitten hand holding the end of the handle
    KP.paper(ctx, KP.jitter(KP.egg(0, 520, 58, 50, .05, 32), 43 + b, 1.2), P.bubble, { shadow: KP.SH.near, mis: P.blue, misOff: [2, -2], misA: .3 });
    KP.paper(ctx, KP.jitter(KP.egg(-44, 492, 18, 26, 0, 20), 44 + b, .8), P.bubble, { shadow: false });
    if (loaded) bite(ctx, 0, -150, 1.1, b, 60);
    ctx.restore();
  }
  // a front-facing paper-doll diner: two eyes, cheeks, a mouth that opens for the bite and chews
  function head(ctx, x, y, b, open, eyes, chew, t) {
    const bob = Math.sin(KP.held(t) * 3) * 4;
    // shoulders + neck
    KP.paper(ctx, KP.jitter(KP.curve([[x - 330, y + 560], [x - 280, y + 320], [x - 120, y + 260], [x + 120, y + 260], [x + 280, y + 320], [x + 330, y + 560]], true, 8), 70 + b, 1.6), P.blue, { shadow: KP.SH.near, mis: P.grape, misOff: [3, -2], misA: .35 });
    KP.paper(ctx, KP.jitter(KP.rr(x - 50, y + 170, 100, 110, 20, 12), 71 + b, 1), P.skin, { shadow: KP.SH.far, shadowA: .25 });
    ctx.save(); ctx.translate(0, bob);
    // hair back, head, ears, hair fringe
    KP.paper(ctx, KP.jitter(KP.egg(x, y - 20, 230, 240, 0, 56), 72 + b, 1.6), P.grape, { shadow: KP.SH.near, shadowA: .3 });
    for (const sx of [-1, 1]) KP.paper(ctx, KP.jitter(KP.egg(x + sx * 196, y + 30, 30, 40, 0, 20), 73 + sx + b, 1), P.skin, { shadow: KP.SH.far, shadowA: .25 });
    KP.paper(ctx, KP.jitter(KP.egg(x, y + 20, 196, 214, .06, 56), 74 + b, 1.5), P.skin, { shadow: KP.SH.near, shadowA: .3, mis: P.coral, misOff: [-3, 2], misA: .35 });
    KP.paper(ctx, KP.jitter(KP.curve([[x - 200, y - 40], [x - 170, y - 150], [x - 60, y - 200], [x + 70, y - 200], [x + 180, y - 140], [x + 204, y - 30], [x + 120, y - 110], [x - 10, y - 130], [x - 120, y - 100]], true, 8), 75 + b, 1.4), P.grape, { shadow: KP.SH.far, shadowA: .3 });
    // cheeks (puffed while chewing), eyes, mouth
    ctx.save(); ctx.globalCompositeOperation = 'multiply'; ctx.globalAlpha = .45; ctx.fillStyle = P.pink;
    for (const sx of [-1, 1]) { ctx.beginPath(); ctx.ellipse(x + sx * 110, y + 90, 42 + chew * 14, 28 + chew * 10, 0, 0, TAU); ctx.fill(); }
    ctx.restore();
    for (const sx of [-1, 1]) {
      const ex = x + sx * 72, ey = y + 10;
      if (eyes === 'happy') KP.brush(ctx, KP.arcPts(ex, ey + 8, 22, 18, Math.PI * 1.1, Math.PI * 1.9, 8), 11, P.ink, { taper: .3 });
      else { ctx.fillStyle = P.ink; ctx.beginPath(); ctx.ellipse(ex, ey, 14, 17, 0, 0, TAU); ctx.fill(); ctx.fillStyle = P.rice; ctx.beginPath(); ctx.arc(ex - 4, ey - 6, 4, 0, TAU); ctx.fill(); }
      KP.brush(ctx, [[ex - 26, ey - 46 - open * 10], [ex + 22, ey - 50 - open * 12]], 9, P.ink, { taper: .4 });
    }
    const my = y + 160;
    if (open > .05) { ctx.fillStyle = P.mouth; ctx.beginPath(); ctx.ellipse(x, my, 34 + open * 26, 10 + open * 46, 0, 0, TAU); ctx.fill(); ctx.fillStyle = P.pink; ctx.beginPath(); ctx.ellipse(x, my + open * 26, 22 + open * 10, 8 + open * 10, 0, 0, TAU); ctx.fill(); }
    else KP.brush(ctx, KP.arcPts(x, my - 14 + chew * 6, 34, 18, Math.PI * .15, Math.PI * .85, 10), 9, P.ink, { taper: .4 });
    ctx.restore();
  }

  // ======================================================== shots 3-4: dive into the gut, style frame B
  const BANDS = [P.pink, P.magenta, P.bubble, P.grape, P.pink30, P.magenta, P.bubble];
  function drawCell(ctx, [cx, cy, cw, id], b, o = {}) {
    const chh = cw * 1.08;
    KP.arm(ctx, cx - cw * .38, cy - chh * .05, cw * .3, 2.3 + (o.wave || 0), cw * .17, P.bubble, { seed: id, boil: b });
    KP.arm(ctx, cx + cw * .38, cy - chh * .05, cw * .3, .85 - (o.wave || 0), cw * .17, P.bubble, { seed: id + 9, boil: b });
    return KP.cell(ctx, cx, cy - chh * .3, cw, chh, { seed: id, boil: b, fill: P.bubble, bib: P.rice, ...o });
  }
  const STREAM = KP.curve([[-120, 470], [80, 520], [240, 640], [180, 820], [262, 990]], false, 30);
  const along = (path, u) => { const n = path.length - 1, i = Math.min(n - 1, Math.floor(u * n)), k = u * n - i, p = path[i], q = path[i + 1]; return [p[0] + (q[0] - p[0]) * k, p[1] + (q[1] - p[1]) * k, Math.atan2(q[1] - p[1], q[0] - p[0])]; };
  function gutB(ctx, t, b, T) {
    const cam = (t - T.l3) * 7;
    ctx.fillStyle = P.turq; ctx.fillRect(-200, -200, KP.W + 400, KP.H + 400);
    ctx.save(); ctx.translate(-cam * .3, 0);
    const vr = KP.rng(12), vil = [[-160, 760]];
    for (let x = -140; x < KP.W + 160; x += 120) { const hgt = 170 + vr() * 120, yb = 760 - x * .05; vil.push([x, yb - hgt * .2], [x + 30, yb - hgt], [x + 85, yb - hgt * .95], [x + 110, yb - hgt * .15]); }
    vil.push([KP.W + 160, 1100], [-160, 1100]);
    KP.paper(ctx, KP.jitter(KP.curve(vil, true, 5), 5 + b, 1.5), P.pink30, { shadow: KP.SH.far, mis: P.blue, misOff: [3, -2], misA: .3 });
    ctx.restore();
    // back tier
    ctx.save(); ctx.translate(-cam * .6, 0);
    const backs = [[350, 852, 134, 5], [520, 846, 150, 4], [690, 862, 168, 3]];
    const backO = [{ eyes: 'dot', mouth: 'o', look: -.06 }, { eyes: 'happy', mouth: 'smile' }, { eyes: 'dot', mouth: 'smile', look: -.08 }];
    backs.forEach((c, i) => { const bob = Math.sin(KP.held(t) * 4 + i * 2) * 3; drawCell(ctx, [c[0], c[1] + bob, c[2], c[3]], b, backO[i]); });
    KP.paper(ctx, KP.jitter(KP.curve([[-160, 870], [200, 868], [500, 852], [760, 880], [960, 940], [1240, 980], [1240, 1400], [-160, 1400]], true, 10), 6 + b, 1.5), P.pink, { shadow: KP.SH.near, mis: P.blue, misOff: [0, -3], misA: .3 });
    backs.forEach(([cx, cy, cw], i) => { for (const sx of [-1, 1]) KP.paper(ctx, KP.jitter(KP.egg(cx + sx * cw * .3, cy + 4, cw * .09, cw * .07, 0, 16), 90 + i * 2 + sx + b, .8), P.bubble, { shadow: KP.SH.far }); });
    ctx.restore();
    ctx.save(); ctx.translate(-cam, 0);
    // the cell burning fuel: flame + sparkles pop on 'burn'
    drawCell(ctx, [830, 1150, 236, 2], b, { eyes: 'happy', mouth: 'smile', look: .03 });
    const fa = KP.arrive(t, T.burn, .3);
    if (fa.on) {
      const fx = 840, fy = 896, fl = 1 + .06 * Math.sin(KP.held(t) * 30);
      ctx.save(); KP.pose(ctx, { ...fa, sy: fa.sy * fl }, fx, fy + 44, 'pop');
      KP.paper(ctx, KP.jitter(KP.curve([[fx, fy - 110], [fx + 22, fy - 60], [fx + 52, fy - 30], [fx + 50, fy + 18], [fx + 22, fy + 44], [fx - 22, fy + 44], [fx - 50, fy + 14], [fx - 44, fy - 34], [fx - 26, fy - 22], [fx - 12, fy - 64]], true, 6), 81 + b, 1.4), P.gold, { shadow: KP.SH.near, mis: P.tomato, misOff: [-2.5, 2], misA: .4 });
      KP.paper(ctx, KP.jitter(KP.curve([[fx + 2, fy - 36], [fx + 22, fy + 2], [fx + 16, fy + 30], [fx - 14, fy + 30], [fx - 22, fy + 6]], true, 6), 82 + b, 1), P.lemon, { shadow: false });
      ctx.restore();
      [[fx + 92, fy - 60, 22, .15], [fx - 88, fy - 16, 16, .3], [fx + 70, fy + 40, 12, .45]].forEach(([x, y, r, d]) => {
        const sa = KP.arrive(t, T.burn + d, .25); if (!sa.on) return; const rr = r * sa.p;
        KP.paper(ctx, KP.jitter([[x, y - rr], [x + rr * .25, y - rr * .25], [x + rr, y], [x + rr * .25, y + rr * .25], [x, y + rr], [x - rr * .25, y + rr * .25], [x - rr, y], [x - rr * .25, y - rr * .25]], 7 + b, .6), P.gold, { shadow: KP.SH.far, mis: P.tomato, misOff: [-2, 2], misA: .4 });
      });
    }
    const front = (yTop, col, sd, mis) => KP.paper(ctx, KP.jitter(KP.curve([[-160, yTop + 30], [180, yTop], [460, yTop - 6], [760, yTop + 20], [1240, yTop + 90], [1240, KP.H + 60], [-160, KP.H + 60]], true, 10), sd + b, 1.8), col, { shadow: [0, -8], shadowA: .2, mis, misOff: [0, -3], misA: .3 });
    front(1140, P.pink, 1, P.blue);
    // the front cell: chomps on 'eaten', 'cells', 'intestines' (and once more before 'fuel')
    const cw = 330, chh = cw * 1.08, ccx = 320, ccy = 1230 - chh * .3;
    const tc = T.chomps.find(c => t >= c - .4 && t < c + .14);
    const chomping = tc !== undefined;
    KP.arm(ctx, ccx - cw * .4, ccy + chh * .14, cw * .32, 3.6 + (chomping ? .2 : 0), cw * .16, P.bubble, { seed: 41, boil: b, thumb: 1 });
    KP.arm(ctx, ccx + cw * .4, ccy + chh * .14, cw * .3, -.55 - (chomping ? .2 : 0), cw * .16, P.bubble, { seed: 42, boil: b, thumb: 1 });
    const m = KP.cell(ctx, ccx, ccy, cw, chh, { seed: 1, boil: b, fill: P.bubble, bib: P.rice, eyes: t > T.eaten - .4 ? 'happy' : 'dot', mouth: chomping ? 'chomp' : 'smile', mouthDx: -.1, look: -.06 });
    front(1320, P.magenta, 2, P.grape);
    front(1520, P.bubble, 3, P.blue);
    front(1720, P.grape, 4, P.pink);
    // the Goldie stream drifting in (starts during the dive)
    const flow = Math.max(0, t - T.l3) * .06;
    for (let i = 0; i < 6; i++) {
      const u = ((i / 6 + flow) % 1) * .9 + .02; if (t < T.l3 + i * .25) continue;
      const [x, y, a] = along(STREAM, u), hgt = 64 + u * 40;
      KP.brush(ctx, [[x - Math.cos(a) * hgt * .75, y - Math.sin(a) * hgt * .75 - 10], [x - Math.cos(a) * hgt * 1.1, y - Math.sin(a) * hgt * 1.1 - 10]], 5, P.ink, { taper: .6 });
      KP.goldie(ctx, x, y, hgt, { seed: 10 + i, boil: b, rot: Math.sin(KP.held(t) * 2 + i) * .15, eyes: i % 3 === 1 ? 'happy' : 'dot', mouth: i % 2 ? 'o' : 'smile', look: 3 });
    }
    // the Goldie going in: clipped to (outside the cell) + (the mouth)
    if (m && chomping && t < tc) {
      const k = KP.ease.in(KP.prog(t, tc - .4, tc));
      ctx.save();
      const clip = new Path2D(); clip.rect(-300, -300, KP.W + 600, KP.H + 600);
      clip.addPath(KP.path(KP.egg(ccx, ccy, cw / 2, chh / 2, .1, 60)));
      clip.addPath(KP.path(KP.ell(m.mx, m.my, m.mw, m.mh, 32)));
      ctx.clip(clip, 'evenodd');
      KP.goldie(ctx, KP.lerp(m.mx - 120, m.mx - 10, k), KP.lerp(m.my - 120, m.my + 4, k), 112, { seed: 3, boil: b, rot: -1.0, eyes: 'happy', mouth: 'open', shadow: KP.SH.far });
      ctx.restore();
      KP.brush(ctx, [[m.mx - 60, m.my - 70], [m.mx - 80, m.my - 96]], 6, P.ink, { taper: .5 });
      KP.brush(ctx, [[m.mx - 88, m.my - 40], [m.mx - 118, m.my - 50]], 6, P.ink, { taper: .5 });
    }
    ctx.restore();
    // foreground scraps
    const deep = KP.mix(P.magenta, P.ink, .2);
    ctx.save(); ctx.translate(-cam * 1.3, 0);
    KP.paper(ctx, KP.torn(KP.curve([[1240, 1600], [960, 1700], [900, 1860], [930, 1980], [1240, 1980]], true, 8), 72 + b), deep, { shadow: [-10, -6] });
    KP.paper(ctx, KP.torn(KP.curve([[-140, 1740], [150, 1770], [230, 1880], [210, 1980], [-140, 1980]], true, 8), 73 + b), deep, { shadow: [10, -6] });
    ctx.restore();
  }
  function shotDive(ctx, t, b, T) {
    // camera dives from 2600 px above the gut, fast then easing out; pink paper layers rush up past it
    const camY = -2600 * (1 - KP.ep(t, T.l3, T.l3 + 1.0, 'out'));
    ctx.save(); ctx.translate(0, -camY); gutB(ctx, t, b, T); ctx.restore();
    BANDS.forEach((c, i) => {
      const y0 = -3300 + i * 380, k = .85 + (i % 3) * .2, y = y0 - camY * k;
      if (y > KP.H || y + 520 < 0) return;
      KP.paper(ctx, KP.torn(KP.rr(-60, y, KP.W + 120, 520, 10, 60), 200 + i + b, 8, 22), c, { shadow: [0, 16], shadowA: .3, mis: i % 2 ? P.blue : P.grape, misOff: [0, 5], misA: .3 });
    });
  }

  // ======================================================== shots 5-7: the map (gut wall -> vessel -> castle)
  const VESSEL = KP.curve([[-160, 1170], [300, 1170], [640, 1160], [790, 1060], [810, 800], [800, 400], [800, -100], [800, -460]], false, 24);
  const VLEN = []; { let L = 0; VESSEL.forEach((p, i) => { if (i) L += Math.hypot(p[0] - VESSEL[i - 1][0], p[1] - VESSEL[i - 1][1]); VLEN.push(L); }); }
  const vAt = s => { s = KP.clamp(s, 0, VLEN[VLEN.length - 1]); let i = 1; while (i < VLEN.length - 1 && VLEN[i] < s) i++; const k = (s - VLEN[i - 1]) / ((VLEN[i] - VLEN[i - 1]) || 1), p = VESSEL[i - 1], q = VESSEL[i]; return [p[0] + (q[0] - p[0]) * k, p[1] + (q[1] - p[1]) * k]; };
  const SGATE = VLEN[VLEN.length - 1];
  const WALLC = [[110, 810, 1], [290, 800, 2], [470, 812, 3], [650, 802, 4]];
  function mapWorld(ctx, t, b, T) {
    ctx.fillStyle = P.turq; ctx.fillRect(-2000, -3000, 5000, 7000);
    // tissue between the gut and the head: torn pink paper banks either side of the vessel, with capillaries
    KP.paper(ctx, KP.torn(KP.rr(-700, -470, 1420, 1060, 10, 60), 310 + b, 10, 26), P.pink30, { shadow: [0, 10], shadowA: .25, mis: P.blue, misOff: [3, -2], misA: .25 });
    KP.paper(ctx, KP.torn(KP.rr(890, -470, 900, 1060, 10, 60), 311 + b, 10, 26), P.pink30, { shadow: [0, 10], shadowA: .25, mis: P.blue, misOff: [3, -2], misA: .25 });
    KP.paper(ctx, KP.torn(KP.curve([[-700, 80], [-200, 20], [300, 140], [640, 60], [640, 590], [-700, 590]], true, 10), 312 + b, 8, 22), P.bubble, { shadow: [0, -8], shadowA: .2 });
    KP.paper(ctx, KP.torn(KP.curve([[900, 200], [1200, 120], [1800, 220], [1800, 590], [900, 590]], true, 10), 313 + b, 8, 22), P.bubble, { shadow: [0, -8], shadowA: .2 });
    [[[700, -260], [520, -320], [380, -250], [220, -330]], [[700, 160], [560, 230], [420, 180]], [[900, -120], [1060, -190], [1200, -120]], [[900, 300], [1040, 360]]].forEach((c, i) => {
      const pts = KP.curve(c, false, 8); KP.brush(ctx, pts, 22, P.coral, { taper: .7 }); KP.brush(ctx, pts, 8, P.pink30, { taper: .7 });
    });
    // the vessel (behind the gut wall and the castle)
    const vs = VESSEL;
    KP.brush(ctx, vs.map(([x, y]) => [x + 10, y + 12]), 130, KP.rgba(P.ink, .25), { taper: 0 });
    KP.brush(ctx, vs, 130, P.coral, { taper: 0 });
    KP.brush(ctx, vs, 66, P.pink30, { taper: 0 });
    KP.brush(ctx, vs.map(([x, y]) => [x - 18, y - 18]), 10, P.rice, { taper: .4 });
    // red blood cells (coral discs) flowing along the vessel
    for (let i = 0; i < 26; i++) {
      const sx = ((i * 140 + KP.held(t) * 160) % (SGATE + 200)) - 100, [x, y] = vAt(sx);
      KP.paper(ctx, KP.jitter(KP.ell(x + (i % 3 - 1) * 12, y + (i % 2) * 6, 22, 15, 18), 320 + i + b, .8), P.tomato, { shadow: KP.SH.far, shadowA: .25 });
    }
    // the castle wall at the head end (cobalt), a peeking brain, the gate + guards
    KP.brain(ctx, 730, -935, 150, { boil: b, eyes: 'happy', mouth: 'smile' });
    const wall = [[-600, -820]];
    for (let x = -600; x < 1700; x += 130) wall.push([x, -900], [x + 70, -900], [x + 70, -820], [x + 130, -820]);
    wall.push([1700, -820], [1700, -450], [-600, -450]);
    const wp = KP.paper(ctx, KP.jitter(wall, 300 + b, 1.6), P.blue, { shadow: [12, 16], shadowA: .35, mis: P.grape, misOff: [-4, 3], misA: .45 });
    ctx.save(); ctx.clip(wp);
    for (let r = 0; r < 4; r++) for (let x = -620 + (r % 2) * 60; x < 1700; x += 120) KP.brush(ctx, [[x + 8, -780 + r * 86], [x + 104, -780 + r * 86]], 7, KP.mix(P.blue, P.turq, .4), { taper: .4 });
    ctx.restore();
    // pennants on the battlements, flapping on the boil
    for (const [px, c] of [[340, P.lemon], [600, P.pink], [1000, P.lemon], [1260, P.pink]]) {
      const fl = Math.sin(KP.held(t) * 7 + px) * 14;
      KP.brush(ctx, [[px, -900], [px, -1040]], 7, P.ink, { taper: .05 });
      KP.paper(ctx, KP.jitter([[px, -1040], [px + 90, -1016 + fl], [px, -990]], 330 + px + b, 1), c, { shadow: KP.SH.far, shadowA: .3 });
    }
    // gate arch
    KP.paper(ctx, KP.jitter(KP.curve([[700, -450], [700, -600], [740, -660], [800, -676], [860, -660], [900, -600], [900, -450]], true, 6), 301 + b, 1.2), P.plum, { shadow: false });
    for (const gx of [730, 770, 810, 850]) KP.brush(ctx, [[gx, -640], [gx, -470]], 7, P.grape, { taper: .1 });
    // the gut wall (MID in shot 5): villi + cells + under-layers
    const vr = KP.rng(31), vil = [[-300, 760]];
    for (let x = -300; x < 1400; x += 110) { const h = 120 + vr() * 90; vil.push([x, 760 - h * .2], [x + 28, 760 - h], [x + 78, 760 - h * .95], [x + 100, 760 - h * .15]); }
    vil.push([1400, 1000], [-300, 1000]);
    KP.paper(ctx, KP.jitter(KP.curve(vil, true, 5), 32 + b, 1.5), P.pink30, { shadow: KP.SH.far, mis: P.blue, misOff: [3, -2], misA: .3 });
    // stream: Goldies flow right above the wall and are caught one by one
    const t0 = T.s5 - 2;
    for (let j = 0; j < 60; j++) {
      const sj = t0 + j * .32; if (sj > t) break;
      const ci = j % 4, [tx] = WALLC[ci], x = -120 + (t - sj) * 520, reach = sj + (tx + 120) / 520;
      if (t < reach) { if (x > -60) KP.goldie(ctx, x, 600 + Math.sin(x * .02) * 14, 52, { seed: j, boil: b, steam: false, shadow: KP.SH.far, eyes: j % 3 ? 'dot' : 'happy' }); }
      else if (t < reach + .16) KP.goldie(ctx, tx - 20, KP.lerp(600, 740, KP.prog(t, reach, reach + .16)), 52 * (1 - KP.prog(t, reach, reach + .16) * .5), { seed: j, boil: b, steam: false, shadow: false });
    }
    const burp = KP.prog(t, T.never, T.never + .7);
    if (burp > 0 && burp < 1) { const [cx, cy] = WALLC[2], sc = Math.sin(Math.PI * burp); KP.paper(ctx, KP.jitter(KP.curve([[cx, cy - 250 - 60 * sc], [cx + 26 * sc, cy - 200], [cx + 20 * sc, cy - 170], [cx - 20 * sc, cy - 170], [cx - 26 * sc, cy - 200]], true, 6), 340 + b, 1), P.gold, { shadow: KP.SH.far, mis: P.tomato, misOff: [-2, 2], misA: .4 }); }
    WALLC.forEach(([cx, cy, id], i) => {
      let chomp = false;
      for (let j = i; j < 60; j += 4) { const sj = t0 + j * .32, reach = sj + (cx + 120) / 520; if (t >= reach && t < reach + .22) chomp = true; }
      drawCell(ctx, [cx, cy, 120, 20 + id], b, { eyes: 'happy', mouth: chomp ? 'chomp' : 'smile', mouthDx: -.05 });
    });
    KP.paper(ctx, KP.jitter(KP.curve([[-300, 790], [300, 780], [800, 800], [1400, 790], [1400, 1000], [-300, 1000]], true, 10), 33 + b, 1.5), P.pink, { shadow: KP.SH.near, mis: P.blue, misOff: [0, -3], misA: .3 });
    KP.paper(ctx, KP.jitter(KP.curve([[-300, 900], [400, 890], [900, 910], [1400, 900], [1400, 1010], [-300, 1010]], true, 10), 34 + b, 1.5), P.magenta, { shadow: [0, -6], shadowA: .2 });
    // the trickle: one or two Goldies in the vessel, from 'blood' on, up to the gate (shots 6-7)
    const lead = goldieS(t, T);
    [[lead, 0], [lead - 240, 1]].forEach(([s, i]) => {
      if (t < T.blood - .2 + i * .35) return;
      const [x, y] = vAt(Math.max(0, s));
      const bounce = i === 0 && t > T.body;
      KP.goldie(ctx, x + (bounce ? Math.sin(KP.held(t) * 9) * 10 * (1 - KP.prog(t, T.body, T.body + 1)) : 0), y, i ? 62 : 80, {
        seed: 50 + i, boil: b, rot: bounce ? .25 * Math.sin(KP.held(t) * 5) : 0, shadow: KP.SH.far,
        eyes: bounce ? 'closed' : 'dot', mouth: bounce ? 'wobble' : 'smile' });
      if (bounce && i === 0 && t > T.body + .35) KP.marks(ctx, x, y - 30, 56, 3, 60 + b, { a0: -2.6, span: 2.2, len: 20, w: 5 });
    });
    // guards at the gate: grape cells in cobalt helmets, mitten arms; the right one raises a mitten
    guard(ctx, 610, -470, b, 0, 101, t);
    guard(ctx, 990, -470, b, KP.ep(t, T.best - .1, T.best + .15, 'out'), 102, t);
  }
  function guard(ctx, x, y, b, raise, id, t) {
    const w = 130, h = 140, bob = Math.sin(KP.held(t) * 3 + id) * 3;
    KP.arm(ctx, x - w * .4, y - h * .55 + bob, w * .32, 2.4, w * .17, P.grape, { seed: id, boil: b });
    KP.arm(ctx, x + w * .4, y - h * .55 + bob, w * .32, KP.lerp(.9, -1.35, raise), w * .17, P.grape, { seed: id + 5, boil: b, thumb: 1 });
    KP.paper(ctx, KP.jitter(KP.egg(x, y - h * .5 + bob, w / 2, h / 2, .1, 48), id + b, 1.4), P.grape, { shadow: KP.SH.near, mis: P.blue, misOff: [2, -2], misA: .35 });
    KP.paper(ctx, KP.jitter(KP.curve([[x - 30, y - h * .45 + bob], [x + 30, y - h * .45 + bob], [x + 34, y - h * .15 + bob], [x, y - h * .05 + bob], [x - 34, y - h * .15 + bob]], true, 6), id + 3 + b, 1), P.rice, { shadow: KP.SH.far, shadowA: .2 });
    KP.paper(ctx, KP.jitter(KP.curve([[x - w * .48, y - h * .78 + bob], [x - w * .36, y - h * 1.02 + bob], [x, y - h * 1.1 + bob], [x + w * .36, y - h * 1.02 + bob], [x + w * .48, y - h * .78 + bob]], true, 6), id + 4 + b, 1.2), P.blue, { shadow: KP.SH.far, mis: P.turq, misOff: [-2, 2], misA: .4 });
    KP.face(ctx, x, y - h * .62 + bob, w * .72, { eyes: raise > .5 ? 'happy' : 'dot', mouth: 'smile', cheek: P.pink });
  }
  // lead Goldie position along the vessel (arc length): rides up to the gate, inches closer, bounces back on 'body'
  function goldieS(t, T) {
    const s1 = KP.ep(t, T.blood - .3, T.bbb + .55, 'inOut') * (SGATE - 260);
    const s2 = KP.ep(t, T.s7 + .1, T.body - .05, 'inOut') * 200;
    const bounce = t > T.body ? Math.sin(Math.PI * KP.prog(t, T.body, T.body + .45)) * 0 - KP.backOut(KP.prog(t, T.body, T.body + .4)) * 170 : 0;
    return s1 + s2 + bounce;
  }
  function shotMap(ctx, t, b, T, D) {
    // camera: pull back from the wall (shot 5), track up the vessel (6), push in to the gate (7)
    const pull = KP.ep(t, T.s5, T.s5 + .45, 'out');
    let s = KP.lerp(2.1, 1, pull), cx = KP.lerp(330, 490, pull), cy = KP.lerp(860, 860, pull);
    const up = KP.ep(t, T.l5, T.bbb + .6, 'inOut');
    cx = KP.lerp(cx, 640, up); cy = KP.lerp(cy, -560, up);
    const pin = KP.ep(t, T.s7, D, 'inOut');
    cx = KP.lerp(cx, 790, pin); cy = KP.lerp(cy, -650, pin); s *= 1 + .38 * pin;
    KP.camera(ctx, cx, cy, s, () => {
      mapWorld(ctx, t, b, T);
      // the one label in shot 5: '95%' rolls up from 80 and lands on 'ninety-five' (never shows '5%')
      if (t >= Math.max(T.around - .1, T.s5 + .4) && t < T.l5 + .2) {
        const v = Math.round(KP.lerp(80, 95, KP.ease.out(KP.prog(KP.held(t), T.around - .1, T.ninety))));
        const la = KP.arrive(t, Math.max(T.around - .1, T.s5 + .4), .3), land = KP.arrive(t, T.ninety, .2);
        const gone = 1 - KP.ease.in(KP.prog(t, T.l5 - .05, T.l5 + .2));   // pops off before the camera tracks up
        ctx.save(); KP.pose(ctx, { ...la, p: la.p * gone, sy: la.sy * (t >= T.ninety ? land.sy : 1) }, 490, 470, 'pop');
        KP.slip(ctx, [{ s: v + '%', size: 190, font: 'Young Serif', weight: 400, ls: 4 }], 220, 200, { rot: -.03 + Math.sin(KP.held(t) * 2.4) * .03, seed: 95, boil: b, tape: P.pink, padX: 50, padY: 26 });
        ctx.restore();
      }
    });
    // shot 6 label: 'Blood-brain barrier' (screen space, TOP)
    if (t >= T.bbb && t < T.s7) {
      const la = KP.arrive(t, T.bbb, .3), out = KP.leave(t, T.s7, .15);
      ctx.save(); ctx.globalAlpha = out > 0 ? 0 : 1; KP.pose(ctx, la, 300, 330, 'pop');
      KP.slip(ctx, [{ s: 'Blood-brain barrier', size: 64, ls: 1 }], 70, 236, { rot: -.03, seed: 77, boil: b, tape: P.lemon, padX: 34, padY: 20 });
      ctx.restore();
    }
  }

  KP.scene('s05_body', {
    sfx(cues, D) {
      const T = times(cues, D), a = [], e = (t, name, gain) => a.push({ t: Math.max(0, +t.toFixed(3)), name, gain });
      e(T.happens, 'whoosh', .35); e(T.eat, 'chomp', .45); e(T.dive, 'whoosh_long', .45);
      e(T.cells, 'chomp', .4); e(T.gut, 'chomp', .35); e(T.burn, 'sizzle', .3); e(T.fuel, 'pop_small', .35);
      e(T.s5 + .1, 'tick', .2); e(T.s5 + .25, 'tick', .2); e(T.ninety, 'ding', .5); e(T.never, 'sizzle', .2); e(T.blood, 'bubble', .3);
      e(T.l5, 'whoosh', .4); e(T.bbb, 'thud', .45); e(T.best, 'click', .35); e(T.body, 'boing', .4); e(D - .45, 'whoosh', .3);
      return a.sort((x, y) => x.t - y.t);
    },
    draw(ctx, t, cues, D) {
      const T = times(cues, D), b = KP.boil(t);
      if (t < T.dive) shotFork(ctx, t, b, T);
      else if (t < T.dive + 1.0) shotDive(ctx, t, b, T);
      else if (t < T.s5) { gutB(ctx, t, b, T); }
      else shotMap(ctx, t, b, T, D);
      // shot 4 label: 'Fuel!' (taped slip, TOP)
      if (t >= T.fuel && t < T.s5) {
        const fa = KP.arrive(t, T.fuel, .3);
        ctx.save(); KP.pose(ctx, fa, 210, 380, 'pop');
        KP.slip(ctx, [{ s: 'Fuel!', size: 72, ls: 1 }], 90, 250, { rot: -.05, seed: 30, boil: b, tape: P.blue, padX: 40, padY: 24 });
        ctx.restore();
      }
      KP.texture(ctx, 'grain', .1);
      // in: revealed by s04's Gold Ripple wipe
      KP.slideTo(ctx, KP.ep(t, D - .5, D, 'inOut'), 's07_tests');      // out: slide carrying s07's first frame
    },
  });
})();
