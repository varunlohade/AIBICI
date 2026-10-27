// s06_origin: How the rumour started (Kombu Paper). Shots s06.01-.09 of design/SHOTLIST.md.
// SENSITIVE: paper, type and objects only. No restaurants, no people, no faux-Asian lettering or signifiers.
// The headline appears only as the quoted, hyphenated journal headline, stamped as the thing being criticised.
// The letter's author is never named or drawn; the journal is never named. All timings come from cues.json words.
//   1 kombu curls into the shaker; tomato spotlight + villain moustache, 'Villain?'
//   2 the lemon letter slides in on the cobalt desk and writes itself in squiggles, '1968'; envelope postmarked
//   3 symptom icons pin on: pins-and-needles mitten (numb), drooping arm (weak)
//   4 the turquoise card: '?' over wine, salt, MSG, merging into one wobbly '?', 'A guess'
//   5 whip to style frame C: the stamp slams the quoted headline; newspapers slide out like an anti-ripple
//   6 lab: newborn mouse pups, a giant syringe lowers (never touches), 'Huge doses'
//   7 split: syringe + pup  ≠  dinner bowl + gut cell chomping a Goldie
//   8 split: pasta + parmesan under a silent bell / plain carton under the same bell, ringing
//   9 the split collapses, the stamped headline slides back, push in on 'name'; paper slide to the TV wall
(function () {
  const P = KP.P, TAU = Math.PI * 2, D2R = Math.PI / 180;
  const times = (cues, D) => {
    const w = (re, o) => KP.word(cues, re, o);
    const T = { seaweed: w(/seaweed/i, { fb: 1.1 }), villain: w(/villain/i, { fb: 2.6 }) };
    T.l2 = w(/1968/i, { line: 1, fb: 3.44 }); T.y1968 = w(/1968/i, { fb: 3.7 }); T.doctor = w(/doctor/i, { fb: 5.3 });
    T.wrote = w(/wrote/i, { fb: 5.8 }); T.journal = w(/journal/i, { fb: 7.1 }); T.after = w(/^after/i, { fb: 8.08 });
    T.numb = w(/numb/i, { fb: 11.07 }); T.weak = w(/weak/i, { fb: 11.52 });
    T.l3 = w(/wine/i, { line: 1, fb: 12.04 }); T.wine = w(/wine/i, { fb: 12.55 }); T.salt = w(/salt/i, { fb: 13.23 });
    T.msg = w(/msg/i, { after: T.salt, fb: 13.9 }); T.itwas = w(/^it$/i, { after: T.msg, fb: 14.74 }); T.guess = w(/guess/i, { fb: 15.22 });
    T.l4 = w(/catchy/i, { line: 1, fb: 15.69 }); T.catchy = w(/catchy/i, { fb: 17.16 });
    T.chinese = w(/chinese/i, { after: T.catchy, fb: 18.25 }); T.syndrome = w(/syndrome/i, { after: T.catchy, fb: 19.31 });
    T.l5 = w(/injected/i, { line: 1, fb: 20.14 }); T.injected = w(/injected/i, { fb: 21.03 }); T.huge = w(/huge/i, { fb: 21.87 });
    T.brain = w(/^brain/i, { after: T.huge, fb: 24.87 }); T.damage = w(/damage/i, { after: T.huge, fb: 25.13 });
    T.l6 = w(/scary/i, { line: 1, fb: 25.91 }); T.nothing = w(/nothing/i, { fb: 29.87 }); T.dinner = w(/dinner/i, { fb: 30.47 });
    T.l7 = w(/nobody/i, { line: 1, fb: 31.27 }); T.tomato = w(/tomato/i, { after: T.l7, fb: 33.3 }); T.parm = w(/parmesan/i, { fb: 34.42 });
    T.fear = w(/fear/i, { fb: 35.6 }); T.name = w(/^name/i, { after: T.fear, fb: 37.87 });
    T.s9 = w(/^the$/i, { after: T.parm, fb: T.fear - .2 }) - .1;
    return T;
  };

  // ======================================================== desk world (style frame C coordinates)
  const LET = { x: 110, y: 240, w: 720, h: 1010, a: -4 * D2R };
  const LINES = [1, 1, 1, 1, .55, 0, .9, 1, 1, 1, .4];
  function letter(ctx, b, write, o = {}) {
    const { x: lx, y: ly, w: lw, h: lh, a } = LET, c = [lx + lw / 2, ly + lh / 2];
    ctx.save(); ctx.translate(c[0], c[1]); ctx.rotate(a); if (o.sy) ctx.scale(1, o.sy); ctx.translate(-c[0], -c[1]);
    KP.paper(ctx, KP.jitter(KP.rr(lx, ly, lw, lh, 8, 30), 9 + b, 1.3), P.lemon, { shadow: KP.SH.near, shadowA: .35, mis: P.coral, misOff: [-3, 3], misA: .5 });
    KP.paper(ctx, [[lx, ly + lh * .34], [lx + lw, ly + lh * .34], [lx + lw, ly + lh * .34 + 3], [lx, ly + lh * .34 + 3]], P.brass, { shadow: false });
    const r = KP.rng(77), x0 = lx + 70, x1 = lx + lw - 70;
    // squiggle 'writing' appears progressively (write 0..1 over date, salutation and the body lines)
    const parts = [[x1 - 170, x1, ly + 90, 1, 5, 7], [x0, x0 + 230, ly + 180, 2, 6, 9]];
    let y = ly + 270;
    LINES.forEach((k, i) => {
      if (k === 0) { y += 36; return; }
      const ind = i === 0 || i === 6 ? 50 : 0;
      if (o.clearZone && y > ly + 440 && y < ly + 790) { y += 54; return; }
      parts.push([x0 + ind, x0 + ind + (x1 - x0 - ind) * k * (.92 + r() * .08), y, 10 + i, 5 + r() * 2, 7]);
      y += 54;
    });
    const n = parts.length, done = write * n;
    parts.forEach(([a0, a1, yy, sd, amp, wdt], i) => {
      const f = KP.clamp(done - i); if (f <= 0) return;
      KP.squiggle(ctx, a0, a0 + (a1 - a0) * f, yy, sd, { amp, wl: 24, w: wdt });
    });
    if (write >= 1) KP.brush(ctx, KP.curve([[x0 + 330, y + 70], [x0 + 360, y + 40], [x0 + 380, y + 80], [x0 + 410, y + 36], [x0 + 440, y + 76], [x0 + 500, y + 52], [x0 + 560, y + 62]], false, 6), 6, P.ink, { taper: .6 });
    if (o.icons) o.icons(ctx);
    ctx.restore();
  }
  function pin(ctx, x, y, b) {
    KP.brush(ctx, [[x, y], [x + 8, y + 18]], 4, P.ink, { taper: .2 });
    KP.paper(ctx, KP.jitter(KP.egg(x, y, 11, 11, 0, 16), 500 + x + b, .6), P.tomato, { shadow: KP.SH.far, shadowA: .35 });
  }
  // symptom icons pinned on the letter (pictures only): numb = mitten with pins-and-needles dots, weak = drooping arm
  function icons(ctx, t, b, T) {
    const n = KP.arrive(t, T.numb, .3), wk = KP.arrive(t, T.weak, .3);
    if (n.on) {
      const x = 700, y = 330;
      ctx.save(); KP.pose(ctx, n, x, y + 60, 'pop');
      KP.paper(ctx, KP.jitter(KP.rr(x - 70, y - 70, 140, 140, 10, 12), 510 + b, 1), P.rice, { shadow: KP.SH.near, shadowA: .3 });
      KP.paper(ctx, KP.jitter(KP.curve([[x - 30, y + 50], [x - 40, y - 10], [x - 22, y - 44], [x + 16, y - 48], [x + 36, y - 18], [x + 34, y + 50]], true, 6), 511 + b, 1), P.coral, { shadow: KP.SH.far });
      KP.paper(ctx, KP.jitter(KP.egg(x - 46, y - 2, 12, 22, 0, 16), 512 + b, .6), P.coral, { shadow: false });
      const r = KP.rng(513 + (b % 4));
      ctx.fillStyle = P.ink; for (let i = 0; i < 10; i++) { ctx.beginPath(); ctx.arc(x - 28 + r() * 56, y - 40 + r() * 80, 3.2, 0, TAU); ctx.fill(); }
      KP.marks(ctx, x, y, 56, 4, 514 + b, { a0: -2.4, span: 1.6, len: 14, w: 4 });
      pin(ctx, x - 50, y - 60, b);
      ctx.restore();
    }
    if (wk.on) {
      const x = 700, y = 500;
      ctx.save(); KP.pose(ctx, wk, x, y + 60, 'pop');
      KP.paper(ctx, KP.jitter(KP.rr(x - 70, y - 70, 140, 140, 10, 12), 520 + b, 1), P.rice, { shadow: KP.SH.near, shadowA: .3 });
      const droop = KP.ep(t, T.weak, T.weak + .5, 'out');
      KP.arm(ctx, x - 40, y - 30, 70, KP.lerp(-.3, 1.1, droop), 24, P.bubble, { seed: 521, boil: b, mis: P.blue });
      if (droop > .5) KP.brush(ctx, [[x + 30, y + 20], [x + 30, y + 50]], 5, P.ink, { taper: .5 });
      pin(ctx, x + 50, y - 60, b);
      ctx.restore();
    }
  }
  function underlay(ctx, b) { KP.paper(ctx, KP.rot(KP.jitter(KP.rr(150, 300, 740, 990, 6, 30), 2 + b, 1.5), 520, 800, .06), P.pink, { shadow: KP.SH.near, shadowA: .3 }); }
  function envelope(ctx, x, y, b, post) {
    ctx.save(); ctx.translate(x, y); ctx.rotate(6 * D2R);
    KP.paper(ctx, KP.jitter(KP.rr(-180, -105, 360, 210, 8, 20), 530 + b, 1.2), P.rice, { shadow: KP.SH.near, shadowA: .35, mis: P.turq, misOff: [-3, 2], misA: .4 });
    KP.brush(ctx, [[-176, -100], [0, 10], [176, -100]], 6, P.lavender, { taper: .1 });
    for (let i = 0; i < 3; i++) KP.squiggle(ctx, -60, 60 - i * 30, 30 + i * 30, 540 + i, { amp: 4, wl: 18, w: 6 });
    // postmark: a plain tomato ring with wavy cancel lines (no text)
    if (post > 0) {
      ctx.save(); ctx.translate(120, -50); ctx.scale(post, post); ctx.rotate(-.2);
      ctx.lineWidth = 7; ctx.strokeStyle = P.tomato; ctx.beginPath(); ctx.arc(0, 0, 40, 0, TAU); ctx.stroke();
      for (let i = 0; i < 3; i++) KP.squiggle(ctx, -110, -48, -18 + i * 18, 550 + i, { amp: 4, wl: 16, w: 5, color: P.tomato });
      ctx.restore();
    }
    ctx.restore();
  }
  // the turquoise card: wine bottle, salt cellar, MSG shaker; '?' per suspect (q = [0..1]x3), merge 0..1
  function card(ctx, b, q, merge) {
    ctx.save(); ctx.translate(250, 1180); ctx.rotate(.06);
    KP.paper(ctx, KP.jitter(KP.rr(-190, -120, 400, 250, 8, 20), 14 + b, 1.2), P.turq, { shadow: KP.SH.near, shadowA: .35 });
    KP.paper(ctx, KP.jitter(KP.curve([[-128, 98], [-128, 10], [-120, -14], [-108, -24], [-108, -60], [-98, -60], [-98, -24], [-86, -14], [-78, 10], [-78, 98]], true, 4), 15 + b, 1), P.grape, { shadow: KP.SH.far });
    KP.paper(ctx, KP.jitter(KP.rr(-124, 30, 42, 36, 4, 8), 16, .8), P.rice, { shadow: false });
    KP.paper(ctx, KP.jitter(KP.curve([[-40, 40], [40, 40], [32, 74], [10, 84], [14, 98], [-14, 98], [-10, 84], [-32, 74]], true, 4), 17 + b, 1), P.rice, { shadow: KP.SH.far });
    KP.paper(ctx, KP.jitter(KP.ell(0, 42, 40, 9, 20), 18, .6), P.pink, { shadow: false });
    KP.shaker(ctx, 116, 98, 96, { boil: b, seed: 7, shadow: KP.SH.far });
    [[-104, -78, -.12], [0, -16, .08], [116, -48, -.05]].forEach(([qx, qy, qr], i) => {
      if (q[i] <= 0 || merge >= 1) return;
      const k = 1 - merge, mx = KP.lerp(qx, 0, merge), my = KP.lerp(qy, -230, merge);
      const dy = (1 - KP.backOut(q[i])) * -180;
      KP.qmark(ctx, mx, my + dy, 64 * (1 + merge), { rot: qr * k });
    });
    if (merge > 0) {
      const wob = Math.sin(KP.held(merge * 30) * 2.4) * .12;
      KP.qmark(ctx, -20, -250, 64 + KP.ease.out(merge) * 170, { rot: wob });
    }
    ctx.restore();
  }
  function stampBodyPts(dx, dy) {
    const lift = pts => KP.rot(pts, 0, 0, -8 * D2R), ox = 700 + dx, oy = 170 + dy;
    return [KP.move(lift(KP.rr(-230, -50, 460, 100, 14, 18)), ox, oy), KP.move(lift(KP.rr(-64, -260, 128, 230, 36, 18)), ox, oy), KP.move(lift(KP.egg(0, -290, 130, 90, 0, 40)), ox, oy)];
  }
  function stampHandle(ctx, b, dx, dy) {
    stampBodyPts(dx + 34, dy + 64).forEach(p => { ctx.fillStyle = KP.rgba(P.ink, .25); ctx.fill(KP.path(p)); });
    const [blk, neck, knob] = stampBodyPts(dx, dy).map((p, i) => KP.jitter(p, 91 + i + b, 1.5));
    KP.paper(ctx, knob, P.ink, { shadow: false }); KP.paper(ctx, neck, P.ink, { shadow: false }); KP.paper(ctx, blk, P.ink, { shadow: false });
    const lift = pts => KP.rot(pts, 0, 0, -8 * D2R);
    KP.paper(ctx, KP.jitter(KP.move(lift(KP.rr(-222, 40, 444, 20, 6, 18)), 700 + dx, 170 + dy), 94 + b, 1), P.tomato, { shadow: false });
    KP.paper(ctx, KP.jitter(KP.move(lift(KP.rr(-196, -20, 392, 10, 4, 18)), 700 + dx, 170 + dy), 95 + b, 1), P.grape, { shadow: false });
  }
  const STW = 800, STH = 310, SCX = 470, SCY = 860;
  const headline = () => KP.stamp('crs', [String.fromCharCode(0x201C) + 'CHINESE-RESTAURANT', 'SYNDROME' + String.fromCharCode(0x201D)], STW, STH, { size: 66, ls: 1.5, seed: 55 });
  function impression(ctx, k = 1) {
    ctx.save(); ctx.translate(SCX, SCY); ctx.rotate(-8 * D2R); ctx.globalAlpha *= k;
    ctx.drawImage(headline(), -STW / 2, -STH / 2); ctx.restore();
  }
  function newspaper(ctx, x, y, w, h, a, sd, col, b, bar = 22) {
    ctx.save(); ctx.translate(x, y); ctx.rotate(a);
    KP.paper(ctx, KP.jitter(KP.rr(-w / 2, -h / 2, w, h, 4, 18), sd + b, 1.2), col, { shadow: KP.SH.near, shadowA: .35 });
    ctx.fillStyle = P.ink; ctx.fillRect(-w / 2 + 18, -h / 2 + 18, w - 36, bar);
    ctx.fillStyle = P.tomato; ctx.fillRect(-w / 2 + 18, -h / 2 + 26 + bar, w - 36, 5 + (bar - 22) * .4);
    const cw = (w - 36 - 24) / 3, top = -h / 2 + 54 + bar;
    for (let c = 0; c < 3; c++) for (let i = 0; i < Math.floor((h / 2 - top - 10) / 17); i++) KP.squiggle(ctx, -w / 2 + 18 + c * (cw + 12), -w / 2 + 18 + c * (cw + 12) + cw * .9, top + i * 17, sd * 50 + c * 20 + i, { amp: 2, wl: 10, w: 4, color: P.rice });
    ctx.restore();
  }
  // newspapers slide out from the stamp like an anti-ripple (p 0..1)
  const PAPERS = [[1040, 1120, 280, 320, .22, 61, P.coral], [-40, 560, 260, 300, -.3, 62, P.grape], [820, 1660, 300, 340, -.12, 63, P.coral], [960, 300, 240, 280, .3, 64, P.coral], [60, 1560, 260, 300, .18, 65, P.coral]];
  function papers(ctx, b, p, bar = 22) {
    PAPERS.forEach(([x, y, w, h, a, sd, col], i) => {
      const k = KP.ease.out(KP.clamp(p * 1.4 - i * .1)); if (k <= 0) return;
      newspaper(ctx, KP.lerp(SCX, x, k), KP.lerp(SCY, y, k), w * (.4 + .6 * k), h * (.4 + .6 * k), a * k, sd, col, b, bar);
    });
  }
  function desk(ctx, t, b, T, o) {
    ctx.fillStyle = P.blue; ctx.fillRect(-1500, -1500, 4000, 5000);
    underlay(ctx, b);
    letter(ctx, b, o.write, { clearZone: o.clearZone, sy: o.sy, icons: c => icons(c, t, b, T) });
    if (o.env) envelope(ctx, 600 + o.env[0], 1110 + o.env[1], b, o.post || 0);
    if (o.card) card(ctx, b, o.q || [0, 0, 0], o.merge || 0);
  }

  // ======================================================== lab, splits
  function graph(ctx, x0, y0, x1, y1, ground, line) {
    ctx.fillStyle = ground; ctx.fillRect(x0, y0, x1 - x0, y1 - y0);
    ctx.fillStyle = line; for (let x = x0 + 30; x < x1; x += 60) ctx.fillRect(x, y0, 3, y1 - y0); for (let y = y0 + 30; y < y1; y += 60) ctx.fillRect(x0, y, x1 - x0, 3);
  }
  function pup(ctx, x, y, s, b, sd, t) {
    const br = 1 + .03 * Math.sin(KP.held(t) * 4 + sd);  // breathing
    KP.brush(ctx, KP.curve([[x + 40 * s, y + 8 * s], [x + 70 * s, y + 20 * s], [x + 84 * s, y - 4 * s]], false, 6), 5 * s, P.pink, { taper: .7 });
    KP.paper(ctx, KP.jitter(KP.egg(x, y, 50 * s * br, 30 * s * br, .1, 30), sd + b, 1), P.pink30, { shadow: KP.SH.far, shadowA: .3, mis: P.pink, misOff: [-2, 2], misA: .4 });
    KP.paper(ctx, KP.jitter(KP.egg(x - 34 * s, y - 22 * s, 13 * s, 11 * s, 0, 14), sd + 1 + b, .6), P.bubble, { shadow: false });
    KP.brush(ctx, KP.arcPts(x - 38 * s, y - 2 * s, 7 * s, 5 * s, Math.PI * .15, Math.PI * .85, 6), 4 * s, P.ink, { taper: .3 });
  }
  function nest(ctx, x, y, b, t, s = 1) {
    KP.paper(ctx, KP.jitter(KP.egg(x, y + 20 * s, 230 * s, 80 * s, .05, 48), 600 + b, 1.6), P.wood, { shadow: KP.SH.near, shadowA: .35, mis: P.coral, misOff: [-3, 2], misA: .4 });
    pup(ctx, x - 90 * s, y - 10 * s, s, b, 610, t); pup(ctx, x + 10 * s, y - 24 * s, s * 1.05, b, 620, t); pup(ctx, x + 100 * s, y - 6 * s, s * .95, b, 630, t);
    const r = KP.rng(640);
    for (let i = 0; i < 14; i++) { const a = -.2 + r() * .4, xx = x - 200 * s + r() * 400 * s, yy = y + 30 * s + r() * 50 * s; KP.brush(ctx, [[xx - 40 * s, yy - Math.sin(a) * 40], [xx + 40 * s, yy + Math.sin(a) * 40]], 5 * s, P.brass, { taper: .5 }); }
  }
  function syringe(ctx, x, tipY, s, b) {
    ctx.save(); ctx.translate(x, tipY); ctx.scale(s, s);
    KP.brush(ctx, [[0, 0], [0, -110]], 8, P.ink, { taper: .1 });                                    // needle (cut ink sliver)
    KP.paper(ctx, KP.jitter(KP.rr(-30, -150, 60, 44, 8, 10), 700 + b, 1), P.grape, { shadow: KP.SH.far });
    KP.paper(ctx, KP.jitter(KP.rr(-80, -620, 160, 480, 16, 20), 701 + b, 1.4), P.rice, { shadow: KP.SH.near, shadowA: .35, mis: P.turq, misOff: [-3, 2], misA: .45 });
    KP.paper(ctx, KP.jitter(KP.rr(-70, -360, 140, 210, 10, 16), 702 + b, 1), P.gold, { shadow: false });
    for (let i = 0; i < 6; i++) KP.brush(ctx, [[-76, -580 + i * 70], [-40, -580 + i * 70]], 6, P.blue, { taper: .2 });
    KP.paper(ctx, KP.jitter(KP.rr(-110, -640, 220, 30, 8, 16), 703 + b, 1), P.blue, { shadow: KP.SH.far });
    KP.paper(ctx, KP.jitter(KP.rr(-16, -900, 32, 270, 8, 16), 704 + b, 1), P.blue, { shadow: KP.SH.far });
    KP.paper(ctx, KP.jitter(KP.rr(-90, -930, 180, 40, 14, 16), 705 + b, 1), P.blue, { shadow: KP.SH.near, shadowA: .3 });
    ctx.restore();
  }
  function bowl(ctx, x, y, b, t, T) {
    KP.paper(ctx, KP.jitter(KP.curve([[x - 190, y - 40], [x + 190, y - 40], [x + 150, y + 70], [x + 60, y + 110], [x - 60, y + 110], [x - 150, y + 70]], true, 6), 800 + b, 1.4), P.turq, { shadow: KP.SH.near, shadowA: .35, mis: P.blue, misOff: [-3, 2], misA: .4 });
    KP.paper(ctx, KP.jitter(KP.ell(x, y - 40, 190, 40, 40), 801 + b, 1.2), P.lemon, { shadow: false });
    KP.brush(ctx, [[x + 120, y - 60], [x + 250, y - 200]], 18, P.rice, { taper: .1 });
    KP.paper(ctx, KP.jitter(KP.egg(x + 110, y - 52, 36, 18, 0, 20), 802 + b, .8), P.rice, { shadow: KP.SH.far });
    for (let i = 0; i < 3; i++) KP.goldie(ctx, x - 100 + i * 70, y - 60 + Math.sin(KP.held(t) * 3 + i) * 4, 54, { seed: 810 + i, boil: b, steam: false, shadow: KP.SH.far, eyes: i === 1 ? 'happy' : 'dot' });
  }
  function bell(ctx, x, y, b, ring, t) {
    const sw = ring ? Math.sin(KP.held(t) * 30) * .32 : 0;
    KP.brush(ctx, [[x, y - 400], [x, y - 90]], 6, P.ink, { taper: .05 });
    ctx.save(); ctx.translate(x, y - 90); ctx.rotate(sw); ctx.translate(-x, -(y - 90));
    KP.paper(ctx, KP.jitter(KP.curve([[x - 20, y - 92], [x + 20, y - 92], [x + 54, y - 40], [x + 64, y + 30], [x + 90, y + 54], [x - 90, y + 54], [x - 64, y + 30], [x - 54, y - 40]], true, 6), 900 + b, 1.2), P.tomato, { shadow: KP.SH.near, shadowA: .35, mis: P.pink, misOff: [-3, 2], misA: .45 });
    KP.paper(ctx, KP.jitter(KP.egg(x, y + 66, 18, 16, 0, 16), 901 + b, .6), P.ink, { shadow: false });
    ctx.restore();
    if (ring) { KP.marks(ctx, x - 70, y, 40, 3, 902 + b, { a0: Math.PI * .8, span: .9, len: 28 }); KP.marks(ctx, x + 70, y, 40, 3, 903 + b, { a0: -Math.PI * .25, span: .9, len: 28 }); }
  }
  function pasta(ctx, x, y, b) {
    KP.paper(ctx, KP.jitter(KP.ell(x, y, 200, 52, 40), 950 + b, 1.2), P.rice, { shadow: KP.SH.near, shadowA: .35, mis: P.turq, misOff: [-3, 2], misA: .4 });
    for (let i = 0; i < 7; i++) KP.brush(ctx, KP.curve([[x - 120 + i * 30, y + 10], [x - 100 + i * 30, y - 40], [x - 130 + i * 32, y - 70], [x - 90 + i * 28, y - 90]], false, 6), 10, P.lemon, { taper: .2 });
    KP.paper(ctx, KP.jitter(KP.egg(x, y - 70, 80, 36, .1, 28), 951 + b, 1), P.tomato, { shadow: KP.SH.far, mis: P.pink, misOff: [-2, 2], misA: .45 });
    // parmesan wedge
    KP.paper(ctx, KP.jitter([[x + 220, y + 20], [x + 340, y - 50], [x + 340, y + 30], [x + 230, y + 60]], 952 + b, 1), P.brass, { shadow: KP.SH.near, shadowA: .3 });
    ctx.fillStyle = KP.mix(P.brass, P.wood, .5); for (const [hx, hy] of [[300, -10], [320, 20], [270, 30]]) { ctx.beginPath(); ctx.arc(x + hx, y + hy, 7, 0, TAU); ctx.fill(); }
  }
  function carton(ctx, x, y, b) {
    // a plain white takeout carton: no print, no lettering
    KP.brush(ctx, KP.curve([[x - 60, y - 140], [x - 30, y - 230], [x + 30, y - 230], [x + 60, y - 140]], false, 8), 6, P.ink, { taper: .05 });
    KP.paper(ctx, KP.jitter([[x - 110, y - 140], [x + 110, y - 140], [x + 80, y + 70], [x - 80, y + 70]], 960 + b, 1.2), P.rice, { shadow: KP.SH.near, shadowA: .35, mis: P.turq, misOff: [-3, 2], misA: .4 });
    KP.paper(ctx, KP.jitter([[x - 110, y - 140], [x - 70, y - 190], [x - 20, y - 140]], 961 + b, .8), P.rice, { shadow: KP.SH.far, shadowA: .25 });
    KP.paper(ctx, KP.jitter([[x + 20, y - 140], [x + 70, y - 190], [x + 110, y - 140]], 962 + b, .8), P.rice, { shadow: KP.SH.far, shadowA: .25 });
    KP.brush(ctx, [[x, y - 140], [x, y + 66]], 4, P.lavender, { taper: .2 });
  }
  function halves(ctx, t, t0, top, bottom, divider, under) {
    // split-wipe: the halves slide in from opposite sides over the previous shot (held on its last frame)
    const p = KP.ep(t, t0, t0 + .35, 'out');
    if (p < 1 && under) { ctx.save(); under(t0 - 1 / 60); ctx.restore(); }
    ctx.save(); ctx.beginPath(); ctx.rect(0, 0, KP.W, 760); ctx.clip(); ctx.translate((1 - p) * -KP.W, 0); top(ctx); ctx.restore();
    ctx.save(); ctx.beginPath(); ctx.rect(0, 760, KP.W, KP.H - 760); ctx.clip(); ctx.translate((1 - p) * KP.W, 0); bottom(ctx); ctx.restore();
    KP.paper(ctx, KP.jitter(KP.torn(KP.rr(-40, 746, KP.W + 80, 28, 6, 40), 990, 3, 12), 991, 1), P.ink, { shadow: [0, 6], shadowA: .3 });
    if (divider) divider(ctx);
  }

  KP.scene('s06_origin', {
    sfx(cues, D) {
      const T = times(cues, D), a = [], e = (t, name, gain) => a.push({ t: Math.max(0, +t.toFixed(3)), name, gain });
      e(T.seaweed, 'whoosh', .35); e(T.villain, 'impact_soft', .45);
      e(T.l2 - .05, 'paper', .35); e(T.y1968, 'pop', .4); e(T.wrote, 'typewriter', .35); e(T.journal, 'paper', .4); e(T.journal + .45, 'stamp', .25);
      e(T.numb, 'pop_small', .35); e(T.weak, 'pop_small', .35);
      e(T.wine, 'blip', .4); e(T.salt, 'blip', .4); e(T.msg, 'blip', .45); e(T.guess, 'boing', .35);
      e(T.l4, 'whoosh', .4); e(T.catchy, 'swoosh_down', .35); e(T.chinese, 'stamp', .8); e(T.chinese + .05, 'impact_big', .45); e(T.syndrome, 'paper', .35);
      e(T.injected, 'swoosh_down', .4); e(T.huge, 'impact_soft', .4); e(T.damage, 'alarm', .25);
      e(T.l6, 'whoosh', .35); e(T.nothing, 'impact_soft', .5); e(T.dinner, 'chomp', .35);
      e(T.l7, 'whoosh', .3); e(T.tomato, 'pop_small', .3); e(T.parm, 'alarm', .3);
      e(T.s9 + .2, 'swoosh_down', .3); e(T.name, 'impact_soft', .4); e(D - .5, 'whoosh', .4);
      return a.sort((x, y) => x.t - y.t);
    },
    draw(ctx, t, cues, D, nested) {
      const T = times(cues, D), b = KP.boil(t);
      const prev = tp => this.draw(ctx, tp, cues, D, true);
      let label = null;
      if (t < T.l2) {
        // ---- 1: kombu -> shaker, villain spotlight + moustache
        const push = 1 + .06 * KP.ep(t, 0, T.l2, 'inOut');
        ctx.fillStyle = P.blue; ctx.fillRect(0, 0, KP.W, KP.H);
        KP.camera(ctx, 490, 900, push, () => {
          const on = t >= T.villain;
          if (on) KP.paper(ctx, KP.jitter([[400, -200], [580, -200], [800, 1220], [180, 1220]], 20 + b, 2), P.tomato, { shadow: false, mis: P.pink, misOff: [-6, 0], misA: .4 });
          KP.paper(ctx, KP.jitter(KP.ell(490, 1210, 300, 60, 44), 21 + b, 1.5), on ? P.coral : KP.mix(P.blue, P.grape, .4), { shadow: false });
          // the kombu strip curls in from the left and into the shaker
          const k = KP.ep(t, T.seaweed - .3, T.villain - .3, 'inOut');
          if (k < 1) {
            const pts = []; for (let i = 0; i <= 24; i++) { const u = i / 24, ph = KP.held(t) * 3; pts.push([KP.lerp(60, 470, k) + u * 360 * (1 - k) + Math.sin(u * 8 + ph) * 30 * (1 - k), KP.lerp(500, 820, k) + Math.sin(u * 6 + ph) * 40 * (1 - k) + u * 200 * k * (1 - k)]); }
            KP.brush(ctx, pts.map(([x, y]) => [x + 6, y + 8]), 70 * (1 - k * .8), KP.rgba(P.ink, .25), { taper: .3 });
            KP.brush(ctx, pts, 70 * (1 - k * .8), P.kombu, { taper: .3 });
          }
          const sh = KP.arrive(t, T.villain, .2);
          KP.shaker(ctx, 490, 1210, 460, { boil: b, seed: 7, sy: t >= T.villain ? sh.sy : 1 });
          // villain moustache drops onto the shaker
          if (on) {
            const ma = KP.arrive(t, T.villain + .08, .3), my = 960 - (1 - ma.p) * 500;
            ctx.save(); ctx.translate(490, my);
            KP.paper(ctx, KP.jitter(KP.curve([[0, -14], [40, -30], [86, -24], [118, -50], [104, -6], [64, 16], [20, 10], [0, 4], [-20, 10], [-64, 16], [-104, -6], [-118, -50], [-86, -24], [-40, -30]], true, 5), 30 + b, 1.2), P.ink, { shadow: KP.SH.far, shadowA: .35 });
            ctx.restore();
          }
        });
        if (t >= T.villain) label = c => { const ar = KP.arrive(t, T.villain, .3); c.save(); KP.pose(c, ar, 330, 380, 'pop'); KP.slip(c, [{ s: 'Villain?', size: 84, ls: 1 }], 160, 260, { rot: -.05, seed: 40, boil: b, tape: P.lemon, padX: 38, padY: 22 }); c.restore(); };
      } else if (t < T.l4) {
        // ---- 2-4: the letter on the cobalt desk; 4 = close on the card
        const slideIn = KP.ep(t, T.l2, T.l2 + .45, 'out');
        const write = KP.ep(t, T.doctor - .3, T.journal, 'inOut') * .55 + KP.ep(t, T.after, T.numb - .2, 'inOut') * .45;
        const fold = t >= T.journal && t < T.journal + .45 ? .92 + .08 * KP.ease.out(KP.prog(t, T.journal + .15, T.journal + .45)) : 1;
        const envOut = KP.ep(t, T.l3 - .3, T.l3 + .1, 'in');
        const cardIn = KP.arrive(t, T.l3 - .1, .35);
        const q = [KP.prog(t, T.wine, T.wine + .35), KP.prog(t, T.salt, T.salt + .35), KP.prog(t, T.msg, T.msg + .35)];
        const merge = KP.ep(t, T.itwas, T.guess, 'inOut');
        const close = KP.ep(t, T.l3 - .2, T.l3 + .4, 'inOut');
        const s = KP.lerp(1 + .05 * KP.prog(t, T.l2, T.l3), 1.75, close), cx = KP.lerp(470, 250, close), cy = KP.lerp(820, 1150, close);
        ctx.fillStyle = P.blue; ctx.fillRect(0, 0, KP.W, KP.H);
        ctx.save(); ctx.translate(0, (1 - slideIn) * 1400);
        KP.camera(ctx, cx, cy, s, () => {
          ctx.save();
          desk(ctx, t, b, T, { write, sy: fold, env: [envOut * 900, 0], post: KP.backOut(KP.prog(t, T.journal + .45, T.journal + .7)), card: false });
          ctx.restore();
          if (cardIn.on) { ctx.save(); KP.pose(ctx, cardIn, 250, 1300, 'up', 600); card(ctx, b, q, merge); ctx.restore(); }
        });
        ctx.restore();
        if (t >= T.y1968 && t < T.l3 - .2) label = c => { const ar = KP.arrive(t, T.y1968, .3), out = KP.leave(t, T.after - .1, .2); c.save(); c.globalAlpha = 1 - out; KP.pose(c, ar, 760, 330, 'pop'); KP.slip(c, [{ s: '1968', size: 96, font: 'Young Serif', weight: 400, ls: 3 }], 620, 226, { rot: .05, seed: 41, boil: b, tape: P.pink, padX: 34, padY: 18 }); c.restore(); };
        if (t >= T.guess) label = c => { const ar = KP.arrive(t, T.guess, .25); c.save(); KP.pose(c, ar, 760, 360, 'pop'); KP.slip(c, [{ s: 'A guess', size: 80, ls: 1 }], 600, 270, { rot: .04, seed: 42, boil: b, tape: P.pink, padX: 36, padY: 20 }); c.restore(); };
      } else if (t < T.l5) {
        // ---- 5: whip to style frame C, the stamp slams the headline, newspapers slide out
        const whip = KP.ep(t, T.l4, T.l4 + .28, 'out');
        const hit = T.chinese, shake = t >= hit && t < hit + 6 / 30 ? KP.wob(t * 2, 3, 14) : [0, 0];
        ctx.fillStyle = P.blue; ctx.fillRect(0, 0, KP.W, KP.H);
        ctx.save(); ctx.translate((1 - whip) * 1300 + shake[0], shake[1]);
        desk(ctx, t, b, T, { write: 1, clearZone: true, card: true, q: [1, 1, 1], merge: 0 });
        ctx.save(); ctx.translate(250, 1180); ctx.restore();
        if (t >= hit) impression(ctx);
        papers(ctx, b, KP.prog(t, T.syndrome - .1, T.syndrome + .9));
        // impact flecks
        if (t >= hit && t < hit + .5) {
          const fr = KP.rng(88), sa = -8 * D2R, k = KP.ease.out(KP.prog(t, hit, hit + .5));
          [[-1.05, .2], [-1, -.6], [-.7, 1.15], [-.2, -1.2], [.3, 1.2], [.75, -1.15], [1.04, .5], [1.06, -.3], [.1, 1.25]].forEach(([u, v], i) => {
            const lx = u * STW / 2, ly = v * STH / 2, x = SCX + lx * Math.cos(sa) - ly * Math.sin(sa), y = SCY + lx * Math.sin(sa) + ly * Math.cos(sa);
            const dx = x - SCX, dy = (y - SCY) * 2.2, dl = Math.hypot(dx, dy), nx = dx / dl, ny = dy / dl, sz = 7 + fr() * 9;
            ctx.save(); ctx.translate(x + nx * 70 * k, y + ny * 50 * k); ctx.rotate(fr() * 3 + k * 2);
            KP.paper(ctx, KP.jitter([[-sz, -sz * .6], [sz, -sz * .4], [sz * .7, sz * .7], [-sz * .8, sz * .5]], i, .8), [P.tomato, P.rice, P.lemon][i % 3], { shadow: KP.SH.far });
            ctx.restore();
          });
        }
        // the stamp: descends from 'catchy', hits on 'Chinese', lifts to rest at the top of frame
        const down = KP.ease.in(KP.prog(t, T.catchy, hit)), lift = KP.ep(t, hit + .12, hit + .5, 'out');
        const sy = t < hit ? KP.lerp(-700, 560, down) : KP.lerp(560, 0, lift), sx = t < hit ? -230 : KP.lerp(-230, 0, lift);
        if (t >= T.catchy - .05) stampHandle(ctx, b, sx, sy);
        ctx.restore();
        if (whip < 1) for (let i = 0; i < 8; i++) KP.brush(ctx, [[100 + i * 120, 300 + i * 140], [100 + i * 120 + 260 * (1 - whip), 300 + i * 140]], 8, P.rice, { taper: .6 });
      } else if (t < T.l6) {
        // ---- 6: the lab; giant syringe lowers (never touches); newspapers at the edges grow bars on 'brain damage'
        const push = 1 + .05 * KP.ep(t, T.l5, T.l6, 'inOut');
        KP.camera(ctx, 490, 860, push, () => {
          graph(ctx, -200, -200, 1300, 2200, P.lemon, KP.mix(P.lemon, P.turq, .35));
          KP.paper(ctx, KP.jitter(KP.rr(-100, 1060, 1300, 900, 10, 40), 650 + b, 1.5), P.turq, { shadow: [0, -10], shadowA: .3 });
          nest(ctx, 400, 1000, b, t, 1.3);
          const low = KP.ep(t, T.injected - .2, T.huge - .1, 'out'), grow = KP.arrive(t, T.huge, .35);
          const tipY = KP.lerp(-300, 720, low), s = t >= T.huge ? 1 + .35 * grow.p : 1;
          syringe(ctx, 400, tipY + (t >= T.huge ? -(s - 1) * 120 : 0), s, b);
          const bar = t >= T.brain ? 22 + 30 * KP.backOut(KP.prog(t, T.brain, T.brain + .3)) : 22;
          const jig = t >= T.brain ? KP.wob(t, 7, 4) : [0, 0];
          ctx.save(); ctx.translate(jig[0], jig[1]);
          newspaper(ctx, 1010, 1150, 260, 300, .2, 66, P.coral, b, bar); newspaper(ctx, -10, 1180, 240, 280, -.22, 67, P.coral, b, bar); newspaper(ctx, 1040, 760, 220, 260, .28, 68, P.coral, b, bar);
          ctx.restore();
        });
        if (t >= T.huge) label = c => { const ar = KP.arrive(t, T.huge, .3); c.save(); KP.pose(c, ar, 760, 320, 'pop'); KP.slip(c, [{ s: 'Huge doses', size: 62, ls: 1 }], 600, 236, { rot: -.04, seed: 43, boil: b, tape: P.pink, padX: 36, padY: 20 }); c.restore(); };
      } else if (t < T.l7) {
        // ---- 7: split: injection  ≠  dinner
        halves(ctx, t, T.l6, c => {
          graph(c, 0, 0, KP.W, 760, P.lemon, KP.mix(P.lemon, P.turq, .35));
          nest(c, 490, 640, b, t, .55);
          syringe(c, 490, 520 + Math.sin(KP.held(t) * 2) * 6, .5, b);
        }, c => {
          c.fillStyle = P.pink; c.fillRect(0, 760, KP.W, KP.H - 760);
          KP.paper(c, KP.jitter(KP.rr(-60, 1180, 1200, 800, 10, 40), 660 + b, 1.5), P.coral, { shadow: [0, -8], shadowA: .3 });
          bowl(c, 380, 1080, b, t, T);
          const ch = t >= T.dinner - .3 && t < T.dinner + .15;
          KP.arm(c, 690, 1150, 40, 2.4, 22, P.bubble, { seed: 670, boil: b });
          KP.cell(c, 760, 1110, 150, 160, { seed: 671, boil: b, fill: P.bubble, bib: P.rice, eyes: 'happy', mouth: ch ? 'chomp' : 'smile', mouthDx: -.1 });
          if (t > T.dinner - .45 && t < T.dinner) KP.goldie(c, KP.lerp(560, 735, KP.prog(t, T.dinner - .45, T.dinner)), 1110, 48, { seed: 672, boil: b, steam: false });
        }, c => {
          const ne = KP.arrive(t, T.nothing, .3); if (!ne.on) return;
          c.save(); KP.pose(c, ne, 490, 760, 'pop');
          KP.paper(c, KP.jitter(KP.egg(490, 760, 96, 96, 0, 40), 680 + b, 1.2), P.rice, { shadow: KP.SH.near, shadowA: .35 });
          for (const dy of [-22, 22]) KP.paper(c, KP.jitter(KP.rr(430, 760 + dy - 12, 120, 24, 8, 12), 681 + dy + b, 1), P.ink, { shadow: false });
          KP.paper(c, KP.jitter(KP.rot(KP.rr(480, 690, 22, 140, 8, 12), 491, 760, .5), 683 + b, 1), P.tomato, { shadow: false });
          c.restore();
        });
      } else if (t < T.s9) {
        // ---- 8: the double standard: same Goldies, same bell; only one rings
        halves(ctx, t, T.l7, c => {
          c.fillStyle = P.turq; c.fillRect(0, 0, KP.W, 760);
          const pa = KP.arrive(t, T.tomato, .3);
          c.save(); c.translate(0, 0); if (t >= T.tomato) KP.pose(c, { ...pa, p: 1 }, 400, 640, 'pop'); pasta(c, 400, 620, b); c.restore();
          bell(c, 760, 430, b, false, t);
          for (let i = 0; i < 2; i++) KP.goldie(c, 300 + i * 150, 420 + Math.sin(KP.held(t) * 3 + i) * 8, 60, { seed: 700 + i, boil: b, steam: false, shadow: KP.SH.far });
        }, c => {
          c.fillStyle = P.pink; c.fillRect(0, 760, KP.W, KP.H - 760);
          KP.paper(c, KP.jitter(KP.rr(-60, 1240, 1200, 700, 10, 40), 690 + b, 1.5), P.magenta, { shadow: [0, -8], shadowA: .3 });
          carton(c, 400, 1200, b);
          bell(c, 760, 1010, b, t >= T.parm, t);
          for (let i = 0; i < 2; i++) KP.goldie(c, 300 + i * 150, 900 + Math.sin(KP.held(t) * 3 + i) * 8, 60, { seed: 700 + i, boil: b, steam: false, shadow: KP.SH.far });
        });
      } else {
        // ---- 9: the split collapses; the stamped headline slides back; push in on 'name'
        const col = KP.ep(t, T.s9, T.s9 + .3, 'in');
        const back = KP.ep(t, T.s9 + .1, T.s9 + .5, 'out');
        const push = 1 + .22 * KP.ep(t, T.name - .2, D, 'inOut');
        ctx.fillStyle = P.blue; ctx.fillRect(0, 0, KP.W, KP.H);
        ctx.save(); ctx.translate((1 - back) * 1300, 0);
        KP.camera(ctx, KP.lerp(490, SCX, push - 1 > 0 ? 1 : 0), KP.lerp(860, SCY, KP.prog(push, 1, 1.22)), push, () => {
          desk(ctx, t, b, T, { write: 1, clearZone: true, card: true, q: [1, 1, 1] });
          impression(ctx);
          papers(ctx, b, 1);
          stampHandle(ctx, b, 0, 0);
        });
        ctx.restore();
        if (col < 1) {
          ctx.save(); ctx.translate(0, 760); ctx.scale(1, 1 - col); ctx.translate(0, -760);
          ctx.fillStyle = P.turq; ctx.fillRect(0, 0, KP.W, 760); ctx.fillStyle = P.pink; ctx.fillRect(0, 760, KP.W, KP.H - 760);
          ctx.restore();
          KP.paper(ctx, KP.jitter(KP.rr(-40, 746, KP.W + 80, 28, 6, 40), 991, 1), P.ink, { shadow: false });
        }
      }
      if (label) label(ctx);
      KP.texture(ctx, 'news', .05);
      KP.texture(ctx, 'grain', .1, 101, 57);
      KP.slide(ctx, KP.ep(t, D - .5, D, 'inOut'), KP.GROUND.s09_outro);   // out: paper slide back to the TV wall
    },
  });
})();
