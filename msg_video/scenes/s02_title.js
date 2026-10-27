// Scene 2 — Title card. Reference scene for style: layered depth, glow, gentle motion, pop-ins.
KZ.scene('s02_title', {
  sfx: (cues, D) => [
    { t: 0.0, name: 'whoosh_long', gain: 0.8 },
    { t: 0.9, name: 'sparkle', gain: 0.7 },
    { t: 1.6, name: 'impact_soft', gain: 0.9 },
    { t: 2.3, name: 'pop', gain: 0.6 },
  ],
  draw(ctx, t, cues, D) {
    const { C, W, H } = KZ;
    // slow push-in camera for the whole scene
    const zoom = KZ.lerp(1.0, 1.08, KZ.ep(t, 0, D, 'linear'));
    KZ.bg(ctx, C.indigo, C.night);
    KZ.camera(ctx, W / 2, H / 2, zoom, () => {
      KZ.stars(ctx, t, 11, 220, 8);
      // distant planet-like plate of food silhouettes
      KZ.glow(ctx, W / 2, 470, 520, C.violet, .5);
      // crystals orbiting
      const r = KZ.rng(5);
      for (let i = 0; i < 14; i++) {
        const a = i / 14 * Math.PI * 2 + t * .25, rad = 330 + r() * 90;
        const x = W / 2 + Math.cos(a) * rad, y = 450 + Math.sin(a) * rad * .35;
        const s = (.6 + r() * .6) * KZ.pop(t, .3 + i * .05);
        const front = Math.sin(a) > 0;
        if (!front) continue;
        KZ.withT(ctx, x, y, s, a * 2, () => KZ.poly(ctx, [[0, -26], [18, 0], [0, 26], [-18, 0]], C.cream));
      }
      // hero molecule
      const ms = KZ.pop(t, .7, .8) * 1.35;
      KZ.glutamate(ctx, W / 2, 440 + Math.sin(t * 1.5) * 10, t, { s: ms, glow: 1, rot: Math.sin(t * .6) * .08, face: true });
    });
    // title
    const tp = KZ.ep(t, 1.4, 2.2, 'outBack');
    KZ.text(ctx, 'MSG', W / 2, 760 + (1 - tp) * 60, { size: 150, weight: 700, alpha: KZ.prog(t, 1.4, 1.8), color: C.yellow, shadow: C.gold, shadowBlur: 40 });
    KZ.text(ctx, 'The Most Misunderstood Molecule', W / 2, 880, { size: 62, weight: 500, alpha: KZ.prog(t, 2.2, 2.8), letterSpacing: 2 });
    KZ.dust(ctx, t, 9, 30, C.cream, .18);
    KZ.vignette(ctx, .55);
    KZ.fade(ctx, 1 - KZ.prog(t, 0, .5));
    KZ.fade(ctx, KZ.prog(t, D - .5, D));
  },
});
