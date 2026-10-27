# Production brief — "MSG: The Most Misunderstood Molecule"

Kurzgesagt-style explainer (~6 min, 1920x1080, 30fps). Read `script.py` (narration + visual direction) and `cues.json`
(exact start/end second of each narration line per scene — sync visuals to these!).

## How rendering works
- Each scene is `scenes/<id>.js` calling `KZ.scene(id, { draw(ctx, t, cues, D), sfx(cues, D) })`.
  `draw` must be a pure function of `t` (seconds since scene start, 0..D). Deterministic: no `Math.random`, no `Date`,
  no state carried between frames (frames are rendered in order, but treat each as independent). Use `KZ.rng(seed)`.
- Shared kit: `lib/kz.js` (palette `KZ.C`, easing `KZ.ep/prog/pop/window`, `KZ.blob`, `KZ.glow`, `KZ.stars`, `KZ.text`,
  `KZ.person`, `KZ.glutamate`, `KZ.shaker`, `KZ.camera`, `KZ.vignette`, `KZ.fade`...). **Do not edit lib/kz.js** — other
  agents rely on it. Put scene-specific helpers inside your own scene file (wrap in an IIFE or prefix names to avoid
  globals colliding — all scene files share one page).
- Reference scene for the target style: `scenes/s02_title.js`.
- Preview: `node render.js --stills <id> 1,5.5,12` → `out/stills/<id>_<t>.png` (view them with the Read tool!),
  `node render.js --contact <id> 12` → contact sheet of 12 evenly spaced frames. `node render.js --video <id> --jobs 1`
  renders the full mp4 (only needed for a final check).

## Style rules (Kurzgesagt look)
- Flat vector shapes, no outlines (or very subtle), saturated colors on deep navy/indigo backgrounds, soft glows,
  rounded everything, cute simple characters with dot eyes. Font: Fredoka (already loaded).
- Constant gentle motion: things bob/breathe/drift; slow camera push/pans. Never a fully static frame.
- Elements pop in with overshoot (`KZ.pop`), move with easing. Change the visual every ~3–6 s to match each narration line.
- Minimal on-screen text: short labels/keywords/numbers only (no subtitles).
- Fade in from black over the first ~0.4s and out over the last ~0.4s of each scene (`KZ.fade`) so cuts are smooth.
- Keep important content inside a 1720x940 safe area. Finish with `KZ.vignette(ctx)` for depth.
- Factual accuracy matters: don't add claims/numbers beyond the script.

## Sound effects
`sfx(cues, D)` returns `[{t, name, gain}]` (t in scene seconds, gain 0..1, default 1). Place SFX on visual events
(pop-ins, whooshes on transitions, stamps...). Keep them tasteful: roughly one every 1–3 s, quieter under narration
(gain 0.3–0.6). Allowed names ONLY:
whoosh, whoosh_long, swoosh_up, swoosh_down, pop, pop_small, bubble, blip, click, tick, ding, chime, sparkle,
impact_soft, impact_big, thud, stamp, boing, riser, record_scratch, tv_on, tv_off, channel_switch, static, glitch,
neon_buzz, neon_flicker, typewriter, paper, heartbeat, alarm, crunch, slurp, chomp, sizzle, bubbling_pot, cash, error, success
