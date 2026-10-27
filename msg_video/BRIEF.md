# Production brief — "MSG: The Most Misunderstood Molecule"

## Client rules (override everything below)
- Format: vertical 9:16, 1080x1920, a single Instagram Reel under 3 minutes (see PARTS in script.py).
  Safe zones: no key content in the top ~220px or bottom ~440px, keep ~140px clear on the right.
- Animated word-by-word captions in a band at about y=1300–1480 (word timings in cues.json `words`).
- Avoid too much text: besides captions, at most one short label (1–3 words) on screen at a time. Tell it with pictures.
- Never use dull colours: no greys, beige, muddy browns or desaturated tones. Everything vivid and saturated.
- Approved style: Kombu Paper — see design/APPROVED.md. Any new design choice still goes to the client first.

Animated explainer Reel (under 3 min, 1080x1920, 30fps). Read `script.py` (narration + visual direction) and `cues.json`
(exact start/end second of each narration line per scene — sync visuals to these!).

## How rendering works
- Each scene is `scenes/<id>.js` calling `KZ.scene(id, { draw(ctx, t, cues, D), sfx(cues, D) })`.
  `draw` must be a pure function of `t` (seconds since scene start, 0..D). Deterministic: no `Math.random`, no `Date`,
  no state carried between frames (frames are rendered in order, but treat each as independent). Use `KZ.rng(seed)`.
- Shared kit: `lib/kz.js` (palette `KZ.C`, easing `KZ.ep/prog/pop/window`, `KZ.blob`, `KZ.glow`, `KZ.stars`, `KZ.text`,
  `KZ.person`, `KZ.glutamate`, `KZ.shaker`, `KZ.camera`, `KZ.vignette`, `KZ.fade`...). **Do not edit lib/kz.js** — other
  agents rely on it. Put scene-specific helpers inside your own scene file (wrap in an IIFE or prefix names to avoid
  globals colliding — all scene files share one page).
- `scenes/s02_title.js` is an old 16:9 placeholder in the rejected style; replace it, do not imitate it.
- Preview: `node render.js --stills <id> 1,5.5,12` → `out/stills/<id>_<t>.png` (view them with the Read tool!),
  `node render.js --contact <id> 12` → contact sheet of 12 evenly spaced frames. `node render.js --video <id> --jobs 1`
  renders the full mp4 (only needed for a final check).

## Style rules
- The approved look is Kombu Paper: follow design/APPROVED.md, the Kombu Paper section of design/DIRECTIONS.md (as
  amended by the client rules above) and the style frames in design/frames/kombu-paper/. Reuse the `KP` helpers.
- Constant gentle motion with the direction's stop-motion "boil"; things pop in and move with easing.
  Change the visual every ~3–6 s to match each narration line. Never a fully static frame.
- Factual accuracy matters: don't add claims or numbers beyond the script.

## Sound effects
`sfx(cues, D)` returns `[{t, name, gain}]` (t in scene seconds, gain 0..1, default 1). Place SFX on visual events
(pop-ins, whooshes on transitions, stamps...). Keep them tasteful: roughly one every 1–3 s, quieter under narration
(gain 0.3–0.6). Allowed names ONLY:
whoosh, whoosh_long, swoosh_up, swoosh_down, pop, pop_small, bubble, blip, click, tick, ding, chime, sparkle,
impact_soft, impact_big, thud, stamp, boing, riser, record_scratch, tv_on, tv_off, channel_switch, static, glitch,
neon_buzz, neon_flicker, typewriter, paper, heartbeat, alarm, crunch, slurp, chomp, sizzle, bubbling_pot, cash, error, success
