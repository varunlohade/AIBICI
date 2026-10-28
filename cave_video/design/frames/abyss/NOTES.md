# Abyss cross-section: design notes

Frames: `A.png` (hook), `B.png` (-180 m), `C.png` (-270 m danger). Render them with
`node design/render_styleframes.js design/frames/abyss/frames.js A,B,C [t]`.
The kit lives in `frames.js` as `window.AB`. Everything is deterministic: it uses `AB.rng(seed)` and sine noise, with no `Math.random` and no `Date`.

## Concept
- The 9:16 frame is the depth axis. One shared "world" (metres: x from the shaft axis, d below the pool surface)
  is drawn through a camera `AB.cam(cx, cy, dRef, pxPerMetre)`. Every shot is the same cave at a different zoom and
  position, so the animation can move the camera down the sinkhole continuously, from the full section (A, 4.2 px/m)
  through the chamber (B, 10 px/m) to the floor (C, 16 px/m).
- The cave is a stylised Boesmansgat: a crater, a small pool, a narrow throat at about 30 m, then a huge chamber that
  leans right as it deepens and has ledges at different depths on each side, so it never reads as a symmetric
  vase or body. The floor is a rubble slope at about 265 to 283 m. The shape is evocative, not a survey.
- It is drawn like a section diagram. The rock is near-black and carries faint strata hatching. A thin teal **cut line** traces
  the rock edge and fades with depth. Three receding wall layers (`layers`) are darkened inside the water, plus two
  rubble layers on the floor. Together they give the "layered silhouettes" depth without detail.

## Colour
| role | value |
|---|---|
| water by depth | `#3a9a93` (0 m) → `#13465e` (35 m) → `#0b2840` (70) → `#071a2c` (120) → `#040e1a` (180) → `#03080f` (240) → `#010305` |
| rock / air | `#07080a` rock, `#05090d`→`#0c1b22` night air |
| cut line | rgb(120,205,205), alpha 0.55 → 0.07 with depth |
| torch (warm accent) | `#ffb347` / hot `#ffe2a8`, always additive (`lighter`) |
| danger (sparingly) | `#ff3b2f`: only the gauge, the CO₂ chip and the edge wash in C |
| marker (cold) | rgb(200,230,255): the only cold light source |

The red only appears at the danger moment. Until then the palette stays teal, blue and black, with amber as the only
warm thing in the frame.

## Type
- **Barlow Condensed Light 300** (downloaded): big depth numbers ("-180 m", "270 m"). It is thin and tall, so it
  reads like instrument numerals, and its thin strokes stay quiet even at 230 px.
- **JetBrains Mono 500** (downloaded) is used for technical labels with wide letter-spacing ("DEPTH", "8 JAN 2005", the ruler).
- **IBM Plex Sans SemiBold 600** (downloaded, variable) is the caption face. It is neutral and slightly technical, it has
  a plain zero (Atkinson's slashed zero looked like code), and it reads well at 56 px.
- Barlow Condensed SemiBold is used for "CO₂" in the warning chip. The subscript is drawn by hand.
- Fonts are in `assets/`. `AB.FONTS` is the font list for loaders.

## Captions (`AB.caption(ctx, words, k, opts)`)
- The caption is centred at y = 1660, the text is 56 px, the lines are balanced (at most 820 px wide, so it clears the right 140 px), and the line height is 1.24.
- It sits on a dark plate, `rgba(2,6,10,.62)`, with an 18 px radius and a 1.5 px hairline at 8% white. The plate is quiet and does not glow.
- The layout is fixed per chunk, so nothing reflows. `k` is the number of words spoken and can be fractional. Each word
  fades and rises in over 10 px. The current word is pure white with a 3 px amber underline, and spoken words settle to `#dfe9ea`.
  Pass `hot:false` to remove the highlight.
- Keep chunks to two lines at most, ideally 3 to 7 words.

## Safe zones
Key content stays between y 220 and 1560 and left of x 940: the gauge is at (80, 300), the date at y 262, the floor
label is right-aligned at 930, and the diver and line stay at x ≤ 800. Only background cave walls and atmosphere bleed to the edges.

## Respectfulness
- People are only ever flat silhouettes with no faces, and there are no bodies or injuries.
- Deon Dreyer is represented only by `AB.marker`, a small cold point of light on the floor with a soft halo. It marks a place,
  never a figure, and it stays cold and still against the warm, moving torch.
- Dave Shaw is a silhouette with an amber rim and a thin cool outline (`edge`) so he reads against black water.
- Danger is shown by the instrument and the tangle, never by drama on the figure.

## Motion guidance for the animation
- Motion should be slow, heavy and inevitable. Camera moves should use long ease-in-out curves, and the depth number should tick continuously.
- Silt drifts slowly. Bubbles rise steadily, and the diver's fins kick slowly (`kickSpeed` about 1.6).
- C's red pulse is `.5+.5*cos((t-1)*5.2)`, so the alarm stays low most of the time and peaks in short beats. Pair it with
  `alarm`/`heartbeat` SFX sparingly.
- Caustics and rays only exist in the top ~30 m and die out with the light. Below 150 m the only light is the torch.

## AB kit reference
- Geometry: `cam`, `wallL/wallR/floorD`, `waterPoly(cam)`, `cave(ctx, cam, t, {layers, cutWidth})` (returns the water poly for clipping), `surface`.
- Light: `rays`, `caustics` (pass the poly to clip), `torch(ctx, {x,y,ang,len,spread}, {intensity, halo})`, `scatter`, `inCone`.
- Particles: `silt(ctx, t, {n, lights, alpha, seed})` (cool, and lit amber inside cones), `coneSilt`, `bubbles(ctx, x, y, t, {n, height, scale})`.
- Figures: `diver(ctx, x, y, ang, scale, t, {rim, edge, kick, flip})` returns the torch origin and head position. `marker` for the cold marker light.
- Line: `line(ctx, pts, {color, width, glow})` and `tangle(x, y, R, t, {n, seed, sx, sy, rot})`.
- HUD: `gauge(ctx, x, y, depth, {size, warn, max, ruleW})`, `ruler(ctx, cam, x, {...})`, `depthMark`, `label`, `warnChip`, `alarm`.
- Finish: `vignette`, `grain(ctx, t, amt)` (a seeded tile that changes at 24 fps), `caption`.
- Order per frame: cave → rays/caustics/surface → lines → scatter → diver → torch → silt → HUD marks → vignette →
  alarm → gauge → grain → caption.

## Review passes
1. **First pass.** The cave read as a symmetric bell or vase. The diver in C was invisible and the tangle looked like a target
   logo. The B camera was too wide (it showed the chamber ceiling, which weakened the sense of vastness). The caption plate was
   sized for a wrapped second line during the reveal.
2. **Second pass.** I added the drift and asymmetric ledges, darkened the water stops, switched captions to IBM Plex with balanced
   line breaks, zoomed B to 10 px/m and made the tangle a smoothed random scribble. The ledges still mirrored each other, the
   floor had a vertical step where it met the wall, the diver drawing was crude, and the red wash muddied C.
3. **Third pass.** I redrew the diver as one streamlined silhouette with a rebreather, hose, bent legs and fins. I moved the ledges
   to different depths, let the walls run down into the floor, and limited the red wash to the edges with the peak timed to the still.
4. **Fourth pass.** In C the diver is larger, has a cool outline and a backscatter glow, and has his legs splayed, so he clearly reads as
   a person struggling. The tangle moved onto the torch hand. I also removed a stray glow box above the pool surface.

Possible next steps: a slow "camera dive" transition from A to B, a CO₂ chip that blinks in on cue, and the
marker brightening very slightly when the torch passes it.
