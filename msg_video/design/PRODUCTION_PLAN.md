# MSG — Production Plan

Companion to `design/SHOTLIST.md` / `design/shotlist.json` (82 shots, 17 hero moments, 359.4 s). This plan is
style-agnostic: it works for whatever look the client approves.

## 1. Workload at a glance

| Scene | Dur (s) | Shots | Hero shots | Complexity / risk | Shared props it depends on |
|---|---|---|---|---|---|
| s01_coldopen | 47.7 | 10 | s01.07, s01.10 | High: 5 talking heads, TV wall, DRAMATIZATION tags | TVWALL, DRAMA_TAG, SIGN, SHAKER |
| s02_title | 6.4 | 2 | s02.02 | Done (`scenes/s02_title.js`) — restyle only | GLU |
| s03_ikeda | 41.5 | 10 | s03.07 | Medium: period setting, one character | IKEDA, TASTE_ICONS, GLU, NA, SHAKER (first canonical) |
| s04_everywhere | 42.8 | 11 | s04.11 | Medium: many food props, one chart | GLU, TOMATO, SHAKER |
| s05_body | 45.2 | 10 | s05.05, s05.06 | High: cutaway body, long camera travel, counter | BODY, GUTCELL, BBB, GLU |
| s06_origin | 78.1 | 17 | s06.07, s06.12, s06.16 | Highest: longest scene + the cultural-sensitivity section | HEADLINE, GUTCELL (callback), SHAKER, SALT, TOMATO, GLU |
| s07_tests | 46.7 | 9 | s07.03, s07.05, s07.08 | Medium: crowd of volunteers, chart, stamp | SHAKER, SALT |
| s08_twist | 18.7 | 5 | s08.03 | Low | SHAKER, SALT, NA |
| s09_outro | 32.3 | 8 | s09.05, s09.06, s09.08 | High: callbacks must match s01 exactly | TVWALL, DRAMA_TAG, SIGN, HEADLINE, TOMATO, SHAKER, GLU |

## 2. Recommended split — 4 animators

The split groups scenes that **share the riskiest recurring props**, so each continuity pair lives with one person.

| Animator | Scenes | Total | Shots | Owns (builds the canonical version of) |
|---|---|---|---|---|
| **A — Bookends** | s01, s09 (+ s02 restyle) | 86.4 s | 20 | TVWALL, DRAMA_TAG, SIGN, end card. s01↔s09 callbacks are the biggest continuity risk in the film, so one person does both. |
| **B — Origins** | s03, s04 | 84.2 s | 21 | GLU usage rules, SHAKER, NA, TOMATO, TASTE_ICONS, IKEDA |
| **C — Science** | s05, s07, s08 | 110.6 s | 24 | BODY, GUTCELL, BBB, SALT, lab/volunteer look. s07→s08 share shakers + lab tone; s08 reuses B's NA. |
| **D — The rumour** | s06 | 78.1 s | 17 | HEADLINE, LETTER, 1960s street, MOUSE/SYRINGE. Fewest seconds on purpose: it is the most complex and the most sensitive scene, and needs a dedicated sensitivity pass. |

Cross-team dependencies (everything else is self-contained):

- **D needs from C:** GUTCELL + 95% badge (s06.13 callback to s05). **D needs from B:** SHAKER, SALT silhouette, TOMATO.
- **A needs from D:** HEADLINE card (mini tag in s09.04). **A needs from B:** SHAKER, TOMATO, GLU.
- **C needs from B:** NA (s08), SHAKER.

**Fallbacks.** 3 animators: A = s01+s09+s02, B = s03+s04+s08, C = s05+s06+s07 (C then owns GUTCELL and its
callback — lowest risk, but C is ~170 s; give C the first slot on the props file). 2 animators: A = s01, s02, s06,
s09 (all "myth" scenes + HEADLINE) / B = s03, s04, s05, s07, s08 (all "science" scenes).

## 3. Shared props — build once, before parallel work

Because every scene is drawn in code and `lib/kz.js` is frozen, the main continuity risk is four people
re-implementing the same prop four different ways. Mitigation:

1. Use existing kit helpers wherever they exist: `KZ.glutamate` (mascot, `face:true`), `KZ.shaker` ('MSG' / 'SALT'),
   `KZ.person` (Ikeda, volunteers, crowd), `KZ.chapter` (TOKYO, 1908 / 1968 / 1969 / 1960s AMERICA).
2. For the props the kit lacks, the producer creates **one shared props file** (proposal: `scenes/_props.js`, exposing
   a single namespace, e.g. `window.KZP`; `render.js` loads `scenes/*.js` alphabetically, so `_props.js` loads before
   `s01…`). Contents: `tvWall` (3x3 layout + speaker slots + per-screen on/off amount), `dramaTag`, `noMsgSign` (separate
   NO / MSG tube intensities), `gutCell`, `sodium`, `tomato`, `headlineCard`, `badge95`. Each prop has one owner
   (table above); others call it, never copy-paste it. Only the owner edits a prop; changes are announced to all.
   _(Requires producer sign-off since it adds a file to `scenes/`; until then, owners publish their prop function in
   their own scene file inside an IIFE and others wait for it rather than drawing their own.)_
3. Props must be pure functions of their parameters and `t` (no internal state), so callbacks render identically.

## 4. Schedule

| Phase | Who | Output | Gate |
|---|---|---|---|
| 0. Lock | Producer + AD | Shot list frozen; cue times frozen (any narration re-record ⇒ regenerate `cues.json` and re-time shot list first) | — |
| 1. Greybox (can start **before** style approval) | All, in parallel | Each scene blocked with placeholder shapes: shot timings, camera moves, text placement, SFX list in `sfx()` | Producer checks timing/sync with `--contact` + one `--video` |
| 2. Props | Prop owners (A, B, C, D) | Shared props in the approved style, plus a contact sheet of each prop | Director approves prop sheet |
| 3. Animate | All, in parallel | Final scenes; hero shots first (send a still of each hero shot for early review) | QA checklist §5 per scene |
| 4. Continuity pass | A + D + C together | Side-by-side stills: s01.07 vs s09.06, s01.01 vs s09.01/05, s05.05 vs s06.13, s03.08 vs s08.03, s04.11 copies identical | Producer |
| 5. Final | Producer | `bash build.sh` full render, full watch-through with audio | Client |

## 5. QA checklist — run on every finished scene

Commands: `node render.js --contact <id> 12`, `node render.js --stills <id> <hero times>`, `node render.js --video <id> --jobs 1`, `node render.js --sfx`.

**Technical**
- [ ] Scene registers with the exact id from `cues.json`; file is `scenes/<id>.js`; no edits to `lib/`, `render.js`, `player.html`, `script.py`, or other animators' scene files.
- [ ] All helpers wrapped in an IIFE / prefixed — no new top-level globals (grep for top-level `const`/`let`/`function`).
- [ ] Deterministic: no `Math.random`, `Date`, `performance.now`, or state carried between frames; randomness only via `KZ.rng(seed)`. Rendering the same still twice gives byte-identical PNGs (`md5sum`).
- [ ] Renders cleanly from t=0 to t=D with no console errors; frames sampled out of order (e.g. stills at 30, 2, 15) look correct.
- [ ] Fade in from black over ~0.4 s and out over ~0.4 s (`KZ.fade`); `KZ.vignette` applied last.
- [ ] All important content and text inside the 1720x940 safe area.
- [ ] Never a fully static frame: something drifts/breathes in every second of the contact sheet/video.

**Timing & sync (against `design/shotlist.json`)**
- [ ] Every shot boundary within ±0.1 s of the shot list; visual changes every ~3–6 s.
- [ ] Hero beats snapped to the spoken word (±2 frames) — e.g. 95% counter, headline stamp, GRAS stamp, NO burning out.
- [ ] Text keywords appear while (or just before) the matching words are spoken, never after the line ends.

**Sound**
- [ ] `sfx()` returns only names from BRIEF.md; all `t` within 0..D; gains 0..1.
- [ ] Density ~1 per 1–3 s; under narration 0.2–0.6; only hero impacts above 0.7; no two SFX within 0.15 s unless intentionally layered.
- [ ] Every SFX lands on a visible event (pop, stamp, cut, whoosh on camera move).

**Content & facts**
- [ ] On-screen text matches the shot list (spelling, capitalisation); short keywords only, no subtitles.
- [ ] Every number/claim on screen appears in the "Allowed" list of SHOTLIST.md guardrails; nothing from the "Forbidden" list (no extra numbers, no axis values, no "5%", no "1909", no % sodium reduction, no health claims).
- [ ] No real logos, mastheads, seals, brands or recognisable real people.
- [ ] Charts: human-milk vs cow's-milk bars ~10:1 (s04.07); MSG vs placebo bars nearly equal, unlabeled (s07.05); sodium ions exactly 3:1 (s08.03).

**Ethics & sensitivity**
- [ ] DRAMATIZATION tag visible and legible on every frame showing a talking head (s01.01–06, s09.01, s09.05), with generic role labels only.
- [ ] 1960s section and any restaurant/takeout imagery pass the guardrail list (no caricatures, no "oriental" fonts or signifiers, prejudice framed as criticised, headline shown as a quoted historical artifact). For s06: second reviewer signs off.
- [ ] Ikeda / Tokyo 1908 depicted respectfully; mice scene contains no gore.

**Continuity**
- [ ] Recurring props come from the shared implementation (or the owner's function), not redrawn: GLU, SHAKER, SALT, TVWALL, DRAMA_TAG, SIGN, GUTCELL, NA, TOMATO, HEADLINE.
- [ ] TV wall slot positions / on-off order match the continuity bible; NO MSG sign same design and angle in s01 and s09.
- [ ] Scene's first and last frames cut cleanly against neighbours (check final still of previous scene vs first still of this one).

**Style (once approved)**
- [ ] Matches the director's style guide: palette, shapes, type (Fredoka), glow/texture, character construction.
- [ ] Text readable at 1080p (minimum ~28 px for labels, ~22 px cap height for the DRAMATIZATION tag) with sufficient contrast.
