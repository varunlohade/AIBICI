# MSG: The Most Misunderstood Molecule: Style Directions

_Director's proposal for client style approval. Three original directions in the spirit of Kurzgesagt (clear, warm, curious, flat vector, cute characters), each with its own identity. Machine-readable twin: `design/directions.json`._

**Recommendation: Direction 1, Kombu Paper.** Reasons are at the end of this document.

| # | Direction | Pitch |
|---|---|---|
| 1 | **Kombu Paper** | A sunlit kitchen-table world of cut paper and two-colour risograph print: science you could pin on the fridge. |
| 2 | **Afterglow** | Late-night neon line-light: one glowing tube that bends itself into every idea, from TV static to a bowl of ramen. |
| 3 | **Lunchbox Lab** | Bold, outlined, poster-graphic science packed into a modular tray: every idea gets its own compartment, and the lids pop off. |

## How to compare

Each direction is described through the same three key frames, so the client can compare like with like:

- **Frame A: Cold-open TV wall, the fictional Daytime TV Doctor mid-claim**: `s01_coldopen` at t = 7.5s. TVDOC: "MSG is an excitotoxin. It overexcites your brain cells. Literally to death." (5.05-10.10s)
- **Frame B: Gut-lining cells eating glutamate**: `s05_body` at t = 18.5s. NARR: "The cells lining your intestines love glutamate. They burn it as fuel." (16.48-21.08s)
- **Frame C: The 1968 letter + headline stamp**: `s06_origin` at t = 29.0s. NARR: "But the journal printed it under a catchy headline: Chinese Restaurant Syndrome." (26.72-31.45s)

Acts used in the colour scripts: **Cold open** = s01-s02, **Discovery** = s03-s04, **Body science** = s05, **1960s rumour** = s06, **Tests** = s07-s08, **Outro** = s09.

## Rules that apply to every direction

**Fictional cold-open characters**

- All five cold-open speakers are fictional composites. No likeness, name, logo, set, catchphrase or signature look of any real presenter, chef, influencer or show.
- Each speaker gets a lower-third: role label (e.g. 'THE DAYTIME TV DOCTOR') plus a smaller 'DRAMATIZATION - fictional character' tag, on screen for their whole line.
- Faces are stylized or silhouetted. No realistic portraits. Who they are comes from props and setting, not from facial likeness.
- TV channels and networks are invented (e.g. 'CH 5', 'WELLNESS LIVE'). No real network bugs.

**The 1960s section: handle with care**

- No 'chop-suey'/'wonton' faux-Asian lettering, anywhere, ever. The headline uses the direction's own neutral typeface.
- No lanterns, dragons, pagodas, conical hats, chopsticks-as-decoration, gongs or caricatured faces. The takeout carton in the split-screen is a plain, generic white carton with no pagoda print.
- Chinese-American restaurants and diners are not drawn as people at all in s06. The story is about a letter, a headline and the press, so the visuals stay on paper, type and newsprint.
- The phrase 'Chinese-Restaurant Syndrome' is shown as a quoted, period-accurate journal headline (hyphenated, in quotation marks) and is always framed as the thing being critiqued. It is never presented as a brand or a joke.
- The prejudice beat (59-77s) is shown with type and objects: the same alarm bell hovers over one carton and stays silent over tomato sauce and parmesan. No people are shown being othered.
- The letter's author is not named or drawn as a specific person. The journal is named in text only because the narration names it. No imitation of its real masthead or logo; use a generic period letter layout.

**Technical**

- 1920x1080, safe area 1720x940 centered; frames are pure functions of t; textures/patterns pre-generated once at load from a fixed seed (deterministic, never mutated per frame).
- Never a fully static frame: idle 'breathing' on every character, slow camera drift.
- On-screen text: keywords, numbers, labels only. No subtitles. Minimum on-screen text height 40px at 1080p.
- Fade-in/out per scene still applies; each direction specifies its own transition flavour on top.

---

## Direction 1: Kombu Paper

> A sunlit kitchen-table world of cut paper and two-colour risograph print: science you could pin on the fridge.

**Mood:** warm, tactile, homemade, curious, cozy, honest

### How this differs from Kurzgesagt

Kurzgesagt lives in deep navy space with glossy vector shapes, soft glows and gradients. Kombu Paper turns that around: the default ground is light, warm rice paper, and everything looks like it was cut with scissors and printed on a risograph. There are no glows and no gradients. Depth comes from stacked paper layers with hard drop shadows, and texture comes from print grain and slight ink misregistration, not airbrushed light. Animation boils on twos like stop-motion, where Kurzgesagt is silky and smooth. The mascot is a gold broth droplet rather than a ball-and-stick molecule, and the signature transition is a golden ripple, not a starfield zoom. The video should feel made by hand at a kitchen table, not seen through a telescope.

### Palette

| Name | Hex | Role |
|---|---|---|
| Rice Paper | `#F3EAD7` | background (default ground) |
| Kraft | `#E2D3B4` | secondary background / back paper layers |
| Squid Ink | `#2B2330` | ink: face details, type, hard drop shadows (at 22% alpha) |
| Plum Dusk | `#3B3152` | dark ground for cold open + outro night scenes |
| Dashi Gold | `#F2A93B` | highlight / umami / the mascot: reserved for 'the good stuff' |
| Riso Pink | `#FF5C7A` | accent 1 (fluorescent riso): body, warmth, emphasis |
| Riso Blue | `#2F7FC1` | accent 2 (riso): science, lab, data |
| Kombu | `#2F6B4F` | accent 3: seaweed, nature, 'safe' signals |
| Tomato | `#E8492F` | alarm / stamps / the rumour (used sparingly) |
| Skin Light | `#F6CBA6` | skin tone 1 |
| Skin Medium | `#C98B5E` | skin tone 2 |
| Skin Deep | `#7A4A30` | skin tone 3 |

### Colour script

- **Cold open (s01-s02):** Plum Dusk wall with TVs as cardboard cut-outs; each screen is a lit Rice Paper rectangle. Only Riso Pink and Tomato for the accusers, so it reads as lurid. The title (s02) is the first moment Dashi Gold appears: one drop falls and makes the gold ripple. That is the turn from fear to curiosity.
- **Discovery (s03-s04):** Full morning light: Rice Paper + Kraft grounds, Kombu green and Dashi Gold dominate. Tokyo 1908 is told in soft kraft/sepia paper with Kombu accents. The food line-up in s04 uses each food's own colour (Tomato, mushroom kraft-brown, parmesan gold).
- **Body science (s05):** The inside of the body is layered Riso Pink paper in 3 tints (100/60/30%) on Rice Paper. Glutamate is the only gold thing in frame, so the eye tracks it. The blood-brain barrier is Riso Blue.
- **1960s rumour (s06):** The palette drains to yellowed newsprint (#E9DFC4) + Squid Ink, with halftone grain doubled. Tomato is the only colour left, used for the stamp, alarm bells and headline rules. Gold returns only when the narration corrects the record (gut 'eats' it, 49-58s).
- **Tests (s07-s08):** Clean, bright: plain white-ish Rice Paper with a faint Riso Blue graph-paper grid. Riso Blue = placebo, Dashi Gold = MSG, so the near-equal bars read instantly. Salt in s08 is Squid Ink grey dots vs MSG gold.
- **Outro (s09):** Back to the Plum Dusk TV wall, now switching off. Then a warm lamp-lit kitchen: Dashi Gold light pooling on a Kraft table, ramen broth with the gold ripple, pulling back to a paper-cut night sky. Warm, not cold, space.

### Shape language & rendering

- **Outline:** None. Shapes are defined by paper edge + hard drop shadow. The only line work is Squid Ink 'brush' strokes for faces, steam and motion marks (round caps, 5-7px, slightly tapered).
- **Corners:** Soft but hand-cut: 10-28px radii, with edge jitter of +/-1.5px from seeded noise so nothing is perfectly geometric. Circles are slightly egg-shaped.
- **Texture:** Two pre-baked tiles generated once at load: (1) riso grain, 512px monochrome speckle multiplied at 10-14%; (2) paper fibre, faint long strands at 6%. Doubled in s06 for newsprint.
- **Misregistration:** Every coloured shape gets a 2px offset 'under-print' copy in a second riso ink at 35% (multiply). This is our signature print look. Faces/type are never misregistered (legibility).
- **Shadows:** Hard paper shadows only: Squid Ink at 22%, offset (+6,+8)px for near layers, (+3,+4) for far. No blur.
- **Lighting:** Flat. The single allowed light effect is a warm 'desk lamp' radial wash (multiply) in kitchen/outro scenes. No glows anywhere.
- **Depth layers:** 4 multiplane layers: back kraft (parallax 0.3), mid set (0.6), action (1.0), foreground paper scraps/steam (1.3, slightly darker). Camera pushes slide layers at different speeds like a real multiplane.

### Characters

- **Proportions:** Chunky paper dolls: head about 40% of total height, short bean body, stub limbs cut as separate pieces that pivot at visible paper-fastener joints (small Kraft circles). Hands are mitten shapes.
- **Eyes & faces:** Two Squid Ink dot eyes, a short curved brush-stroke mouth, Riso Pink cheek circles (multiply, 40%). No noses, no pupils/whites. Expressions come from eye squash (dots become arcs for joy) and the mouth stroke.
- **Glutamate mascot:** 'Goldie': a Dashi Gold paper droplet (rounded teardrop, about 1:1.2) with dot eyes, pink cheeks and a tiny lighter-gold curl of steam on top. When chemistry matters (s03, s04 equals-sign beat) Goldie stands beside a flat cut-paper skeletal diagram of glutamate (five-carbon zigzag, labelled), so we never imply the molecule is literally a droplet. In the gut, many small Goldies travel as a stream.
- **Cold-open 'celebrities' (fictional):** Shadow-puppet theatre: each fictional speaker is a flat Squid Ink silhouette puppet on a visible wooden stick, held up in front of a backlit paper TV screen. The puppet's jaw is a separate hinged piece that flaps with the voice. Identity comes from props only: TV Doctor = head mirror + stethoscope; Chef = tall toque + ladle; Wellness Influencer = ring-light halo behind, topknot, phone; Talk-Show Host = big hair + desk mug + cue cards; Aunt = round glasses + phone with share icons bursting off it. The puppetry makes the artifice obvious, which is the point. A torn paper lower-third reads the role, plus 'DRAMATIZATION - fictional character'.
- **Other humans:** Ikeda (s03) is the only named real person drawn. He is shown respectfully as a scientist in period dress (high collar, round glasses, moustache from period photographs, simplified), never comedic. Volunteers in s07 use all three skin tones evenly.

### Typography

- **Display:** Young Serif (Regular): Title, scene keywords ('UMAMI', '95%', 'GRAS'), headline in s06. Source: `https://raw.githubusercontent.com/google/fonts/main/ofl/youngserif/YoungSerif-Regular.ttf`
- **Label:** Nunito (ExtraBold, Bold): Labels, lower-thirds, chart axes, numbers. Source: `assets/Nunito.ttf (local)`
- **Document:** Courier Prime (Regular, Bold): The 1968 letter, lab sheets. Source: `https://raw.githubusercontent.com/google/fonts/main/ofl/courierprime/CourierPrime-Regular.ttf`
- **Sizes (px @1080p):** title 150, keyword 104, big number 180, label 46, lower third role 40, dramatization tag 28, letter body 30
- **Rules:** Text is always printed on something: a torn paper slip, a label-tape strip, a stamp, a page. Never floating type. Squid Ink on paper by default; Dashi Gold only for the umami/correct answer. Max 4 words per on-screen phrase. Sentence case for labels, ALL CAPS only for stamps.

### Motion language

- **Easing:** Placed by hand: pieces arrive with a quick ease-out and a small overshoot (back-out, 1.3), then a 'paper settle' squash (scaleY 0.94 to 1 over 4 frames) as if pressed onto the table.
- **Frame rate feel:** Characters and props 'boil' on twos: edge jitter and tiny positional wobble update at 15fps from floor(t*15) seeds, while camera moves stay smooth at 30fps. This is the stop-motion character of the style.
- **Camera:** Multiplane pushes and lateral trucks only; no rotation, no zoom-blur. Moves are slow (3-8% scale over a line) with parallax doing the work.
- **Transitions:** Default: a sheet of paper slides over (whoosh) with its shadow leading. Act changes: the Gold Ripple wipe, a drop lands centre-frame and a gold ring expands to reveal the next scene inside it.
- **Signature move:** The Gold Ripple: a single drop, then 5 concentric rings; the fifth is gold and wider than the others.

### Recurring motif: The Fifth Ripple

A drop hitting broth sends out five rings; four are paper-coloured (sweet, sour, salty, bitter) and the fifth is Dashi Gold (umami). It recurs as: the title reveal (s02); Ikeda's four taste icons riding rings 1-4 and the mystery fifth (s03); a small gold ripple each time glutamate docks on a tongue receptor (s05); a dark Tomato 'anti-ripple' of newspapers spreading outward from the stamp (s06, the rumour spreads like the same physics); a flattened ripple becoming the placebo chart baseline (s07); and the ramen bowl's surface on the final shot (s09).

### Canvas 2D feasibility

- **Cheap:** Flat fills + offset shadow (draw the path twice); Pre-baked grain/fibre patterns as CanvasPattern with multiply; Boil-on-twos jitter (seeded by floor(t*15)); Ripples (stroked circles); Paper-slip text plates.
- **Expensive / watch:** Per-vertex edge jitter on very complex paths: limit to silhouettes, max ~80 vertices/shape; Misregistration doubles every draw call: apply only to hero shapes, not background crowds; Full-frame multiply passes: max 2 per frame (grain + lamp).
- **Approach:** Build a small helper `paperShape(path, seed)` in each scene file that caches the jittered vertex list per seed+boilFrame. Grain tiles generated once from KZ.rng and reused; no per-frame noise generation.
- **Style risk:** Light grounds can look 'flat'. Solved by the Plum Dusk cold open/outro bookends and strong multiplane parallax.

### Key frames

**Frame A: Cold-open TV wall, the fictional Daytime TV Doctor mid-claim** (`s01_coldopen` @ 7.5s)

Plum Dusk wall, a 3x3 grid of chunky cardboard TV cut-outs (Kraft bodies, rounded 24px, slightly mismatched sizes, hard shadows) with gentle multiplane depth. 8 screens are dim grey-plum paper; the centre-left screen is lit warm Rice Paper. On it: the TV Doctor shadow puppet in Squid Ink silhouette, head mirror and stethoscope cut out, stick visible from the bottom edge, jaw piece mid-flap. Behind the puppet a simple cut-paper brain with jagged Riso Pink 'zap' shapes (overexcited) printed with misregistration. Torn Rice Paper lower-third across the screen: 'THE DAYTIME TV DOCTOR' (Nunito ExtraBold 40px) + 'DRAMATIZATION - fictional character' (28px, 70% ink). Faint channel tag 'CH 5' in the corner. Grain at 14%, boil visible on the puppet edges.

**Frame B: Gut-lining cells eating glutamate** (`s05_body` @ 18.5s)

Cutaway of the intestine wall as a big curving landscape of Riso Pink paper layers (3 tints), the gut lining shown as a row of 6-7 round, rosy cell characters (pink paper blobs with dot eyes and cheeks, bibs of lighter pink) standing shoulder-to-shoulder like a lunch counter. A stream of small Dashi Gold Goldies drifts in from the left on a gently curving path. The nearest cell has its mouth open mid-chomp with one Goldie going in, eyes squeezed to happy arcs; the next one has a tiny gold flame/sparkle cut-out above its head (burning fuel). Paper-slip label top-left: 'Gut cells: fuel!' Kraft background behind, foreground pink scraps blurred-free but darker. Shadows hard, lamp wash off.

**Frame C: The 1968 letter + headline stamp** (`s06_origin` @ 29.0s)

Overhead desk view, drained newsprint palette. A Courier Prime typewritten letter on yellowed paper fills centre-left at a slight -4 degree angle: '1968' date line, 'To the Editor:' then greeked lines (only legible fragments: 'numbness', 'weakness', 'palpitations'). Pinned to its corner, three small Squid Ink question-mark cut-outs over a wine bottle, a salt cellar and a shaker (left over from the 19-26s beat). The stamp has just hit: a rectangular Tomato rubber-stamp impression, rotated -8 degrees, rough ink texture with missing patches, Young Serif all caps: '"CHINESE-RESTAURANT SYNDROME"' in quotes. A flat Squid Ink rubber stamp handle lifts out of the top of frame. Small paper flecks fly from the impact. Tiny newspaper sheets are beginning to slide in at the edges (next beat). Nothing else, no restaurant imagery.

---

## Direction 2: Afterglow

> Late-night neon line-light: one glowing tube that bends itself into every idea, from TV static to a bowl of ramen.

**Mood:** nocturnal, witty, electric, sleek, a little noir, warm-hearted

### How this differs from Kurzgesagt

Kurzgesagt fills shapes; Afterglow draws with lines. Almost everything is a single-weight luminous stroke (a neon tube or a CRT phosphor trace) on a warm soy-black ground, not a filled vector on navy. There are no gradient skies, starfields or blobby planets. Colour is emitted rather than painted, and fills are rare tinted glass at 15-25%. The mascot is drawn in the chemist's own notation (the skeletal zigzag formula), not ball-and-stick. Transitions draw on and burn out like real neon tubes instead of zooming. It's a late-night show about a late-night-TV panic, with the 'NO MSG' sign as the main visual idea.

### Palette

| Name | Hex | Role |
|---|---|---|
| Soy | `#17100F` | background (warm near-black) |
| Aubergine | `#2A1730` | secondary background / far depth layer / walls |
| Chalk | `#F5EDE3` | ink: type, neutral line work, unlit tube glass |
| Tube Core | `#FFF6E0` | highlight: hot centre line of every lit tube |
| Umami Amber | `#FFB547` | signature accent / glutamate / the good stuff |
| Kelp Cyan | `#3FE0C5` | accent: science, nature, the body |
| Fluoro Pink | `#FF6FB5` | accent: organs, warmth, characters |
| Static Violet | `#9C7BFF` | accent: TV / media / brain |
| Alarm Red | `#FF4F4F` | rumour, stamps, the 'NO' of the sign |
| Skin Light | `#F7C6A3` | skin tone 1 (tube colour; 20% glass fill) |
| Skin Medium | `#C68A62` | skin tone 2 |
| Skin Deep | `#8A5236` | skin tone 3 (tube drawn one step lighter, #A86A48, for legibility on Soy) |

### Colour script

- **Cold open (s01-s02):** Overstimulated: Static Violet, Fluoro Pink and Alarm Red CRTs, scanlines, channel-switch tears. Ends on the red 'NO MSG' sign buzzing on. Title (s02): everything dies to black except one Umami Amber zigzag that strikes on.
- **Discovery (s03-s04):** Amber and Kelp Cyan only, slower flicker, like a quiet lab at night. Tokyo 1908 is drawn as a single-line amber sketch: Ikeda, steaming bowl, a lab burner. Foods in s04 each get their own tube colour, all sharing the amber glutamate zigzag inside.
- **Body science (s05):** Fluoro Pink anatomy line drawing over Aubergine; Kelp Cyan for receptors and the barrier; glutamate the only Amber. Clean, almost medical-illustration linework.
- **1960s rumour (s06):** Tubes go out. Monochrome amber-phosphor CRT look (like a 1960s terminal/newsreel): Chalk at 80% on Soy, heavy scanlines, slight barrel vignette. Alarm Red is the only other colour (stamp, alarm bells).
- **Tests (s07-s08):** Cool Kelp Cyan lab, steady and flicker-free (science is calm and unflashy). Amber = MSG, Chalk = placebo bars.
- **Outro (s09):** The TV wall powers off tube by tube; the 'NO MSG' sign: the 'NO' sputters and dies, leaving a warm amber 'MSG'. Final ramen bowl in amber + pink, steam as rising tube curls; pull back to city lights then stars as dots of tube light.

### Shape language & rendering

- **Outline:** Line IS the shape. One stroke weight family: 8px hero tubes, 5px secondary, 3px detail. Round caps and joins always. Tubes may have small visible 'gaps' where a real neon tube would turn (a 6px break at sharp corners), which is the craft detail that sells it.
- **Corners:** Bends, not corners: minimum bend radius 14px (a tube can't make a sharp angle). Exceptions: the glutamate zigzag (deliberate 120-degree chemistry angles, softened to 6px radius).
- **Texture:** CRT scanlines (2px period, 8% darken) as a cached pattern over TV/1960s scenes only; elsewhere a very faint brick/wall texture on Aubergine walls behind signs. No paper grain.
- **Glow:** Faked cheaply: each tube drawn as 3 strokes: wide (24px, 10% alpha, colour), mid (12px, 30%), core (3px, Tube Core). Composite 'lighter' for overlaps. shadowBlur reserved for at most one hero element per frame.
- **Fills:** Rare: tinted 'glass' fills at 15-25% of the tube colour for faces/bodies/organs that need mass.
- **Lighting:** Lit tubes cast a soft coloured pool on the wall behind them (one big radial, low alpha), so the world is lit by its own drawings.
- **Depth layers:** 3 layers: Aubergine wall (dim, reflections), mid tubes, foreground 'unlit' tubes in Chalk at 25% that pass near camera and defocus via a wider, dimmer stroke.

### Characters

- **Proportions:** Single-line figures: big rounded head (circle), pill body, noodle arms. Head about 35% of height. Drawn as if bent from one or two tubes.
- **Eyes & faces:** Two short vertical dash eyes (tube segments) that compress to dots when blinking, an optional curved mouth line. Blush as a tiny pink tube dash.
- **Glutamate mascot:** 'Zig': glutamate's real skeletal formula (the five-carbon zigzag with the COOH ends and the NH2 branch, simplified to its silhouette) bent from one Umami Amber tube, with two dash eyes on the middle vertex. It walks on the ends of the zigzag like legs. It is chemically recognisable, which is what earns it the 'equals' beat in s04: the tomato-Zig and shaker-Zig are literally the same line.
- **Cold-open 'celebrities' (fictional):** Each fictional speaker is a one-line neon portrait inside a curved CRT bezel, faces deliberately abstract: head outline + one identifying prop line (stethoscope loop; toque; ring-light circle; big hair swoop + desk; glasses + phone). No eyes, so no likeness; only a mouth line that flickers open/closed in sync. Each speaker has their own colour. Scanlines and a tiny chromatic split sell 'broadcast'. Lower-third in Space Mono: role + 'DRAMATIZATION - fictional character'.
- **Other humans:** Ikeda drawn in respectful single-line amber portrait style (period collar, round glasses), no flicker on him. Volunteers in s07 use skin-tone glass fills for diversity in an otherwise line-only world.

### Typography

- **Display:** Righteous (Regular): Title, keywords and signs, rendered as tubes (stroked text, triple-stroke glow). Source: `https://raw.githubusercontent.com/google/fonts/main/ofl/righteous/Righteous-Regular.ttf`
- **Label:** Nunito (Bold, ExtraBold): Labels on charts, body text that must be very legible (filled, not stroked). Source: `assets/Nunito.ttf (local)`
- **Mono:** Space Mono (Regular, Bold): Lower-thirds, numbers, data, lab labels. Source: `https://raw.githubusercontent.com/google/fonts/main/ofl/spacemono/SpaceMono-Bold.ttf`
- **Document:** Courier Prime (Regular, Bold): 1968 letter (amber phosphor). Source: `https://raw.githubusercontent.com/google/fonts/main/ofl/courierprime/CourierPrime-Regular.ttf`
- **Sizes (px @1080p):** title 170, keyword 110, big number 200, label 44, lower third role 38, dramatization tag 26, letter body 32
- **Rules:** Big words are neon signs: they strike on with the flicker pattern and buzz faintly. Small text is filled Chalk, never glowing (legibility). Max 3 words per sign. No more than two glowing words on screen at once.

### Motion language

- **Easing:** Snappy in, long decay out: positions use expo-out (fast arrival, long settle); light uses a deterministic 'strike' flicker (on-off-on-off-ON over 0.25s from a seeded pattern), then a slow 1-2% brightness hum.
- **Camera:** Smooth slow dolly/drift; occasional whip-pan between ideas with a fake motion smear (draw 3 offset copies at 15/25/40% alpha). Mild perspective tilt on the TV wall only.
- **Transitions:** Line-draw: the outgoing drawing un-draws (setLineDash offset shrinking back along the path) into a single moving spark, which travels across frame and draws the next scene. Act changes: hard 'power cut' to black for 4 frames with tv_off/neon_flicker.
- **Signature move:** The One-Line Morph: the amber zigzag stretches and re-bends into the next object (zigzag to kombu frond to tongue receptor to heartbeat line to chart line), keeping one continuous stroke on screen across cuts.

### Recurring motif: The Glutamate Line

Glutamate's own zigzag is the thread through the video. It is the title, the kombu strand in Ikeda's bowl, the line inside every food in s04, the stream down the gut in s05, the jagged fever-line in the newspapers in s06 (same shape turned sinister, in red), the flat calm trace of the placebo chart in s07, and the steam over the ramen in s09. It is book-ended by the 'NO MSG' sign: buzzing on in s01, losing its 'NO' in s09.

### Canvas 2D feasibility

- **Cheap:** Stroked paths (3 strokes per tube); Line-draw-on via setLineDash/lineDashOffset with path lengths precomputed; Flicker = alpha multiply from seeded table; Scanline cached pattern; Dark ground hides compositing seams.
- **Expensive / watch:** shadowBlur on many strokes (avoid; use triple stroke); Measuring path lengths of complex bezier paths every frame: precompute at load via sampled polylines; 'lighter' composite on full-frame overlaps: restrict to tube layer; One-Line Morph needs both shapes resampled to the same point count (do once at load).
- **Approach:** Represent every drawable as a pre-sampled polyline (e.g. 200 points) so draw-on, morph and glow all use one cheap code path. Keep tube count under ~60 per frame.
- **Style risk:** Dark-background-with-glow is the trait people most associate with Kurzgesagt. Afterglow avoids it through line-only drawing and the warm soy ground, but of the three it is the closest in general feel. Line-only art can also go cold or abstract in the body-science section if the glass fills are skipped.

### Key frames

**Frame A: Cold-open TV wall, the fictional Daytime TV Doctor mid-claim** (`s01_coldopen` @ 7.5s)

A slightly curved 3x3 wall of CRT bezels drawn as Chalk outline tubes (25% dim) on an Aubergine brick wall, faint scanlines everywhere. Eight screens show violet/pink static noise (cached pattern, scrolling). The centre screen is lit Static Violet: a one-line neon portrait of the TV Doctor: head outline, stethoscope loop in Kelp Cyan, no eyes, mouth line open. Beside the head, a small Fluoro Pink neuron sign with crackling Alarm Red zig-zag sparks (overexcited). Chromatic split of 2px on the screen. Lower-third in Space Mono across the bottom of the screen: 'THE DAYTIME TV DOCTOR' / 'DRAMATIZATION - fictional character'. Corner bug 'CH 5'. Coloured light pool from the lit screen on the wall around it.

**Frame B: Gut-lining cells eating glutamate** (`s05_body` @ 18.5s)

A Fluoro Pink line cross-section of the intestine running diagonally lower-left to upper-right, with finger-like villi drawn as a wavy tube; each villus tip is a round cell character (pink glass fill 20%, dash eyes). A stream of small Amber Zigs flows along the gut channel. Front cell: mouth line open, swallowing a Zig, which breaks into amber sparks inside it, and a little Amber flame tube flickers above its head ('fuel'). Neon sign top-right: 'FUEL' in Righteous, Kelp Cyan. Aubergine background with soft pink light pool behind the gut. Nothing else glows.

**Frame C: The 1968 letter + headline stamp** (`s06_origin` @ 29.0s)

Amber-phosphor CRT world: a typewritten letter (Courier Prime, amber Chalk at 80%) on screen with scanlines and barrel vignette; date line '1968', 'To the Editor:', greeked text with 'numbness... weakness... palpitations' legible. The question marks (from the previous beat) hang as three small amber tubes above sketches of a wine bottle, salt, and a shaker in the margin. The headline has just slammed: an Alarm Red stroked rectangle stamp rotated -6 degrees, Righteous all caps '"CHINESE-RESTAURANT SYNDROME"' in quotes, with a hard red flash (one frame at 100%, decaying) and a glitch tear across the CRT. No restaurant imagery, no faux-Asian type.

---

## Direction 3: Lunchbox Lab

> Bold, outlined, poster-graphic science packed into a modular tray: every idea gets its own compartment, and the lids pop off.

**Mood:** bright, punchy, playful, organised, graphic, confident

### How this differs from Kurzgesagt

Lunchbox Lab is the most graphic-design-led of the three. It uses thick ink outlines, which Kurzgesagt avoids, plus flat poster colours on bright solid grounds and halftone-dot shading instead of gradients and glows. The frame is often split into a grid of rounded compartments like a lunch tray or lab sample tray, so several ideas sit side by side. That suits the comparisons in this script: tomato vs shaker, MSG vs placebo, salt vs MSG. Characters have a rubbery, bouncy, cartoon-print energy rather than Kurzgesagt's gentle bob. Bold editorial type is part of the picture, where Kurzgesagt keeps text small and rounded. It's closer to a museum-exhibit poster or a science-magazine spread brought to life than to a space documentary.

### Palette

| Name | Hex | Role |
|---|---|---|
| Cream Tray | `#FFF4E2` | background (default) / compartment floors |
| Liquorice | `#1E1A1D` | ink: all outlines (6px), type, halftone dots |
| Butter | `#FFD95A` | highlight / umami / glutamate mascot |
| Tomato | `#FF5A3C` | accent: alarm, rumour, stamps |
| Pool | `#4FB3E8` | accent: science, lab, placebo |
| Pea | `#7BC96F` | accent: nature, 'safe', kombu |
| Bubblegum | `#FF9EC4` | accent: body, organs, warmth |
| Lilac | `#B9A4F2` | accent: media, brain, TV |
| Tray Grey | `#D8D2C8` | tray walls / dividers / inactive compartments |
| Skin Light | `#FFD1B0` | skin tone 1 |
| Skin Medium | `#D99A6C` | skin tone 2 |
| Skin Deep | `#8C5636` | skin tone 3 |

### Colour script

- **Cold open (s01-s02):** A clashing TV-guide grid: every compartment a different saturated colour (Lilac, Tomato, Bubblegum, Pool) for overload. The lids slam shut one by one at 39-46s. Title (s02): a single Butter compartment opens with the mascot inside.
- **Discovery (s03-s04):** Butter + Pea dominant, warm and sunny. 1908 Tokyo in a 2-colour print treatment (Pea + Liquorice on Cream) to signal history. s04 foods each fill a compartment of a tray in their own colours.
- **Body science (s05):** A Bubblegum body cutaway as one big compartment, Pool for receptors/barrier, Butter for glutamate. Halftone shading heavier for 'inside' depth.
- **1960s rumour (s06):** Colour drains to a newsprint print: Cream + Liquorice only, coarse halftone, with Tomato as the single spot colour (stamp, alarm bells). Compartments become newspaper columns.
- **Tests (s07-s08):** Pool + white: a clean lab tray, the A/B cups in two compartments, a crisp bar chart. Butter = MSG, Pool = placebo. s08 balance scale in Butter vs Tray Grey (salt).
- **Outro (s09):** The tray rebuilds in full colour; the 'NO MSG' sign is a Tomato-outlined compartment whose 'NO' tile pops out. Final: a ramen bowl in the centre compartment, then the tray zooms out to become one of many trays/windows, ending on a Butter end card.

### Shape language & rendering

- **Outline:** Yes, as the signature: Liquorice outline on everything, 6px for heroes, 4px secondary, 3px detail; round joins. Outlines sit slightly off the fill (fill offset 3px down-right inside the line) for a printed, off-register feel.
- **Corners:** Generous and consistent: 24px on compartments, 16px on props, full pill-rounding on characters. Geometric construction (circles, capsules, rounded rects) with no hand-jitter.
- **Texture:** Halftone dots as the only shading: a cached Liquorice dot pattern (8px grid) clipped into shadow shapes at 18%. A coarser 14px dot for the 1960s section. No grain.
- **Shadows:** Shading via halftone on the shadow side + a solid Liquorice 'stacked' drop shadow (offset 0,8px, 100% ink) under compartments and stickers, like a sticker on a page. No blur.
- **Lighting:** Flat; light direction fixed top-left for all halftone shading.
- **Depth layers:** Shallow and graphic: tray (background) > compartments > contents > stickers/labels on top. Depth is shown by overlap and the hard ink shadow, not by parallax. Camera mostly flat-on.
- **Grid:** Compartment layouts on a 12-column grid with 24px gutters inside the safe area. Common splits: 2-up (comparison), 3-up, 2x2, 1 big + 2 small, full-frame single.

### Characters

- **Proportions:** Rubbery and chunky: head about 45% of height, pear body, noodle limbs with no elbows (rubber-hose arms), oversized hands with 3 fingers + thumb. Big feet for bounce.
- **Eyes & faces:** Large vertical-oval eyes (Liquorice, with one white sparkle each), eyebrows as short thick dashes that do most of the acting, small open mouth shapes. Cheeks as halftone pink patches.
- **Glutamate mascot:** 'Umi': a chubby Butter rod-crystal (MSG crystals really are prism/rod shaped): a rounded hexagonal capsule with a white facet highlight stripe, big oval eyes, rubber-hose arms. In molecule contexts Umi holds up a clean outlined skeletal-formula card. Crowds of glutamate in the body are mini Umis (no arms, eyes only) so they stay cheap.
- **Cold-open 'celebrities' (fictional):** Pictogram archetypes in a TV-guide grid: each is a bold outlined figure whose face is a blank flat oval (no eyes, no features, only a mouth shape that changes), so they read as types, not people. Prop-based identity: TV Doctor (white coat, stethoscope, clipboard), Chef (toque, ladle, crossed arms), Influencer (ring light, phone on selfie stick), Host (desk, mug, cue cards), Aunt (cardigan, glasses, phone with hearts/share sticker). A sticker-style lower-third overlaps each compartment: role name + 'DRAMATIZATION - fictional character'.
- **Other humans:** Ikeda shown in the 2-colour history print style, respectful, no rubber-hose comedy on him (straight limbs, calm pose). Volunteers in s07 use all three skin tones and varied hair silhouettes.

### Typography

- **Display:** Dela Gothic One (Regular): Title, keywords, big numbers, stamps (heavy poster weight). Source: `https://raw.githubusercontent.com/google/fonts/main/ofl/delagothicone/DelaGothicOne-Regular.ttf`
- **Label:** Atkinson Hyperlegible (Bold, Regular): Labels, lower-thirds, chart axes: maximally legible. Source: `https://raw.githubusercontent.com/google/fonts/main/ofl/atkinsonhyperlegible/AtkinsonHyperlegible-Bold.ttf`
- **Mono:** Space Mono (Bold): Lab codes ('SAMPLE A'), data, dates. Source: `https://raw.githubusercontent.com/google/fonts/main/ofl/spacemono/SpaceMono-Bold.ttf`
- **Document:** Courier Prime (Regular, Bold): 1968 letter. Source: `https://raw.githubusercontent.com/google/fonts/main/ofl/courierprime/CourierPrime-Regular.ttf`
- **Sizes (px @1080p):** title 160, keyword 120, big number 240, label 44, lower third role 40, dramatization tag 28, letter body 30
- **Rules:** Type is a graphic element: big keywords can fill a compartment edge to edge. Text sits on outlined sticker plates with the hard ink shadow. ALL CAPS for display, sentence case for labels. Max 3 words in display, 6 in labels.

### Motion language

- **Easing:** Bouncy: elastic-out for arrivals (one visible overshoot + one smaller), squash-and-stretch on landing (scale 1.15x0.85 to 1 over 6 frames). Characters idle with a 2-beat bounce.
- **Camera:** Mostly locked-off and flat-on, as in print. Moves are graphic: the whole tray slides left/right to the next compartment (a 'tray slide'), or the camera punches in to fill frame with one compartment.
- **Transitions:** The Lid Pop: compartments are covered by Tray Grey lids that pop off (boing, 0.3s, flip up out of frame) to reveal content; closing a scene = lids slam back down in a staggered cascade (pop_small per lid). Act changes: the full frame becomes one lid that flips over.
- **Signature move:** Lid Pop + sticker slap: labels arrive as stickers that slap down with a 3-frame squash and a tiny rotation settle (+-3 degrees).

### Recurring motif: The Five-Compartment Tray

A five-section tray holds the five tastes: four filled (sweet, sour, salty, bitter) and an empty fifth that Ikeda fills with Butter (s03). It returns as the tongue's receptors in s05 (the tray maps onto the tongue), as a sinister one-compartment newspaper column in s06 where the fear gets put in one cuisine's box (visual rhyme with 'the name said it out loud'), as the A/B test tray in s07, and fully lit for the outro bowl in s09. The video's argument is about what gets put in which box.

### Canvas 2D feasibility

- **Cheap:** Fill + stroke primitives (rounded rects, circles, capsules); Hard offset shadows; Halftone via cached CanvasPattern + clip; Lid pops (rect transforms); Grid layout math.
- **Expensive / watch:** Clip + pattern for halftone on many small shapes: batch shadow shapes into one path per layer and clip once; Rubber-hose limbs as beziers: fine, but avoid per-frame path-length math; Thick outlines on text (strokeText at 6px) can look lumpy at small sizes: only on display type 100px+.
- **Approach:** Use a per-scene `tray(ctx, layout, t)` helper for compartments; construct characters from 8-12 primitives each. Halftone pattern built once.
- **Style risk:** Outlines + bright grounds can feel more kids' TV than curious-adult science. Kept in check by the restrained Cream ground, generous whitespace and the serious 2-colour print treatment for history sections.

### Key frames

**Frame A: Cold-open TV wall, the fictional Daytime TV Doctor mid-claim** (`s01_coldopen` @ 7.5s)

Full-frame TV-guide grid of 6 rounded compartments (2 rows x 3) on a Liquorice tray, each a different flat colour with a Tray Grey lid half-off or closed. The large centre-left compartment (spanning 2 cells) is open and Lilac: inside, a chunky outlined TV set with the TV Doctor pictogram: white coat, Pool-blue stethoscope, clipboard, blank flat face with an open mouth shape, one rubber-hose finger raised. Behind him on the TV, an outlined cartoon brain with Tomato zig-zag 'zap' bolts and halftone shading. A Butter sticker lower-third slaps over the compartment corner: 'THE DAYTIME TV DOCTOR' (Atkinson Bold 40px) + 'DRAMATIZATION - fictional character' (28px). Other compartments show closed lids with channel numbers in Space Mono ('CH 2', 'CH 9').

**Frame B: Gut-lining cells eating glutamate** (`s05_body` @ 18.5s)

One big Bubblegum compartment containing a cutaway of the gut wall: a row of 6 round, outlined intestinal-cell characters (Bubblegum fills, halftone on lower right, big oval eyes, dash eyebrows raised with delight) along a wavy wall, mouths wide. Mini Butter Umis (eyes only) flow in from the left in a tidy queue. The front cell is mid-chomp with a squash pose (1.15x0.85) and a Butter 'spark' sticker above it. A Pool sticker label at top-left reads 'FUEL!' in Dela Gothic One. Small inset compartment (top-right, 1/6 size) shows the tongue tray from the previous beat, lid closing, for continuity.

**Frame C: The 1968 letter + headline stamp** (`s06_origin` @ 29.0s)

Newsprint treatment: the tray becomes a newspaper layout of compartments/columns in Cream + Liquorice with coarse 14px halftone. Centre: the typewritten Courier Prime letter on an outlined page sticker, '1968' and 'To the Editor:' legible, key words 'numbness', 'weakness', 'palpitations' underlined in Liquorice. Left column: three small outlined icons, a wine bottle, a salt cellar and a shaker, each topped with a Dela Gothic '?' sticker. The stamp: a chunky Tomato rubber stamp mid-rebound above frame, leaving a squashed, slightly rotated (-7 degrees) Tomato stamp plate with ink-splat edges: '"CHINESE-RESTAURANT SYNDROME"' in Dela Gothic One, quotes included. Impact lines (thick Liquorice dashes) around the stamp. No restaurant or cultural imagery.

---

## Recommendation

**Go with Kombu Paper.** Runner-up: Lunchbox Lab.

- It keeps what the client likes about Kurzgesagt (clear, warm, curious, cute flat characters) while flipping the most recognisable traits: light paper instead of navy space, hard paper shadows instead of glows, stop-motion boil instead of silky motion. Nobody will mistake it for a copy.
- It suits the subject. This is a story about kitchens, seaweed soup, a letter and newspapers. Paper, print and a gold broth ripple are made of the same stuff as the story, so the 1908 history, the 1968 letter and the newsprint montage need no change of art style.
- The Fifth Ripple motif carries the argument: umami is the fifth ring, and later the rumour spreads with the same physics in red. Viewers get the thesis visually.
- It handles the sensitive 1960s section with restraint. Draining to newsprint with one red spot colour keeps the focus on the letter, the headline and the press, not on any culture.
- It is very feasible in canvas 2D: flat fills, offset shadows and two cached textures. The 'boil on twos' adds a lot of charm and costs almost nothing to render.

**Watch-outs:** Keep the Plum Dusk bookends so the piece still has cinematic contrast, and keep grain restrained (10-14%) so it reads as crafted rather than dirty when compressed for streaming.

**If not Kombu Paper:** Choose Lunchbox Lab if the client wants maximum punch and clarity for social cut-downs (its compartments also reframe well to 9:16). Choose Afterglow if they want a bolder, late-night editorial tone and accept being closer to the dark-background family.

**Next step after approval:** render Frames A, B and C for the chosen direction (or all three, for a side-by-side) as single stills, lock the palette and type into a style sheet, and only then start animating.
