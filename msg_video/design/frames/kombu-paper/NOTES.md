# Kombu Paper: style-frame notes

- **Client rules (these override DIRECTIONS.md):**
  - *Vertical 9:16 Reel (1080x1920).* The frames are stacked top/middle/bottom, not cropped. Key content stays inside y 220–1300 and x < 940. The caption band is at y ~1320–1530. The bottom ~440px is decor only: a sideboard and sofa in A, gut layers in B, a sliding newspaper in C.
  - *Little text.* Besides the caption, each frame has at most one label of 1–3 words: A "TV DOCTOR" (plus the required DRAMATIZATION tag), B "Fuel!", C the stamped headline. The 1968 letter is now squiggle lines only: no date, no salutation, no legible symptom words. The newspapers are squiggle columns. "CH 5" is gone.
  - *Vivid colour only.* Rice Paper, Kraft, Plum Dusk and newsprint are replaced by bright riso stock:
    - indigo-violet wall and grape TVs, with a turquoise hero set on a lemon screen (A);
    - a turquoise background behind hot-pink, bubblegum, magenta and grape layers of gut tissue (B);
    - a cobalt desk, lemon letter, hot-pink underlay, turquoise card and coral newspapers (C).

    Ink is a saturated violet-black (#1E1033). Gold (#FFB000) is still reserved for glutamate: Goldie, the flame, the gold caption chip. Grain and print texture stay but are lighter (10–12%) so nothing reads faded.
- **Caption style (part of the pitch):** a torn white paper strip set in Nunito Black at 64px. Words already spoken are ink and upcoming words are lavender. The current word sits on a raised chip with a hard shadow and a misregistered under-print. The chip colour follows the act: hot pink for the cold open, gold on "glutamate", tomato red with white text for the rumour. In motion, the chip should hop from word to word using the "paper settle" squash.
- **What's built:** `frames.js` exposes `window.KP`, a reusable kit for the animator. It includes seeded hand-cut edge jitter, cut paper with a hard shadow and riso misregistration, tapered brush strokes, squiggle "writing", and torn slips and captions. It also has Goldie, cell characters, mitten arms with paper fasteners, cardboard TVs, the TV Doctor shadow puppet, the Fifth Ripple, the distressed stamp, and grain/fibre/halftone tiles baked once. Everything is deterministic: there is no `Math.random`, and boil runs on twos from `floor(t*15)`.
- **Other deviations:**
  - A has 8 dim sets plus a hero that spans two columns, to fill the tall frame.
  - Shadows on dark or vivid grounds are 30–50% ink, because 22% disappears on saturated colour.
  - B's 6 cells sit on two tiers (3 behind, plus the eater and the "burning" cell in front), because a single row is illegible in 9:16.
  - The stamp text breaks over two lines and is printed opaque. The squiggles leave a clear patch where the stamp lands so the headline reads.
- **Guardrails:** The TV Doctor is a generic puppet in profile with a head mirror and stethoscope, and carries the DRAMATIZATION tag. Frame C shows only paper, a bottle, a salt cellar, a shaker and type. The headline is in Young Serif, hyphenated and in quotes, stamped as the thing being criticised. No author is named.
- **Open decision 1: vivid level vs. "kitchen-table warmth".** The rule pushed this direction away from its sunlit cream paper. To get some warmth back, we could use saffron or coral as the default "paper" (for example, a saffron desk in C instead of cobalt) and keep cobalt and violet for night and media scenes.
- **Open decision 2: caption chip colour.** It currently changes by act. The alternative is always gold, which is a stronger signature but weakens the rule that gold means "the good stuff".
- **Open decision 3: the mascot and the puppet.** Goldie could get stub mitten arms for carrying things in s04 and s07. The TV Doctor's mouth is a wedge cut; a separately hinged jaw piece that drops is the alternative if the wedge reads too "Pac-Man". The lavender for upcoming caption words could also be pushed darker if Instagram compression hurts it.
