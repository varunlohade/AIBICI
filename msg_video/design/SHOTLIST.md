# MSG — Shot List / Storyboard

_Assistant-director shot list for "MSG: The Most Misunderstood Molecule". Machine-readable twin: `design/shotlist.json`
(generated from the same data — keep them in sync; edit the JSON first if anything changes)._

**How to read this**

- All times are **scene-relative seconds** (same clock as `draw(ctx, t, ...)` and `sfx(cues, D)`), taken from
  `cues.json`. Shots tile each scene from 0 to its duration with no gaps.
- Each shot starts on a narration line start (or on an estimated word boundary inside a long line). Times marked
  "estimate" in notes, and any SFX inside a line, must be **snapped to the narration WAV** (`audio/narration/`)
  when animating — the shot boundaries at line starts are exact.
- **Style-agnostic**: this document fixes content, framing, motion, timing and sound only. Palette, rendering,
  textures and character styling follow the director's approved style guide. Words like "warning-colored" or
  "calm" refer to roles in that palette, not specific colors.
- Shot sizes: WIDE / MEDIUM / CLOSE refer to subject size in the 1920x1080 frame; keep all key content inside the
  1720x940 safe area. "Locked" camera still means gentle drift/breathing — never a fully static frame.
- Every scene fades in from black (~0.4 s) and out to black (~0.4 s); the first/last shot transitions reflect that.
- SFX format: `time name gain` — names from BRIEF.md only; gains 0.2–0.6 under narration, up to 0.8 on hero impacts.
- `L#` = narration line number within the scene (1-based, as in `cues.json`); "partial" = the shot covers only part
  of that line.
- ★ = hero moment (see summary table at the end).

**Totals:** 9 scenes · 82 shots · 17 hero moments · 359.4 s (~5 min 59 s).

## s01_coldopen — Cold Open: The Accusations  ·  47.709 s  ·  10 shots

### s01.01  `0.00 → 5.05` (5.0 s) · WIDE -> MEDIUM

> L1 HOST [0.60–4.70]: Coming up after the break: the hidden chemical in your takeout that could be making you sick!  

- **On screen:** Dark wall of 9 retro TVs (3x3 grid, see continuity TVWALL). Static flickers on all screens; the CENTER TV (B2) switches on: HOST silhouette behind a talk-show desk, pointing at camera. Off-screens keep faint static.
- **Camera:** Slow push-in from full wall toward B2 (ends ~70% frame on B2).
- **Text:** DRAMATIZATION (tag on B2), TALK SHOW HOST (role chyron), HIDDEN CHEMICAL?
- **Transition:** in: Fade from black (0.4 s) / out: Hard cut on channel_switch
- **SFX:** `0.20 static 0.3`, `0.45 tv_on 0.7`, `2.60 pop_small 0.35`
- **Continuity props:** TVWALL, DRAMA_TAG

### s01.02  `5.05 → 10.45` (5.4 s) · MEDIUM (single TV)

> L2 TVDOC [5.05–10.10]: MSG is an excitotoxin. It overexcites your brain cells. Literally to death.  

- **On screen:** Top-left TV (A1): TV DOCTOR silhouette (white coat, stethoscope shape). Inset on screen: a simple brain icon with jittery zig-zag 'overexcited' lines that fizzle out on 'to death'. No gore, no skulls.
- **Camera:** Whip-pan from B2 to A1, then gentle drift right.
- **Text:** DRAMATIZATION, DAYTIME TV DOCTOR, EXCITOTOXIN?!
- **Transition:** in: Channel-switch glitch (2-4 frames) / out: Channel-switch glitch
- **SFX:** `4.95 channel_switch 0.6`, `5.10 glitch 0.3`, `8.30 blip 0.3`
- **Continuity props:** TVWALL, DRAMA_TAG

### s01.03  `10.45 → 15.35` (4.9 s) · MEDIUM (single TV)

> L3 CHEF [10.45–15.00]: We never use MSG in my kitchen. Never! It's poison, darling.  

- **On screen:** Top-right TV (C1): CELEBRITY CHEF silhouette (toque), arms crossed, wagging finger. On 'Never!' an MSG shaker on screen gets a big warning-colored X slammed over it.
- **Camera:** Whip-pan A1 -> C1; small punch-in (5%) on 'Never!'.
- **Text:** DRAMATIZATION, CELEBRITY CHEF, POISON!
- **Transition:** in: Channel-switch glitch / out: Channel-switch glitch
- **SFX:** `10.40 channel_switch 0.6`, `12.20 stamp 0.4`, `13.90 pop_small 0.3`
- **Continuity props:** TVWALL, DRAMA_TAG, SHAKER

### s01.04  `15.35 → 21.46` (6.1 s) · MEDIUM (single TV, vertical phone frame inside)

> L4 GURU [15.35–21.11]: Every time I eat Chinese food I get headaches, my heart races... it's the MSG, you guys!  

- **On screen:** Bottom-left TV (A3) shows a vertical phone-video frame: WELLNESS INFLUENCER silhouette with ring-light halo. Pop-up icons: headache (zig-zag at temple), racing heart (pulsing). A PLAIN generic takeout box icon only (no lettering, no pagoda/lantern/chopstick-font cliches).
- **Camera:** Whip-pan C1 -> A3 (diagonal); slow drift up.
- **Text:** DRAMATIZATION, WELLNESS INFLUENCER, HEADACHES?
- **Transition:** in: Channel-switch glitch / out: Channel-switch glitch
- **SFX:** `15.30 channel_switch 0.6`, `17.60 pop_small 0.35`, `18.60 heartbeat 0.35`
- **Continuity props:** TVWALL, DRAMA_TAG

### s01.05  `21.46 → 26.92` (5.5 s) · MEDIUM -> CLOSE (social post)

> L5 AUNT [21.46–26.02]: Sharing this for everyone. MSG is basically brain poison! Please read!!  

- **On screen:** Bottom-right TV (C3) shows a social-media post card typing itself out (illegible placeholder lines + shouty caps keyword). Share-arrow icon bounces. Generic UI only - no real platform logo/colors.
- **Camera:** Whip-pan A3 -> C3, slow push into the post card.
- **Text:** DRAMATIZATION, YOUR AUNT, ON SOCIAL MEDIA, BRAIN POISON!!
- **Transition:** in: Channel-switch glitch / out: Fast pull-back (no cut)
- **SFX:** `21.40 channel_switch 0.55`, `21.70 typewriter 0.3`, `24.90 ding 0.35`
- **Continuity props:** TVWALL, DRAMA_TAG

### s01.06  `26.92 → 32.45` (5.5 s) · EXTREME WIDE

> L6 NARR [26.92–32.15]: For more than fifty years, three little letters have terrified diners around the world.  

- **On screen:** All five speaker TVs are on simultaneously, looping their keywords; the 4 filler TVs show static. The wall shrinks into a dark room/void as the narrator takes over; the three letters 'MSG' float up big in front.
- **Camera:** Fast pull-back from C3 to full wall, then continuous slow pull-back.
- **Text:** 50+ YEARS, MSG
- **Transition:** in: Continuous (camera move) / out: Cut
- **SFX:** `26.95 whoosh 0.55`, `27.30 static 0.2`, `29.60 pop 0.45`
- **Continuity props:** TVWALL, DRAMA_TAG
- **Note:** Keep DRAMATIZATION tags visible on any TV whose talking head is still readable.

### s01.07  `32.45 → 35.60` (3.2 s) · MEDIUM  ★ HERO 1 - first appearance of the NO MSG sign (paid off in s09).

> L7 NARR [32.45–38.72, partial]: Restaurants hang signs promising they don't use it. Food labels proudly brag about leaving it out.  

- **On screen:** Night street: restaurant window. The NO MSG neon sign buzzes on letter by letter (NO, then MSG), stutters once, then holds.
- **Camera:** Slow push-in toward the sign.
- **Text:** NO MSG (the sign itself)
- **Transition:** in: Cut / out: Pan right / wipe to supermarket shelf
- **SFX:** `32.60 neon_flicker 0.6`, `33.10 neon_buzz 0.4`, `34.20 neon_flicker 0.3`
- **Continuity props:** SIGN

### s01.08  `35.60 → 39.02` (3.4 s) · MEDIUM -> WIDE

> L7 NARR [32.45–38.72, partial]: Restaurants hang signs promising they don't use it. Food labels proudly brag about leaving it out.  

- **On screen:** Supermarket shelf of generic snack/soup packages; badge stickers stamp onto them one by one (fictional brands only, no real logos).
- **Camera:** Lateral truck left->right along the shelf.
- **Text:** NO MSG, NO ADDED MSG
- **Transition:** in: Wipe / pan from sign / out: Cut
- **SFX:** `36.00 stamp 0.4`, `36.70 stamp 0.35`, `37.40 stamp 0.3`

### s01.09  `39.02 → 42.92` (3.9 s) · CLOSE

> L8 NARR [39.02–42.62]: But what actually is MSG? Is it really bad for you?  

- **On screen:** MSG shaker alone, centered in a spotlight on an empty table, crystals glinting. A large '?' pops above it on 'Is it really bad for you?'.
- **Camera:** Very slow push-in.
- **Text:** MSG?
- **Transition:** in: Cut / out: Freeze-frame on record scratch
- **SFX:** `39.30 pop 0.45`, `40.60 sparkle 0.35`, `41.60 pop_small 0.35`
- **Continuity props:** SHAKER

### s01.10  `42.92 → 47.71` (4.8 s) · CLOSE -> WIDE  ★ HERO 2 - record scratch + 'WHY?' (the video's central question).

> L9 NARR [42.92–46.51]: And if it isn't... why does almost everyone think it is?  

- **On screen:** Record scratch: frame freezes for ~0.3 s. Then pull back: the shaker sits small in front of the (now dim) TV wall and the NO MSG sign; a big 'WHY?' rises. Fade out.
- **Camera:** Freeze, then smooth pull-back to wide.
- **Text:** WHY?
- **Transition:** in: Freeze (no cut) / out: Fade to black (0.4 s)
- **SFX:** `42.95 record_scratch 0.7`, `44.90 pop 0.4`, `46.90 whoosh 0.35`
- **Continuity props:** SHAKER, TVWALL, SIGN

## s02_title — Title Card  ·  6.360 s  ·  2 shots

### s02.01  `0.00 → 1.40` (1.4 s) · WIDE

> L1 NARR [0.60–4.76, partial]: MSG. The most misunderstood molecule in your kitchen.  

- **On screen:** Starfield; small crystals orbit; the glutamate mascot (face on) pops in at center with a glow (~0.7 s).
- **Camera:** Slow linear push-in over whole scene (1.00 -> 1.08).
- **Text:** —
- **Transition:** in: Fade from black (0.5 s) / out: Continuous
- **SFX:** `0.00 whoosh_long 0.8`, `0.90 sparkle 0.7`
- **Continuity props:** GLU
- **Note:** Already implemented in scenes/s02_title.js - this entry documents it for continuity.

### s02.02  `1.40 → 6.36` (5.0 s) · WIDE  ★ HERO 3 - title card; mascot design is canonical from here on.

> L1 NARR [0.60–4.76, partial]: MSG. The most misunderstood molecule in your kitchen.  

- **On screen:** Title type rises under the bobbing mascot: 'MSG' (large) then subtitle.
- **Camera:** Continued push-in.
- **Text:** MSG, The Most Misunderstood Molecule
- **Transition:** in: Continuous / out: Fade to black (0.5 s)
- **SFX:** `1.60 impact_soft 0.9`, `2.30 pop 0.6`
- **Continuity props:** GLU

## s03_ikeda — 1908: The Discovery of Umami  ·  41.495 s  ·  10 shots

### s03.01  `0.00 → 4.57` (4.6 s) · EXTREME WIDE (establishing)

> L1 NARR [0.60–4.27]: Our story begins in Tokyo, in 1908.  

- **On screen:** Tokyo, 1908, at dusk: period-accurate wooden two-storey buildings, telegraph poles, lanterns over a street, a lit university-lab window. Respectful, not exoticized (no generic 'Asia' cliches).
- **Camera:** Slow descending crane toward the lit window.
- **Text:** TOKYO, 1908 (chapter label)
- **Transition:** in: Fade from black / out: Push through the window (match to interior)
- **SFX:** `0.20 whoosh_long 0.5`, `1.00 chime 0.35`, `3.40 tick 0.2`

### s03.02  `4.57 → 10.93` (6.4 s) · MEDIUM

> L2 NARR [4.57–10.63]: A chemist named Kikunae Ikeda was eating a bowl of soup made with kombu, a kind of seaweed.  

- **On screen:** Interior: Professor Ikeda (period suit, moustache, round glasses - see continuity IKEDA) at a table, blowing on a steaming bowl. A kombu strip floats up out of the bowl as an icon.
- **Camera:** Slow push-in from medium to medium-close.
- **Text:** KIKUNAE IKEDA (name tag), KOMBU
- **Transition:** in: Continuous from window push / out: Cut
- **SFX:** `5.00 pop_small 0.4`, `7.40 slurp 0.4`, `8.90 pop_small 0.35`
- **Continuity props:** IKEDA

### s03.03  `10.93 → 15.40` (4.5 s) · MEDIUM-CLOSE

> L3 NARR [10.93–18.37, partial]: He noticed a taste that wasn't sweet, or sour, or salty, or bitter. It was something else. Savory. Meaty. Deep.  

- **On screen:** Four taste icons pop into a ring around Ikeda's head as each word is spoken: SWEET, SOUR, SALTY, BITTER. Each dims/greys out right after it appears ('wasn't...').
- **Camera:** Locked with gentle drift.
- **Text:** SWEET, SOUR, SALTY, BITTER
- **Transition:** in: Cut / out: Continuous
- **SFX:** `11.80 pop 0.4`, `12.60 pop 0.4`, `13.30 pop 0.4`, `14.10 pop 0.4`
- **Continuity props:** TASTE_ICONS, IKEDA
- **Note:** Word times are estimates; snap each pop to the spoken word in the WAV.

### s03.04  `15.40 → 18.77` (3.4 s) · CLOSE

> L3 NARR [10.93–18.37, partial]: He noticed a taste that wasn't sweet, or sour, or salty, or bitter. It was something else. Savory. Meaty. Deep.  

- **On screen:** A fifth empty slot in the ring glows with a mysterious '?' that pulses; three words drift up in sequence.
- **Camera:** Push-in on the fifth slot.
- **Text:** SAVORY, MEATY, DEEP
- **Transition:** in: Continuous / out: Cut
- **SFX:** `15.50 sparkle 0.45`, `16.60 pop_small 0.3`, `17.40 pop_small 0.3`, `18.00 pop_small 0.3`
- **Continuity props:** TASTE_ICONS

### s03.05  `18.77 → 22.60` (3.8 s) · WIDE

> L4 NARR [18.77–26.52, partial]: So he went to his lab, boiled down huge amounts of seaweed, and found the source: a substance called glutamic acid.  

- **On screen:** Lab: a comically huge pile of seaweed tumbles into a giant pot over a burner; it starts boiling.
- **Camera:** Tilt down following the falling seaweed.
- **Text:** —
- **Transition:** in: Cut / out: Time-lapse dissolve
- **SFX:** `19.00 thud 0.5`, `19.80 bubbling_pot 0.4`
- **Continuity props:** IKEDA

### s03.06  `22.60 → 26.82` (4.2 s) · WIDE -> CLOSE

> L4 NARR [18.77–26.52, partial]: So he went to his lab, boiled down huge amounts of seaweed, and found the source: a substance called glutamic acid.  

- **On screen:** Time-lapse: the pile shrinks, steam rises, the pot boils down to a small dish with just a few glinting crystals.
- **Camera:** Push-in from pot to dish (ends close on crystals).
- **Text:** GLUTAMIC ACID
- **Transition:** in: Time-lapse dissolve / out: Cut
- **SFX:** `22.80 swoosh_down 0.4`, `24.60 sparkle 0.5`, `25.40 pop 0.4`

### s03.07  `26.82 → 30.51` (3.7 s) · MEDIUM  ★ HERO 4 - UMAMI is named (fifth taste completes the ring).

> L5 NARR [26.82–30.21]: He named the taste umami. Roughly: delicious taste.  

- **On screen:** The fifth taste slot fills: big word UMAMI pops in center with glow; the four other taste icons re-appear around it as equals. Small translation line below.
- **Camera:** Slight push-in with breathing scale.
- **Text:** UMAMI, 'delicious taste'
- **Transition:** in: Cut / out: Cut
- **SFX:** `27.20 impact_soft 0.7`, `27.45 chime 0.45`
- **Continuity props:** TASTE_ICONS

### s03.08  `30.51 → 34.40` (3.9 s) · CLOSE

> L6 NARR [30.51–38.19, partial]: To turn it into a seasoning, he paired it with sodium, making a stable, salt-like crystal. Monosodium glutamate.  

- **On screen:** The glutamate mascot (first time in the story) meets a small round sodium ion character; they click together.
- **Camera:** Locked, gentle drift.
- **Text:** + Na (sodium)
- **Transition:** in: Cut / out: Continuous
- **SFX:** `31.60 click 0.45`, `33.00 pop 0.4`
- **Continuity props:** GLU, NA

### s03.09  `34.40 → 38.49` (4.1 s) · CLOSE -> MEDIUM

> L6 NARR [30.51–38.19, partial]: To turn it into a seasoning, he paired it with sodium, making a stable, salt-like crystal. Monosodium glutamate.  

- **On screen:** The pair tiles into a growing, salt-like crystal lattice. Full name appears with M, S and G highlighted.
- **Camera:** Pull-back as the crystal grows.
- **Text:** MONOSODIUM GLUTAMATE, M-S-G
- **Transition:** in: Continuous / out: Cut
- **SFX:** `35.00 sparkle 0.5`, `36.60 ding 0.4`
- **Continuity props:** GLU, NA

### s03.10  `38.49 → 41.49` (3.0 s) · MEDIUM

> L7 NARR [38.49–40.49]: Within a year, it was on sale.  

- **On screen:** Crystals pour into the MSG shaker (first canonical appearance), which lands on a period shop counter.
- **Camera:** Small push-in, then hold.
- **Text:** ON SALE
- **Transition:** in: Cut / out: Fade to black
- **SFX:** `38.70 pop_small 0.35`, `39.30 cash 0.45`, `41.00 whoosh 0.3`
- **Continuity props:** SHAKER
- **Note:** Do NOT show a year here: script says only 'within a year'.

## s04_everywhere — Glutamate Is Everywhere  ·  42.752 s  ·  11 shots

### s04.01  `0.00 → 5.08` (5.1 s) · CLOSE

> L1 NARR [0.60–4.78]: Here's the thing. Glutamate isn't some strange lab-made chemical.  

- **On screen:** Glutamate mascot center. Behind it a lab flask and beaker pop in; on 'lab-made' they get a big X and slide away; mascot shrugs.
- **Camera:** Gentle drift.
- **Text:** LAB-MADE?
- **Transition:** in: Fade from black / out: Zoom-out (continuous)
- **SFX:** `0.70 pop 0.5`, `2.30 pop_small 0.3`, `3.40 stamp 0.35`
- **Continuity props:** GLU

### s04.02  `5.08 → 8.80` (3.7 s) · CLOSE -> MEDIUM

> L2 NARR [5.08–11.87, partial]: It's one of the most common amino acids on Earth, a building block of the proteins in every living thing.  

- **On screen:** Zoom out: the mascot is one bead in a long chain of differently-colored beads (amino acids) that folds into a protein blob.
- **Camera:** Smooth zoom-out.
- **Text:** AMINO ACID, PROTEIN
- **Transition:** in: Continuous / out: Continuous
- **SFX:** `5.20 swoosh_up 0.45`, `6.20 pop_small 0.3`, `7.40 bubble 0.3`
- **Continuity props:** GLU

### s04.03  `8.80 → 12.16` (3.4 s) · WIDE

> L2 NARR [5.08–11.87, partial]: It's one of the most common amino acids on Earth, a building block of the proteins in every living thing.  

- **On screen:** Protein blobs drift into silhouettes of living things: a leaf, a fish, a cell, a person.
- **Camera:** Continued slow zoom-out / pan.
- **Text:** EVERY LIVING THING
- **Transition:** in: Continuous / out: Cut
- **SFX:** `9.00 whoosh 0.4`, `9.80 pop_small 0.3`, `10.60 pop_small 0.3`, `11.30 pop_small 0.3`

### s04.04  `12.16 → 15.40` (3.2 s) · MEDIUM

> L3 NARR [12.16–21.38, partial]: Ripe tomatoes are full of it. So are mushrooms, soy sauce, and parmesan cheese, one of the richest sources of glutamate there is.  

- **On screen:** Ripe tomato (cut-away) glowing softly with tiny glutamate mascots bobbing inside.
- **Camera:** Push-in.
- **Text:** TOMATO
- **Transition:** in: Cut / out: Truck right
- **SFX:** `12.40 pop 0.5`, `14.00 bubble 0.3`
- **Continuity props:** TOMATO, GLU

### s04.05  `15.40 → 18.30` (2.9 s) · WIDE (row of foods)

> L3 NARR [12.16–21.38, partial]: Ripe tomatoes are full of it. So are mushrooms, soy sauce, and parmesan cheese, one of the richest sources of glutamate there is.  

- **On screen:** Mushrooms and a soy-sauce bottle pop in, forming a row to the right of the tomato, each with a mini mascot.
- **Camera:** Truck right along the row.
- **Text:** MUSHROOMS, SOY SAUCE
- **Transition:** in: Truck / out: Continuous
- **SFX:** `15.60 pop 0.4`, `16.80 pop 0.4`
- **Continuity props:** TOMATO, GLU

### s04.06  `18.30 → 21.68` (3.4 s) · MEDIUM

> L3 NARR [12.16–21.38, partial]: Ripe tomatoes are full of it. So are mushrooms, soy sauce, and parmesan cheese, one of the richest sources of glutamate there is.  

- **On screen:** Parmesan wedge pops in last and largest, sparkles; it is the only food given an emphasis badge.
- **Camera:** Push-in on parmesan.
- **Text:** PARMESAN, ONE OF THE RICHEST
- **Transition:** in: Continuous / out: Cut
- **SFX:** `18.60 pop 0.45`, `19.50 sparkle 0.45`
- **Continuity props:** GLU
- **Note:** No glutamate amounts or ranking between tomato/mushroom/soy - script only says parmesan is one of the richest.

### s04.07  `21.68 → 25.60` (3.9 s) · MEDIUM

> L4 NARR [21.68–29.25, partial]: Human breast milk contains around ten times more free glutamate than cow's milk. Our very first meal is umami.  

- **On screen:** Two simple milk icons side by side: HUMAN MILK (droplet/bottle) and COW'S MILK (glass). Bars rise beneath them: the human-milk bar grows to ~10x the cow's-milk bar.
- **Camera:** Locked; slight tilt up following the tall bar.
- **Text:** HUMAN MILK, COW'S MILK, FREE GLUTAMATE, ~10x
- **Transition:** in: Cut / out: Cut
- **SFX:** `22.00 pop_small 0.35`, `22.60 swoosh_up 0.45`, `24.50 ding 0.4`
- **Note:** Bar ratio must be ~10:1. No axis values. Icon-level depiction only (no nudity / breastfeeding scene).

### s04.08  `25.60 → 29.66` (4.1 s) · MEDIUM-CLOSE

> L4 NARR [21.68–29.25, partial]: Human breast milk contains around ten times more free glutamate than cow's milk. Our very first meal is umami.  

- **On screen:** Swaddled baby smiles with a warm umami glow; a tiny mascot floats by.
- **Camera:** Gentle push-in.
- **Text:** FIRST MEAL = UMAMI
- **Transition:** in: Cut / out: Cut
- **SFX:** `26.40 chime 0.4`, `27.80 pop_small 0.3`
- **Continuity props:** GLU

### s04.09  `29.66 → 33.75` (4.1 s) · MEDIUM

> L5 NARR [29.66–33.45]: Your body even makes its own glutamate, every single day.  

- **On screen:** Simple human silhouette with glowing interior; little 'cell factories' produce mascots. A sun/moon arc cycles overhead twice to say 'every day'.
- **Camera:** Locked, gentle drift.
- **Text:** MADE IN YOUR BODY, EVERY DAY
- **Transition:** in: Cut / out: Split-wipe
- **SFX:** `30.00 bubble 0.35`, `31.20 tick 0.25`, `32.40 tick 0.25`
- **Continuity props:** GLU

### s04.10  `33.75 → 37.80` (4.0 s) · WIDE (split screen)

> L6 NARR [33.75–41.75, partial]: And to your body, glutamate from a tomato and glutamate from a shaker of MSG are exactly the same molecule.  

- **On screen:** Left half: tomato releases a glutamate mascot. Right half: MSG shaker shakes out a glutamate mascot.
- **Camera:** Locked split; both halves drift slightly.
- **Text:** FROM TOMATO, FROM MSG
- **Transition:** in: Split-wipe / out: Continuous
- **SFX:** `33.90 whoosh 0.45`, `35.00 pop 0.45`, `36.30 pop 0.45`
- **Continuity props:** TOMATO, SHAKER, GLU

### s04.11  `37.80 → 42.75` (5.0 s) · MEDIUM  ★ HERO 5 - tomato glutamate = MSG glutamate (the two mascots must be IDENTICAL).

> L6 NARR [33.75–41.75, partial]: And to your body, glutamate from a tomato and glutamate from a shaker of MSG are exactly the same molecule.  

- **On screen:** Split line dissolves; both mascots slide to center and overlap perfectly (pixel-identical). Big '=' between then over them.
- **Camera:** Slow push-in.
- **Text:** =, SAME MOLECULE
- **Transition:** in: Continuous / out: Fade to black
- **SFX:** `38.40 impact_soft 0.6`, `39.60 success 0.45`
- **Continuity props:** GLU

## s05_body — What Happens When You Eat It  ·  45.164 s  ·  10 shots

### s05.01  `0.00 → 2.78` (2.8 s) · MEDIUM

> L1 NARR [0.60–2.48]: So what happens when you eat it?  

- **On screen:** A fork lifts a bite of seasoned food from a plate toward a mouth (profile).
- **Camera:** Tilt up with the fork.
- **Text:** —
- **Transition:** in: Fade from black / out: Match-cut into the cutaway body
- **SFX:** `0.60 whoosh 0.4`, `2.20 chomp 0.5`

### s05.02  `2.78 → 7.00` (4.2 s) · MEDIUM -> CLOSE

> L2 NARR [2.78–11.03, partial]: First, glutamate docks onto special receptors on your tongue, and sends your brain a message: this food has protein in it. Yum.  

- **On screen:** Cutaway body (side profile head + torso, see BODY). Push into the tongue: taste receptors as little docking cups; glutamate mascots dock with a click and the receptor glows.
- **Camera:** Zoom into tongue.
- **Text:** RECEPTORS
- **Transition:** in: Match-cut / out: Continuous
- **SFX:** `4.20 click 0.5`, `4.60 click 0.4`, `5.00 click 0.35`
- **Continuity props:** BODY, GLU

### s05.03  `7.00 → 11.33` (4.3 s) · MEDIUM

> L2 NARR [2.78–11.03, partial]: First, glutamate docks onto special receptors on your tongue, and sends your brain a message: this food has protein in it. Yum.  

- **On screen:** A glowing signal pulse travels up a nerve to the brain; the brain pops a speech bubble.
- **Camera:** Tilt/track up following the pulse.
- **Text:** PROTEIN!, YUM
- **Transition:** in: Continuous / out: Continuous
- **SFX:** `7.30 swoosh_up 0.45`, `9.00 ding 0.45`, `10.50 pop 0.4`
- **Continuity props:** BODY

### s05.04  `11.33 → 16.48` (5.1 s) · WIDE (vertical travel)

> L3 NARR [11.33–16.18]: Then it reaches your gut. And this is where the scary story starts to fall apart.  

- **On screen:** Camera dives down the esophagus into the gut tunnel. On 'scary story starts to fall apart' a small '!' alarm bubble (TV-wall style from s01) cracks and crumbles.
- **Camera:** Long downward dive, ease-in-out.
- **Text:** GUT
- **Transition:** in: Continuous / out: Cut
- **SFX:** `11.40 whoosh_long 0.5`, `14.60 crunch 0.35`
- **Continuity props:** BODY

### s05.05  `16.48 → 21.38` (4.9 s) · CLOSE  ★ HERO 6 - gut cells eat glutamate as fuel (character sheet reused in s06).

> L4 NARR [16.48–21.08]: The cells lining your intestines love glutamate. They burn it as fuel.  

- **On screen:** Intestine-wall close-up: round gut-cell characters (see GUTCELL) gobble passing glutamate mascots and light up with a small energy spark.
- **Camera:** Lateral truck along the cells.
- **Text:** FUEL
- **Transition:** in: Cut / out: Pull-back
- **SFX:** `17.20 chomp 0.45`, `18.00 chomp 0.4`, `18.90 chomp 0.4`, `20.00 sparkle 0.3`
- **Continuity props:** GUTCELL, GLU

### s05.06  `21.38 → 25.60` (4.2 s) · MEDIUM  ★ HERO 7 - the 95% counter.

> L5 NARR [21.38–30.36, partial]: Almost all the glutamate you eat, around ninety-five percent, is used up right there in your gut, before it ever reaches your bloodstream.  

- **On screen:** A stream of glutamate particles enters; most are eaten. A percentage counter climbs and lands on 95% exactly as 'ninety-five percent' is spoken.
- **Camera:** Pull-back to medium, locked.
- **Text:** 95%, USED BY YOUR GUT
- **Transition:** in: Continuous / out: Continuous
- **SFX:** `22.00 tick 0.2`, `22.60 tick 0.2`, `23.20 tick 0.2`, `23.80 tick 0.2`, `24.50 ding 0.5`
- **Continuity props:** GUTCELL, GLU
- **Note:** Counter landing time (~24.5 s) is an estimate - snap to the spoken 'ninety-five' in the WAV.

### s05.07  `25.60 → 30.77` (5.2 s) · WIDE

> L5 NARR [21.38–30.36, partial]: Almost all the glutamate you eat, around ninety-five percent, is used up right there in your gut, before it ever reaches your bloodstream.  

- **On screen:** Pull back: gut wall as a border; only a few stray particles slip through into a blood vessel beyond.
- **Camera:** Continued pull-back.
- **Text:** BLOODSTREAM
- **Transition:** in: Continuous / out: Track along vessel
- **SFX:** `27.00 bubble 0.3`, `28.60 swoosh_down 0.35`
- **Continuity props:** GUTCELL, GLU
- **Note:** Do not print '5%' - only 95% is stated.

### s05.08  `30.77 → 35.30` (4.5 s) · WIDE

> L6 NARR [30.77–39.70, partial]: And the little that slips through faces another wall: the blood-brain barrier, a tightly guarded border that controls what gets into your brain.  

- **On screen:** Follow the few particles up a vessel to the head. A castle-wall barrier rises, with cute guard characters.
- **Camera:** Track upward with particles.
- **Text:** BLOOD-BRAIN BARRIER
- **Transition:** in: Track / out: Continuous
- **SFX:** `30.90 whoosh 0.45`, `32.80 thud 0.5`, `34.00 pop_small 0.3`
- **Continuity props:** BBB, GLU

### s05.09  `35.30 → 40.00` (4.7 s) · MEDIUM

> L6 NARR [30.77–39.70, partial]: And the little that slips through faces another wall: the blood-brain barrier, a tightly guarded border that controls what gets into your brain.  

- **On screen:** At the gate a guard checks arrivals; a glutamate particle approaches, the guard raises a palm, it bounces back.
- **Camera:** Push-in to gate.
- **Text:** CONTROLLED BORDER
- **Transition:** in: Continuous / out: Pull-back
- **SFX:** `36.40 click 0.35`, `38.20 boing 0.4`
- **Continuity props:** BBB, GLU

### s05.10  `40.00 → 45.16` (5.2 s) · WIDE

> L7 NARR [40.00–44.16]: The glutamate on your dinner plate doesn't just stroll into your neurons.  

- **On screen:** Full stack: dinner plate at bottom, gut, wall, brain with calm glowing neurons at top. Mascot at the gate shrugs.
- **Camera:** Slow pull-back to full body.
- **Text:** —
- **Transition:** in: Continuous / out: Fade to black
- **SFX:** `41.00 pop 0.35`, `44.40 whoosh 0.3`
- **Continuity props:** BODY, BBB, GLU

## s06_origin — How the Rumour Started  ·  78.143 s  ·  17 shots

### s06.01  `0.00 → 5.04` (5.0 s) · MEDIUM

> L1 NARR [0.60–4.74]: So how did a seaweed seasoning become public enemy number one?  

- **On screen:** A kombu strip morphs into the MSG shaker; spotlight snaps on and a target-reticle frames it.
- **Camera:** Push-in.
- **Text:** PUBLIC ENEMY #1
- **Transition:** in: Fade from black / out: Cut
- **SFX:** `0.50 whoosh 0.45`, `2.60 impact_soft 0.5`, `3.60 pop_small 0.3`
- **Continuity props:** SHAKER

### s06.02  `5.04 → 8.67` (3.6 s) · MEDIUM

> L2 NARR [5.04–8.37]: It started in 1968, with a letter.  

- **On screen:** Chapter label 1968. A typewritten letter flutters down onto a desk.
- **Camera:** Tilt down following the letter.
- **Text:** 1968 (chapter label)
- **Transition:** in: Cut / out: Push-in
- **SFX:** `5.30 paper 0.45`, `6.00 swoosh_down 0.35`, `7.40 paper 0.35`
- **Continuity props:** LETTER

### s06.03  `8.67 → 13.80` (5.1 s) · CLOSE

> L3 NARR [8.67–19.06, partial]: A doctor wrote to the New England Journal of Medicine, describing numbness, weakness and palpitations after eating at Chinese restaurants in the United States.  

- **On screen:** Letter close-up: lines type themselves (illegible placeholder text); envelope addressed to the journal.
- **Camera:** Slow push along the letter.
- **Text:** TO: NEW ENGLAND JOURNAL OF MEDICINE
- **Transition:** in: Push-in / out: Continuous
- **SFX:** `9.00 typewriter 0.4`, `12.60 paper 0.25`
- **Continuity props:** LETTER
- **Note:** Plain text address only - do not recreate the journal's real masthead/logo.

### s06.04  `13.80 → 19.36` (5.6 s) · MEDIUM

> L3 NARR [8.67–19.06, partial]: A doctor wrote to the New England Journal of Medicine, describing numbness, weakness and palpitations after eating at Chinese restaurants in the United States.  

- **On screen:** Three symptom icons pop out of the letter: tingling hand (numbness), drooping arm (weakness), fluttering heart (palpitations). Beside them a small, NEUTRAL restaurant-front icon with a US flag pin.
- **Camera:** Gentle pull-back.
- **Text:** NUMBNESS, WEAKNESS, PALPITATIONS
- **Transition:** in: Continuous / out: Cut
- **SFX:** `14.60 pop 0.4`, `15.50 pop 0.4`, `16.60 pop 0.4`, `17.10 heartbeat 0.3`
- **Continuity props:** LETTER
- **Note:** No stereotyped restaurant/people imagery (see guardrails).

### s06.05  `19.36 → 23.50` (4.1 s) · WIDE (line-up)

> L4 NARR [19.36–26.42, partial]: He wondered what might be causing it. The cooking wine? The salt? Maybe the MSG? He didn't know. It was a guess.  

- **On screen:** Suspect line-up: cooking-wine bottle, salt shaker, MSG shaker. A '?' pops above each as it is named.
- **Camera:** Truck left->right.
- **Text:** ?, ?, ?
- **Transition:** in: Cut / out: Continuous
- **SFX:** `20.50 blip 0.4`, `21.30 blip 0.4`, `22.30 blip 0.45`
- **Continuity props:** SALT, SHAKER

### s06.06  `23.50 → 26.72` (3.2 s) · MEDIUM

> L4 NARR [19.36–26.42, partial]: He wondered what might be causing it. The cooking wine? The salt? Maybe the MSG? He didn't know. It was a guess.  

- **On screen:** The three '?' drift together and merge into one wobbly label; the letter-writer (a simple shrugging hand/silhouette, not a portrait) shrugs.
- **Camera:** Locked.
- **Text:** A GUESS
- **Transition:** in: Continuous / out: Whip to the headline
- **SFX:** `24.60 pop 0.4`, `25.60 boing 0.3`
- **Continuity props:** SALT, SHAKER

### s06.07  `26.72 → 31.75` (5.0 s) · CLOSE  ★ HERO 8 - the headline stamp.

> L5 NARR [26.72–31.45]: But the journal printed it under a catchy headline: Chinese Restaurant Syndrome.  

- **On screen:** The printed headline stamp slams across the letter/page with a camera shake.
- **Camera:** Whip-in, then 6-frame shake on impact, then hold.
- **Text:** "CHINESE-RESTAURANT SYNDROME"
- **Transition:** in: Whip-pan / out: Cut
- **SFX:** `27.40 swoosh_down 0.35`, `29.30 stamp 0.8`, `29.35 impact_big 0.45`
- **Continuity props:** HEADLINE
- **Note:** Stamp lands on the spoken name (~29.3 s, verify). Plain neutral typeface - never a 'chop-suey'/wonton font.

### s06.08  `31.75 → 35.00` (3.2 s) · WIDE

> L6 NARR [31.75–38.32, partial]: Letters poured in. Newspapers ran with it. Almost overnight, a guess had become a diagnosis.  

- **On screen:** Envelopes pour in from the top; newspapers multiply and spin toward camera (classic spinning-paper gag).
- **Camera:** Push-in into the spinning paper.
- **Text:** —
- **Transition:** in: Cut / out: Spin transition
- **SFX:** `31.90 paper 0.4`, `33.20 whoosh 0.45`, `34.30 whoosh 0.35`
- **Continuity props:** HEADLINE

### s06.09  `35.00 → 38.72` (3.7 s) · MEDIUM

> L6 NARR [31.75–38.32, partial]: Letters poured in. Newspapers ran with it. Almost overnight, a guess had become a diagnosis.  

- **On screen:** The 'A GUESS' label morphs into a clipboard-style 'DIAGNOSIS' stamp.
- **Camera:** Locked.
- **Text:** GUESS -> DIAGNOSIS
- **Transition:** in: Spin transition / out: Cut
- **SFX:** `35.80 swoosh_up 0.35`, `36.90 stamp 0.55`

### s06.10  `38.72 → 44.00` (5.3 s) · MEDIUM

> L7 NARR [38.72–49.39, partial]: A year later, a scientist injected huge doses of MSG under the skin of newborn mice, and found brain damage. The headlines got even scarier.  

- **On screen:** Chapter label 1969. A cute lab with newborn mouse pups in a nest; a comically GIANT syringe looms in (scale gag = huge dose). The injection is implied, never shown; no gore.
- **Camera:** Slow push-in.
- **Text:** 1969 (chapter label), HUGE DOSE
- **Transition:** in: Cut / out: Cut
- **SFX:** `39.00 swoosh_down 0.4`, `41.20 impact_soft 0.4`, `42.60 pop_small 0.3`
- **Continuity props:** MOUSE, SYRINGE

### s06.11  `44.00 → 49.69` (5.7 s) · WIDE

> L7 NARR [38.72–49.39, partial]: A year later, a scientist injected huge doses of MSG under the skin of newborn mice, and found brain damage. The headlines got even scarier.  

- **On screen:** Newspapers return with bigger, scarier headlines (illegible except 'MSG' and '!'); a muted alarm bell shakes.
- **Camera:** Push-in through the papers.
- **Text:** MSG!?
- **Transition:** in: Cut / out: Split-wipe
- **SFX:** `44.30 paper 0.4`, `46.40 alarm 0.25`, `48.20 whoosh 0.35`
- **Continuity props:** HEADLINE

### s06.12  `49.69 → 54.20` (4.5 s) · WIDE (split screen)  ★ HERO 9 - injection ≠ meal.

> L8 NARR [49.69–58.69, partial]: But injecting a massive dose into a baby mouse is nothing like eating a seasoned meal. Remember: your gut eats almost all of it first.  

- **On screen:** Left: giant syringe over a tiny mouse pup ('injection'). Right: a person enjoying a seasoned bowl ('meal'). A big not-equal sign between.
- **Camera:** Locked split, slight drift.
- **Text:** INJECTION, MEAL, ≠
- **Transition:** in: Split-wipe / out: Push into the person's belly (cutaway)
- **SFX:** `49.90 whoosh 0.4`, `51.80 impact_soft 0.55`
- **Continuity props:** MOUSE, SYRINGE

### s06.13  `54.20 → 59.09` (4.9 s) · CLOSE

> L8 NARR [49.69–58.69, partial]: But injecting a massive dose into a baby mouse is nothing like eating a seasoned meal. Remember: your gut eats almost all of it first.  

- **On screen:** Callback: the s05 gut cells gobble glutamate mascots; the 95% badge from s05 reappears small.
- **Camera:** Lateral truck (same as s05 shot 05).
- **Text:** 95% (callback)
- **Transition:** in: Push-in / cutaway / out: Desaturating dissolve
- **SFX:** `55.40 chomp 0.4`, `56.30 chomp 0.35`, `57.60 pop_small 0.3`
- **Continuity props:** GUTCELL, GLU
- **Note:** Must reuse GUTCELL and 95% badge exactly as designed in s05.

### s06.14  `59.09 → 63.20` (4.1 s) · WIDE

> L9 NARR [59.09–67.09, partial]: And there was something else going on. In 1960s America, Chinese food was seen as foreign and suspicious.  

- **On screen:** Mood shift (slightly muted palette, still the house style): a 1960s American street with a varied crowd of ordinary diners walking past a NEUTRAL restaurant facade.
- **Camera:** Slow lateral pan.
- **Text:** 1960s AMERICA
- **Transition:** in: Desaturating dissolve / out: Continuous
- **SFX:** `59.20 whoosh_long 0.35`
- **Continuity props:** CROWD
- **Note:** Highest-sensitivity shot - see guardrails (no stereotypes, no caricatured Asian people).

### s06.15  `63.20 → 67.39` (4.2 s) · MEDIUM

> L9 NARR [59.09–67.09, partial]: And there was something else going on. In 1960s America, Chinese food was seen as foreign and suspicious.  

- **On screen:** Suspicion shown abstractly: cold side-eye glances and gray tag-labels drift onto a plain takeout box (the prejudice is visualized as something put ON the food, not as a trait of people).
- **Camera:** Slow push-in on the box.
- **Text:** 'FOREIGN', 'SUSPICIOUS'
- **Transition:** in: Continuous / out: Split-wipe
- **SFX:** `64.00 static 0.2`, `65.60 pop_small 0.3`
- **Continuity props:** TAKEOUT
- **Note:** Words shown in quotes / as tags so they read as the attitude being criticised.

### s06.16  `67.39 → 72.00` (4.6 s) · WIDE (split screen)  ★ HERO 10 - double standard split screen.

> L10 NARR [67.39–77.14, partial]: Nobody blamed the tomato sauce. Nobody worried about Italian Restaurant Syndrome. The fear stuck to one cuisine, and the name said it out loud.  

- **On screen:** Left: pasta with tomato sauce and parmesan - calm, nothing happens. Right: the takeout box - alarm bells ring. Identical glutamate mascots bob above BOTH sides. A fake headline 'ITALIAN RESTAURANT SYNDROME' fades in on the left and dissolves away (it never existed).
- **Camera:** Locked split, gentle drift.
- **Text:** TOMATO SAUCE, ITALIAN RESTAURANT SYNDROME? (dissolves)
- **Transition:** in: Split-wipe / out: Continuous push
- **SFX:** `67.60 whoosh 0.4`, `69.00 alarm 0.3`, `70.60 pop_small 0.3`
- **Continuity props:** TOMATO, TAKEOUT, GLU

### s06.17  `72.00 → 78.14` (6.1 s) · CLOSE

> L10 NARR [67.39–77.14, partial]: Nobody blamed the tomato sauce. Nobody worried about Italian Restaurant Syndrome. The fear stuck to one cuisine, and the name said it out loud.  

- **On screen:** Split collapses onto the right; the original HEADLINE card from shot 07 slides back in; camera pushes in on the name; hold, then fade.
- **Camera:** Slow push-in, ease-out.
- **Text:** "CHINESE-RESTAURANT SYNDROME"
- **Transition:** in: Continuous / out: Fade to black
- **SFX:** `73.40 swoosh_down 0.3`, `75.80 impact_soft 0.4`, `77.40 whoosh 0.25`
- **Continuity props:** HEADLINE
- **Note:** Same HEADLINE card as shot 07, identical typography.

## s07_tests — Putting MSG to the Test  ·  46.740 s  ·  9 shots

### s07.01  `0.00 → 5.57` (5.6 s) · WIDE

> L1 NARR [0.60–5.27]: Scientists took the claims seriously, and tested them. Again, and again.  

- **On screen:** Clean modern lab, a diverse team of scientists. Clipboards and study folders stack up, each getting a check as 'again, and again' is said.
- **Camera:** Slow lateral pan.
- **Text:** TESTED, AGAIN & AGAIN
- **Transition:** in: Fade from black / out: Cut
- **SFX:** `0.60 whoosh 0.35`, `2.40 pop 0.4`, `3.90 click 0.35`, `4.60 click 0.35`
- **Continuity props:** LAB
- **Note:** No study count is given in the script - do not number the folders.

### s07.02  `5.57 → 9.80` (4.2 s) · MEDIUM

> L2 NARR [5.57–15.60, partial]: The gold standard is a double-blind study: people get either MSG or a placebo, and nobody, not even the researchers, knows who got which.  

- **On screen:** A small gold-standard badge, then two identical capsules labeled A and B on a tray.
- **Camera:** Push-in.
- **Text:** DOUBLE-BLIND, A, B
- **Transition:** in: Cut / out: Continuous
- **SFX:** `6.20 ding 0.4`, `8.00 pop 0.4`, `8.50 pop 0.4`
- **Continuity props:** CAPSULES

### s07.03  `9.80 → 15.90` (6.1 s) · MEDIUM  ★ HERO 11 - the double-blind shuffle.

> L2 NARR [5.57–15.60, partial]: The gold standard is a double-blind study: people get either MSG or a placebo, and nobody, not even the researchers, knows who got which.  

- **On screen:** Capsules drop under two cups and get shuffled (shell-game). Volunteer AND researcher both wear cute blindfolds - nobody knows which is which.
- **Camera:** Locked; slight push on blindfolds.
- **Text:** MSG?, PLACEBO?
- **Transition:** in: Continuous / out: Cut
- **SFX:** `10.40 whoosh 0.35`, `11.10 whoosh 0.3`, `11.80 whoosh 0.3`, `13.60 pop_small 0.3`, `14.80 boing 0.3`
- **Continuity props:** CAPSULES

### s07.04  `15.90 → 19.30` (3.4 s) · WIDE

> L3 NARR [15.90–22.60, partial]: Over and over, people who said they were sensitive to MSG reacted just as often to the fake.  

- **On screen:** A row of diverse volunteers wearing 'SENSITIVE' badges, thought bubbles with worried symptom icons; each takes a cup.
- **Camera:** Truck left->right.
- **Text:** SENSITIVE
- **Transition:** in: Cut / out: Continuous
- **SFX:** `16.40 bubble 0.35`, `17.20 bubble 0.3`, `18.00 bubble 0.3`
- **Continuity props:** VOLUNTEERS

### s07.05  `19.30 → 22.90` (3.6 s) · MEDIUM  ★ HERO 12 - MSG vs placebo bars nearly equal.

> L3 NARR [15.90–22.60, partial]: Over and over, people who said they were sensitive to MSG reacted just as often to the fake.  

- **On screen:** Results chart: two bars, MSG vs PLACEBO, rise to nearly the same height. Volunteers' reaction icons split evenly between the two groups.
- **Camera:** Locked, gentle drift.
- **Text:** MSG, PLACEBO
- **Transition:** in: Cut / out: Cut
- **SFX:** `19.60 swoosh_up 0.4`, `21.00 ding 0.35`
- **Continuity props:** VOLUNTEERS
- **Note:** No axis numbers / percentages - the script gives none.

### s07.06  `22.90 → 28.50` (5.6 s) · MEDIUM

> L4 NARR [22.90–35.02, partial]: Huge doses on an empty stomach caused mild, short-lived symptoms in some people. But the results weren't consistent, and nothing like that showed up with normal amounts in normal food.  

- **On screen:** Comically huge spoonful next to an empty plate/stomach icon; SOME volunteers get small symptom icons that fade away quickly (a short timer runs out).
- **Camera:** Push-in.
- **Text:** HUGE DOSE, EMPTY STOMACH, MILD · SHORT-LIVED
- **Transition:** in: Cut / out: Cut
- **SFX:** `23.40 thud 0.4`, `25.40 pop_small 0.3`, `26.60 tick 0.25`, `27.30 tick 0.25`
- **Continuity props:** VOLUNTEERS

### s07.07  `28.50 → 35.41` (6.9 s) · WIDE -> MEDIUM

> L4 NARR [22.90–35.02, partial]: Huge doses on an empty stomach caused mild, short-lived symptoms in some people. But the results weren't consistent, and nothing like that showed up with normal amounts in normal food.  

- **On screen:** Result cards scatter with mixed, inconsistent marks; then they clear and a normal meal plate appears with a relaxed volunteer and a calm check mark.
- **Camera:** Pull-back then push-in on the plate.
- **Text:** NOT CONSISTENT, NORMAL FOOD ✓
- **Transition:** in: Cut / out: Cut
- **SFX:** `28.80 paper 0.35`, `30.20 error 0.25`, `33.40 success 0.4`
- **Continuity props:** VOLUNTEERS

### s07.08  `35.41 → 40.50` (5.1 s) · MEDIUM  ★ HERO 13 - GRAS stamp.

> L5 NARR [35.41–45.74, partial]: Today, the US Food and Drug Administration lists MSG as generally recognized as safe, the same category as salt, sugar, and baking powder.  

- **On screen:** Official-looking (generic, not a real seal/logo) document slides in; a big GRAS stamp slams onto it.
- **Camera:** Push-in, 4-frame shake on stamp.
- **Text:** US FDA, GRAS, GENERALLY RECOGNIZED AS SAFE
- **Transition:** in: Cut / out: Pull-back
- **SFX:** `35.60 paper 0.35`, `39.00 stamp 0.8`, `39.05 impact_soft 0.4`
- **Continuity props:** GRAS
- **Note:** Stamp on 'generally recognized as safe' (~39.0 s, verify). Plain-text 'US FDA' only; no agency logo.

### s07.09  `40.50 → 46.74` (6.2 s) · WIDE

> L5 NARR [35.41–45.74, partial]: Today, the US Food and Drug Administration lists MSG as generally recognized as safe, the same category as salt, sugar, and baking powder.  

- **On screen:** The MSG shaker joins a row with a salt shaker, a sugar bowl and a baking-powder tin under the GRAS label.
- **Camera:** Pull-back, then hold.
- **Text:** SALT, SUGAR, BAKING POWDER
- **Transition:** in: Pull-back / out: Fade to black
- **SFX:** `41.60 pop 0.4`, `42.50 pop 0.4`, `43.60 pop 0.4`, `45.00 success 0.35`
- **Continuity props:** SHAKER, SALT, GRAS

## s08_twist — Plot Twist: MSG vs Salt  ·  18.733 s  ·  5 shots

### s08.01  `0.00 → 4.33` (4.3 s) · MEDIUM

> L1 NARR [0.60–4.04]: Here's a plot twist. MSG might actually help.  

- **On screen:** Swirl/twist transition; MSG shaker in spotlight with a curling twist-arrow and a small heart-shaped spark.
- **Camera:** Quick spin-in, then hold.
- **Text:** PLOT TWIST
- **Transition:** in: Fade from black + spin / out: Cut
- **SFX:** `0.40 swoosh_up 0.5`, `2.00 sparkle 0.4`, `3.00 pop_small 0.3`
- **Continuity props:** SHAKER

### s08.02  `4.33 → 6.80` (2.5 s) · WIDE

> L2 NARR [4.33–9.09, partial]: Gram for gram, it has only about a third of the sodium of table salt.  

- **On screen:** Balance scale; SALT shaker lands on the left pan, MSG shaker on the right ('gram for gram' - equal weights).
- **Camera:** Locked, gentle drift.
- **Text:** SALT, MSG, 1 g = 1 g
- **Transition:** in: Cut / out: Continuous
- **SFX:** `4.60 thud 0.45`, `5.30 thud 0.45`
- **Continuity props:** SALT, SHAKER
- **Note:** Scale stays level: equal mass is the premise.

### s08.03  `6.80 → 9.39` (2.6 s) · MEDIUM  ★ HERO 14 - about a third of the sodium (3:1 ions).

> L2 NARR [4.33–9.09, partial]: Gram for gram, it has only about a third of the sodium of table salt.  

- **On screen:** Sodium ions (same Na character as s03) pop out above each shaker: three under SALT, one under MSG.
- **Camera:** Push-in.
- **Text:** Na, ≈ 1/3 THE SODIUM
- **Transition:** in: Continuous / out: Cut
- **SFX:** `7.00 pop_small 0.35`, `7.30 pop_small 0.35`, `7.60 pop_small 0.35`, `8.20 pop_small 0.35`, `8.80 ding 0.45`
- **Continuity props:** SALT, SHAKER, NA

### s08.04  `9.39 → 13.50` (4.1 s) · MEDIUM

> L3 NARR [9.39–17.73, partial]: Swap some of the salt in a recipe for MSG, and you can cut the sodium in a dish noticeably, while keeping all the flavor.  

- **On screen:** A cooking pot; a salt spoon is partly swapped for an MSG spoon; a SODIUM meter beside the pot drops.
- **Camera:** Locked.
- **Text:** SODIUM ↓
- **Transition:** in: Cut / out: Continuous
- **SFX:** `10.40 swoosh_down 0.4`, `11.20 bubbling_pot 0.3`
- **Continuity props:** SALT, SHAKER, NA
- **Note:** No specific % reduction - script only says 'noticeably'.

### s08.05  `13.50 → 18.73` (5.2 s) · MEDIUM

> L3 NARR [9.39–17.73, partial]: Swap some of the salt in a recipe for MSG, and you can cut the sodium in a dish noticeably, while keeping all the flavor.  

- **On screen:** A FLAVOR meter next to the sodium meter stays full; a heart icon pops up and smiles.
- **Camera:** Slow pull-back.
- **Text:** FLAVOR ✓
- **Transition:** in: Continuous / out: Fade to black
- **SFX:** `14.60 pop 0.45`, `16.00 success 0.4`
- **Continuity props:** SHAKER
- **Note:** Heart is a mood icon; no 'heart health' text or claim.

## s09_outro — So... Is MSG Bad for You?  ·  32.319 s  ·  8 shots

### s09.01  `0.00 → 2.95` (2.9 s) · WIDE

> L1 NARR [0.60–2.65]: So, is MSG bad for you?  

- **On screen:** Callback: the exact s01 TV wall, all five talking heads frozen mid-claim (DRAMATIZATION tags visible); MSG shaker in front with a '?'.
- **Camera:** Slow push-in.
- **Text:** ?, DRAMATIZATION (on visible heads)
- **Transition:** in: Fade from black / out: Cut
- **SFX:** `0.30 static 0.25`, `1.00 pop 0.35`
- **Continuity props:** TVWALL, DRAMA_TAG, SHAKER

### s09.02  `2.95 → 7.60` (4.7 s) · MEDIUM

> L2 NARR [2.95–12.46, partial]: In absurd amounts, almost anything is. Even water. But in food, the evidence says it's about as dangerous as the tomato on your pizza.  

- **On screen:** Absurd-dose gag: a comically gigantic water tank next to a normal glass of water.
- **Camera:** Tilt up the giant tank.
- **Text:** EVEN WATER
- **Transition:** in: Cut / out: Continuous
- **SFX:** `3.60 bubble 0.35`, `5.80 boing 0.35`

### s09.03  `7.60 → 12.86` (5.3 s) · MEDIUM

> L2 NARR [2.95–12.46, partial]: In absurd amounts, almost anything is. Even water. But in food, the evidence says it's about as dangerous as the tomato on your pizza.  

- **On screen:** Balance scale: a pizza slice with tomato on one side, a pinch of MSG on the other - level.
- **Camera:** Locked; gentle drift.
- **Text:** ≈
- **Transition:** in: Continuous / out: Cut
- **SFX:** `8.40 pop 0.4`, `10.60 impact_soft 0.45`
- **Continuity props:** TOMATO, SHAKER

### s09.04  `12.86 → 18.60` (5.7 s) · MEDIUM

> L3 NARR [12.86–22.06, partial]: The story of MSG is really a story about how one guess, a catchy name, and a little bit of prejudice can outlive decades of science.  

- **On screen:** Three tags tumble in as each phrase is spoken: A GUESS (shape from s06), A CATCHY NAME (HEADLINE card mini), A LITTLE PREJUDICE.
- **Camera:** Locked.
- **Text:** ONE GUESS, A CATCHY NAME, A LITTLE PREJUDICE
- **Transition:** in: Cut / out: Pull-back
- **SFX:** `15.00 pop 0.4`, `16.10 pop 0.4`, `17.60 pop 0.4`
- **Continuity props:** HEADLINE
- **Note:** Word times estimated; snap to WAV.

### s09.05  `18.60 → 22.46` (3.9 s) · WIDE  ★ HERO 15 - TVs switch off one by one.

> L3 NARR [12.86–22.06, partial]: The story of MSG is really a story about how one guess, a catchy name, and a little bit of prejudice can outlive decades of science.  

- **On screen:** Pull back to the TV wall: on 'outlive decades of science' the TVs switch off one by one (reverse of s01 order: C3, A3, C1, A1, B2) while a tall stack of research papers rises beside them.
- **Camera:** Slow pull-back.
- **Text:** DECADES OF SCIENCE
- **Transition:** in: Pull-back / out: Cut
- **SFX:** `18.90 tv_off 0.45`, `19.50 tv_off 0.45`, `20.10 tv_off 0.45`, `20.70 tv_off 0.45`, `21.40 tv_off 0.5`
- **Continuity props:** TVWALL

### s09.06  `22.46 → 27.47` (5.0 s) · MEDIUM  ★ HERO 16 - 'NO' burns out (payoff of s01 shot 07).

> L4 NARR [22.46–27.17]: So the next time you see a No MSG sign, you'll know the real story.  

- **On screen:** The same NO MSG neon sign from s01; it flickers and the 'NO' burns out, leaving 'MSG' glowing warmly.
- **Camera:** Slow push-in.
- **Text:** NO MSG -> MSG
- **Transition:** in: Cut / out: Dissolve through the window
- **SFX:** `22.60 neon_buzz 0.35`, `24.00 neon_flicker 0.5`, `24.90 neon_flicker 0.45`, `25.50 click 0.4`, `26.40 sparkle 0.35`
- **Continuity props:** SIGN

### s09.07  `27.47 → 29.82` (2.3 s) · CLOSE

> L5 NARR [27.47–29.82]: And maybe... you'll reach for the shaker.  

- **On screen:** Cozy steaming bowl of ramen; a hand reaches for the MSG shaker and gives it two shakes.
- **Camera:** Gentle push-in.
- **Text:** —
- **Transition:** in: Dissolve / out: Continuous
- **SFX:** `27.90 slurp 0.3`, `28.60 pop_small 0.35`, `29.00 pop_small 0.35`
- **Continuity props:** SHAKER

### s09.08  `29.82 → 32.32` (2.5 s) · CLOSE -> EXTREME WIDE  ★ HERO 17 - pull-back to space / end card.

> (no narration)

- **On screen:** Camera pulls back: bowl -> table -> window -> city -> planet -> stars. End card fades in.
- **Camera:** Fast-then-slow pull-back into space.
- **Text:** MSG: The Most Misunderstood Molecule (end card)
- **Transition:** in: Continuous / out: Fade to black (end of film)
- **SFX:** `29.90 whoosh_long 0.55`, `31.00 chime 0.45`
- **Continuity props:** GLU
- **Note:** Optional: small glutamate mascot in the starfield mirrors s02.

---

## Hero moments

| # | Shot | Time (scene s) | Beat |
|---|---|---|---|
| 1 | s01.07 | 32.45–35.60 | first appearance of the NO MSG sign (paid off in s09). |
| 2 | s01.10 | 42.92–47.71 | record scratch + 'WHY?' (the video's central question). |
| 3 | s02.02 | 1.40–6.36 | title card; mascot design is canonical from here on. |
| 4 | s03.07 | 26.82–30.51 | UMAMI is named (fifth taste completes the ring). |
| 5 | s04.11 | 37.80–42.75 | tomato glutamate = MSG glutamate (the two mascots must be IDENTICAL). |
| 6 | s05.05 | 16.48–21.38 | gut cells eat glutamate as fuel (character sheet reused in s06). |
| 7 | s05.06 | 21.38–25.60 | the 95% counter. |
| 8 | s06.07 | 26.72–31.75 | the headline stamp. |
| 9 | s06.12 | 49.69–54.20 | injection ≠ meal. |
| 10 | s06.16 | 67.39–72.00 | double standard split screen. |
| 11 | s07.03 | 9.80–15.90 | the double-blind shuffle. |
| 12 | s07.05 | 19.30–22.90 | MSG vs placebo bars nearly equal. |
| 13 | s07.08 | 35.41–40.50 | GRAS stamp. |
| 14 | s08.03 | 6.80–9.39 | about a third of the sodium (3:1 ions). |
| 15 | s09.05 | 18.60–22.46 | TVs switch off one by one. |
| 16 | s09.06 | 22.46–27.47 | 'NO' burns out (payoff of s01 shot 07). |
| 17 | s09.08 | 29.82–32.32 | pull-back to space / end card. |

Hero shots get the most animation time: extra secondary motion, the strongest SFX of their scene, and a producer review of a still before full animation.

## Continuity bible (recurring props & characters)

Style details (palette, line, texture) come from the approved style guide; the rules below fix *identity*, *layout* and *behaviour* so scenes built by different animators cut together.

### GLU — Glutamate mascot
- **Rules:** Always drawn with KZ.glutamate(ctx,x,y,t,{face:true}) — same atom layout, same face, same proportions as s02. Swarms/particles may use small simplified versions but must read as the same molecule. Never redesign per scene; never give it a menacing face (it is the friendly hero). In s04.11 the two copies must be pixel-identical.
- **Appears in:** s02, s03.08–09, s04 (most shots), s05 (tongue, gut, barrier), s06.13, s06.16, s09.08

### SHAKER — MSG shaker
- **Rules:** Always KZ.shaker(ctx,x,y,s,'MSG') with the same body/cap colors in every scene. Label reads exactly 'MSG'. Same height ratio relative to the SALT shaker whenever both appear. Negative scenes (s01, s06.01) change the lighting/overlays (X, reticle), never the prop itself.
- **Appears in:** s01.03, s01.09–10, s03.10 (first canonical), s04.10, s06.01, s06.05–06, s07.09, s08 (all), s09.01, s09.03, s09.07

### SALT — Salt shaker
- **Rules:** KZ.shaker(...,'SALT') — a visibly different body or cap tone than MSG, otherwise identical silhouette (fair comparison). Reused unchanged in s06.05, s07.09, s08.
- **Appears in:** s06.05–06, s07.09, s08.02–04

### TVWALL — Retro TV wall
- **Rules:** 3x3 grid of rounded retro TVs, fixed layout reused verbatim in s09. Speaker positions: HOST=B2 (center), TVDOC=A1 (top-left), CHEF=C1 (top-right), GURU=A3 (bottom-left), AUNT=C3 (bottom-right); the 4 remaining TVs show static only. Lighting order in s01: B2, A1, C1, A3, C3. Switch-off order in s09: C3, A3, C1, A1, B2. Talking heads are flat silhouettes with role props (desk, coat, toque, ring light, post card) — no facial likeness to any real person. Build once and share (see PRODUCTION_PLAN.md).
- **Appears in:** s01.01–06, s01.10 (dim), s09.01, s09.05

### DRAMA_TAG — DRAMATIZATION tag
- **Rules:** Small pill tag reading 'DRAMATIZATION' in the same corner of every TV screen whenever a talking head is visible, legible at 1080p (cap height >= 22 px), present for the entire time the head is on screen.
- **Appears in:** s01.01–06, s09.01 (and any other frame showing a talking head)

### SIGN — NO MSG neon sign
- **Rules:** One design: two words on one line, 'NO' and 'MSG' as separate neon tubes so 'NO' can burn out alone in s09. Same window/frame, same camera angle in both appearances (s09.06 should feel like a return to s01.07).
- **Appears in:** s01.07, s01.10 (background), s09.06

### GUTCELL — Gut cells
- **Rules:** Round intestine-cell characters with dot eyes, sitting in a row along the gut wall; gobble animation (mouth opens, mascot vanishes, small energy spark). s06.13 is a callback and must reuse the s05 design, motion and the same '95%' badge styling.
- **Appears in:** s05.05–07, s06.13

### NA — Sodium ion
- **Rules:** Small round 'Na' character (no face or tiny face), identical in s03 (pairing with glutamate) and s08 (counting sodium). In s08 the count is exactly 3 (salt) : 1 (MSG).
- **Appears in:** s03.08–09, s08.03–04

### TOMATO — Tomato
- **Rules:** Same ripe tomato design (cut-away variant allowed) in s04, s06.16 (sauce), s09.03 (on pizza).
- **Appears in:** s04.04–05, s04.10, s06.16, s09.03

### HEADLINE — 'Chinese-Restaurant Syndrome' headline card
- **Rules:** One typographic card in a plain, neutral newspaper typeface (never a stereotyped 'oriental' font). Reused in s06.07, s06.08, s06.11 (variants), s06.17 and as a mini tag in s09.04.
- **Appears in:** s06.07–08, s06.11, s06.17, s09.04

### TASTE_ICONS — Five taste icons
- **Rules:** SWEET, SOUR, SALTY, BITTER + UMAMI in a ring; consistent icon set and order.
- **Appears in:** s03.03–04, s03.07

### IKEDA — Professor Kikunae Ikeda
- **Rules:** Respectful, simple character (period suit, moustache, round glasses), built from KZ.person proportions; same design across s03.
- **Appears in:** s03.02–05

### BODY — Cutaway body
- **Rules:** Side-profile head + torso; tongue, nerve, gut, vessels, brain in one consistent map.
- **Appears in:** s05.02–04, s05.10

### BBB — Blood-brain barrier castle wall + guards
- **Rules:** Cute guards, friendly not militaristic.
- **Appears in:** s05.08–10

### CHAPTER — Chapter labels
- **Rules:** Year/place labels use KZ.chapter (TOKYO, 1908 / 1968 / 1969 / 1960s AMERICA) for a consistent look.
- **Appears in:** s03.01, s06.02, s06.10, s06.14

## Factual guardrails

### Allowed on-screen numbers / claims (only these)

- '50+ YEARS' / 'more than fifty years' (s01)
- 1908, Tokyo, Kikunae Ikeda, kombu, glutamic acid, 'umami ~ delicious taste', sodium + glutamate = monosodium glutamate, on sale 'within a year' (s03) — NO explicit year for the sale
- Glutamate: one of the most common amino acids; building block of proteins; in tomatoes, mushrooms, soy sauce, parmesan ('one of the richest sources') (s04)
- Human breast milk ~10x more FREE glutamate than cow's milk (s04) — show '~10x', bars ~10:1
- Body makes its own glutamate every day; tomato glutamate = MSG glutamate (s04)
- ~95% of dietary glutamate used up in the gut (s05) — the ONLY percentage in the film; do not show '5%'
- 1968 letter to the New England Journal of Medicine; symptoms numbness, weakness, palpitations; suspects wine, salt, MSG; 'a guess'; printed under the headline 'Chinese Restaurant Syndrome' (s06)
- 1969 (a year later): huge doses injected under the skin of newborn mice, brain damage found (s06) — no gore
- Double-blind placebo-controlled testing; MSG-sensitive people reacted as often to placebo; huge doses on an empty stomach -> mild, short-lived symptoms in some people, inconsistent; nothing with normal amounts in normal food (s07)
- US FDA: 'generally recognized as safe' (GRAS), same category as salt, sugar, baking powder (s07)
- MSG has ~1/3 the sodium of table salt gram for gram; swapping some salt cuts sodium 'noticeably' while keeping flavor (s08) — no % reduction
- 'In absurd amounts almost anything is [bad], even water'; 'about as dangerous as the tomato on your pizza' (s09)

### Forbidden

- Any number, percentage, chart value, axis label, study count, dose amount or date not listed above (e.g. no '1909', no 'mg/kg', no 'n=...', no glutamate content per food).
- Health claims beyond the script: no 'heart healthy', 'cures', 'MSG makes you healthier', 'zero side effects'.
- Real logos, mastheads or seals (NEJM, FDA, social platforms, TV networks, food brands). Plain text names only.
- Real or recognisable people (celebrity look-alikes, real TV doctors/chefs). The cold-open talking heads are fictional composites.
- Showing Olney or Kwok as portraits — use hands/silhouettes or objects (letter, lab).
- Gore: no needles piercing animals, no dead mice, no brain lesions imagery.

### DRAMATIZATION rule

Every frame that shows a cold-open talking head (s01.01–06, s09.01, s09.05 if heads still visible) must carry the 'DRAMATIZATION' tag on that screen. Role chyrons use generic labels only: TALK SHOW HOST, DAYTIME TV DOCTOR, CELEBRITY CHEF, WELLNESS INFLUENCER, YOUR AUNT, ON SOCIAL MEDIA.

### 1960s section / cultural sensitivity

- Applies to s01.04 (takeout mention), s06.04, s06.07, s06.11, s06.14–17 and s09.04.
- No caricatured Asian people: no slanted-eye shorthand, buck teeth, conical hats, queues, 'coolie' imagery.
- No 'oriental' signifiers as shorthand: no wonton/chop-suey fonts, pagoda roofs, gongs, dragons, paper-lantern clusters, chopsticks-in-hair.
- Restaurant = neutral storefront; food = plain takeout box / bowl. Crowds are varied ordinary diners.
- The prejudice is the subject being criticised: show it as cold tags/glances put ON the food, and frame 'foreign'/'suspicious' in quotes; it should never look endorsed.
- The headline text is shown verbatim as a historical artifact in quotation marks; it is not used as a joke or punchline.
- Ikeda / Tokyo 1908 (s03) depicted respectfully and period-plausibly, not exoticized.
