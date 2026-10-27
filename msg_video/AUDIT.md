# Release audit: "MSG: The Most Misunderstood Molecule" (Instagram Reel)

**Audited:** 2026-10-27, final cut (`out/msg_reel.mp4`, rebuilt with `./build.sh`).

**Verdict: ready to publish**, after the three human checks at the end of this page.

## What it is

- **Format:** a vertical 1080×1920 Reel, 2:08 long, in the client-approved "Kombu Paper" style.
- **Narration:** ElevenLabs voice "George".
- **Opening:** two real, attributed clips of Dr. Russell Blaylock.
- **Order:**
  1. the claim
  2. how the rumour started (1968)
  3. the science (Ikeda, the same molecule, the gut, the blood-brain barrier)
  4. the tests
  5. the salt twist
  6. the verdict
- **Ending:** the last frame loops seamlessly into the first.

## 1. Technical checks (`python3 audit_tech.py out/msg_reel.mp4`)

| Check | Result |
|---|---|
| Length under 3:00 (Reel limit) | PASS, 128.3 s |
| 1080×1920, 30 fps, H.264 + AAC, yuv420p | PASS |
| Audio and video lengths match | PASS (128.23 s / 128.25 s) |
| Loudness around -14 LUFS | PASS, -14.1 LUFS |
| True peak below -1 dBTP | PASS, -1.7 dBTP (was -0.6; limiter ceiling lowered) |
| No silent gaps over 1.5 s | PASS |
| No black frames | PASS |
| No blank single-colour frames (10 fps scan) | PASS |
| Seamless loop | PASS (0.3/255 mean difference; pixel-identical before encoding) |
| File size | PASS, 81 MB |

On top of the automated checks, I reviewed every scene by eye on contact sheets:
- Safe zones: nothing important is in the top 220 px or the bottom 440 px.
- Captions: inside the y 1300–1480 band, and they never fade or ghost.
- Labels: one short label on screen at a time.
- Colour: no dull colours.

## 2. Fact-check (full report: `FACTCHECK.md`)

An independent fact-checker verified 22 claims against primary sources: NEJM 1968, Olney 1969, Geha 2000, Reeds 2000, Hawkins 2009, Mosby 2009, 21 CFR 182.1 and EFSA 2017.

**Every required fix is applied:**
- **Placebo line.** It said people "reacted just as often to a placebo", which overstated the evidence. It now says people "couldn't reliably tell it from a placebo". The on-screen chart shows inconsistent reactions instead of equal bars.
- **Salt line.** "Same flavor, less sodium" was overstated, because MSG is savoury, not salty. It now says "Big flavor, less sodium."
- **Blaylock credit.** It is now "Retired neurosurgeon · lecture video posted 2012". He is retired, and 2012 is the upload date, not the lecture date.

**Optional fixes also applied:**
- "For **over** fifty years, **people** have been scared", with the label "50+ YEARS".
- "a huge **dose** in a baby mouse".
- "it **ran under** a catchy headline". Who wrote the headline has never been confirmed.

**Everything else** is accurate, or accurate with nuance noted in FACTCHECK.md: the 1908 Ikeda story and umami, glutamate in tomatoes, parmesan and breast milk, "exact same molecule", "~95%", the blood-brain barrier, "generally recognized as safe, just like salt", a third of the sodium, the 1968 letter and "guessing", the mice study, and "the fear stuck to one cuisine".

## 3. The real interview clips

- **Source:** Dr. Russell Blaylock, "Nutrition & Behavior Dangers of Aspartame & MSG". The YouTube upload of 2012-10-04 has been removed; an archived copy survives at https://archive.org/details/youtube-Y0kvaulUIfc.
- **Quotes:**
  - Clip 1: "That's why they put so much MSG in food because it makes you hungry and you eat more of their food." (22:57–23:01)
  - Clip 2: "Excitotoxicity destroys brain cells." (24:05–24:08)
- **Verification:** two independent speech-recognition models (Whisper small.en and medium.en) agree with both quotes word for word, and the captions match the audio exactly.
- **Context:** both quotes are representative of his argument. I deliberately avoided a nearby "rage" line, because it describes injecting MSG directly into an animal's brain and would mislead out of context.
- **Edits to the footage:**
  - Where the source cuts away to a slide, the picture holds on his last on-camera frame.
  - The background is blurred so his lecture-slide text isn't readable.
  - His words and audio are untouched.
- **Rejected source:** the WMAR-TV 1986 news clip was not used, because the Internet Archive lists it as stream-only at the rights holder's request.

## 4. Viewer testing

A cold "19-year-old, short attention span" reviewer watched each cut:

| Round | Score |
|---|---|
| Early cut | 3.5/10 |
| | 4/10 |
| | 5/10 |
| Final | **6.5/10** (would like it, might share) |

**Fixes from her notes:**
- rumour-first order
- the mid-video title card cut
- captions highlight key words only
- no blank or glitchy frames
- tests scene recomposed
- clear loop
- "MYTH?" stamp in the first 2 seconds so the opening reads as a debunk, not scare content
- "GRAS" became "SAFE"

**Notes deliberately left as they are:**
- **Length.** She'd prefer about 90 s. The client chose about 2 minutes; a tighter cut-down is a quick follow-up if wanted.
- **The Blaylock footage looks low-res.** The source is a low-resolution 2012 upload, so this can't be fixed.

## 5. Costs

- **ElevenLabs:** 3,324 of 10,000 free monthly characters used, leaving 6,676.
- **Re-renders:** every narration line is cached in `audio/tts_cache/`, so re-rendering costs nothing unless wording changes.

## Human checks before posting (things I can't do)

1. **Listen to both Blaylock clips once** to confirm the quotes by ear: 0:00–0:07 of the Reel.
2. **Re-open the FDA MSG Q&A page once.** fda.gov was blocked from this environment, so the FDA wording was checked via search results.
3. **Rights call on the Blaylock footage.** It's used briefly, credited, and for criticism and commentary, which is the strongest fair-use position. Even so, it's someone else's footage: Instagram may flag or mute it, and the final call is the publisher's.
