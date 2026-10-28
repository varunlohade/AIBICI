# Release audit: "270 Metres Down" (Boesmansgat, 2005), Instagram Reel

**Audited:** 2026-10-28, final cut (`out/cave_reel.mp4`, rebuilt with `./build.sh`).

**Verdict: ready to publish.**

## What it is

- **Format:** a 1:35 vertical (1080×1920) animated true-incident Reel in an original "Abyss cross-section" style.
- **Story:** Dave Shaw's 2005 dive to recover Deon Dreyer from Boesmansgat, and what went wrong. Narrated by ElevenLabs "George", with an original dark score and sound design.
- **Ending:** it loops seamlessly from the last frame to the first.

## 1. Technical checks (`python3 audit_tech.py out/cave_reel.mp4`)

| Check | Result |
|---|---|
| Length under 3:00 | PASS, 95.1 s |
| 1080×1920, 30 fps, H.264 + AAC, yuv420p | PASS |
| Audio and video lengths match | PASS (95.10 s / 95.11 s) |
| Loudness about -14 LUFS | PASS, -14.1 LUFS |
| True peak below -1 dBTP | PASS, -1.6 dBTP |
| No silent gaps over 1.5 s | PASS |
| No black frames / no blank single-colour frames | PASS. The dark palette was lifted so every frame keeps visible structure (mean luma of at least 19/255). |
| Seamless loop | PASS (0.3/255; pixel-identical before encoding) |
| File size | PASS, 74 MB |

## 2. Facts

- **Sources:** every narrated claim traces to `RESEARCH.md`, which carries a confidence label and source for each fact:
  - Zimmermann, *Outside* (2005)
  - Mitchell & Cronjé et al., *Aviat Space Environ Med* 78:81 (2007), the peer-reviewed analysis of the helmet-camera video
  - IOL, Mail & Guardian and Globe and Mail reports from January 2005
- **Disputed numbers:** where sources disagree, the script uses the safe form. For example, it says "about 270 metres" for the depth of the body, and it doesn't claim whether he cut the line.
- **Myths deliberately avoided:**
  - he "ran out of air" (it was CO₂ build-up on a rebreather)
  - "the corpse pulled him down"
  - "the deepest cave in the world"
  - "the family asked him to go"
- **On-screen labels** all match the research:
  - 8 JAN 2005 (the dive)
  - 1994 → 2004
  - 6 MIN (the planned time at the bottom: about 5–6 minutes)
  - 12 HRS (about 12 hours of decompression)
  - CO₂
  - -250 m RIG FAILURE (Shirley's controller crushed at about 250 m)
  - 12 JAN 2005 (the documented recovery date)

## 3. Respect for real people

- **Depiction:** no faces, no bodies and no helmet-camera imagery. Deon is only ever a cold point of light, and the people are plain silhouettes.
- **Sound:** no breathing sounds in the scene where Shaw dies. On "He never came back up." the music drops to silence and the torch goes still.
- **CO₂ warning:** the "video-game health bar" the viewer test flagged was removed; a quiet warning chip remains.
- **Left out on purpose:** graphic details, private family moments and any blaming of individuals, as listed in RESEARCH.md.
- **Viewer verdict:** the cold viewer judged it "tasteful… no gore, no clickbait energy".

## 4. Viewer testing

- **First cut:** 7/10. The viewer watched to the end, liked it, and called the "stranger home" → "did bring Deon home" bookend the chills moment.
- **Fixes applied:**
  - a readable opening frame with the depth counter ticking
  - full-chunk captions that read on mute
  - a visible "Nobody asked him… He offered." scene
  - a shorter geography line
  - the reel ends on "Dave Shaw did bring Deon home." with no lecture afterwards
  - the line clearly wrapping the diver, with nothing graphic
  - a big year counter
  - confusing icons and stray depth readouts removed

## 5. Costs

- **This Reel:** about 1,900 ElevenLabs characters.
- **Remaining:** about 3,560 of 10,000 this month.
- **Re-renders:** all lines are cached in `audio/tts_cache/`, so re-renders are free unless wording changes.

## Before posting

1. Watch it once with sound. The machine checks can't judge the emotional pacing of the silence on "He never came back up."
2. Suggested caption credit: "Sources: Outside (2005), Aviat Space Environ Med (2007), contemporary SA news reports." This is optional, but it fits the factual framing.
